"""Segmented shrink: Z[rows, :RB] = X[rows] @ A[slot, :rank].T for each tile of one segment (HANDOFF 3.4).

Two modes (TASKS decision 2):
  uniform  every segment has N tokens; pid -> row = pid // cdiv(N, BM), rows [row*N + t*BM, row*N + N).
           Only row_slot [rows] is read; no tile tables.
  varlen   tile tables (tile_seg, tile_row0, tile_row_end) built on the host with a fixed BM (packed inputs).

Per-layer rank (decision 3): `rank` is the layer's int32 [S]; rank == 0 (layer not in that adapter) and
slot == -1 exit early. A is read only below the rank, so stale slot data is never touched. The rank loop spans
RB = kernel_rb(r_bucket) columns, a constexpr (decision 4).

Z comes from torch.empty: every row of an active tile is written (all RB columns, zeros beyond the rank), and
expand never reads rows of slot -1 / rank-0 rows or columns >= rank.

FP8 weights (FP8W): A is float8_e4m3fn with an fp32 scale per A row (L1.5 format), upcast in registers; the
row scale is applied to the accumulated column after the K loop.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._tcommon import INTERP_BK, INTERP_BM, configs, interpreting, kernel_rb


@triton.jit
def _shrink_kernel(X, A, Z, SA, rank_ptr, slot_ptr, tile_seg, tile_r0, tile_r1,
                   N, d_in, d_out, stride_xm, stride_as, stride_ar, stride_sas, stride_zm,
                   UNIFORM: tl.constexpr, FP8W: tl.constexpr, RB: tl.constexpr, BM: tl.constexpr,
                   BK: tl.constexpr):
    pid = tl.program_id(0)
    if UNIFORM:
        tpr = tl.cdiv(N, BM)
        row = pid // tpr
        r0 = row * N + (pid % tpr) * BM
        r1 = row * N + N
    else:
        row = tl.load(tile_seg + pid)
        r0 = tl.load(tile_r0 + pid)
        r1 = tl.load(tile_r1 + pid)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    rk = tl.load(rank_ptr + slot)
    if rk == 0:
        return
    slot64 = slot.to(tl.int64)
    c = tl.arange(0, RB)
    cmask = c < rk
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    rows64 = rows.to(tl.int64)
    acc = tl.zeros((BM, RB), dtype=tl.float32)
    for k in range(0, d_in, BK):
        kk = k + tl.arange(0, BK)
        kmask = kk < d_in
        x = tl.load(X + rows64[:, None] * stride_xm + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        a = tl.load(A + slot64 * stride_as + c[None, :] * stride_ar + kk[:, None],
                    mask=cmask[None, :] & kmask[:, None], other=0.0)
        acc += tl.dot(x, a.to(x.dtype))
    if FP8W:
        sa = tl.load(SA + slot64 * stride_sas + c, mask=cmask, other=0.0)
        acc = acc * sa[None, :]
    tl.store(Z + rows64[:, None] * stride_zm + c[None, :], acc.to(Z.dtype.element_ty), mask=rmask[:, None])


_KEY = ["d_in", "d_out", "RB"]
_shrink_uniform = triton.autotune(configs(bm=True, bn=False, bk=True), key=_KEY)(_shrink_kernel)
_shrink_varlen = triton.autotune(configs(bm=False, bn=False, bk=True), key=_KEY)(_shrink_kernel)


def _launch(x, A, z, sa, rank, row_slot, N, tiles, d_out, RB, fp8w):
    d_in = x.shape[1]
    if sa is None:
        sa, stride_sas = A, 0
    else:
        stride_sas = sa.stride(0)
    uniform = tiles is None
    if uniform:
        rows = row_slot.numel()
        tile_seg = tile_r0 = tile_r1 = row_slot          # unused in uniform mode
    else:
        tile_seg, tile_r0, tile_r1 = tiles.tile_seg, tiles.tile_row0, tiles.tile_row_end
        if tile_seg.numel() == 0:
            return z
    args = (x, A, z, sa, rank, row_slot, tile_seg, tile_r0, tile_r1, N, d_in, d_out,
            x.stride(0), A.stride(0), A.stride(1), stride_sas, z.stride(0))
    kw = {"UNIFORM": uniform, "FP8W": fp8w, "RB": RB}
    if interpreting():
        bm = INTERP_BM if uniform else tiles.bm
        grid = (rows * triton.cdiv(N, bm),) if uniform else (tile_seg.numel(),)
        _shrink_kernel[grid](*args, **kw, BM=bm, BK=INTERP_BK)
    elif uniform:
        _shrink_uniform[lambda META: (rows * triton.cdiv(N, META["BM"]),)](*args, **kw)
    else:
        _shrink_varlen[(tile_seg.numel(),)](*args, **kw, BM=tiles.bm)
    return z


def _check(x, A):
    if x.stride(-1) != 1 or A.stride(-1) != 1:
        raise ValueError("shrink needs unit inner stride on x and A")


def shrink(x, A, rank, row_slot, *, r_bucket: int, N: int = 0, tiles=None, z=None, z_dtype=torch.float32,
           a_scale=None, d_out: int = 0):
    """Z [T, RB] (torch.empty; rows of inactive tiles are left unwritten).

    x [T, d_in]; A [S, R, d_in] (any slot stride, unit inner stride); rank int32 [S] (this layer);
    row_slot int32 [rows]; uniform mode: N > 0 and tiles None (T == rows * N); varlen: tiles = TileMeta.
    a_scale [S, R] fp32 selects the FP8-weight variant. d_out only feeds the autotune key."""
    _check(x, A)
    RB = kernel_rb(r_bucket)
    T = x.shape[0]
    if z is None:
        z = torch.empty(T, RB, dtype=z_dtype, device=x.device)
    if T == 0:
        return z
    if tiles is None and (N <= 0 or T != row_slot.numel() * N):
        raise ValueError(f"uniform mode needs N > 0 and T == rows * N, got T={T}, rows={row_slot.numel()}, N={N}")
    return _launch(x, A, z, a_scale, rank, row_slot, N, tiles, d_out, RB, a_scale is not None)

