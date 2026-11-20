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

Grouped shrink (G > 1): G linears that share X, with A concatenated along rank (the pool's group buffers,
A_cat [S, G*R, d_in], ranks [G, S]). One pass over X computes Z [T, G*RB]; member g is Z[:, g*RB:(g+1)*RB].

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
                   N, d_in, d_out, R, stride_xm, stride_as, stride_ar, stride_sas, stride_rg, stride_zm,
                   UNIFORM: tl.constexpr, FP8W: tl.constexpr, G: tl.constexpr, GRB: tl.constexpr,
                   RB: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
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
    c = tl.arange(0, GRB)
    g = c // RB
    rr = c % RB
    cvalid = c < G * RB
    rk = tl.load(rank_ptr + g * stride_rg + slot, mask=cvalid, other=0)
    if tl.max(rk, axis=0) == 0:
        return
    slot64 = slot.to(tl.int64)
    cmask = cvalid & (rr < rk)
    arow = g * R + rr
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    rows64 = rows.to(tl.int64)
    acc = tl.zeros((BM, GRB), dtype=tl.float32)
    for k in range(0, d_in, BK):
        kk = k + tl.arange(0, BK)
        kmask = kk < d_in
        x = tl.load(X + rows64[:, None] * stride_xm + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        a = tl.load(A + slot64 * stride_as + arow[None, :] * stride_ar + kk[:, None],
                    mask=cmask[None, :] & kmask[:, None], other=0.0)
        acc += tl.dot(x, a.to(x.dtype))
    if FP8W:
        sa = tl.load(SA + slot64 * stride_sas + arow, mask=cmask, other=0.0)
        acc = acc * sa[None, :]
    tl.store(Z + rows64[:, None] * stride_zm + c[None, :], acc.to(Z.dtype.element_ty),
             mask=rmask[:, None] & cvalid[None, :])


_KEY = ["d_in", "d_out", "RB"]
_shrink_uniform = triton.autotune(configs(bm=True, bn=False, bk=True), key=_KEY)(_shrink_kernel)
_shrink_varlen = triton.autotune(configs(bm=False, bn=False, bk=True), key=_KEY)(_shrink_kernel)


def _launch(x, A, z, sa, rank, row_slot, N, tiles, d_out, R, stride_rg, G, RB, fp8w):
    d_in = x.shape[1]
    if sa is None:
        sa, stride_sas = A, 0
    else:
        stride_sas = sa.stride(0)
    grb = 1 << (G * RB - 1).bit_length()
    uniform = tiles is None
    if uniform:
        rows = row_slot.numel()
        tile_seg = tile_r0 = tile_r1 = row_slot          # unused in uniform mode
    else:
        tile_seg, tile_r0, tile_r1 = tiles.tile_seg, tiles.tile_row0, tiles.tile_row_end
        if tile_seg.numel() == 0:
            return z
    args = (x, A, z, sa, rank, row_slot, tile_seg, tile_r0, tile_r1, N, d_in, d_out, R,
            x.stride(0), A.stride(0), A.stride(1), stride_sas, stride_rg, z.stride(0))
    kw = {"UNIFORM": uniform, "FP8W": fp8w, "G": G, "GRB": grb, "RB": RB}
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
    if tiles is None and N <= 0:
        raise ValueError("uniform mode needs N > 0 (or pass tiles for varlen mode)")
    return _launch(x, A, z, a_scale, rank, row_slot, N, tiles, d_out, A.shape[1], 0, 1, RB, a_scale is not None)


def shrink_grouped(x, A_cat, ranks, row_slot, *, G: int, r_bucket: int, N: int = 0, tiles=None,
                   z_dtype=torch.float32, a_scale=None, d_out: int = 0):
    """One pass over x for G linears sharing it. A_cat [S, G*R, d_in]; ranks int32 [G, S] (row stride any);
    a_scale [S, G*R] for FP8. Returns Z [T, G*RB]; member g reads columns [g*RB, (g+1)*RB)."""
    _check(x, A_cat)
    if A_cat.shape[1] % G:
        raise ValueError(f"A_cat rank dim {A_cat.shape[1]} is not a multiple of G={G}")
    RB = kernel_rb(r_bucket)
    T = x.shape[0]
    z = torch.empty(T, G * RB, dtype=z_dtype, device=x.device)
    if T == 0:
        return z
    if tiles is None and N <= 0:
        raise ValueError("uniform mode needs N > 0 (or pass tiles for varlen mode)")
    return _launch(x, A_cat, z, a_scale, ranks, row_slot, N, tiles, d_out, A_cat.shape[1] // G, ranks.stride(0),
                   G, RB, a_scale is not None)
