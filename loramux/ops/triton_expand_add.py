"""Segmented expand-add, in place: Y[rows, n-tile] += scale[row] * Z[rows, :rank] @ B[slot, n-tile, :rank].T.

Same modes, early exits and rank masks as triton_shrink (TASKS decisions 2-4). Z needs a unit column stride
(any row stride). FP8W: B is float8_e4m3fn with an fp32 scale per B row (= per output column of Y), upcast in
registers and applied after the dot.
"""
from __future__ import annotations

import triton
import triton.language as tl

from ._tcommon import INTERP_BM, INTERP_BN, configs, interpreting, kernel_rb


@triton.jit
def _expand_kernel(Y, Z, B, SB, rank_ptr, slot_ptr, scale_ptr, tile_seg, tile_r0, tile_r1,
                   N, d_in, d_out, stride_ym, stride_zm, stride_bs, stride_bn, stride_sbs,
                   UNIFORM: tl.constexpr, FP8W: tl.constexpr, RB: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
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
    r = tl.load(rank_ptr + slot)
    if r == 0:
        return
    slot64 = slot.to(tl.int64)
    scale = tl.load(scale_ptr + row)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    rows64 = rows.to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < d_out
    rc = tl.arange(0, RB)
    kmask = rc < r
    z = tl.load(Z + rows64[:, None] * stride_zm + rc[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
    b = tl.load(B + slot64 * stride_bs + cols[None, :] * stride_bn + rc[:, None],
                mask=cmask[None, :] & kmask[:, None], other=0.0)
    cdt = Y.dtype.element_ty
    acc = tl.dot(z.to(cdt), b.to(cdt), out_dtype=tl.float32)
    if FP8W:
        sb = tl.load(SB + slot64 * stride_sbs + cols, mask=cmask, other=0.0)
        acc = acc * sb[None, :]
    acc = acc * scale
    yp = Y + rows64[:, None] * stride_ym + cols[None, :]
    m = rmask[:, None] & cmask[None, :]
    y = tl.load(yp, mask=m, other=0.0)
    tl.store(yp, (y.to(tl.float32) + acc).to(y.dtype), mask=m)


_KEY = ["d_in", "d_out", "RB"]
# Y is updated in place, so the autotuner must restore it between timing runs.
_expand_uniform = triton.autotune(configs(bm=True, bn=True, bk=False), key=_KEY, restore_value=["Y"])(_expand_kernel)
_expand_varlen = triton.autotune(configs(bm=False, bn=True, bk=False), key=_KEY, restore_value=["Y"])(_expand_kernel)


def expand_add_(y, z, B, rank, row_slot, row_scale, *, r_bucket: int, N: int = 0, tiles=None, b_scale=None,
                d_in: int = 0):
    """y [T, d_out] += per-row LoRA expand, in place. B [S, d_out, R] (unit inner stride); z [T, >=RB] from
    shrink (unit column stride); b_scale [S, d_out] fp32 selects the FP8-weight variant. Returns y."""
    if y.stride(-1) != 1 or z.stride(-1) != 1 or B.stride(-1) != 1:
        raise ValueError("expand needs unit inner stride on y, z and B")
    RB = kernel_rb(r_bucket)
    T, d_out = y.shape
    if T == 0:
        return y
    uniform = tiles is None
    if uniform:
        if N <= 0:
            raise ValueError("uniform mode needs N > 0 (or pass tiles for varlen mode)")
        rows = row_slot.numel()
        tile_seg = tile_r0 = tile_r1 = row_slot
    else:
        tile_seg, tile_r0, tile_r1 = tiles.tile_seg, tiles.tile_row0, tiles.tile_row_end
        if tile_seg.numel() == 0:
            return y
    fp8w = b_scale is not None
    sb, stride_sbs = (b_scale, b_scale.stride(0)) if fp8w else (B, 0)
    args = (y, z, B, sb, rank, row_slot, row_scale, tile_seg, tile_r0, tile_r1, N, d_in, d_out,
            y.stride(0), z.stride(0), B.stride(0), B.stride(1), stride_sbs)
    kw = {"UNIFORM": uniform, "FP8W": fp8w, "RB": RB}
    if interpreting():
        bm = INTERP_BM if uniform else tiles.bm
        n0 = rows * triton.cdiv(N, bm) if uniform else tile_seg.numel()
        _expand_kernel[(n0, triton.cdiv(d_out, INTERP_BN))](*args, **kw, BM=bm, BN=INTERP_BN)
    elif uniform:
        _expand_uniform[lambda META: (rows * triton.cdiv(N, META["BM"]), triton.cdiv(d_out, META["BN"]))](*args, **kw)
    else:
        _expand_varlen[lambda META: (tile_seg.numel(), triton.cdiv(d_out, META["BN"]))](*args, **kw, BM=tiles.bm)
    return y
