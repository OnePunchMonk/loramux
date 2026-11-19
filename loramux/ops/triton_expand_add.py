import torch
import triton
import triton.language as tl


@triton.jit
def _expand_kernel(Y, Z, B, rank_ptr, slot_ptr, scale_ptr, tile_seg, tile_r0, tile_r1,
                   d_out, stride_ym, stride_zm, stride_bs, stride_bn,
                   BM: tl.constexpr, BN: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    seg = tl.load(tile_seg + pid)
    slot = tl.load(slot_ptr + seg)
    if slot < 0:
        return
    r = tl.load(rank_ptr + slot)
    scale = tl.load(scale_ptr + seg)
    r0 = tl.load(tile_r0 + pid)
    r1 = tl.load(tile_r1 + pid)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < d_out
    rc = tl.arange(0, RB)
    z = tl.load(Z + rows[:, None] * stride_zm + rc[None, :], mask=rmask[:, None] & (rc[None, :] < r), other=0.0)
    b = tl.load(B + slot * stride_bs + cols[None, :] * stride_bn + rc[:, None],
                mask=cmask[None, :] & (rc[:, None] < r), other=0.0)
    acc = tl.dot(z.to(b.dtype), b, out_dtype=tl.float32) * scale
    yp = Y + rows[:, None] * stride_ym + cols[None, :]
    m = rmask[:, None] & cmask[None, :]
    y = tl.load(yp, mask=m, other=0.0)
    tl.store(yp, (y.to(tl.float32) + acc).to(y.dtype), mask=m)


def expand_add_(y, z, B, rank, slot_id, scale, tiles, r_max, BN=128):
    n = tiles.tile_seg.numel()
    if n == 0:
        return y
    grid = (n, triton.cdiv(y.shape[1], BN))
    _expand_kernel[grid](y, z, B, rank, slot_id, scale, tiles.tile_seg, tiles.tile_row0, tiles.tile_row_end,
                         y.shape[1], y.stride(0), z.stride(0), B.stride(0), B.stride(1),
                         BM=tiles.bm, BN=BN, RB=r_max)
    return y
