import torch
import triton
import triton.language as tl


@triton.jit
def _shrink_kernel(X, A, Z, rank_ptr, slot_ptr, tile_seg, tile_r0, tile_r1,
                   d_in, stride_xm, stride_as, stride_ar, stride_zm,
                   BM: tl.constexpr, RB: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    seg = tl.load(tile_seg + pid)
    slot = tl.load(slot_ptr + seg)
    if slot < 0:
        return
    r = tl.load(rank_ptr + slot)
    r0 = tl.load(tile_r0 + pid)
    r1 = tl.load(tile_r1 + pid)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    rc = tl.arange(0, RB)
    cmask = rc < r
    acc = tl.zeros((BM, RB), dtype=tl.float32)
    for k in range(0, d_in, BK):
        kk = k + tl.arange(0, BK)
        kmask = kk < d_in
        x = tl.load(X + rows[:, None] * stride_xm + kk[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        a = tl.load(A + slot * stride_as + rc[None, :] * stride_ar + kk[:, None],
                    mask=cmask[None, :] & kmask[:, None], other=0.0)
        acc += tl.dot(x, a)
    tl.store(Z + rows[:, None] * stride_zm + rc[None, :], acc, mask=rmask[:, None] & (rc[None, :] < RB))


def shrink(x, A, rank, slot_id, tiles, r_max, z=None, BK=64):
    T, d_in = x.shape
    if z is None:
        z = torch.zeros(T, r_max, dtype=torch.float32, device=x.device)
    n = tiles.tile_seg.numel()
    if n == 0:
        return z
    _shrink_kernel[(n,)](x, A, z, rank, slot_id, tiles.tile_seg, tiles.tile_row0, tiles.tile_row_end,
                         d_in, x.stride(0), A.stride(0), A.stride(1), z.stride(0),
                         BM=tiles.bm, RB=r_max, BK=BK)
    return z
