"""Scatter one staged adapter payload into its pool slot in a single launch (HANDOFF 3.8, 11.1).

The staging buffer holds the descriptor table, the per-layer ranks, the weights (native rank, blob order) and,
for FP8 pools, the per-row scales. Each descriptor row is

    (src_off, dst_off, rows, cols, src_ld, dst_ld, kind)      int64, offsets/strides in elements

and copies a `rows x cols` block from `src[src_off + i*src_ld + j]` to `dst[dst_off + i*dst_ld + j]`, where the
(src, dst) pair is chosen by `kind`: 0 weights -> arena, 1 fp32 scales -> scale arena, 2 int32 ranks -> rank
table. Values are cast to the destination dtype (a bf16 blob into an fp32 pool, for instance).

`scatter` runs the Triton kernel for CUDA tensors (and for CPU tensors under the interpreter when
custom_op.TRITON_ON_CPU is set); otherwise `scatter_torch` applies the same table with torch copies.
"""
from __future__ import annotations

import torch

KIND_W, KIND_S, KIND_R = 0, 1, 2
NCOL = 7
BR, BC = 32, 128

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _scatter_kernel(desc, srcW, dstW, srcS, dstS, srcR, dstR, BR: tl.constexpr, BC: tl.constexpr):
        d = tl.program_id(0)
        rb = tl.program_id(1)
        base = desc + d * 7
        src_off = tl.load(base)
        dst_off = tl.load(base + 1)
        rows = tl.load(base + 2)
        cols = tl.load(base + 3)
        src_ld = tl.load(base + 4)
        dst_ld = tl.load(base + 5)
        kind = tl.load(base + 6)
        r0 = rb * BR
        if r0 >= rows:
            return
        rr = r0 + tl.arange(0, BR).to(tl.int64)
        rm = rr < rows
        for c0 in range(0, cols, BC):
            cc = c0 + tl.arange(0, BC).to(tl.int64)
            m = rm[:, None] & (cc < cols)[None, :]
            so = src_off + rr[:, None] * src_ld + cc[None, :]
            do = dst_off + rr[:, None] * dst_ld + cc[None, :]
            if kind == 0:
                vw = tl.load(srcW + so, mask=m)
                tl.store(dstW + do, vw.to(dstW.dtype.element_ty), mask=m)
            elif kind == 1:
                vs = tl.load(srcS + so, mask=m)
                tl.store(dstS + do, vs.to(dstS.dtype.element_ty), mask=m)
            else:
                vr = tl.load(srcR + so, mask=m)
                tl.store(dstR + do, vr.to(dstR.dtype.element_ty), mask=m)

    HAVE_TRITON = True
except Exception:  # noqa: BLE001  no triton
    HAVE_TRITON = False


def scatter_torch(desc_rows, srcW, dstW, srcS, dstS, srcR, dstR) -> None:
    """Apply the descriptor table (host list of 7-tuples) with torch copies. All tensors are 1-D."""
    for so, do, rows, cols, sld, dld, kind in desc_rows:
        src, dst = ((srcW, dstW), (srcS, dstS), (srcR, dstR))[kind]
        s = src.as_strided((rows, cols), (sld, 1), src.storage_offset() + so)
        t = dst.as_strided((rows, cols), (dld, 1), dst.storage_offset() + do)
        t.copy_(s)


def scatter(desc_dev, desc_rows, srcW, dstW, srcS, dstS, srcR, dstR, *, use_triton: bool) -> None:
    """One launch over every descriptor. desc_dev: int64 [n, 7] on the tensors' device (the staged copy);
    desc_rows: the same table on the host (sizes the grid; used by the torch path). srcS/dstS may be None
    when the table has no scale rows."""
    if not desc_rows:
        return
    if srcS is None:
        srcS, dstS = srcW, dstW
    if not (use_triton and HAVE_TRITON):
        scatter_torch(desc_rows, srcW, dstW, srcS, dstS, srcR, dstR)
        return
    max_rows = max(r[2] for r in desc_rows)
    grid = (len(desc_rows), triton.cdiv(max_rows, BR))
    _scatter_kernel[grid](desc_dev, srcW, dstW, srcS, dstS, srcR, dstR, BR=BR, BC=BC)


def desc_tensor(desc_rows) -> torch.Tensor:
    return torch.tensor(desc_rows, dtype=torch.int64).reshape(-1, NCOL)
