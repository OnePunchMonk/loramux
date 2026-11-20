"""Fused base GEMM + LoRA expand epilogue (HANDOFF 3.5, "v2"):

    Y[rows] = X[rows] @ W.T (+ bias) + scale_row * Z[rows, :r] @ B[slot_row, :, :r].T

Uniform-row mode, bf16/fp32 weights only. Each M-tile lies inside one segment so it reads exactly one adapter,
and the LoRA term joins the accumulator in registers before the single store: the read-modify-write of Y that
the v1 expand-add pays (HANDOFF 3.6) disappears. Z comes from the v1 shrink kernel. Slot -1 and rank-0 rows get
the base GEMM only. Rows of Z for such tiles may be unwritten (torch.empty) and are never read.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._tcommon import INTERP_BK, INTERP_BM, INTERP_BN, configs, interpreting, kernel_rb


@triton.jit
def _gemm_lora_kernel(X, W, BIAS, Y, Z, B, rank_ptr, row_slot_ptr, row_scale_ptr,
                      N, d_in, d_out, stride_xm, stride_wn, stride_ym, stride_zm, stride_bs, stride_bn,
                      HAS_BIAS: tl.constexpr, RB: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    tpr = tl.cdiv(N, BM)
    row = pid_m // tpr
    r0 = row * N + (pid_m % tpr) * BM
    rows = r0 + tl.arange(0, BM)
    rmask = rows < row * N + N
    rows64 = rows.to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    cmask = cols < d_out
    cols64 = cols.to(tl.int64)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, d_in, BK):
        kk = k + tl.arange(0, BK)
        x = tl.load(X + rows64[:, None] * stride_xm + kk[None, :], mask=rmask[:, None] & (kk[None, :] < d_in), other=0.0)
        w = tl.load(W + cols64[None, :] * stride_wn + kk[:, None], mask=cmask[None, :] & (kk[:, None] < d_in), other=0.0)
        acc += tl.dot(x, w)

    slot = tl.load(row_slot_ptr + row)
    if slot >= 0:
        r = tl.load(rank_ptr + slot)
        if r > 0:
            rc = tl.arange(0, RB)
            z = tl.load(Z + rows64[:, None] * stride_zm + rc[None, :], mask=rmask[:, None] & (rc[None, :] < r), other=0.0)
            b = tl.load(B + slot.to(tl.int64) * stride_bs + cols64[None, :] * stride_bn + rc[:, None],
                        mask=cmask[None, :] & (rc[:, None] < r), other=0.0)
            acc += tl.dot(z.to(b.dtype), b, out_dtype=tl.float32) * tl.load(row_scale_ptr + row)
    if HAS_BIAS:
        acc += tl.load(BIAS + cols, mask=cmask, other=0.0)[None, :].to(tl.float32)
    tl.store(Y + rows64[:, None] * stride_ym + cols[None, :], acc.to(Y.dtype.element_ty),
             mask=rmask[:, None] & cmask[None, :])


# Large tiles + the epilogue tiles can exceed shared memory; the autotuner skips configs that fail to compile.
_gemm_lora_tuned = triton.autotune(configs(bm=True, bn=True, bk=True), key=["d_in", "d_out", "RB", "N"])(_gemm_lora_kernel)


def gemm_lora(x, weight, bias, z, B, rank, row_slot, row_scale, *, N: int, r_bucket: int):
    """x [rows*N, d_in]; weight [d_out, d_in] (nn.Linear layout); z [rows*N, >= kernel_rb(r_bucket)] from shrink;
    B [S, d_out, R]; rank int32 [S] (this layer); row_slot int32 / row_scale fp32 [rows]. Returns Y [rows*N, d_out]."""
    T, d_in = x.shape
    d_out = weight.shape[0]
    if N <= 0 or T % N or row_slot.numel() != T // N:
        raise ValueError(f"gemm_lora: {T} tokens, N={N}, {row_slot.numel()} row slots do not describe uniform rows")
    for t, name in ((x, "x"), (weight, "weight"), (z, "z"), (B, "B")):
        if t.stride(-1) != 1:
            raise ValueError(f"gemm_lora: {name} must have unit inner stride")
    RB = kernel_rb(r_bucket)
    if z.shape[1] < RB:
        raise ValueError(f"gemm_lora: z has {z.shape[1]} columns, r_bucket {r_bucket} needs {RB}")
    y = torch.empty(T, d_out, device=x.device, dtype=x.dtype)
    if T == 0:
        return y
    args = (x, weight, bias if bias is not None else x, y, z, B, rank, row_slot, row_scale, N, d_in, d_out,
            x.stride(0), weight.stride(0), y.stride(0), z.stride(0), B.stride(0), B.stride(1))
    kw = {"HAS_BIAS": bias is not None, "RB": RB}
    rows = T // N
    if interpreting():
        _gemm_lora_kernel[(rows * triton.cdiv(N, INTERP_BM), triton.cdiv(d_out, INTERP_BN))](
            *args, **kw, BM=INTERP_BM, BN=INTERP_BN, BK=INTERP_BK)
    else:
        _gemm_lora_tuned[lambda m: (rows * triton.cdiv(N, m["BM"]), triton.cdiv(d_out, m["BN"]))](*args, **kw)
    return y
