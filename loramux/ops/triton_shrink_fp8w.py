"""FP8-weight shrink: A stored float8_e4m3fn with one fp32 scale per A row (the L1.5 `quantize_fp8_rows` format,
A[s, j, :] ~ q[s, j, :] * a_scale[s, j]). The tile is upcast in registers and the row scale is applied to Z's
column j after the K loop. Same uniform/varlen modes, rank masks and early exits as `triton_shrink` (it is the
same kernel with FP8W=True)."""
from __future__ import annotations

import torch

from .triton_shrink import shrink


def shrink_fp8w(x, A_q, a_scale, rank, row_slot, *, r_bucket: int, N: int = 0, tiles=None, z=None,
                z_dtype=torch.float32, d_out: int = 0):
    """A_q [S, R, d_in] float8_e4m3fn, a_scale [S, R] fp32. See `triton_shrink.shrink`."""
    _check(A_q, a_scale)
    return shrink(x, A_q, rank, row_slot, r_bucket=r_bucket, N=N, tiles=tiles, z=z, z_dtype=z_dtype,
                  a_scale=a_scale, d_out=d_out)


def _check(q, scale):
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(f"FP8-weight kernel needs float8_e4m3fn weights, got {q.dtype}")
    if scale.dtype != torch.float32 or scale.shape != q.shape[:2] or scale.stride(-1) != 1:
        raise ValueError(f"scale must be fp32 {tuple(q.shape[:2])} with unit inner stride, got "
                         f"{scale.dtype} {tuple(scale.shape)}")
