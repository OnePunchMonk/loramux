"""FP8-weight expand-add: B stored float8_e4m3fn [S, d_out, R] with one fp32 scale per B row (= per output
column of Y; the L1.5 `quantize_fp8_rows` format). Upcast in registers, scale applied after the dot. Same modes
and masks as `triton_expand_add` (it is the same kernel with FP8W=True)."""
from __future__ import annotations

import torch

from .triton_expand_add import expand_add_


def expand_add_fp8w_(y, z, B_q, b_scale, rank, row_slot, row_scale, *, r_bucket: int, N: int = 0, tiles=None,
                     d_in: int = 0):
    """B_q [S, d_out, R] float8_e4m3fn, b_scale [S, d_out] fp32. See `triton_expand_add.expand_add_`."""
    if B_q.dtype != torch.float8_e4m3fn:
        raise TypeError(f"FP8-weight kernel needs float8_e4m3fn weights, got {B_q.dtype}")
    if b_scale.dtype != torch.float32 or b_scale.shape != B_q.shape[:2] or b_scale.stride(-1) != 1:
        raise ValueError(f"b_scale must be fp32 {tuple(B_q.shape[:2])}, got {b_scale.dtype} {tuple(b_scale.shape)}")
    return expand_add_(y, z, B_q, rank, row_slot, row_scale, r_bucket=r_bucket, N=N, tiles=tiles, b_scale=b_scale,
                       d_in=d_in)
