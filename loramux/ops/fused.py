"""loramux::lora_linear: base linear and every row's LoRA correction as one op (uniform-row mode, bf16/fp32 weights).

CUDA (or the interpreter with TRITON_ON_CPU): v1 shrink, then the fused GEMM+LoRA-epilogue kernel. Otherwise the
torch reference: F.linear then the in-place reference. Returns a new tensor [T, d_out]; no graph breaks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import custom_op
from .torch_ref import lora_apply_ref_

if custom_op.HAVE_TRITON:
    from .triton_gemm_lora import gemm_lora
    from .triton_shrink import shrink


@torch.library.custom_op("loramux::lora_linear", mutates_args=())
def lora_linear(x: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, A: torch.Tensor, B: torch.Tensor,
                rank: torch.Tensor, row_slot: torch.Tensor, row_scale: torch.Tensor, N: int, r_bucket: int,
                z_bf16: bool = False) -> torch.Tensor:
    if custom_op.use_triton(x):
        z = shrink(x, A, rank, row_slot, r_bucket=r_bucket, N=N, d_out=W.shape[0],
                   z_dtype=torch.bfloat16 if z_bf16 else torch.float32)
        return gemm_lora(x, W, bias, z, B, rank, row_slot, row_scale, N=N, r_bucket=r_bucket)
    y = F.linear(x, W, bias)
    lora_apply_ref_(y, x, A, B, rank, row_slot, row_scale, N=N, r_bucket=r_bucket)
    return y


@lora_linear.register_fake
def _(x, W, bias, A, B, rank, row_slot, row_scale, N, r_bucket, z_bf16=False):
    return x.new_empty(x.shape[0], W.shape[0])
