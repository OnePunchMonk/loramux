"""torch.library custom op so torch.compile sees one opaque node, no graph breaks.

loramux::lora_apply_(y, x, A, B, rank, row_slot, row_scale, N, r_bucket, seg_start=None, seg_len=None,
                     tile_seg=None, tile_row0=None, tile_row_end=None, bm=64, z_bf16=False) -> None

  Uniform-row mode: N > 0, the tile/segment tensors are None (rows = row_slot.numel(), T = rows * N).
  Varlen mode:      N == 0, seg_start/seg_len and the tile table are given (tile height bm).
  rank is this layer's int32 [S]; r_bucket the batch's host-side rank bucket (constexpr for the kernels).
  z_bf16 keeps the shrink workspace in bf16 instead of fp32.

CUDA tensors run the Triton kernels; CPU tensors run the torch reference (same semantics). With
TRITON_INTERPRET=1 and TRITON_ON_CPU = True, CPU tensors run the Triton kernels in the interpreter (tests).
"""
from __future__ import annotations

import torch

from .torch_ref import lora_apply_ref_

try:
    from ._tcommon import interpreting
    from .triton_expand_add import expand_add_
    from .triton_shrink import shrink
    HAVE_TRITON = True
except Exception:  # noqa: BLE001  CPU-only env without triton
    HAVE_TRITON = False

TRITON_ON_CPU = False   # tests flip this (with TRITON_INTERPRET=1) to send CPU tensors through the kernels


def use_triton(t: torch.Tensor) -> bool:
    return HAVE_TRITON and (t.is_cuda or (TRITON_ON_CPU and interpreting()))


def _tiles(tile_seg, tile_row0, tile_row_end, bm):
    if tile_seg is None:
        return None
    from ..context import TileMeta
    return TileMeta(tile_seg, tile_row0, tile_row_end, bm)


@torch.library.custom_op("loramux::lora_apply_", mutates_args=("y",))
def lora_apply_(y: torch.Tensor, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rank: torch.Tensor,
                row_slot: torch.Tensor, row_scale: torch.Tensor, N: int, r_bucket: int,
                seg_start: torch.Tensor | None = None, seg_len: torch.Tensor | None = None,
                tile_seg: torch.Tensor | None = None, tile_row0: torch.Tensor | None = None,
                tile_row_end: torch.Tensor | None = None, bm: int = 64, z_bf16: bool = False) -> None:
    if N <= 0 and (seg_start is None or tile_seg is None):
        raise ValueError("lora_apply_: varlen mode (N == 0) needs seg_start/seg_len and the tile table")
    if use_triton(y):
        tiles = None if N > 0 else _tiles(tile_seg, tile_row0, tile_row_end, bm)
        n = max(0, N)
        z = shrink(x, A, rank, row_slot, r_bucket=r_bucket, N=n, tiles=tiles,
                   z_dtype=torch.bfloat16 if z_bf16 else torch.float32, d_out=y.shape[1])
        expand_add_(y, z, B, rank, row_slot, row_scale, r_bucket=r_bucket, N=n, tiles=tiles, d_in=x.shape[1])
    elif N > 0:
        lora_apply_ref_(y, x, A, B, rank, row_slot, row_scale, N=N, r_bucket=r_bucket)
    else:
        lora_apply_ref_(y, x, A, B, rank, row_slot, row_scale, seg_start, seg_len, r_bucket=r_bucket)


@lora_apply_.register_fake
def _(y, x, A, B, rank, row_slot, row_scale, N, r_bucket, seg_start=None, seg_len=None, tile_seg=None,
      tile_row0=None, tile_row_end=None, bm=64, z_bf16=False):
    return None
