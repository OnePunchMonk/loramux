"""torch.library custom op so torch.compile sees one opaque node, no graph breaks."""
import torch
from .torch_ref import lora_apply_ref_

try:
    from .triton_shrink import shrink
    from .triton_expand_add import expand_add_
    HAVE_TRITON = True
except Exception:  # CPU-only env
    HAVE_TRITON = False


@torch.library.custom_op("loramux::lora_apply_", mutates_args=("y",))
def lora_apply_(y: torch.Tensor, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rank: torch.Tensor,
                slot_id: torch.Tensor, scale: torch.Tensor, tile_seg: torch.Tensor, tile_row0: torch.Tensor,
                tile_row_end: torch.Tensor, bm: int, seg_start: torch.Tensor, seg_len: torch.Tensor) -> None:
    if HAVE_TRITON and y.is_cuda:
        from ..context import TileMeta
        tiles = TileMeta(tile_seg, tile_row0, tile_row_end, bm)
        r_max = A.shape[1]
        z = shrink(x, A, rank, slot_id, tiles, r_max)
        expand_add_(y, z, B, rank, slot_id, scale, tiles, r_max)
    else:
        lora_apply_ref_(y, x, A, B, rank, slot_id, scale, seg_start, seg_len)


@lora_apply_.register_fake
def _(y, x, A, B, rank, slot_id, scale, tile_seg, tile_row0, tile_row_end, bm, seg_start, seg_len):
    return None
