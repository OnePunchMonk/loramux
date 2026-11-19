from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import torch


@dataclass
class TileMeta:
    tile_seg: torch.Tensor      # int32 [n_tiles]
    tile_row0: torch.Tensor     # int32 [n_tiles]
    tile_row_end: torch.Tensor  # int32 [n_tiles]
    bm: int


@dataclass
class BatchLoRAMeta:
    seg_start: torch.Tensor   # int32 [nseg]
    seg_len: torch.Tensor     # int32 [nseg]
    slot_id: torch.Tensor     # int32 [nseg], -1 = base model
    scale: torch.Tensor       # fp32 [nseg]
    tiles: TileMeta | None
    any_lora: bool            # host-side flag, decided when the batch is formed


_current: BatchLoRAMeta | None = None


@contextmanager
def lora_batch(meta: BatchLoRAMeta):
    global _current
    prev, _current = _current, meta
    try:
        yield meta
    finally:
        _current = prev


def current() -> BatchLoRAMeta | None:
    return _current
