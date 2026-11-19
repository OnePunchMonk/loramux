"""Per-step batch metadata. Everything is built from host lists, so no GPU sync."""
from __future__ import annotations
import torch
from .context import BatchLoRAMeta, TileMeta


def expand_cfg(slots, scales, layout: str, cfg: bool):
    """Per-request (slot, scale) -> per-row lists after CFG duplication.

    layout 'concat':      [cond_0..cond_n, uncond_0..uncond_n]
    layout 'interleaved': [cond_0, uncond_0, cond_1, uncond_1, ...]
    Both halves of a request use the same adapter.
    """
    if not cfg:
        return list(slots), list(scales)
    if layout == "concat":
        return list(slots) * 2, list(scales) * 2
    if layout == "interleaved":
        return [s for s in slots for _ in (0, 1)], [c for c in scales for _ in (0, 1)]
    raise ValueError(f"unknown CFG layout {layout!r}")


def build_tiles(seg_start, seg_len, bm: int, device) -> TileMeta:
    seg, r0, r1 = [], [], []
    for i, (s, n) in enumerate(zip(seg_start, seg_len)):
        for t in range(0, n, bm):
            seg.append(i)
            r0.append(s + t)
            r1.append(s + min(t + bm, n))
    mk = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    return TileMeta(mk(seg), mk(r0), mk(r1), bm)


def build_meta(slots, scales, tokens_per_row, *, cfg=False, layout="concat",
               seg_len=None, bm=64, device="cpu") -> BatchLoRAMeta:
    """slots[i] = pool slot of request i (-1 = base). One segment per row unless seg_len given."""
    slots, scales = expand_cfg(slots, scales, layout, cfg)
    n = len(slots)
    lens = list(seg_len) if seg_len is not None else [tokens_per_row] * n
    assert len(lens) == n
    starts, acc = [], 0
    for l in lens:
        starts.append(acc)
        acc += l
    mk = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return BatchLoRAMeta(
        seg_start=mk(starts, torch.int32),
        seg_len=mk(lens, torch.int32),
        slot_id=mk(slots, torch.int32),
        scale=mk(scales, torch.float32),
        tiles=build_tiles(starts, lens, bm, device),
        any_lora=any(s >= 0 for s in slots),
    )
