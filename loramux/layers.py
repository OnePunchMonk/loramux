from __future__ import annotations

import re

import torch
from torch import nn

from . import ops  # noqa: F401  (registers the custom op)
from .context import current


class MuxLinear(nn.Module):
    """y = base(x), then add each row's own LoRA correction in place."""

    def __init__(self, base: nn.Linear, layer_key: str, pool):
        super().__init__()
        self.base = base
        self.layer_key = layer_key
        self.pool = pool

    def forward(self, x):
        y = self.base(x)
        m = current()
        if m is None or not m.any_lora or not self.pool.has_layer(self.layer_key):
            return y
        A, B, rank = self.pool.buffers(self.layer_key)
        v = m.view_for(x)
        t = v.tiles  # varlen mode until L2 adds the uniform-row kernel path
        torch.ops.loramux.lora_apply_(
            y.view(-1, y.shape[-1]), x.reshape(-1, x.shape[-1]), A, B, rank, v.row_slot, v.row_scale,
            t.tile_seg, t.tile_row0, t.tile_row_end, t.bm, v.seg_start, v.seg_len)
        return y


def patch_model(model: nn.Module, target_patterns: list[str], pool) -> list[str]:
    """Replace linears whose qualified name matches any regex; return their keys."""
    pats = [re.compile(p) for p in target_patterns]
    keys = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(p.search(full) for p in pats):
                setattr(mod, cname, MuxLinear(child, full, pool))
                keys.append(full)
    return keys


def unpatch_model(model: nn.Module) -> list[str]:
    """Undo patch_model: put the original linears back. Returns the restored keys."""
    keys = []
    for mod in list(model.modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, MuxLinear):
                setattr(mod, cname, child.base)
                keys.append(child.layer_key)
    return keys
