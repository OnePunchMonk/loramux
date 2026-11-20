from __future__ import annotations

import re

import torch
from torch import nn

from . import ops  # noqa: F401  (registers the custom op)
from .context import current


class MuxLinear(nn.Module):
    """y = base(x), then add each row's own LoRA correction in place.

    Uniform-row inputs ([rows, N, d], or [rows, d] with per_row_2d) take the kernels' uniform mode with the
    meta's prebuilt per-row tensors; packed inputs (meta.seg_len) take varlen mode. Nothing here reads device
    memory or branches on per-request values, so a compiled forward has no host syncs (TASKS decision 5).
    """

    def __init__(self, base: nn.Linear, layer_key: str, pool, z_dtype: torch.dtype = torch.float32,
                 fused: bool = False):
        super().__init__()
        if z_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"z_dtype must be float32 or bfloat16, got {z_dtype}")
        self.base = base
        self.layer_key = layer_key
        self.pool = pool
        self.z_dtype = z_dtype
        self.fused = fused   # uniform-row inputs and unquantized pool weights run base GEMM + LoRA as one kernel

    def forward(self, x):
        m = current()
        if self.fused and m is not None and m.any_lora and self.pool.has_layer(self.layer_key) and x.numel():
            us = m.uniform_shape(x)
            if us is not None and getattr(self.pool, "scales", lambda k: None)(self.layer_key) is None:
                A, B, rank = self.pool.buffers(self.layer_key)
                rt = m.rows_for(us[0])
                y = torch.ops.loramux.lora_linear(x.reshape(-1, x.shape[-1]), self.base.weight, self.base.bias, A, B,
                                                  rank, rt.slot, rt.scale, us[1], m.r_bucket,
                                                  self.z_dtype == torch.bfloat16)
                return y.view(*x.shape[:-1], -1)
        y = self.base(x)
        if m is None or not m.any_lora or not self.pool.has_layer(self.layer_key) or x.numel() == 0:
            return y
        A, B, rank = self.pool.buffers(self.layer_key)
        sc = self.pool.scales(self.layer_key) if hasattr(self.pool, "scales") else None
        sa, sb = sc if sc is not None else (None, None)
        y2, x2 = y.view(-1, y.shape[-1]), x.reshape(-1, x.shape[-1])
        z_bf16 = self.z_dtype == torch.bfloat16
        us = m.uniform_shape(x)
        if us is not None:
            rows, N = us
            rt = m.rows_for(rows)
            torch.ops.loramux.lora_apply_(y2, x2, A, B, rank, rt.slot, rt.scale, N, m.r_bucket, z_bf16=z_bf16,
                                          a_scale=sa, b_scale=sb)
        else:
            v = m.view_for(x)
            t = v.tiles
            torch.ops.loramux.lora_apply_(y2, x2, A, B, rank, v.row_slot, v.row_scale, 0, m.r_bucket,
                                          v.seg_start, v.seg_len, t.tile_seg, t.tile_row0, t.tile_row_end, t.bm,
                                          z_bf16=z_bf16, a_scale=sa, b_scale=sb)
        return y


def patch_model(model: nn.Module, target_patterns: list[str], pool, **mux_kwargs) -> list[str]:
    """Replace linears whose qualified name matches any regex; return their keys. mux_kwargs go to MuxLinear."""
    pats = [re.compile(p) for p in target_patterns]
    keys = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(p.search(full) for p in pats):
                setattr(mod, cname, MuxLinear(child, full, pool, **mux_kwargs))
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
