from __future__ import annotations

import re

import torch
from torch import nn

from . import ops  # noqa: F401  (registers the custom op)
from .context import current
from .ops import custom_op


class MuxLinear(nn.Module):
    """y = base(x), then add each row's own LoRA correction in place.

    Uniform-row inputs ([rows, N, d], or [rows, d] with per_row_2d) take the kernels' uniform mode with the
    meta's prebuilt per-row tensors; packed inputs (meta.seg_len) take varlen mode. Nothing here reads device
    memory or branches on per-request values, so a compiled forward has no host syncs (TASKS decision 5).

    Grouped shrink: when the pool groups this layer with siblings that share x (pool.add_group), eager mode
    computes Z for the whole group in one pass and the siblings reuse it. Under torch.compile (and on the CPU
    reference path) every linear does its own shrink through the custom op.
    """

    def __init__(self, base: nn.Linear, layer_key: str, pool, z_dtype: torch.dtype = torch.float32):
        super().__init__()
        if z_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"z_dtype must be float32 or bfloat16, got {z_dtype}")
        self.base = base
        self.layer_key = layer_key
        self.pool = pool
        self.z_dtype = z_dtype

    def forward(self, x):
        y = self.base(x)
        m = current()
        if m is None or not m.any_lora or not self.pool.has_layer(self.layer_key) or x.numel() == 0:
            return y
        A, B, rank = self.pool.buffers(self.layer_key)
        y2, x2 = y.view(-1, y.shape[-1]), x.reshape(-1, x.shape[-1])
        z_bf16 = self.z_dtype == torch.bfloat16
        us = m.uniform_shape(x)
        if us is not None:
            rows, N = us
            rt = m.rows_for(rows)
            if not torch.compiler.is_compiling() and self._grouped(m, x, x2, y2, B, rank, rt, N):
                return y
            torch.ops.loramux.lora_apply_(y2, x2, A, B, rank, rt.slot, rt.scale, N, m.r_bucket, z_bf16=z_bf16)
        else:
            v = m.view_for(x)
            t = v.tiles
            torch.ops.loramux.lora_apply_(y2, x2, A, B, rank, v.row_slot, v.row_scale, 0, m.r_bucket,
                                          v.seg_start, v.seg_len, t.tile_seg, t.tile_row0, t.tile_row_end, t.bm,
                                          z_bf16=z_bf16)
        return y

    def _grouped(self, m, x, x2, y2, B, rank, rt, N) -> bool:
        group_of = getattr(self.pool, "group_of", None)
        gi = group_of(self.layer_key) if group_of is not None else None
        if gi is None or not custom_op.use_triton(x2):
            return False
        from .ops.triton_expand_add import expand_add_
        from .ops.triton_shrink import kernel_rb, shrink_grouped
        name, idx, G = gi
        ent = m._zcache.get(name)
        if ent is None or ent[0] is not x or ent[1] != x._version:
            A_cat, ranks = self.pool.group_buffers(name)
            z = shrink_grouped(x2, A_cat, ranks, rt.slot, G=G, r_bucket=m.r_bucket, N=N, z_dtype=self.z_dtype,
                               d_out=y2.shape[1])
            ent = m._zcache[name] = [x, x._version, z, set()]
        ent[3].add(idx)
        RB = kernel_rb(m.r_bucket)
        expand_add_(y2, ent[2][:, idx * RB:(idx + 1) * RB], B, rank, rt.slot, rt.scale, r_bucket=m.r_bucket, N=N,
                    d_in=x2.shape[1])
        if len(ent[3]) == G:          # every member consumed it: drop the workspace (and the reference to x)
            del m._zcache[name]
        return True


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
