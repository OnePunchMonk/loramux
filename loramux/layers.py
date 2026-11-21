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
                 fused: bool = False, in_slice: tuple[int, int] | None = None,
                 out_slice: tuple[int, int] | None = None):
        super().__init__()
        if z_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"z_dtype must be float32 or bfloat16, got {z_dtype}")
        self.base = base
        self.layer_key = layer_key
        self.pool = pool
        self.z_dtype = z_dtype
        self.fused = fused   # uniform-row inputs and unquantized pool weights run base GEMM + LoRA as one kernel
        # Column windows for fused projections where the adapter covers only part of the layer, e.g. FLUX.2
        # single-stream `to_qkv_mlp_proj` (attention LoRA = the q/k/v output columns only) and `to_out` (the
        # attention input columns only). The pool stores this layer at the window's width; the kernels run on
        # strided views, so the untouched columns cost nothing.
        self.in_slice, self.out_slice = in_slice, out_slice

        # Resolve the layer once. Forward must only read tensor attributes so repeated blocks can share a graph.
        A = B = rank = a_scale = b_scale = None
        if pool.has_layer(layer_key):
            A, B, rank = pool.buffers(layer_key)
            scales = pool.scales(layer_key) if hasattr(pool, "scales") else None
            if scales is not None:
                a_scale, b_scale = scales
        self.register_buffer("pool_A", A, persistent=False)
        self.register_buffer("pool_B", B, persistent=False)
        self.register_buffer("pool_rank", rank, persistent=False)
        self.register_buffer("pool_a_scale", a_scale, persistent=False)
        self.register_buffer("pool_b_scale", b_scale, persistent=False)
        self._has_pool_layer = A is not None
        bind = getattr(pool, "bind", None)
        if callable(bind):
            bind(self, layer_key)

    def forward(self, x):
        m = current()
        if self.fused and self.in_slice is None and self.out_slice is None and m is not None and m.any_lora and self._has_pool_layer and x.numel():
            us = m.uniform_shape(x)
            if us is not None and self.pool_a_scale is None:
                rt = m.rows_for(us[0])
                y = torch.ops.loramux.lora_linear(
                    x.reshape(-1, x.shape[-1]).contiguous(), self.base.weight, self.base.bias,
                    self.pool_A, self.pool_B, self.pool_rank, rt.slot, rt.scale, us[1], m.r_bucket,
                    self.z_dtype == torch.bfloat16)
                return y.view(*x.shape[:-1], -1)
        y = self.base(x)
        if m is None or not m.any_lora or not self._has_pool_layer or x.numel() == 0:
            return y
        A, B, rank = self.pool_A, self.pool_B, self.pool_rank
        sa, sb = self.pool_a_scale, self.pool_b_scale
        y2, x2 = y.view(-1, y.shape[-1]), x.reshape(-1, x.shape[-1])
        if self.out_slice is not None:
            y2 = y2[:, self.out_slice[0]:self.out_slice[1]]
        if self.in_slice is not None:
            x2 = x2[:, self.in_slice[0]:self.in_slice[1]]
        if x2.stride(-1) != 1:   # e.g. FLUX.2 packed latents reach x_embedder as a permuted view
            x2 = x2.contiguous()
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


def select_linears(model: nn.Module, patterns: list[str]) -> dict[str, tuple[int, int]]:
    """{qualified name: (d_in, d_out)} of the nn.Linear modules whose name matches any regex (re.search)."""
    pats = [re.compile(p) for p in patterns]
    return {n: (m.in_features, m.out_features) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and any(p.search(n) for p in pats)}


def patch_keys(model: nn.Module, keys, pool, slices=None, **mux_kwargs) -> list[str]:
    """patch_model on exactly these qualified names."""
    return patch_model(model, ["^(?:" + "|".join(re.escape(k) for k in keys) + ")$"], pool, slices=slices,
                       **mux_kwargs)


def patch_model(model: nn.Module, target_patterns: list[str], pool, slices=None, **mux_kwargs) -> list[str]:
    """Replace linears whose qualified name matches any regex; return their keys. mux_kwargs go to MuxLinear.
    slices: optional {key: (in_slice, out_slice)} column windows (see MuxLinear)."""
    pats = [re.compile(p) for p in target_patterns]
    matches = []
    layer_shapes = {}
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(p.search(full) for p in pats):
                ins, outs = (slices or {}).get(full, (None, None))
                d_in = (ins[1] - ins[0]) if ins is not None else child.in_features
                d_out = (outs[1] - outs[0]) if outs is not None else child.out_features
                matches.append((name, cname, full, child, ins, outs))
                layer_shapes[full] = (d_in, d_out)
    if hasattr(pool, "add_layers"):
        pool.add_layers(layer_shapes)
    keys = []
    modules = dict(model.named_modules())
    for name, cname, full, child, ins, outs in matches:
        setattr(modules[name], cname, MuxLinear(child, full, pool, in_slice=ins, out_slice=outs, **mux_kwargs))
        keys.append(full)
    return keys


def unpatch_model(model: nn.Module) -> list[str]:
    """Undo patch_model: put the original linears back. Returns the restored keys."""
    keys = []
    for mod in list(model.modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, MuxLinear):
                unbind = getattr(child.pool, "unbind", None)
                if callable(unbind):
                    unbind(child)
                setattr(mod, cname, child.base)
                keys.append(child.layer_key)
    return keys
