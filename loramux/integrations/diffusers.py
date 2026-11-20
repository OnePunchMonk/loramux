"""diffusers integration: patch a pipeline's transformer and set per-step batch metadata with a forward pre-hook.

    pool = AdapterPool({}, slots=8, r_max=64, device="cuda")
    enable(pipe, pool, "attn")                 # registers layer shapes in the pool, wraps the linears
    pool.register_blob("a", "a.lmx"); pool.ensure_resident("a"); pool.wait("a")  # acquire needs it loaded
    with mixed_lora(pipe, [pool.acquire("a"), -1], [1.0, 1.0]):
        pipe(...)                              # request 0 uses adapter "a", request 1 the base model
    disable(pipe)

CFG is detected from the batch dimension of the transformer's `hidden_states` at call time: B == n_req is a plain
call (this also covers pipelines that call the transformer twice, cond then uncond, as Qwen-Image does), and
B == 2 * n_req is a CFG-duplicated batch (diffusers' torch.cat([x] * 2) = layout "concat", or "interleaved").
Anything else raises. A fresh BatchLoRAMeta is built for every transformer call, so its cached segment views
live for exactly one denoise step.
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import torch
from torch import nn

from ..layers import patch_model, unpatch_model
from ..metadata import build_meta

_ATTN = r"(^|\.)attn\."
_STATE = "_loramux_state"


def _model(pipe) -> nn.Module:
    if isinstance(pipe, nn.Module):
        return pipe
    tr = getattr(pipe, "transformer", None)
    if tr is None:
        raise ValueError("pipeline has no .transformer; pass the transformer module itself")
    return tr


def _patterns(targets) -> list[str]:
    if targets == "attn":
        return [_ATTN]
    if targets == "all":
        return [""]
    if isinstance(targets, str):
        raise TypeError(f"targets must be 'attn', 'all' or a list of regexes, got {targets!r}")
    return list(targets)


def enable(pipe, pool, targets="attn") -> list[str]:
    """Wrap the targeted nn.Linear modules of pipe.transformer; returns their keys.

    targets: "attn" (linears under an `.attn.` module), "all" (every nn.Linear, including embedders and
    modulation), or a list of regexes searched against the qualified module name. The pool's layer shapes are
    registered from the model first, so adapters can be registered right after."""
    model = _model(pipe)
    if getattr(pipe, _STATE, None) is not None or getattr(model, _STATE, None) is not None:
        raise RuntimeError("loramux is already enabled on this pipeline; call disable() first")
    pats = [re.compile(p) for p in _patterns(targets)]
    shapes = {n: (m.in_features, m.out_features) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and any(p.search(n) for p in pats)}
    if not shapes:
        raise ValueError(f"targets {targets!r} match no nn.Linear in the model")
    pool.add_layers(shapes)
    keys = patch_model(model, ["^(?:" + "|".join(re.escape(k) for k in shapes) + ")$"], pool)
    setattr(model, _STATE, {"pool": pool, "keys": keys})
    return keys


def disable(pipe) -> list[str]:
    """Restore the original linears (the same module objects, so outputs are bit-exact)."""
    model = _model(pipe)
    if getattr(model, _STATE, None) is None:
        return []
    keys = unpatch_model(model)
    delattr(model, _STATE)
    return keys


@contextmanager
def mixed_lora(pipe, req_slots, req_scales, *, cfg_layout: str = "concat"):
    """Per-request adapters for every transformer call inside the block.

    req_slots[i]: pool slot of request i (-1 = base model); req_scales[i]: its LoRA strength."""
    from .. import context

    model = _model(pipe)
    state = getattr(model, _STATE, None)
    if state is None:
        raise RuntimeError("call enable(pipe, pool) first")
    pool = state["pool"]
    slots, scales = list(req_slots), [float(s) for s in req_scales]
    n = len(slots)
    saved: list = []

    # The hooks run eagerly even when the transformer is torch.compile'd: they only build host-side metadata,
    # and tracing them would guard on every slot value (a recompile per batch composition).
    @torch.compiler.disable
    def pre(module, args, kwargs):
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        b = hs.shape[0]
        if b not in (n, 2 * n):
            raise ValueError(f"transformer batch is {b} rows but mixed_lora has {n} requests; "
                             f"expected {n} or {2 * n} (CFG)")
        meta = build_meta(slots, scales, cfg_layout=cfg_layout, rank_of_slot=pool.slot_rank, r_max=pool.r_max,
                          device=pool.device, per_row_2d=True)
        saved.append(context._current)
        context._current = meta

    @torch.compiler.disable
    def post(module, args, kwargs, output):
        if saved:
            context._current = saved.pop()

    h1 = model.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = model.register_forward_hook(post, with_kwargs=True, always_call=True)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()
        while saved:                       # a pre-hook that ran but whose forward never completed
            context._current = saved.pop()
