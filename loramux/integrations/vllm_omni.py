"""vLLM-Omni mixed-LoRA mode (L3.3): requests with different adapters share one denoise batch.

Four pieces, none of which needs vllm_omni at import time:
  1. `drop_lora_identity` / `patch_scheduler_keys`: the batch-compatibility keys stop splitting on
     lora_int_id / lora_scale, so the scheduler co-batches any adapters.
  2. `patch_vllm_linears`: vLLM linears (forward returns `(out, bias)`; fused QKV / add_kv_proj included) get a
     MuxLinear wrapper that adds each row's LoRA correction to `out`.
  3. `MixedLoRA`: resolves a request's LoRARequest to a pool key (loads and normalizes the adapter on first sight,
     composes fused groups from `model.packed_modules_mapping`, member dims from `output_sizes`), prefetches it,
     pins it for the request's lifetime, and builds the BatchLoRAMeta for a batch from request order.
  4. `MixedLoRA.attach(worker)`: replaces the worker's LoRA activation, so `set_request_adapter` is never called,
     and wraps `pipeline.denoise_step` (step mode) and `pipeline.forward` (request mode).
Tensor parallel size 1 only. Rows of a request repeat `num_outputs_per_prompt` times.
"""
from __future__ import annotations

import dataclasses
import re
import sys
from contextlib import contextmanager

import torch
from torch import nn

from ..context import lora_batch
from ..formats import load_lora, normalize
from ..formats.keymaps.generic import make_keymap
from ..layers import MuxLinear
from ..metadata import build_meta


def drop_lora_identity(key):
    """A batch key with the adapter identity neutralised (frozen dataclass in, new instance out)."""
    return dataclasses.replace(key, lora_int_id=None, lora_scale=1.0)


def patch_scheduler_keys() -> None:  # pragma: no cover - needs vllm_omni
    """Make the step and request batch keys ignore the adapter. Idempotent."""
    from vllm_omni.diffusion.sched import base_scheduler, request_scheduler

    orig_step = base_scheduler.BaseScheduler._build_sampling_params_key
    if not getattr(orig_step, "_loramux", False):
        def step_key(self, request):
            return drop_lora_identity(orig_step(self, request))
        step_key._loramux = True
        base_scheduler.BaseScheduler._build_sampling_params_key = step_key
    orig_req = request_scheduler.build_request_batch_sampling_params_key
    if not getattr(orig_req, "_loramux", False):
        def req_key(request):
            return drop_lora_identity(orig_req(request))
        req_key._loramux = True
        for mod in list(sys.modules.values()):   # also the names imported elsewhere (DP dispatch)
            if getattr(mod, "build_request_batch_sampling_params_key", None) is orig_req:
                mod.build_request_batch_sampling_params_key = req_key


def _is_vllm_linear(m: nn.Module) -> bool:
    return not isinstance(m, MuxLinear) and all(hasattr(m, a) for a in ("input_size", "output_size", "weight"))


class _Unpack(nn.Module):
    """Presents a vLLM linear (returns `(out, bias)`) as a tensor-returning module; remembers the bias slot."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner, self.tail = inner, None
        self.weight, self.bias = inner.weight, None

    def forward(self, x):
        out = self.inner(x)
        if isinstance(out, tuple):
            out, self.tail = out
            return out
        self.tail = None
        return out


class VLLMMuxLinear(MuxLinear):
    def __init__(self, inner: nn.Module, key: str, pool):
        self.returns_tuple = None
        super().__init__(_Unpack(inner), key, pool)

    def forward(self, x):
        y = super().forward(x)
        return (y, self.base.tail) if self.base.tail is not None or self.returns_tuple else y


def patch_vllm_linears(model: nn.Module, pool, targets: list[str]) -> list[str]:
    pats = [re.compile(p) for p in targets]
    keys, dims = [], {}
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if _is_vllm_linear(child) and any(p.search(full) for p in pats):
                if getattr(child, "output_size_per_partition", child.output_size) != child.output_size:
                    raise NotImplementedError(f"{full}: tensor parallel linears are not supported in mixed mode")
                dims[full] = (child.input_size, child.output_size)
                w = VLLMMuxLinear(child, full, pool)
                w.returns_tuple = getattr(child, "return_bias", True)
                setattr(mod, cname, w)
                keys.append(full)
    pool.add_layers({k: (dims[k][0], dims[k][1]) for k in keys})
    return keys


def fused_groups(model: nn.Module) -> dict[str, tuple[list[str], list[int]]]:
    """{fused module name: (member names in output order, member output sizes)} from packed_modules_mapping."""
    packed = getattr(model, "packed_modules_mapping", {}) or {}
    out = {}
    for name, mod in model.named_modules():
        prefix, _, leaf = name.rpartition(".")
        if leaf in packed and _is_vllm_linear(mod):
            members = [f"{prefix}.{m}" if prefix else m for m in packed[leaf]]
            sizes = list(getattr(mod, "output_sizes", None) or [mod.output_size // len(members)] * len(members))
            out[name] = (members, sizes)
    return out


class MixedLoRA:
    def __init__(self, model: nn.Module, pool, targets: list[str], load_fn=load_lora):
        self.pool, self.load_fn = pool, load_fn
        self.groups = fused_groups(model)
        member_of = {m for ms, _ in self.groups.values() for m in ms}
        self.keys = patch_vllm_linears(model, pool, targets)
        names = [k for k in self.keys if k not in self.groups] + sorted(member_of)
        self.keymap = make_keymap(names)
        self._req: dict[str, tuple[object, float]] = {}   # request id -> (LoRARequest | None, scale)
        self._pinned: dict[str, str] = {}                  # request id -> pool key held for its lifetime

    # ---- adapters ---------------------------------------------------------------------------------------------
    def resolve(self, lora_request) -> str:
        key = f"lora:{lora_request.lora_int_id}"
        if key not in self.pool.cpu and key not in self.pool.blobs:
            sd, cfg = self.load_fn(lora_request.lora_path)
            canon, _ = normalize(sd, self.keymap, config=cfg)
            for fused, (members, sizes) in self.groups.items():
                if not any(m in canon for m in members):
                    continue
                d_in = self.pool.layer_shapes[fused][0]
                parts = [canon.pop(m, None) or (torch.zeros(0, d_in, dtype=torch.bfloat16),
                                                torch.zeros(n, 0, dtype=torch.bfloat16))
                         for m, n in zip(members, sizes, strict=True)]
                canon[fused] = _compose(parts)
            self.pool.register(key, canon)
        return key

    def note_request(self, request_id: str, lora_request, scale: float = 1.0) -> None:
        """Remember a request's adapter and start loading it (hidden behind the request's text encoding)."""
        self._req[request_id] = (lora_request, float(scale))
        if lora_request is not None:
            self.pool.ensure_resident(self.resolve(lora_request))

    def forget(self, request_id: str) -> None:
        self._req.pop(request_id, None)
        key = self._pinned.pop(request_id, None)
        if key is not None:
            self.pool.release(key)

    def _slot(self, request_id: str) -> tuple[int, float]:
        lora_request, scale = self._req.get(request_id, (None, 1.0))
        if lora_request is None or scale == 0.0:
            return -1, 1.0
        key = self.resolve(lora_request)
        if request_id not in self._pinned:
            self.pool.ensure_resident(key)
            self.pool.wait(key)
            self.pool.acquire(key)
            self._pinned[request_id] = key
        return self.pool.slot_of[key], scale

    @contextmanager
    def batch(self, request_ids: list[str], repeat: int = 1):
        """Set the per-request meta for one forward over `request_ids` (rows repeat `repeat` times each)."""
        pairs = [self._slot(r) for r in request_ids]
        slots = [s for s, _ in pairs for _ in range(repeat)]
        scales = [c for _, c in pairs for _ in range(repeat)]
        with lora_batch(build_meta(slots, scales, r_max=self.pool.r_max, device=self.pool.device)):
            yield

    # ---- worker hooks -----------------------------------------------------------------------------------------
    def attach(self, worker) -> None:
        """Mixed mode for a vllm_omni DiffusionWorker: no LoRA manager, per-request adapters, wrapped pipeline."""
        worker.lora_manager = None

        def activate(scheduler_output):
            for rid in scheduler_output.finished_req_ids:
                self.forget(rid)
            for new in scheduler_output.scheduled_new_reqs:
                sp = new.req.sampling_params
                self.note_request(new.request_id, sp.lora_request, sp.lora_scale)

        worker._activate_step_lora = activate
        pipe = worker.model_runner.pipeline
        step, fwd = pipe.denoise_step, pipe.forward

        def denoise_step(input_batch, *a, **kw):
            with self.batch(list(input_batch.request_ids)):
                return step(input_batch, *a, **kw)

        def forward(req_batch, *a, **kw):
            for r in req_batch.requests:
                sp = r.sampling_params
                self.note_request(r.request_id, sp.lora_request, sp.lora_scale)
            n = max(getattr(sp, "num_outputs_per_prompt", 1) for sp in req_batch.sampling_params_list)
            try:
                with self.batch(req_batch.request_ids, repeat=n):
                    return fwd(req_batch, *a, **kw)
            finally:
                for rid in req_batch.request_ids:
                    self.forget(rid)

        pipe.denoise_step, pipe.forward = denoise_step, forward


def _compose(parts):
    from ..formats.normalize import compose_fused
    return compose_fused(parts)
