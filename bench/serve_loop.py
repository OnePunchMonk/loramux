"""Continuous-batching serve loop for flow-matching DiTs with per-request adapters (L2.5, HANDOFF §2.3, §11.3-§11.4).

Control flow (one iteration = one step boundary):

1. Arrivals whose trace time has passed are taken in. `pool.ensure_resident(key)` is issued for each at once
   (mixed policy), then the new prompts are text-encoded in one batch on the main timeline.
2. Waiting requests are admitted only at a step boundary, only when their text is encoded and their adapter is
   ready, and only while the batch has fewer than `max_bs` requests.
   - `mixed` (P2): any ready request; the batch mixes adapters through `BatchLoRAMeta`.
   - `keyed` (P1): only requests whose (adapter, scale) group equals the running group. When the batch drains, the
     group of the oldest ready request is chosen; switching groups unmerges the old ΔW = scale·B·A from the base
     weights and merges the new one, for real (the time lands in queue_wait and in the step CSV's swap_ms).
3. One denoise forward over every running row; each row takes a flow-matching Euler step with its own sigma
   schedule (`x += (σ_next − σ)·v`). Rows that finished leave the batch immediately and release their slot.
4. Finished latents go to a post worker (VAE decode + JPEG encode) off the denoise path: a worker thread (on CUDA
   with its own stream) for real-clock runs, a deferred queue with its own virtual timeline for virtual runs.

Clocks. `VirtualClock`: compute still runs (so outputs are real), but time advances by a `CostModel`, so tests are
fast and deterministic. `RealClock`: wall time (perf_counter), the trace is replayed in real time, and CUDA is
synchronized at each step boundary so step_ms is real (never inside forward). All timestamps are seconds relative
to the start of the run.

Outputs (bench.common.write_csv):
- latency CSV: one row per request = `LatencyLog.to_rows()` (L0.3: request_id, adapter_key, adapter_source,
  n_steps, the 8 timestamps in s, the 8 breakdown terms in ms) plus `policy`.
- step CSV: one row per denoise step: t (s, step start), bs, k_distinct (distinct adapters, base excluded),
  step_ms, plus policy, swap_ms (adapter switch time spent right before this step, keyed policy only) and post_ms
  (main-thread time spent handing finished requests to the post worker; 0 on the virtual clock).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import re
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from bench.common import collect_env, write_csv
from bench.trace.synth_trace import Request
from loramux.context import lora_batch
from loramux.latency import AdapterSource, LatencyLog, RequestTimeline
from loramux.metadata import build_meta
from loramux.pool import composite_key

POLICIES = ("mixed", "keyed")


# ---------------------------------------------------------------- pool helpers (old and new pool API)

def _resident(pool, key) -> bool:
    return key in getattr(pool, "slot_of", {})


def _source(pool, key) -> AdapterSource:
    if _resident(pool, key):
        return AdapterSource.GPU_HIT
    if key in getattr(pool, "cpu", {}):
        return AdapterSource.CPU_HIT
    if key in getattr(pool, "blobs", {}):
        return AdapterSource.DISK_HIT
    return AdapterSource.MISS


def default_key_fn(req: Request) -> tuple[str | None, float]:
    """(pool key, scale). No adapter -> (None, 1). One -> (id, scale). Several -> (composite_key, 1.0): the composite
    adapter must be registered in the pool under that key with its scales already folded in."""
    if not req.adapters:
        return None, 1.0
    if len(req.adapters) == 1:
        return req.adapters[0], float(req.scales[0]) if req.scales else 1.0
    return composite_key(zip(req.adapters, req.scales or [1.0] * len(req.adapters))), 1.0


# ---------------------------------------------------------------- clocks and costs

class VirtualClock:
    virtual = True

    def __init__(self) -> None:
        self.t = 0.0

    def start(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += max(0.0, float(dt))

    def sleep_until(self, t: float) -> None:
        self.t = max(self.t, t)

    def sync(self, device) -> None:
        pass


class RealClock:
    virtual = False
    poll_s = 1e-3

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def advance(self, dt: float) -> None:
        pass

    def sleep_until(self, t: float) -> None:
        dt = t - self.now()
        if dt > 0:
            time.sleep(dt)

    def sync(self, device) -> None:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()


@dataclass
class CostModel:
    """Seconds charged to a VirtualClock. step(bs, k) = step_base_s + step_per_req_s·bs + step_per_adapter_s·k."""
    step_base_s: float = 0.10
    step_per_req_s: float = 0.02
    step_per_adapter_s: float = 0.0
    text_s: float = 0.05             # per text-encode call (one batch per tick)
    text_per_prompt_s: float = 0.0
    h2d_s: float = 0.0               # per adapter load from the CPU tier / disk (GPU hit = 0)
    swap_s: float = 0.0              # keyed: one unmerge + merge
    vae_s: float = 0.05              # per decode call (batched)
    encode_s: float = 0.01           # per image
    transfer_s: float = 0.0          # per image

    def step(self, bs: int, k: int) -> float:
        return self.step_base_s + self.step_per_req_s * bs + self.step_per_adapter_s * k

    def text(self, n: int) -> float:
        return self.text_s + self.text_per_prompt_s * n

    def h2d(self, source: AdapterSource, key) -> float:
        return 0.0 if source in (AdapterSource.GPU_HIT, AdapterSource.NONE) else self.h2d_s

    def swap(self) -> float:
        return self.swap_s

    def vae(self, n: int) -> float:
        return self.vae_s

    def encode(self, n: int) -> float:
        return self.encode_s * n

    def transfer(self, n: int) -> float:
        return self.transfer_s * n


# ---------------------------------------------------------------- step models

@runtime_checkable
class StepModel(Protocol):
    device: Any
    module: nn.Module          # the denoiser whose linears carry adapters (patched for `mixed`)

    def encode_text(self, prompts: list[str]) -> list: ...

    def init_latents(self, req: Request) -> torch.Tensor: ...

    def denoise(self, latents: torch.Tensor, timesteps: torch.Tensor, embeds: list) -> torch.Tensor: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


def _seed_of(x) -> int:
    return int(x) if isinstance(x, int) else zlib.crc32(str(x).encode())


def flow_sigmas(steps: int, shift: float = 1.0) -> list[float]:
    """steps + 1 sigmas from 1 to 0; shift > 1 spends more steps at high noise (SD3/FLUX time shift)."""
    s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    s = shift * s / (1 + (shift - 1) * s)
    return [float(v) for v in s]


class ToyStepModel:
    """CPU toy: bench.step_bench.ToyDiT velocity field, latents [n_img, d], embeds [n_txt, d] from the prompt hash.

    Timestep conditioning: hidden + σ·t_vec. decode: sigmoid of the first 3 channels on a sqrt(n_img) grid."""

    def __init__(self, d: int = 32, layers: int = 2, n_img: int = 16, n_txt: int = 4, seed: int = 0,
                 device="cpu", dtype=torch.float32, shift: float = 1.0):
        from bench.step_bench import ToyDiT
        torch.manual_seed(seed)
        self.d, self.n_img, self.n_txt, self.device, self.dtype, self.shift = d, n_img, n_txt, device, dtype, shift
        self.module = ToyDiT(d, layers).to(device=device, dtype=dtype).requires_grad_(False)
        self.t_vec = torch.randn(d, generator=torch.Generator().manual_seed(seed + 1)).to(device, dtype)
        self.side = round(n_img ** 0.5)
        if self.side ** 2 != n_img:
            raise ValueError("n_img must be a square")

    def encode_text(self, prompts):
        out = []
        for p in prompts:
            g = torch.Generator().manual_seed(_seed_of(p))
            out.append(torch.randn(self.n_txt, self.d, generator=g).to(self.device, self.dtype))
        return out

    def init_latents(self, req):
        g = torch.Generator().manual_seed(_seed_of(req.id) + 12345)
        return torch.randn(self.n_img, self.d, generator=g).to(self.device, self.dtype)

    def denoise(self, latents, timesteps, embeds):
        h = latents + timesteps.to(latents.dtype)[:, None, None] * self.t_vec
        return self.module(h, torch.stack(list(embeds)))

    def decode(self, latents):
        img = torch.sigmoid(latents[..., :3]).transpose(1, 2).reshape(latents.shape[0], 3, self.side, self.side)
        return (img * 255).round().to(torch.uint8)


class DiffusersStepModel:
    """Qwen-Image / FLUX.2 transformer from diffusers with per-row timesteps (σ in [0, 1], as the pipelines pass
    timestep / 1000). No CFG (the batch key is CFG off).

    Text: `encode_fn(prompts) -> list of (embeds [L, D], mask [L] or None)` or, with `pipe`, `pipe.encode_prompt`.
    Rows with different text lengths are right-padded; Qwen gets the padding masked out. decode needs `pipe`
    (its VAE and unpack helpers) and real weights; it follows the diffusers 0.43 pipelines and is only exercised on
    GPU. Build with `from_pretrained` for real weights.
    """

    def __init__(self, kind: str, transformer: nn.Module, *, pipe=None, encode_fn=None, resolution: int = 1024,
                 guidance: float = 4.0, device="cuda", dtype=torch.bfloat16, shift: float = 3.0):
        if kind not in ("qwen_image", "flux2"):
            raise ValueError(f"unknown kind {kind!r}")
        if pipe is None and encode_fn is None:
            raise ValueError("pass pipe or encode_fn")
        self.kind, self.pipe, self.encode_fn, self.resolution = kind, pipe, encode_fn, resolution
        self.guidance, self.device, self.dtype, self.shift = guidance, device, dtype, shift
        self.module = transformer.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.grid = max(1, resolution // 16)

    @classmethod
    def from_pretrained(cls, kind: str, model_id: str, *, device="cuda", dtype=torch.bfloat16, **kw):
        try:
            import diffusers
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError("DiffusersStepModel needs diffusers (pip install loramux[diffusers])") from e
        name = {"qwen_image": "QwenImagePipeline", "flux2": "Flux2Pipeline"}[kind]
        pipe = getattr(diffusers, name).from_pretrained(model_id, torch_dtype=dtype).to(device)
        return cls(kind, pipe.transformer, pipe=pipe, device=device, dtype=dtype, **kw)

    def encode_text(self, prompts):
        if self.encode_fn is not None:
            return list(self.encode_fn(prompts))
        with torch.no_grad():
            if self.kind == "qwen_image":
                emb, mask = self.pipe.encode_prompt(list(prompts), device=self.device)
                return [(emb[i], None if mask is None else mask[i]) for i in range(len(prompts))]
            emb, _ids = self.pipe.encode_prompt(list(prompts), device=self.device)
            return [(emb[i], None) for i in range(len(prompts))]

    def init_latents(self, req):
        g = torch.Generator().manual_seed(_seed_of(req.id) + 12345)
        c = self.module.config.in_channels
        return torch.randn(self.grid * self.grid, c, generator=g).to(self.device, self.dtype)

    def _text(self, embeds):
        L = max(e.shape[0] for e, _ in embeds)
        D = embeds[0][0].shape[-1]
        rows = len(embeds)
        txt = torch.zeros(rows, L, D, device=self.device, dtype=self.dtype)
        mask = torch.zeros(rows, L, device=self.device, dtype=torch.long)
        for i, (e, m) in enumerate(embeds):
            txt[i, :e.shape[0]] = e.to(self.device, self.dtype)
            mask[i, :e.shape[0]] = 1 if m is None else m.to(self.device, torch.long)
        return txt, mask

    def denoise(self, latents, timesteps, embeds):
        rows, g = latents.shape[0], self.grid
        txt, mask = self._text(embeds)
        kw = {"hidden_states": latents, "encoder_hidden_states": txt,
              "timestep": timesteps.to(self.device, self.dtype), "return_dict": False}
        if self.kind == "qwen_image":
            kw["encoder_hidden_states_mask"] = mask
            kw["img_shapes"] = [[(1, g, g)]] * rows
        else:
            hw = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1))
            kw["img_ids"] = hw.unsqueeze(0).expand(rows, -1, -1).to(self.device)
            tx = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(txt.shape[1]))
            kw["txt_ids"] = tx.unsqueeze(0).expand(rows, -1, -1).to(self.device)
            if getattr(self.module.config, "guidance_embeds", False):
                kw["guidance"] = torch.full((rows,), self.guidance, device=self.device, dtype=self.dtype)
        out = self.module(**kw)
        return out[0] if isinstance(out, tuple) else out

    def decode(self, latents):  # pragma: no cover - needs real VAE weights
        if self.pipe is None:
            raise RuntimeError("decode needs the pipeline (VAE)")
        p, vae = self.pipe, self.pipe.vae
        with torch.no_grad():
            if self.kind == "qwen_image":
                x = p._unpack_latents(latents, self.resolution, self.resolution, p.vae_scale_factor).to(vae.dtype)
                mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(x)
                std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(x)
                img = vae.decode(x * std + mean, return_dict=False)[0][:, :, 0]
            else:
                c = latents.shape[-1]
                ids = p._prepare_latent_ids(torch.zeros(latents.shape[0], c, self.grid, self.grid)).to(latents.device)
                x = p._unpack_latents_with_ids(latents, ids)
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(x)
                bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(x)
                img = vae.decode(p._unpatchify_latents(x * bn_std + bn_mean), return_dict=False)[0]
        return ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def encode_image(img: torch.Tensor, quality: int = 95) -> bytes:
    """uint8 [3, H, W] -> JPEG bytes: torchvision.io.encode_jpeg on CUDA tensors when available, else PIL."""
    if img.is_cuda:
        try:
            from torchvision.io import encode_jpeg
            return bytes(encode_jpeg(img, quality=quality).cpu().numpy().tobytes())
        except ImportError:
            pass
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img.permute(1, 2, 0).cpu().numpy()).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------- admission (shared with bench.sim)

def select_admits(policy: str, eligible: list, n_running: int, running_group, max_bs: int):
    """eligible: ready requests in arrival order, each with `.group`. Returns (admitted, group of the batch).

    mixed: the first max_bs - n_running. keyed: only the running group; an empty batch takes the oldest ready
    request's group. running_group is ignored by `mixed`."""
    cap = max_bs - n_running
    if cap <= 0 or not eligible:
        return [], running_group
    if policy == "mixed":
        return eligible[:cap], running_group
    if policy != "keyed":
        raise ValueError(f"unknown policy {policy!r}")
    g = running_group if n_running else eligible[0].group
    return [r for r in eligible if r.group == g][:cap], g


@dataclass
class _Live:
    req: Request
    key: str | None
    scale: float
    t_arrive: float
    source: AdapterSource
    sigmas: list[float]
    t_text: float | None = None
    t_issue: float | None = None
    ready_at: float | None = None
    t_resident: float | None = None
    t_first: float | None = None
    t_last: float | None = None
    slot: int = -1
    step: int = 0
    latents: torch.Tensor | None = None
    embeds: Any = None
    post: dict = field(default_factory=dict)

    @property
    def group(self):
        return None if self.key is None else (self.key, self.scale)


@dataclass
class ServeResult:
    policy: str
    log: LatencyLog
    step_rows: list[dict]
    step_groups: list[list]                  # per step: the (key, scale) group (None = base) of every row
    latents: dict = field(default_factory=dict)   # request id -> final latents (CPU)
    images: dict = field(default_factory=dict)    # request id -> JPEG bytes
    swaps: int = 0

    def latency_rows(self) -> list[dict]:
        return [dict(r, policy=self.policy) for r in self.log.to_rows()]

    def write(self, latency_csv, step_csv, env: dict | None = None) -> None:
        env = collect_env() if env is None else env
        write_csv(latency_csv, self.latency_rows(), env)
        write_csv(step_csv, self.step_rows, env)


# ---------------------------------------------------------------- post worker (VAE + encode)

class _PostWorker:
    """Virtual clock: runs at once but on its own timeline (start = max(step end, worker free)). Real clock: one
    worker thread; on CUDA it uses a side stream that waits on the main stream's event."""

    def __init__(self, loop: ServeLoop):
        self.loop = loop
        self.free_at = 0.0
        self.pool = None if loop.clock.virtual else cf.ThreadPoolExecutor(max_workers=1)
        self.futs: list = []
        cuda = loop.model is not None and str(loop.model.device).startswith("cuda")
        self.stream = torch.cuda.Stream() if cuda and not loop.clock.virtual else None

    def submit(self, done: list[_Live], t_end: float) -> None:
        loop = self.loop
        if loop.clock.virtual:
            start = max(t_end, self.free_at)
            imgs = loop._decode(done)
            t_vae = start + loop.cost.vae(len(done))
            t = t_vae
            for r, img in zip(done, imgs):
                t += loop.cost.encode(1)
                r.post = {"t_vae": t_vae, "t_enc": t, "t_sent": t + loop.cost.transfer(1),
                          "jpeg": loop._encode_img(img)}
            self.free_at = t
            return
        ev = None
        if self.stream is not None:
            ev = torch.cuda.Event()
            ev.record()
        self.futs.append(self.pool.submit(self._run_real, done, ev))

    def _run_real(self, done, ev):
        loop, clock = self.loop, self.loop.clock
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else _Null()
        with ctx:
            if ev is not None:
                self.stream.wait_event(ev)
            imgs = loop._decode(done)
            if self.stream is not None:
                self.stream.synchronize()
            t_vae = clock.now()
            for r, img in zip(done, imgs):
                jpeg = loop._encode_img(img)
                t = clock.now()
                r.post = {"t_vae": t_vae, "t_enc": t, "t_sent": t, "jpeg": jpeg}

    def drain(self) -> None:
        for f in self.futs:
            f.result()
        if self.pool is not None:
            self.pool.shutdown()


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------------------------------------------------------------- the loop

class ServeLoop:
    """model=None runs without tensors (the simulator); costs then come only from `cost`."""

    def __init__(self, model: StepModel | None, pool, *, policy: str = "mixed", max_bs: int = 8,
                 clock=None, cost: CostModel | None = None, key_fn=default_key_fn, prompt_fn=None,
                 shift: float | None = None, keep_latents: bool = True):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.model, self.pool, self.policy, self.max_bs = model, pool, policy, max_bs
        self.clock = clock if clock is not None else VirtualClock()
        self.cost = cost if cost is not None else CostModel()
        self.key_fn = key_fn
        self.prompt_fn = prompt_fn or (lambda r: f"prompt {r.prompt_id}")
        self.shift = shift if shift is not None else getattr(model, "shift", 1.0)
        self.keep_latents = keep_latents
        self.merged = None          # keyed: (key, scale) currently merged into the base weights
        self._linears = None
        self._key_ready: dict = {}  # key -> time its current load completes

    # ---- compute hooks (no-ops without a model)

    def _timed(self, cost_s: float, fn):
        t0 = self.clock.now()
        out = fn() if self.model is not None else None
        if self.clock.virtual:
            self.clock.advance(cost_s)
        elif self.model is not None:
            self.clock.sync(self.model.device)
        return out, self.clock.now() - t0

    def _decode(self, done: list[_Live]):
        if self.model is None:
            return [None] * len(done)
        with torch.no_grad():
            return list(self.model.decode(torch.stack([r.latents for r in done])))

    def _encode_img(self, img):
        return None if img is None else encode_image(img)

    def _linear_map(self) -> dict[str, nn.Linear]:
        if self._linears is None:
            from loramux.layers import MuxLinear
            m = {}
            for name, mod in self.model.module.named_modules():
                if isinstance(mod, MuxLinear):
                    m[mod.layer_key] = mod.base
                elif isinstance(mod, nn.Linear):
                    m.setdefault(name, mod)
            self._linears = m
        return self._linears

    def _merge(self, group, sign: float) -> None:
        key, scale = group
        lin = self._linear_map()
        with torch.no_grad():
            for k, (a, b) in self.pool.canonical(key).items():
                w = lin[k].weight
                delta = (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32)) * (sign * scale)
                w.copy_((w.float() + delta).to(w.dtype))

    def _swap(self, new_group) -> float:
        old = self.merged

        def fn():
            if old is not None:
                self._merge(old, -1.0)
            if new_group is not None:
                self._merge(new_group, +1.0)
        _, dt = self._timed(self.cost.swap(), fn)
        self.merged = new_group
        return dt

    def restore(self) -> None:
        """Keyed: unmerge whatever is merged, so the base weights are back."""
        if self.merged is not None and self.model is not None:
            self._merge(self.merged, -1.0)
        self.merged = None

    def _step(self, batch: list[_Live]):
        bs = len(batch)
        k = len({r.key for r in batch if r.key is not None})

        def fn():
            x = torch.stack([r.latents for r in batch])
            dev = x.device
            s = torch.tensor([r.sigmas[r.step] for r in batch], dtype=torch.float32, device=dev)
            s_next = torch.tensor([r.sigmas[r.step + 1] for r in batch], dtype=torch.float32, device=dev)
            with torch.no_grad():
                if self.policy == "mixed":
                    meta = build_meta([r.slot for r in batch], [r.scale for r in batch],
                                      rank_of_slot=getattr(self.pool, "slot_rank", None),
                                      r_max=getattr(self.pool, "r_max", None), device=dev, per_row_2d=True)
                    with lora_batch(meta):
                        v = self.model.denoise(x, s, [r.embeds for r in batch])
                else:
                    v = self.model.denoise(x, s, [r.embeds for r in batch])
                dt = (s_next - s).to(x.dtype).view(-1, *([1] * (x.dim() - 1)))
                x = x + dt * v.to(x.dtype)
            for i, r in enumerate(batch):
                r.latents = x[i]
        _, dt = self._timed(self.cost.step(bs, k), fn)
        return dt, k

    # ---- adapter residency

    def _issue(self, r: _Live) -> None:
        """ensure_resident + acquire (mixed only). The reference pins the slot from issue until the request
        finishes, so waiting requests cannot evict each other's adapters (no thrash). Retried at later step
        boundaries while every slot is pinned."""
        now = self.clock.now()
        if r.key is None or self.policy == "keyed":
            r.t_issue = r.ready_at = r.t_resident = now
            return
        src = _source(self.pool, r.key)
        try:
            self.pool.ensure_resident(r.key)
        except RuntimeError:
            return                              # no evictable slot right now
        r.slot = self.pool.acquire(r.key, loading_ok=True)   # admitted only once is_ready (below)
        r.t_issue = now
        if src is not AdapterSource.GPU_HIT or r.key not in self._key_ready:
            self._key_ready[r.key] = now + (self.cost.h2d(src, r.key) if self.clock.virtual else 0.0)
        r.ready_at = self._key_ready[r.key]     # an in-flight load started by another request counts

    def _adapter_ready(self, r: _Live) -> bool:
        if r.t_resident is not None:
            return True
        if r.t_issue is None:
            self._issue(r)
            if r.t_issue is None:
                return False
        now = self.clock.now()
        if now >= r.ready_at and self.pool.is_ready(r.key):
            r.t_resident = max(r.ready_at, r.t_issue) if self.clock.virtual else now
            return True
        return False

    # ---- main loop

    def run(self, trace: list[Request]) -> ServeResult:
        self.clock.start()
        self._key_ready = {}
        pending = deque(sorted(trace, key=lambda q: (q.t_arrive, str(q.id))))
        waiting: list[_Live] = []
        running: list[_Live] = []
        group = None
        res = ServeResult(self.policy, LatencyLog(), [], [])
        post = _PostWorker(self)
        finished: list[_Live] = []
        pending_swap = 0.0
        while pending or waiting or running:
            now = self.clock.now()
            new = []
            while pending and pending[0].t_arrive <= now:
                q = pending.popleft()
                key, scale = self.key_fn(q)
                src = AdapterSource.NONE if key is None else (
                    AdapterSource.CPU_HIT if self.policy == "keyed" else _source(self.pool, key))
                new.append(_Live(q, key, scale, q.t_arrive, src, flow_sigmas(q.steps, self.shift)))
            if new:
                for r in new:
                    self._issue(r)
                embeds, _ = self._timed(self.cost.text(len(new)),
                                        lambda new=new: self.model.encode_text([self.prompt_fn(r.req) for r in new]))
                t_text = self.clock.now()
                for i, r in enumerate(new):
                    r.t_text = t_text
                    if self.model is not None:
                        r.embeds = embeds[i]
                        r.latents = self.model.init_latents(r.req)
                waiting.extend(new)
            eligible = [r for r in waiting if r.t_text is not None and self._adapter_ready(r)]
            admits, g = select_admits(self.policy, eligible, len(running), group, self.max_bs)
            if admits:
                if self.policy == "keyed" and g != self.merged:
                    pending_swap += self._swap(g)
                    res.swaps += 1
                group = g
                for r in admits:
                    waiting.remove(r)
                running.extend(admits)
            if not running:
                nxt = [pending[0].t_arrive] if pending else []
                nxt += [r.ready_at for r in waiting if r.ready_at is not None and r.ready_at > now]
                if not nxt:
                    if waiting and not self.clock.virtual:
                        time.sleep(RealClock.poll_s)
                        continue
                    if waiting:
                        raise RuntimeError(f"{len(waiting)} requests can never be admitted (all slots pinned?)")
                    continue
                target = min(nxt)
                if not self.clock.virtual:
                    target = min(target, now + RealClock.poll_s) if waiting else target
                self.clock.sleep_until(target)
                continue
            t_start = self.clock.now()
            for r in running:
                if r.t_first is None:
                    r.t_first = t_start
            step_s, k = self._step(running)
            t_end = self.clock.now()
            res.step_groups.append([r.group for r in running])
            res.step_rows.append({"t": t_start, "bs": len(running), "k_distinct": k, "step_ms": 1e3 * step_s,
                                  "policy": self.policy, "swap_ms": 1e3 * pending_swap, "post_ms": 0.0})
            pending_swap = 0.0
            done = []
            for r in running:
                r.step += 1
                if r.step >= r.req.steps:
                    r.t_last = t_end
                    done.append(r)
            if done:
                running = [r for r in running if r.t_last is None]
                for r in done:
                    if self.policy == "mixed" and r.key is not None:
                        self.pool.release(r.key)
                t_post = time.perf_counter()
                post.submit(done, t_end)
                if not self.clock.virtual:  # main-thread time handing off VAE + encode (L5); virtual: 0 by design
                    res.step_rows[-1]["post_ms"] = 1e3 * (time.perf_counter() - t_post)
                finished.extend(done)
            if not running:
                group = None
        post.drain()
        for r in finished:
            p = r.post
            res.log.append(RequestTimeline(
                t_arrive=r.t_arrive, t_admit_ready_text=r.t_text, t_adapter_resident=r.t_resident,
                t_first_step=r.t_first, t_last_step=r.t_last, t_vae_done=p["t_vae"], t_encode_done=p["t_enc"],
                t_sent=p["t_sent"], adapter_source=r.source, n_steps=r.req.steps, request_id=str(r.req.id),
                adapter_key=r.key))
            if self.model is not None and self.keep_latents:
                res.latents[r.req.id] = r.latents.detach().cpu()
            if p.get("jpeg") is not None:
                res.images[r.req.id] = p["jpeg"]
        return res


def serve(model, pool, trace, **kw) -> ServeResult:
    return ServeLoop(model, pool, **kw).run(trace)


# ---------------------------------------------------------------- CLI (toy model; real-clock replay)

def main(argv: list[str] | None = None) -> int:
    from bench.step_bench import linear_shapes
    from bench.synth_adapters import make_adapter
    from bench.trace.synth_trace import synth_trace
    from loramux import AdapterPool, patch_model

    ap = argparse.ArgumentParser(prog="python -m bench.serve_loop")
    ap.add_argument("--policy", choices=POLICIES, default="mixed")
    ap.add_argument("--n-requests", type=int, default=32)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-bs", type=int, default=8)
    ap.add_argument("--n-adapters", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--virtual", action="store_true", help="virtual clock (default: real-time replay)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/serve")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToyStepModel(device=dev)
    shapes = linear_shapes(model.module, "attn")
    pool = AdapterPool(shapes, slots=a.n_adapters, r_max=a.rank, device=dev, dtype=torch.float32)
    for i in range(a.n_adapters):
        pool.register(f"ad_{i:04d}", make_adapter(shapes, a.rank, "all", seed=i))
    if a.policy == "mixed":
        patch_model(model.module, ["^(?:" + "|".join(re.escape(k) for k in shapes) + ")$"], pool)
    trace = synth_trace(a.n_requests, a.rate, n_adapters=a.n_adapters, lora_frac=0.8, multi_frac=0.0,
                        top5pct_share=None, seed=a.seed, steps=a.steps)
    loop = ServeLoop(model, pool, policy=a.policy, max_bs=a.max_bs,
                     clock=VirtualClock() if a.virtual else RealClock())
    res = loop.run(trace)
    loop.restore()
    res.write(f"{a.out}_{a.policy}_latency.csv", f"{a.out}_{a.policy}_steps.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
