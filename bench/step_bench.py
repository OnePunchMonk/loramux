"""Step bench: one denoise forward of a whole DiT with and without mixed-adapter LoRA (K2b / K2c).

Model-agnostic through the `StepTarget` protocol:

    target.name, target.model (nn.Module whose nn.Linear modules get patched), target.device, target.dtype
    target.make_inputs(bs, rows_per_req=1) -> dict   # rows = bs * rows_per_req (2 = CFG, concat layout)
    target.forward(**inputs) -> Tensor

Implementations: `ToyDiTTarget` (CPU smoke tests), `DiffusersStepTarget` (Qwen-Image / FLUX.2 transformers from
diffusers, random latents and text embeddings of the right shapes; real weights loaded by id; needs diffusers and,
for real weights, a GPU, so it is import-guarded and not exercised in the CPU tests beyond a tiny config).

`run_step_bench` times, for bs requests with `n_adapters` distinct adapters mixed in the batch, base vs
attention-only adapters vs all-linear adapters, eager vs `torch.compile`, and reports overhead_pct against the base
model in the same mode. Rows: model, mode, adapters (base|attn|all), bs, rows_per_req, k, rank, median_ms, p90_ms,
mean_ms, n, overhead_pct.
"""
from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from bench.common import collect_env, interleave, seed_all, summarize, write_csv
from bench.synth_adapters import make_adapter
from loramux import AdapterPool, build_meta, lora_batch
from loramux.layers import patch_keys, select_linears, unpatch_model

# bench.synth_adapters.ATTN_PAT plus the MMDiT/FLUX.2 attention projections it does not name.
ATTN_STEP_PAT = re.compile(
    r"(?:^|\.)(?:to_q|to_k|to_v|to_out(?:\.0)?|attn|add_[qkv]_proj|to_add_out|to_qkv_mlp_proj)$")


@runtime_checkable
class StepTarget(Protocol):
    name: str
    model: nn.Module
    device: str
    dtype: torch.dtype

    def make_inputs(self, bs: int, rows_per_req: int = 1) -> dict: ...

    def forward(self, **inputs) -> torch.Tensor: ...


# ---------------------------------------------------------------- toy target

class _ToyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q, self.attn.to_k = nn.Linear(d, d), nn.Linear(d, d)
        self.attn.to_v, self.attn.to_out = nn.Linear(d, d), nn.Linear(d, d)
        self.ff_in, self.ff_out = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, x):
        q, k, v = self.attn.to_q(x), self.attn.to_k(x), self.attn.to_v(x)
        h = torch.softmax(q @ k.transpose(1, 2) / q.shape[-1] ** 0.5, -1) @ v
        x = x + self.attn.to_out(h)
        return x + self.ff_out(torch.relu(self.ff_in(x)))


class ToyDiT(nn.Module):
    def __init__(self, d=32, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(_ToyBlock(d) for _ in range(layers))

    def forward(self, hidden_states, encoder_hidden_states):
        n_txt = encoder_hidden_states.shape[1]
        x = torch.cat([encoder_hidden_states, hidden_states], 1)
        for b in self.blocks:
            x = b(x)
        return x[:, n_txt:]


class ToyDiTTarget:
    """Tiny single-stream DiT with attention and MLP linears; resolution sets the token count."""

    def __init__(self, d=32, layers=2, n_img=16, n_txt=4, device="cpu", dtype=torch.float32, seed=0):
        torch.manual_seed(seed)
        self.name, self.device, self.dtype = "toy", device, dtype
        self.d, self.n_img, self.n_txt = d, n_img, n_txt
        self.model = ToyDiT(d, layers).to(device=device, dtype=dtype).requires_grad_(False)

    def make_inputs(self, bs, rows_per_req=1):
        rows = bs * rows_per_req
        g = torch.Generator().manual_seed(7)
        mk = lambda n: torch.randn(rows, n, self.d, generator=g).to(self.device, self.dtype)
        return {"hidden_states": mk(self.n_img), "encoder_hidden_states": mk(self.n_txt)}

    def forward(self, **inputs):
        return self.model(**inputs)


# ---------------------------------------------------------------- diffusers target

class DiffusersStepTarget:
    """Qwen-Image or FLUX.2 transformer from diffusers with random inputs of the right shapes.

    kind: 'qwen_image' | 'flux2'. Pass `model` (an already built transformer, e.g. a tiny config) or `model_id`
    (loaded with `from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)`). `resolution` is the image
    side in pixels; the latent grid is resolution/16 per side for both models (VAE /8 times 2x2 patchify), so
    1024 px -> 4096 image tokens. `txt_len` text tokens (default 256).
    Not runnable without the model's weights or a tiny config; diffusers is imported lazily.
    """

    def __init__(self, kind: str, *, model: nn.Module | None = None, model_id: str | None = None,
                 subfolder: str = "transformer", resolution: int = 1024, txt_len: int = 256,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        if kind not in ("qwen_image", "flux2"):
            raise ValueError(f"unknown kind {kind!r}")
        if model is None:
            if model_id is None:
                raise ValueError("pass model or model_id")
            try:
                import diffusers
            except ImportError as e:  # pragma: no cover - environment dependent
                raise ImportError("DiffusersStepTarget needs diffusers (pip install loramux[diffusers])") from e
            cls = {"qwen_image": "QwenImageTransformer2DModel", "flux2": "Flux2Transformer2DModel"}[kind]
            model = getattr(diffusers, cls).from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
        self.kind, self.resolution, self.txt_len = kind, resolution, txt_len
        self.device, self.dtype = device, dtype
        self.name = f"{kind}_{resolution}"
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    @property
    def grid(self) -> int:
        return max(1, self.resolution // 16)

    def make_inputs(self, bs, rows_per_req=1):
        rows, g, cfg = bs * rows_per_req, self.grid, self.model.config
        gen = torch.Generator().manual_seed(7)
        rnd = lambda *s: torch.randn(*s, generator=gen).to(self.device, self.dtype)
        n_img = g * g
        out = {"hidden_states": rnd(rows, n_img, cfg.in_channels),
               "encoder_hidden_states": rnd(rows, self.txt_len, cfg.joint_attention_dim),
               "timestep": torch.full((rows,), 0.5, device=self.device, dtype=self.dtype),
               "return_dict": False}
        if self.kind == "qwen_image":
            out["encoder_hidden_states_mask"] = torch.ones(rows, self.txt_len, device=self.device, dtype=torch.long)
            out["img_shapes"] = [[(1, g, g)]] * rows
        else:
            hw = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1))
            out["img_ids"] = hw.unsqueeze(0).expand(rows, -1, -1).to(self.device)
            tx = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(self.txt_len))
            out["txt_ids"] = tx.unsqueeze(0).expand(rows, -1, -1).to(self.device)
            if getattr(cfg, "guidance_embeds", False):
                out["guidance"] = torch.full((rows,), 4.0, device=self.device, dtype=self.dtype)
        return out

    def forward(self, **inputs):
        out = self.model(**inputs)
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------- LoRA harness

def linear_shapes(model: nn.Module, targets: str) -> dict[str, tuple[int, int]]:
    """{qualified name: (d_in, d_out)} of the nn.Linear modules selected by targets ('attn' | 'all')."""
    if targets not in ("attn", "all"):
        raise ValueError("targets must be 'attn' or 'all'")
    return select_linears(model, [""] if targets == "all" else [ATTN_STEP_PAT.pattern])


def token_linears(target: StepTarget, shapes: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Keep the linears that one forward actually calls, with a [rows, N, d] or [rows, d] input.

    2-D inputs (timestep/guidance embedders, AdaLN modulation) are one token per row (`per_row_2d`), so "all" really
    is every linear the forward uses. Linears that the forward never calls are dropped.
    """
    seen: dict[str, bool] = {}
    mods = dict(target.model.named_modules())
    hooks = [mods[k].register_forward_pre_hook(
        lambda _m, args, k=k: seen.__setitem__(k, seen.get(k, True) and args[0].dim() >= 2)) for k in shapes]
    try:
        with torch.no_grad():
            target.forward(**target.make_inputs(1))
    finally:
        for h in hooks:
            h.remove()
    return {k: v for k, v in shapes.items() if seen.get(k)}


class LoRAHarness:
    """Registers `n_adapters` synthetic adapters in a pool and patches the token-stream linears of `target.model`.

    `close()` unpatches.
    """

    def __init__(self, target: StepTarget, *, targets: str, n_adapters: int, rank: int, seed: int = 0):
        self.target, self.targets, self.n_adapters, self.rank = target, targets, n_adapters, rank
        shapes = token_linears(target, linear_shapes(target.model, targets))
        if not shapes:
            raise ValueError(f"no linear layers match targets={targets!r}")
        self.pool = AdapterPool(shapes, slots=n_adapters, r_max=rank, device=target.device, dtype=target.dtype)
        for a in range(n_adapters):
            self.pool.register(f"ad{a}", make_adapter(shapes, rank, "all", seed=seed + a))
            self.pool.ensure_resident(f"ad{a}")
        if str(target.device).startswith("cuda"):
            torch.cuda.synchronize()
        patch_keys(target.model, shapes, self.pool)

    def close(self) -> None:
        unpatch_model(self.target.model)

    def step_fn(self, inputs: dict, bs: int, k: int, rows_per_req: int = 1) -> Callable[[], torch.Tensor]:
        """Callable running one forward with bs requests spread over k distinct adapters (k=1: uniform)."""
        if not 1 <= k <= min(bs, self.n_adapters):
            raise ValueError(f"need 1 <= k <= min(bs={bs}, n_adapters={self.n_adapters}), got k={k}")
        slots = [self.pool.slot_of[f"ad{i % k}"] for i in range(bs)]
        fwd = self.target.forward
        dev, rank = self.target.device, self.rank
        if rows_per_req not in (1, 2):
            raise ValueError("rows_per_req must be 1 or 2")

        def fn():
            meta = build_meta(slots, [1.0] * bs, cfg_layout="concat", device=dev, r_max=rank, per_row_2d=True)
            with lora_batch(meta):
                return fwd(**inputs)
        return fn


def _compiled(fn_factory: Callable[[], Callable], backend: str):
    torch._dynamo.reset()
    return torch.compile(fn_factory(), backend=backend)


def _need_warmup(modes, warmup: int) -> None:
    """torch.compile compiles on the first call; the warmup call must absorb it or it lands in the timings."""
    if "compile" in modes and warmup < 1:
        raise ValueError("mode 'compile' needs warmup >= 1 so compile time is excluded from the timings")


def run_step_bench(target: StepTarget, *, bs: int = 8, k: int | None = None, rank: int = 16,
                   adapter_sets: tuple[str, ...] = ("attn", "all"), modes: tuple[str, ...] = ("eager", "compile"),
                   rows_per_req: int = 1, compile_backend: str = "inductor", rounds: int = 3, warmup: int = 3,
                   iters: int = 10, seed: int = 0) -> list[dict]:
    """Overhead of mixed-adapter LoRA on one denoise step. k = distinct adapters (default bs)."""
    _need_warmup(modes, warmup)
    seed_all(seed)
    k = k or bs
    rows: list[dict] = []
    common = {"model": target.name, "bs": bs, "rows_per_req": rows_per_req, "k": k, "rank": rank}
    with torch.no_grad():
        inputs = target.make_inputs(bs, rows_per_req)
        base_fns: dict[str, Callable] = {}
        for mode in modes:
            f = (lambda: target.forward(**inputs))
            base_fns[mode] = _compiled(lambda f=f: f, compile_backend) if mode == "compile" else f
        base_t = interleave({f"base/{m}": f for m, f in base_fns.items()}, rounds=rounds, warmup=warmup, iters=iters)
        base_med = {m: summarize(base_t[f"base/{m}"])["median"] for m in modes}
        for m in modes:
            s = summarize(base_t[f"base/{m}"])
            rows.append(dict(common, mode=m, adapters="base", median_ms=s["median"], p90_ms=s["p90"],
                             mean_ms=s["mean"], n=s["n"], overhead_pct=0.0))
        for aset in adapter_sets:
            h = LoRAHarness(target, targets=aset, n_adapters=min(k, bs), rank=rank, seed=seed)
            try:
                fns = {}
                for mode in modes:
                    fn = h.step_fn(inputs, bs, k, rows_per_req)
                    fns[mode] = _compiled(lambda fn=fn: fn, compile_backend) if mode == "compile" else fn
                t = interleave(fns, rounds=rounds, warmup=warmup, iters=iters)
            finally:
                h.close()
            for m in modes:
                s = summarize(t[m])
                rows.append(dict(common, mode=m, adapters=aset, median_ms=s["median"], p90_ms=s["p90"],
                                 mean_ms=s["mean"], n=s["n"],
                                 overhead_pct=(s["median"] - base_med[m]) / base_med[m] * 100.0))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.step_bench")
    ap.add_argument("--kind", choices=["toy", "qwen_image", "flux2"], default="toy")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--cfg", action="store_true", help="2 rows per request (concat CFG layout)")
    ap.add_argument("--modes", default="eager,compile")
    ap.add_argument("--out", default="results/k2b.csv")
    a = ap.parse_args(argv)
    if a.kind == "toy":
        target: StepTarget = ToyDiTTarget(device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        target = DiffusersStepTarget(a.kind, model_id=a.model_id, subfolder=a.subfolder, resolution=a.resolution)
    rows = run_step_bench(target, bs=a.bs, rank=a.rank, modes=tuple(a.modes.split(",")),
                          rows_per_req=2 if a.cfg else 1)
    write_csv(a.out, rows, collect_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
