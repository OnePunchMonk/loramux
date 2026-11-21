"""vLLM-Omni diffusion adapter switch cost (L2.7). GPU-only; needs vllm-omni==0.32.0 installed.

Measures what the simulator (bench/sim) needs as the swap cost of the adapter-keyed baseline: the extra
per-request latency when the next request uses a different adapter than the previous one (activation of a
cached adapter), and the cost of going back to the base model (deactivation).

vLLM-Omni 0.32.0 APIs used / relied on (paths relative to the vllm_omni package):
  entrypoints/omni.py            `Omni(model, **kwargs)` — synchronous offline entrypoint (OmniBase.__init__ in
                                 entrypoints/omni_base.py takes `model` + engine kwargs); `Omni.generate(prompts,
                                 sampling_params_list, use_tqdm=...)` returns list[OmniRequestOutput].
  inputs/data.py                 `OmniDiffusionSamplingParams` (height, width, num_inference_steps, seed,
                                 lora_request, lora_scale).
  lora/request.py                re-exports vLLM's `LoRARequest(lora_name, lora_int_id, lora_path)` (positional
                                 form used in entrypoints/openai/utils.py).
  lora/utils.py                  `stable_lora_int_id(lora_path)` — deterministic adapter id (the cache key).
  diffusion/data.py              OmniDiffusionConfig.max_cpu_loras (adapter cache size; default 1, which would
                                 turn every switch into a reload — we pass the number of adapters).
  diffusion/lora/manager.py      DiffusionLoRAManager.set_request_adapter -> set_active_adapter: add_adapter on a
                                 cache miss (load), else _touch_adapter_info, then _activate_adapter (re-binds
                                 weights into the LoRA layers); None / scale 0 -> _deactivate_all_adapters.
                                 remove_adapter evicts. One adapter is active per forward.
  diffusion/worker/diffusion_worker.py
                                 execute_model / execute_model_batch call set_request_adapter once per batch;
                                 execute_stepwise -> _activate_step_lora once per denoise step.
  diffusion/sched/interface.py   Request/Step batch keys carry lora_int_id and lora_scale, so requests with
                                 different adapters never share a batch.

The manager lives in the worker process, so activation cannot be timed in-process from here; it is measured as
the latency difference between a request that switches adapter and one that repeats it (same shapes, steps).
Each request is its own `generate` call, so batches are single-request and the only variable is the adapter.

Patterns (each after a 'register' pass that loads every adapter once, recorded with cold=1):
  repeat       adapter 0 every time (no switch)                       -> baseline latency
  alternate    0,1,...,N-1,0,1,... (every request switches)           -> activation cost
  base_toggle  0, base, 0, base, ... (adapter <-> base model)         -> activation + deactivation cost

CSV (bench.common.write_csv; env header lines '# key=value'). One row per request (kind=request) followed by
summary rows (kind=summary). Columns:
  kind            'request' | 'summary'
  pattern         register | repeat | alternate | base_toggle (summary rows: also 'all')
  metric          summary only: latency_ms | swap_cost_ms | deactivate_cost_ms | cold_load_ms
  idx             request index within the pattern
  adapter         adapter name, or 'base'
  prev_adapter    adapter of the previous request ('' for the first)
  switched        1 if adapter != prev_adapter
  cold            1 if this is the adapter's first use in the engine (includes the load)
  latency_ms      wall time of the generate call (request rows)
  median_ms, p90_ms, mean_ms, n   summary statistics (summary rows)
  model, steps, height, width, lora_scale, n_adapters   run configuration (every row)
The simulator reads `kind=summary, metric=swap_cost_ms` (median switching minus median repeat latency) as the
per-switch penalty, and `deactivate_cost_ms` for switching to the base model.
"""
from __future__ import annotations

import argparse
import time

from bench.common import collect_env, summarize, write_csv

PATTERNS = ("repeat", "alternate", "base_toggle")
COLUMNS = ("kind", "pattern", "metric", "idx", "adapter", "prev_adapter", "switched", "cold", "latency_ms",
           "median_ms", "p90_ms", "mean_ms", "n", "model", "steps", "height", "width", "lora_scale", "n_adapters")
BASE = "base"


def import_vllm_omni():
    """(Omni, OmniDiffusionSamplingParams, LoRARequest, stable_lora_int_id); ImportError with a hint otherwise."""
    try:
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.lora.request import LoRARequest
        from vllm_omni.lora.utils import stable_lora_int_id
    except ImportError as e:
        raise ImportError("bench.vllm_omni_swap_cost needs vllm-omni==0.32.0 (and vllm) on a CUDA machine: "
                          f"pip install vllm-omni==0.32.0 ({e})") from e
    return Omni, OmniDiffusionSamplingParams, LoRARequest, stable_lora_int_id


def build_schedule(pattern: str, names: list[str], n: int) -> list[str]:
    """Adapter name per request; BASE means no LoRA."""
    if not names:
        raise ValueError("need at least one adapter")
    if pattern == "repeat":
        return [names[0]] * n
    if pattern == "alternate":
        if len(names) < 2:
            raise ValueError("pattern 'alternate' needs >= 2 adapters")
        return [names[i % len(names)] for i in range(n)]
    if pattern == "base_toggle":
        return [names[0] if i % 2 == 0 else BASE for i in range(n)]
    raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")


def request_rows(pattern: str, schedule: list[str], latencies_ms: list[float], seen: set[str], config: dict,
                 prev: str = "") -> list[dict]:
    """Rows for one pattern. `seen` (adapters already used in the engine) is updated in place."""
    rows = []
    for i, (ad, lat) in enumerate(zip(schedule, latencies_ms, strict=True)):
        cold = int(ad != BASE and ad not in seen)
        seen.add(ad)
        rows.append(dict(kind="request", pattern=pattern, metric="", idx=i, adapter=ad, prev_adapter=prev,
                         switched=int(bool(prev) and ad != prev), cold=cold, latency_ms=float(lat), **config))
        prev = ad
    return rows


_RAW = object()   # no derived value: report the samples' own median / p90 / mean


def _stats_row(pattern: str, metric: str, xs: list[float], config: dict, value=_RAW) -> dict:
    """value=None means "derived but unmeasurable" and stays None; it never falls back to the raw latency median."""
    s = summarize(xs) if xs else {"median": None, "p90": None, "mean": None, "n": 0}
    raw = value is _RAW
    return dict(kind="summary", pattern=pattern, metric=metric, median_ms=s["median"] if raw else value,
                p90_ms=s["p90"] if raw else None, mean_ms=s["mean"] if raw else None, n=s["n"], **config)


def summary_rows(rows: list[dict], config: dict) -> list[dict]:
    """Per-pattern latency stats plus derived swap_cost_ms, deactivate_cost_ms and cold_load_ms (median deltas
    vs warm repeat requests). A derived value is None when its inputs are missing."""
    warm = [r for r in rows if r["kind"] == "request" and not r["cold"]]
    by = lambda pred: [r["latency_ms"] for r in warm if pred(r)]
    out = [_stats_row(p, "latency_ms", by(lambda r, p=p: r["pattern"] == p), config)
           for p in dict.fromkeys(r["pattern"] for r in rows if r["kind"] == "request")]
    rep = by(lambda r: r["pattern"] == "repeat" and r["idx"] > 0)
    rep_med = summarize(rep)["median"] if rep else None

    def delta(xs):
        return None if rep_med is None or not xs else summarize(xs)["median"] - rep_med

    sw = by(lambda r: r["pattern"] == "alternate" and r["switched"])
    to_lora = by(lambda r: r["pattern"] == "base_toggle" and r["switched"] and r["adapter"] != BASE)
    to_base = by(lambda r: r["pattern"] == "base_toggle" and r["switched"] and r["adapter"] == BASE)
    base_med = summarize(to_base)["median"] if to_base else None
    cold = [r["latency_ms"] for r in rows if r["kind"] == "request" and r["cold"]]
    out.append(_stats_row("alternate", "swap_cost_ms", sw, config, delta(sw)))
    # A base request also skips the LoRA math, so deactivate_cost_ms = base-after-adapter median minus warm
    # repeat median can be negative (LoRA compute saved > deactivation cost). Read it as an upper bound.
    out.append(_stats_row("base_toggle", "deactivate_cost_ms", to_base, config,
                          None if rep_med is None or base_med is None else base_med - rep_med))
    out.append(_stats_row("base_toggle", "swap_cost_ms", to_lora, config, delta(to_lora)))
    out.append(_stats_row("register", "cold_load_ms", cold, config, delta(cold)))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m bench.vllm_omni_swap_cost",
                                 description="vLLM-Omni diffusion adapter switch cost (GPU only)")
    ap.add_argument("--model", default="Qwen/Qwen-Image")
    ap.add_argument("--adapter", action="append", required=True, dest="adapters",
                    help="LoRA path (PEFT dir or .safetensors); repeat for N adapters (>= 2 for 'alternate')")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--requests", type=int, default=12, help="requests per pattern")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="a photo of a red fox in the snow")
    ap.add_argument("--max-cpu-loras", type=int, default=None, help="engine adapter cache (default: N adapters)")
    ap.add_argument("--out", default="results/vllm_omni_swap_cost.csv")
    a = ap.parse_args(argv)
    a.patterns = [p for p in a.patterns.split(",") if p]
    bad = [p for p in a.patterns if p not in PATTERNS]
    if bad:
        ap.error(f"unknown pattern(s) {bad}; choose from {PATTERNS}")
    if "alternate" in a.patterns and len(a.adapters) < 2:
        ap.error("pattern 'alternate' needs >= 2 --adapter")
    if a.requests < 2:
        ap.error("--requests must be >= 2")
    return a


def run(a: argparse.Namespace) -> list[dict]:  # pragma: no cover - needs a GPU and vllm-omni
    Omni, SP, LoRARequest, stable_id = import_vllm_omni()
    names = [f"ad{i}" for i in range(len(a.adapters))]
    reqs = {n: LoRARequest(n, stable_id(p), p) for n, p in zip(names, a.adapters)}
    omni = Omni(model=a.model, max_cpu_loras=a.max_cpu_loras or len(names))
    config = {"model": a.model, "steps": a.steps, "height": a.height, "width": a.width, "lora_scale": a.lora_scale,
                  "n_adapters": len(names)}

    def one(ad: str) -> float:
        sp = SP(height=a.height, width=a.width, num_inference_steps=a.steps, seed=a.seed,
                lora_request=None if ad == BASE else reqs[ad], lora_scale=a.lora_scale)
        t0 = time.perf_counter()
        omni.generate([a.prompt], [sp], use_tqdm=False)
        return (time.perf_counter() - t0) * 1e3

    try:
        one(BASE)                                            # engine / compile warmup, not recorded
        seen: set[str] = set()
        rows = request_rows("register", names, [one(n) for n in names], seen, config)
        prev = names[-1]
        for p in a.patterns:
            sched = build_schedule(p, names, a.requests)
            rows += request_rows(p, sched, [one(n) for n in sched], seen, config, prev=prev)
            prev = sched[-1]
    finally:
        close = getattr(omni, "close", None)
        if close:
            close()
    return rows + summary_rows(rows, config)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU and vllm-omni
    a = parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("bench.vllm_omni_swap_cost needs a CUDA GPU")
    rows = run(a)
    write_csv(a.out, [{c: r.get(c) for c in COLUMNS} for r in rows], collect_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
