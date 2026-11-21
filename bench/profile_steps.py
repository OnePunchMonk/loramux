"""Step-time profiler: measures the tables the replay simulator (L2.6) interpolates over.

Output CSV (written with bench.common.write_csv; `# key=value` env header, then one row per measurement):

    model          str    target name (e.g. 'qwen_image_1024', 'toy')
    resolution     int    image side in pixels (0 when the target has no resolution)
    bs             int    requests in the batch (rows are bs * rows_per_req)
    rows_per_req   int    1, or 2 when CFG doubles the rows
    kind           str    'base'    no LoRA (T_base(bs));
                          'uniform' LoRA on, every request uses the same adapter, k = 1 (T_uniform(bs));
                          'mixed'   LoRA on, k >= 2 distinct adapters spread round-robin (T_mixed(bs, k))
    k              int    distinct adapters in the batch: 0 for base, 1 for uniform, >= 2 for mixed (k <= bs)
    targets        str    'base' for kind=base, else 'attn' or 'all' (which linears carry adapters)
    rank           int    LoRA rank of every adapter (0 for base)
    mode           str    'eager' or 'compile'
    median_ms, p90_ms, mean_ms  float   per-forward time in milliseconds
    n              int    number of samples
    timer          str    'cuda' (CUDA events) or 'cpu' (perf_counter); CPU timings are for smoke tests only
    compile_scope  str    'none' (eager), 'regional' (repeated blocks compiled) or 'full' (whole forward)
    first_call_s   float  wall time of the config's first call, including its compile if it needed one
    n_patched      int    linears carrying LoRA in this row (0 for base); must match the eager patch set

Grid: bs in BATCH_SIZES = (1, 2, 4, 8, 16), resolutions (512, 1024), k in (2, 4, 8, 16) restricted to k <= bs,
for each targets in `adapter_sets`. Requests are always (bs, k) points that exist; the simulator interpolates
between them and must not extrapolate beyond bs=16 or k=bs. Rows for one (model, resolution, mode) share the
base measurement, so T_base is repeated across targets and appears once per (bs, mode).
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from bench.common import collect_env, interleave, seed_all, summarize, timer_kind, write_csv
from bench.step_bench import (DiffusersStepTarget, LoRAHarness, StepTarget, ToyDiTTarget, _need_warmup,
                               linear_shapes, token_linears)

BATCH_SIZES = (1, 2, 4, 8, 16)
RESOLUTIONS = (512, 1024)
K_VALUES = (2, 4, 8, 16)


def count_shapes(batch_sizes, k_values, n_adapter_sets: int) -> tuple[int, int]:
    """(timed configs, compiles) per resolution and mode. A compile is one (bs, patched-or-not) graph: adapter
    variants that differ only in k reuse it (the adapter table is data, not part of the graph)."""
    configs = len(batch_sizes)   # base
    for bs in batch_sizes:
        configs += n_adapter_sets * (1 + len([k for k in k_values if k <= bs]))
    compiles = len(batch_sizes) * (1 + n_adapter_sets)
    return configs, compiles


def setup_compile(target: StepTarget, scope: str, backend: str) -> tuple[Callable, str]:
    """Compile once per target, not once per shape (the old path reset dynamo for every shape).

    scope 'regional': diffusers' compile_repeated_blocks (one transformer block compiled, reused by all of them);
    falls back to 'full' (torch.compile of the whole forward) for models without `_repeated_blocks`.
    Shapes stay static (automatic dynamic shapes off): each batch size gets its own graph, like serving buckets."""
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.config.automatic_dynamic_shapes = False
    dynamo.config.cache_size_limit = 256
    dynamo.config.accumulated_cache_size_limit = 4096
    model = target.model
    if scope == "regional" and getattr(model, "_repeated_blocks", None) and hasattr(model, "compile_repeated_blocks"):
        model.compile_repeated_blocks(backend=backend)
        return target.forward, "regional"
    return torch.compile(target.forward, backend=backend), "full"


def profile_target(target: StepTarget, *, resolution: int = 0, batch_sizes=BATCH_SIZES, k_values=K_VALUES,
                   adapter_sets=("all",), rank: int = 16, modes=("eager",), rows_per_req: int = 1,
                   compile_backend: str = "inductor", compile_scope: str = "regional", rounds: int = 2,
                   warmup: int = 2, iters: int = 5, seed: int = 0,
                   on_row: Callable[[dict], None] | None = None) -> list[dict]:
    """Rows of the CSV described in the module docstring for one target (one resolution).

    Modes run one after another (eager first: regional compile changes the model in place). Every row is passed
    to on_row as soon as it is measured, so a stopped run keeps what it measured. compile_s is the wall time of
    the first call of each config (includes the compile, if any)."""
    import time
    _need_warmup(modes, warmup)
    seed_all(seed)
    timer = timer_kind()
    rows: list[dict] = []
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def first_call(fn) -> float:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        return time.perf_counter() - t0

    def emit(bs, kind, k, targets, mode, times, compile_s, scope, n_patched=0):
        s = summarize(times)
        row = {"model": target.name, "resolution": resolution, "bs": bs, "rows_per_req": rows_per_req, "kind": kind,
               "k": k, "targets": targets, "rank": 0 if kind == "base" else rank, "mode": mode,
               "median_ms": s["median"], "p90_ms": s["p90"], "mean_ms": s["mean"], "n": s["n"], "timer": timer,
               "compile_scope": scope, "first_call_s": round(compile_s, 3), "n_patched": n_patched}
        rows.append(row)
        if on_row:
            on_row(row)

    def run(fn) -> tuple[list[float], float]:
        c = first_call(fn)
        return interleave({"x": fn}, rounds=rounds, warmup=warmup, iters=iters)["x"], c

    order = [m for m in ("eager", "compile") if m in modes] + [m for m in modes if m not in ("eager", "compile")]
    with torch.no_grad():
        inputs = {bs: target.make_inputs(bs, rows_per_req) for bs in batch_sizes}
        n_ad = max([1] + [k for k in k_values if k <= max(batch_sizes)])
        # Find the linears to patch once, in eager mode, before anything is compiled (token_linears uses forward
        # hooks; regionally compiled blocks do not run hooks added later, so it would find only the linears
        # outside the blocks).
        patch_sets = {a: token_linears(target, linear_shapes(target.model, a)) for a in adapter_sets}
        for mode in order:
            if mode == "compile":
                fwd, scope = setup_compile(target, compile_scope, compile_backend)
            elif mode == "eager":
                fwd, scope = target.forward, "none"
            else:
                raise ValueError(f"profile_steps supports modes eager and compile, got {mode!r}")
            for bs in batch_sizes:  # base first: the model must be unpatched
                t, c = run(lambda bs=bs: fwd(**inputs[bs]))
                emit(bs, "base", 0, "base", mode, t, c, scope)
            for aset in adapter_sets:
                h = LoRAHarness(target, targets=aset, n_adapters=n_ad, rank=rank, seed=seed, shapes=patch_sets[aset])
                try:
                    for bs in batch_sizes:
                        for k in (1, *[k for k in k_values if k <= bs]):
                            t, c = run(h.step_fn(inputs[bs], bs, k, rows_per_req, forward=fwd))
                            emit(bs, "uniform" if k == 1 else "mixed", k, aset, mode, t, c, scope, h.n_patched)
                finally:
                    h.close()
    return rows


def profile_steps(target_factory: Callable[[int], StepTarget], *, resolutions=RESOLUTIONS, **kw) -> list[dict]:
    """target_factory(resolution) -> StepTarget. Concatenates profile_target rows over the resolutions."""
    rows: list[dict] = []
    for res in resolutions:
        rows.extend(profile_target(target_factory(res), resolution=res, **kw))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.profile_steps")
    ap.add_argument("--kind", choices=["toy", "qwen_image", "flux2"], default="toy")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--resolutions", default="512,1024")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--adapter-sets", default="all", help="comma list of attn,all")
    ap.add_argument("--k-values", default=",".join(map(str, K_VALUES)), help="distinct-adapter counts for mixed rows")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--modes", default="eager")
    ap.add_argument("--cfg", action="store_true")
    ap.add_argument("--compile-scope", choices=("regional", "full"), default="regional")
    ap.add_argument("--min-per-compile", type=float, default=None,
                    help="estimate input: minutes per compile (default 1.0 regional, 3.5 full)")
    ap.add_argument("--log-compiles", action="store_true", help="log dynamo recompiles and graph breaks")
    ap.add_argument("--out", default="results/step_profile.csv")
    a = ap.parse_args(argv)
    if a.log_compiles:
        torch._logging.set_logs(recompiles=True, graph_breaks=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def factory(res: int) -> StepTarget:
        if a.kind == "toy":
            return ToyDiTTarget(n_img=max(4, (res // 16) ** 2 // 64), device=dev)
        return DiffusersStepTarget(a.kind, model_id=a.model_id, subfolder=a.subfolder, resolution=res)

    import json
    import time
    res_list = [int(r) for r in a.resolutions.split(",")]
    bss = tuple(int(b) for b in a.batch_sizes.split(","))
    asets = tuple(a.adapter_sets.split(","))
    modes = tuple(a.modes.split(","))
    kvals = tuple(int(k) for k in a.k_values.split(",") if k)
    n_cfg, n_comp = count_shapes(bss, kvals, len(asets))
    n_comp_total = n_comp * len(res_list) * sum(m == "compile" for m in modes)
    per = a.min_per_compile if a.min_per_compile is not None else (1.0 if a.compile_scope == "regional" else 3.5)
    print(f"PLAN {n_cfg * len(res_list) * len(modes)} timed configs, {n_comp_total} compiles "
          f"({a.compile_scope}); compile estimate {n_comp_total * per:.0f} min at {per} min each", flush=True)
    partial = str(a.out) + ".partial.csv"
    t0 = time.perf_counter()
    streamed: list[dict] = []

    def on_row(row):
        streamed.append(row)
        write_csv(partial, streamed, collect_env())   # rewritten each row: a stopped run keeps every finished row
        print("ROW " + json.dumps(row), flush=True)
        print(f"PROGRESS {len(streamed)}/{n_cfg * len(res_list) * len(modes)} rows, "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    rows = profile_steps(factory, resolutions=res_list, batch_sizes=bss, k_values=kvals, adapter_sets=asets, rank=a.rank,
                         modes=modes, rows_per_req=2 if a.cfg else 1, compile_scope=a.compile_scope,
                         on_row=on_row)
    write_csv(a.out, rows, collect_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
