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
from bench.step_bench import DiffusersStepTarget, LoRAHarness, StepTarget, ToyDiTTarget, _compiled, _need_warmup

BATCH_SIZES = (1, 2, 4, 8, 16)
RESOLUTIONS = (512, 1024)
K_VALUES = (2, 4, 8, 16)


def profile_target(target: StepTarget, *, resolution: int = 0, batch_sizes=BATCH_SIZES, k_values=K_VALUES,
                   adapter_sets=("all",), rank: int = 16, modes=("eager",), rows_per_req: int = 1,
                   compile_backend: str = "inductor", rounds: int = 2, warmup: int = 2, iters: int = 5,
                   seed: int = 0) -> list[dict]:
    """Rows of the CSV described in the module docstring for one target (one resolution)."""
    _need_warmup(modes, warmup)
    seed_all(seed)
    timer = timer_kind()
    rows: list[dict] = []

    def emit(bs, kind, k, targets, mode, times):
        s = summarize(times)
        rows.append({"model": target.name, "resolution": resolution, "bs": bs, "rows_per_req": rows_per_req, "kind": kind,
                         "k": k, "targets": targets, "rank": 0 if kind == "base" else rank, "mode": mode,
                         "median_ms": s["median"], "p90_ms": s["p90"], "mean_ms": s["mean"], "n": s["n"], "timer": timer})

    def timed(fns: dict[str, Callable]) -> dict[str, list[float]]:
        built = {m: (_compiled(lambda f=f: f, compile_backend) if m == "compile" else f) for m, f in fns.items()}
        return interleave(built, rounds=rounds, warmup=warmup, iters=iters)

    with torch.no_grad():
        inputs = {bs: target.make_inputs(bs, rows_per_req) for bs in batch_sizes}
        for bs in batch_sizes:  # base first: the model must be unpatched
            t = timed({m: (lambda bs=bs: target.forward(**inputs[bs])) for m in modes})
            for m in modes:
                emit(bs, "base", 0, "base", m, t[m])
        n_ad = max([1] + [k for k in k_values if k <= max(batch_sizes)])
        for aset in adapter_sets:
            h = LoRAHarness(target, targets=aset, n_adapters=n_ad, rank=rank, seed=seed)
            try:
                for bs in batch_sizes:
                    for k in (1, *[k for k in k_values if k <= bs]):
                        fn = h.step_fn(inputs[bs], bs, k, rows_per_req)
                        t = timed({m: fn for m in modes})
                        for m in modes:
                            emit(bs, "uniform" if k == 1 else "mixed", k, aset, m, t[m])
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
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--modes", default="eager")
    ap.add_argument("--cfg", action="store_true")
    ap.add_argument("--out", default="results/step_profile.csv")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def factory(res: int) -> StepTarget:
        if a.kind == "toy":
            return ToyDiTTarget(n_img=max(4, (res // 16) ** 2 // 64), device=dev)
        return DiffusersStepTarget(a.kind, model_id=a.model_id, subfolder=a.subfolder, resolution=res)

    rows = profile_steps(factory, resolutions=[int(r) for r in a.resolutions.split(",")],
                         batch_sizes=tuple(int(b) for b in a.batch_sizes.split(",")),
                         adapter_sets=tuple(a.adapter_sets.split(",")), rank=a.rank,
                         modes=tuple(a.modes.split(",")), rows_per_req=2 if a.cfg else 1)
    write_csv(a.out, rows, collect_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
