"""Kill test (HANDOFF §4 Phase 1, §5): run every stage, evaluate the pre-registered bars, write the verdict.

    python -m bench.killtest run      --model-id Qwen/Qwen-Image --kind qwen_image --adapter a.safetensors ...
    python -m bench.killtest evaluate --results results/killtest     # writes RESULTS_killtest.md

`run` executes the stages below in order (each is an existing bench CLI), logs each stage's outcome to
`<results>/runlog.csv`, and checks the spend caps once, before the first GPU stage, against the summed estimates.
A failed stage does not stop the others; `evaluate` marks whatever is missing as not measured.

Verdict (HANDOFF §5): GO = K1, K2 and K3 pass; GREY = K1 and K2 pass, K3 fails; NO-GO = K1 or K2 fails;
INCOMPLETE = a deciding bar was not measured. Bars come from `bench.bars` and are never edited here.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from bench.bars import BARS, K2, check, check_group
from bench.common import collect_env, read_csv, write_csv
from bench.k1_correctness import summarize_k1
from bench.spend import check_caps, estimate, read_rows, spent

H100_USD_PER_HOUR = 3.95  # keep in sync with modal_app.py
F_OVER_BW = 230.0         # achievable bf16 FLOP/s / achievable HBM B/s on H100 (HANDOFF §3.6)
K3_RHOS = (0.7, 0.9)
K2A_TOKENS = 4096


@dataclass
class Config:
    results: Path
    kind: str = "qwen_image"
    model_id: str = "Qwen/Qwen-Image"
    adapters: list[str] = field(default_factory=list)
    resolutions: tuple[int, ...] = (512, 1024)
    dim: int = 3072
    vllm_omni: bool = False


@dataclass
class Stage:
    name: str
    module: str
    args: list[str]
    gpu_minutes: float  # 0 = CPU-only stage


def stages(c: Config) -> list[Stage]:
    r, ad = c.results, [x for p in c.adapters for x in ("--adapter", p)]
    res = ",".join(map(str, c.resolutions))
    out = [
        Stage("k1a", "bench.k1_correctness", ["--checks", "k1a", "--out", f"{r}/k1a.csv"], 2),
        Stage("k2a", "bench.linear_chain", ["--dim", str(c.dim), "--out", f"{r}/k2a.csv"], 4),
        Stage("k2b", "bench.step_bench", ["--kind", c.kind, "--model-id", c.model_id, "--resolution", "1024",
                                          "--modes", "eager,compile", "--out", f"{r}/k2b_{c.kind}.csv"], 8),
        Stage("profile", "bench.profile_steps", ["--kind", c.kind, "--model-id", c.model_id, "--resolutions", res,
                                                 "--adapter-sets", "all", "--modes", "compile",
                                                 "--out", f"{r}/profile_{c.kind}.csv"], 15),
        Stage("k1bc", "bench.k1_correctness", ["--checks", "k1b,k1c", "--kind", c.kind, "--model-id", c.model_id,
                                               *ad, "--out", f"{r}/k1bc.csv"], 20),
    ]
    if c.vllm_omni:
        out.append(Stage("swap", "bench.vllm_omni_swap_cost", ["--model", c.model_id, *ad,
                                                               "--out", f"{r}/swap.csv"], 15))
    for res_ in c.resolutions:  # CPU-only: replays the trace on the measured tables
        out.append(Stage(f"k3_{res_}", "bench.sim.replay_sim",
                         ["--profile", f"{r}/profile_{c.kind}.csv", "--resolution", str(res_), "--mode", "compile",
                          "--targets", "all", "--rhos", ",".join(map(str, K3_RHOS)),
                          "--out", f"{r}/sim_{c.kind}_{res_}"], 0))
    return out


def measured_swap_ms(results: Path) -> float | None:
    """Median vLLM-Omni switch cost from the swap stage, if it ran (bench.vllm_omni_swap_cost summary rows)."""
    p = results / "swap.csv"
    if not p.exists():
        return None
    vals = [float(r["median_ms"]) for r in read_csv(p)[1] if r.get("metric") == "swap_cost_ms"
            and r.get("median_ms") not in (None, "")]
    return vals[0] if vals else None


def run(c: Config, only: set[str] | None = None, dry_run: bool = False, spend_path: str = "SPEND.md") -> list[dict]:
    c.results.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stages(c) if only is None or s.name in only]
    gpu_min = sum(s.gpu_minutes for s in todo)
    if gpu_min:
        check_caps(spend_path, new_est=estimate("H100", gpu_min, H100_USD_PER_HOUR))
    log = []
    for s in todo:
        args = list(s.args)
        if s.name.startswith("k3_") and (sw := measured_swap_ms(c.results)) is not None:
            args += ["--swap-ms", f"{sw:.3f}"]  # measured vLLM-Omni switch cost beats the simulator default
        cmd = [sys.executable, "-m", s.module, *args]
        t0 = time.time()
        rc = 0 if dry_run else subprocess.run(cmd, check=False).returncode
        log.append({"stage": s.name, "returncode": rc, "seconds": round(time.time() - t0, 1),
                    "est_gpu_minutes": s.gpu_minutes, "cmd": " ".join(cmd)})
    write_csv(c.results / "runlog.csv", log, collect_env())
    return log


# ---------------------------------------------------------------- evaluation

def _rows(path: Path) -> list[dict]:
    return read_csv(path)[1] if path.exists() else []


def k2a_value(rows: list[dict]) -> tuple[float | None, dict]:
    """loramux (B4) overhead / vLLM-Omni-pattern (B2) overhead at 4096 tokens."""
    at = {r["baseline"]: r for r in rows if r.get("status") == "ok" and int(r.get("tokens_requested") or 0) == K2A_TOKENS}
    b4, b2 = (at.get(b, {}).get("overhead_pct") for b in ("B4", "B2"))
    if b4 is None or b2 is None:
        return None, at
    return (math.inf if b2 <= 0 else b4 / b2), at


def k2b_values(results: Path) -> tuple[dict[str, float | None], list[dict]]:
    """Worst compiled overhead over every measured model, per adapter set (each model must meet the bar)."""
    per_model = [dict(r, file=p.name) for p in sorted(results.glob("k2b_*.csv")) for r in _rows(p)
                 if r.get("mode") == "compile" and r.get("adapters") in ("attn", "all")]
    worst = {}
    for aset, bar in (("attn", "K2b_attn"), ("all", "K2b_all")):
        xs = [float(r["overhead_pct"]) for r in per_model if r["adapters"] == aset]
        worst[bar] = max(xs) if xs else None
    return worst, per_model


def k3_regimes(results: Path) -> list[dict]:
    """Per (model, resolution) file: P2/P1 throughput and P99 ratios at each rho, and whether the regime passes
    (at every rho in K3_RHOS: throughput >= 1.5x or P99 <= 0.5x)."""
    thr_bar, p99_bar = BARS["K3_throughput"], BARS["K3_p99"]
    out = []
    for p in sorted(results.glob("sim_*_summary.csv")):
        by = {(float(r["rho"]), r["policy"]): r for r in _rows(p)}
        per_rho = []
        for rho in K3_RHOS:
            p1, p2 = by.get((rho, "P1")), by.get((rho, "P2"))
            if not (p1 and p2):
                per_rho = None
                break
            thr = float(p2["throughput_rps"]) / float(p1["throughput_rps"])
            p99 = float(p2["p99_ms"]) / float(p1["p99_ms"])
            per_rho.append({"rho": rho, "thr_ratio": thr, "p99_ratio": p99,
                            "ok": bool(check(thr_bar, thr).passed or check(p99_bar, p99).passed)})
        out.append({"regime": p.name.removeprefix("sim_").removesuffix("_summary.csv"), "per_rho": per_rho,
                    "passed": None if per_rho is None else all(x["ok"] for x in per_rho)})
    return out


def roofline(rows: list[dict], dim: int) -> list[dict]:
    """Measured B4 overhead per token count vs the v1 prediction (F/BW)·(2/d_in + 1/d_out) for a d->d chain."""
    pred = F_OVER_BW * (2 / dim + 1 / dim) * 100
    return [{"tokens": r["tokens"], "measured_pct": r["overhead_pct"], "predicted_pct": pred,
             "ratio": r["overhead_pct"] / pred, "profile_with_ncu": r["overhead_pct"] / pred > 1.3}
            for r in rows if r.get("baseline") == "B4" and r.get("status") == "ok" and r.get("overhead_pct") is not None]


def verdict(k1: bool | None, k2: bool | None, k3: bool | None) -> str:
    if k1 is False or k2 is False:
        return "NO-GO"
    if k1 is None or k2 is None or k3 is None:
        return "INCOMPLETE"
    return "GO" if k3 else "GREY"


def evaluate(results: Path, dim: int = 3072, spend_path: str = "SPEND.md") -> dict:
    k1 = summarize_k1(_rows(results / "k1a.csv") + _rows(results / "k1bc.csv"))
    k2a, k2a_rows = k2a_value(_rows(results / "k2a.csv"))
    k2b, k2b_rows = k2b_values(results)
    k2 = check_group(K2, {"K2a": k2a, **k2b})
    regimes = k3_regimes(results)
    k3 = None if not regimes or all(r["passed"] is None for r in regimes) else any(r["passed"] for r in regimes)
    spend = spent(read_rows(spend_path), "1")
    return {"verdict": verdict(k1.passed, k2.passed, k3), "k1": k1, "k2": k2, "k2a_rows": k2a_rows,
            "k2b_rows": k2b_rows, "k3": k3, "k3_regimes": regimes, "roofline": roofline(list(k2a_rows.values()), dim),
            "swap_ms": measured_swap_ms(results), "spend_usd": spend}


def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def render(ev: dict) -> str:
    tag = {True: "PASS", False: "FAIL", None: "N/A"}
    reason = {"GO": "K1, K2 and K3 pass.", "GREY": "K1 and K2 pass; K3 (system value) fails. Upstream the kernel only.",
              "NO-GO": "K1 (correctness) or K2 (kernel overhead) fails.",
              "INCOMPLETE": "A deciding bar was not measured; see the N/A rows."}[ev["verdict"]]
    L = [f"# Kill test results\n\n**Verdict: {ev['verdict']}.** {reason}\n",
         f"GPU spend, phase 1: {ev['spend_usd']:.2f} USD (cap 25).\n", "## Bars\n", "| group | bar | result |", "|---|---|---|"]
    for g in (ev["k1"], ev["k2"]):
        L += [f"| {g.group_id} | {v.bar_id} | {tag[v.passed]}: {v.detail} |" for v in g.verdicts]
    L += [f"| K3 | any regime | {tag[ev['k3']]} |", "", "## K2b per model (compiled)\n",
          "| file | adapters | overhead % |", "|---|---|---|"]
    L += [f"| {r['file']} | {r['adapters']} | {_fmt(r['overhead_pct'])} |" for r in ev["k2b_rows"]]
    L += ["", "## K3 per regime (P2 vs P1)\n", "| regime | rho | throughput ratio | P99 ratio | ok |", "|---|---|---|---|---|"]
    for g in ev["k3_regimes"]:
        L += [f"| {g['regime']} | {x['rho']} | {x['thr_ratio']:.3g} | {x['p99_ratio']:.3g} | {x['ok']} |"
              for x in g["per_rho"] or []] or [f"| {g['regime']} | — | — | — | missing rows |"]
    L += ["", "## Roofline (K2a chain, B4)\n", "| tokens | measured % | predicted % | ratio | ncu? |", "|---|---|---|---|---|"]
    L += [f"| {r['tokens']} | {r['measured_pct']:.3g} | {r['predicted_pct']:.3g} | {r['ratio']:.2f} | "
          f"{'yes' if r['profile_with_ncu'] else 'no'} |" for r in ev["roofline"]]
    b2 = ev["k2a_rows"].get("B2", {})
    b5 = ev["k2a_rows"].get("B5", {})
    L += ["", "## Threats to validity\n",
          f"- B2 is {b2.get('note') or 'not measured'}.",
          f"- B5 (vLLM LoRA ops): status {b5.get('status', 'not measured')}.",
          f"- K3 swap cost: {'measured ' + _fmt(ev['swap_ms']) + ' ms' if ev['swap_ms'] is not None else 'simulator default (not measured)'}.",
          "- K3 uses a synthetic trace fitted to the published GenTD26 statistics unless a real trace was loaded.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.killtest")
    ap.add_argument("cmd", choices=("run", "evaluate", "plan"))
    ap.add_argument("--results", default="results/killtest")
    ap.add_argument("--kind", default="qwen_image", choices=("qwen_image", "flux2"))
    ap.add_argument("--model-id", default="Qwen/Qwen-Image")
    ap.add_argument("--adapter", action="append", default=[], dest="adapters")
    ap.add_argument("--resolutions", default="512,1024")
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--vllm-omni", action="store_true", help="also measure vLLM-Omni switch cost (needs vllm-omni)")
    ap.add_argument("--only", default=None, help="comma list of stage names")
    ap.add_argument("--out", default="RESULTS_killtest.md")
    a = ap.parse_args(argv)
    c = Config(Path(a.results), a.kind, a.model_id, a.adapters, tuple(int(x) for x in a.resolutions.split(",")),
               a.dim, a.vllm_omni)
    if a.cmd == "plan":
        for s in stages(c):
            print(f"{s.name:10s} {s.gpu_minutes:5.0f} GPU-min  python -m {s.module} {' '.join(s.args)}")
        return 0
    if a.cmd == "run":
        log = run(c, set(a.only.split(",")) if a.only else None)
        return 0 if all(r["returncode"] == 0 for r in log) else 1
    ev = evaluate(c.results, a.dim)
    Path(a.out).write_text(render(ev))
    print(f"{ev['verdict']} -> {a.out}")
    return 0


__all__ = ["Config", "evaluate", "render", "run", "stages", "verdict"]

if __name__ == "__main__":
    raise SystemExit(main())
