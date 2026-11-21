"""Phase 2b evaluation (HANDOFF §11.5): bars L1-L5 and P1-P3 from result CSVs, lossy-gate decisions,
RESULTS_latency.md.

    python -m bench.latency_bars --mixed-07 m07.csv --mixed-09 m09.csv --keyed-09 k09.csv \
        --steps m07_steps.csv --payload payload.csv --lossy lossy.csv --out RESULTS_latency.md

Inputs (any may be omitted; the bars it feeds are then "not measured"):
  latency CSVs   bench.serve_loop latency rows (LatencyLog.to_rows + policy), one run per (policy, rho)
  steps CSV      bench.serve_loop step rows (t, bs, k_distinct, step_ms, policy, swap_ms, post_ms), mixed, rho=0.7
  payload CSV    bench.payload_bench: adapter, rank, targets, source_bytes, blob_bytes, load_ms, h2d_gbps
  lossy CSV      bench.lossy_quality: variant (fp8|svd), adapter, prompt_idx, psnr, lpips, bytes_exact,
                 bytes_variant, slots_per_gb_exact, slots_per_gb_variant

Metric definitions (one place):
  L1  P50 e2e of LoRA requests / P50 e2e of base requests, mixed, rho=0.7 (same run)
  L2  P99 e2e mixed / P99 e2e keyed, rho=0.9
  L3  P90 adapter_wait of CPU-hit requests / median denoise step_ms, mixed, rho=0.7
  L4  rank-32 attention-only adapter: median load_ms, and achieved bandwidth (blob_bytes / load) / measured H2D
  L5  share of the denoise loop's main-thread time spent handing off VAE + encode: sum(post_ms) / sum(step_ms + post_ms)
  P1  median blob_bytes / source_bytes
  P2  FP8 vs exact: mean PSNR, mean LPIPS, median slots-per-GB ratio
  P3  SVD-truncated vs exact: mean PSNR, mean LPIPS, median bytes saved
A lossy variant becomes a default only if its quality bars pass (HANDOFF §11.2: off by default otherwise).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bench.bars import BARS, GROUPS, check, check_group
from bench.common import read_csv

PCTL = {"p50": 50, "p90": 90, "p99": 99}


def _rows(path) -> list[dict]:
    return read_csv(path)[1] if path and Path(path).exists() else []


def _pct(xs, q) -> float | None:
    xs = [float(x) for x in xs if x not in (None, "")]
    return float(np.percentile(xs, PCTL[q])) if xs else None


def _ratio(a, b) -> float | None:
    return None if a is None or b in (None, 0) else a / b


def _is_lora(r) -> bool:
    return r.get("adapter_key") not in (None, "")


def l1(mixed07):
    return _ratio(_pct([r["e2e"] for r in mixed07 if _is_lora(r)], "p50"),
                  _pct([r["e2e"] for r in mixed07 if not _is_lora(r)], "p50"))


def l2(mixed09, keyed09):
    return _ratio(_pct([r["e2e"] for r in mixed09], "p99"), _pct([r["e2e"] for r in keyed09], "p99"))


def l3(mixed07, steps):
    return _ratio(_pct([r["adapter_wait"] for r in mixed07 if r.get("adapter_source") == "cpu_hit"], "p90"),
                  _pct([r["step_ms"] for r in steps], "p50"))


def l4(payload):
    rows = [r for r in payload if int(r.get("rank") or 0) == 32 and r.get("targets") == "attn"]
    ms = _pct([r["load_ms"] for r in rows], "p50")
    bw = _pct([(r["blob_bytes"] / (r["load_ms"] / 1e3)) / (r["h2d_gbps"] * 1e9) for r in rows if r["load_ms"]], "p50")
    return ms, bw


def l5(steps):
    if not steps or any("post_ms" not in r for r in steps):
        return None
    post = sum(float(r["post_ms"]) for r in steps)
    return post / (post + sum(float(r["step_ms"]) for r in steps))


def p_values(lossy, variant):
    rows = [r for r in lossy if r.get("variant") == variant]
    mean = lambda c: float(np.mean([r[c] for r in rows])) if rows else None
    return {"psnr": mean("psnr"), "lpips": mean("lpips"),
            "slots": _pct([r["slots_per_gb_variant"] / r["slots_per_gb_exact"] for r in rows], "p50"),
            "saved": _pct([r["bytes_exact"] - r["bytes_variant"] for r in rows], "p50")}


def evaluate(*, mixed07=None, mixed09=None, keyed09=None, steps=None, payload=None, lossy=None) -> dict:
    m07, m09, k09, st = _rows(mixed07), _rows(mixed09), _rows(keyed09), _rows(steps)
    pay, los = _rows(payload), _rows(lossy)
    l4_ms, l4_bw = l4(pay)
    fp8, svd = p_values(los, "fp8"), p_values(los, "svd")
    singles = {b: check(BARS[b], v) for b, v in (("L1", l1(m07)), ("L2", l2(m09, k09)), ("L3", l3(m07, st)),
                                                ("L5", l5(st)),
                                                ("P1", _pct([r["blob_bytes"] / r["source_bytes"] for r in pay], "p50")))}
    groups = {"L4": check_group(GROUPS["L4"], {"L4_ms": l4_ms, "L4_bw": l4_bw}),
              "P2": check_group(GROUPS["P2"], {"P2_psnr": fp8["psnr"], "P2_lpips": fp8["lpips"], "P2_slots": fp8["slots"]}),
              "P3": check_group(GROUPS["P3"], {"P3_psnr": svd["psnr"], "P3_lpips": svd["lpips"], "P3_bytes": svd["saved"]})}
    decisions = {"fp8_storage_default": groups["P2"].passed is True, "svd_truncation_default": groups["P3"].passed is True}
    return {"singles": singles, "groups": groups, "decisions": decisions}


def render(ev: dict) -> str:
    tag = {True: "PASS", False: "FAIL", None: "N/A"}
    L = ["# Phase 2b latency and payload results\n", "| bar | result |", "|---|---|"]
    L += [f"| {v.bar_id} | {tag[v.passed]}: {v.detail} |" for v in ev["singles"].values()]
    for g in ev["groups"].values():
        L += [f"| {v.bar_id} ({g.group_id}: {tag[g.passed]}) | {tag[v.passed]}: {v.detail} |" for v in g.verdicts]
    d = ev["decisions"]
    L += ["", "## Lossy-gate decisions\n",
          f"- FP8 adapter storage on by default: **{'yes' if d['fp8_storage_default'] else 'no'}** (P2 must pass).",
          f"- SVD rank truncation on by default: **{'yes' if d['svd_truncation_default'] else 'no'}** (P3 must pass).",
          "", "L5 is `== 0`: any main-thread time handing off VAE + encode counts against it, however small.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.latency_bars")
    for f in ("mixed-07", "mixed-09", "keyed-09", "steps", "payload", "lossy"):
        ap.add_argument(f"--{f}", default=None)
    ap.add_argument("--out", default="RESULTS_latency.md")
    a = ap.parse_args(argv)
    ev = evaluate(mixed07=a.mixed_07, mixed09=a.mixed_09, keyed09=a.keyed_09, steps=a.steps, payload=a.payload,
                  lossy=a.lossy)
    Path(a.out).write_text(render(ev))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
