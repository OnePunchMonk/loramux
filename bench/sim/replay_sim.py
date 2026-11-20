"""Discrete-event replay of a trace under P1/P2/P3, driven by measured step-time tables (L2.6).

The simulator is the serve loop itself (`bench.serve_loop.ServeLoop`) run with no model on a `VirtualClock`, a
tensor-free `SimPool`, and a `TableCost` that charges table times. So arrivals, step-boundary admission, text
encoding per tick, adapter residency/pinning, continuous batching, keyed group switching and the off-path post
worker are the same code as the real loop; only the time of each operation comes from tables.

Policies:
- P1 adapter-keyed (`keyed`): a batch holds one (adapter, scale) group, up to `max_bs` (vLLM's max_num_seqs).
  Step time T_uniform(bs) (T_base(bs) for base requests); every group switch costs `swap_ms` (measured: L2.5's
  step CSV swap_ms or L2.7). Adapters are merged from the CPU tier, so no H2D wait.
- P2 mixed (`mixed`): T_mixed(bs, k) (T_uniform for k = 1, T_base for k = 0). An adapter that is not GPU-resident
  costs bytes / bandwidth of H2D, issued at arrival, so it hides behind the text encode (adapter_wait =
  max(0, ready - text done)). LRU slots with refcounts, as the pool.
- P3 ideal mixed: as P2 but every step costs T_base(bs) (zero kernel overhead). H2D is still modelled, so
  P2 - P3 isolates the per-step LoRA overhead.

Step table input: the profile_steps CSV (bench/profile_steps.py docstring): columns model, resolution, bs,
rows_per_req, kind (base|uniform|mixed), k, targets, rank, mode, median_ms (used), ... Rows are filtered to one
(model, resolution, mode, targets, rows_per_req); base rows match any targets. Interpolation is piecewise linear
in bs, and in k at each measured bs (k = 1 anchors on T_uniform); a query with bs beyond the largest measured bs
raises instead of extrapolating.

Load: rho is offered load relative to the base model's saturation throughput,
rate(rho) = rho * max_bs / (steps * T_base(max_bs)), the same rate for every policy.

Outputs: summary rows (policy, rho, rate_rps, n, throughput_rps, p50_ms, p90_ms, p99_ms, swaps) and per-request
timelines in the L0.3 format (LatencyLog.to_rows()) plus policy and rho.
"""
from __future__ import annotations

import argparse
import bisect
from collections import OrderedDict

import numpy as np

from bench.common import collect_env, read_csv, write_csv
from bench.serve_loop import CostModel, ServeLoop, ServeResult, VirtualClock
from bench.trace.synth_trace import synth_trace

POLICY_MODE = {"P1": "keyed", "P2": "mixed", "P3": "mixed"}


def _interp(xs: list, ys: list, x: float) -> float:
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        raise ValueError(f"{x} is outside the measured range [{xs[0]}, {xs[-1]}]; not extrapolating")
    return float(np.interp(x, xs, ys))


class StepTable:
    """T_base(bs), T_uniform(bs), T_mixed(bs, k) in ms."""

    def __init__(self, base: dict[int, float], uniform: dict[int, float], mixed: dict[tuple[int, int], float]):
        if not base:
            raise ValueError("step table has no base rows")
        self.base, self.uniform, self.mixed = dict(base), dict(uniform), dict(mixed)
        self.bs_grid = sorted(self.base)

    @classmethod
    def from_rows(cls, rows: list[dict], *, model=None, resolution=None, mode="eager", targets="all",
                  rows_per_req=1) -> StepTable:
        want = {"model": model, "resolution": resolution, "mode": mode, "rows_per_req": rows_per_req}
        sel = [r for r in rows if all(v is None or r.get(f) == v for f, v in want.items())]
        sel = [r for r in sel if r["kind"] == "base" or r.get("targets") == targets]
        for f in ("model", "resolution"):
            vals = {r[f] for r in sel}
            if len(vals) > 1:
                raise ValueError(f"profile has several {f} values {sorted(map(str, vals))}; pass {f}=")
        base, uni, mix = {}, {}, {}
        for r in sel:
            bs, ms = int(r["bs"]), float(r["median_ms"])
            if r["kind"] == "base":
                base[bs] = ms
            elif r["kind"] == "uniform":
                uni[bs] = ms
            elif r["kind"] == "mixed":
                mix[(bs, int(r["k"]))] = ms
        return cls(base, uni, mix)

    @classmethod
    def from_csv(cls, path, **kw) -> StepTable:
        return cls.from_rows(read_csv(path)[1], **kw)

    def _bs_bracket(self, bs: int):
        g = self.bs_grid
        if bs < 1 or bs > g[-1]:
            raise ValueError(f"bs={bs} is outside the measured range [1, {g[-1]}]; not extrapolating")
        if bs <= g[0]:
            return g[0], g[0]
        i = bisect.bisect_left(g, bs)
        return (g[i], g[i]) if g[i] == bs else (g[i - 1], g[i])

    def _lin_bs(self, f, bs: int) -> float:
        b0, b1 = self._bs_bracket(bs)
        if b0 == b1:
            return f(b0) if bs == b0 else f(b0) * bs / b0     # below the grid: scale (only if grid starts > 1)
        w = (bs - b0) / (b1 - b0)
        return (1 - w) * f(b0) + w * f(b1)

    def T_base(self, bs: int) -> float:
        return self._lin_bs(lambda b: self.base[b], bs)

    def T_uniform(self, bs: int) -> float:
        if not self.uniform:
            raise ValueError("step table has no uniform rows")
        return self._lin_bs(lambda b: self.uniform[b], bs)

    def T_mixed(self, bs: int, k: int) -> float:
        if k > bs:
            raise ValueError(f"k={k} > bs={bs}")
        if k <= 0:
            return self.T_base(bs)
        if k == 1:
            return self.T_uniform(bs)

        def at(b):
            ks = [1] + sorted(kk for (bb, kk) in self.mixed if bb == b)
            ys = [self.uniform[b]] + [self.mixed[(b, kk)] for kk in ks[1:]]
            return _interp(ks, ys, min(k, ks[-1]))   # k beyond the largest measured k at b: k clipped to it
        return self._lin_bs(at, bs)


class _Everything:
    def __contains__(self, _):
        return True


class SimPool:
    """Tensor-free stand-in for AdapterPool: S slots, LRU over refcount-0 keys, every key in the CPU tier."""

    def __init__(self, slots: int, rank: int = 16):
        self.S, self.rank, self.r_max = slots, rank, rank
        self.slot_of: dict[str, int] = {}
        self.lru: OrderedDict[str, None] = OrderedDict()
        self.refs: dict[str, int] = {}
        self.free = list(range(slots))
        self.cpu = _Everything()
        self.hits = self.misses = self.evictions = 0

    def ensure_resident(self, key):
        if key in self.slot_of:
            self.hits += 1
            self.lru.move_to_end(key)
            return
        self.misses += 1
        if not self.free:
            for k in self.lru:
                if self.refs.get(k, 0) == 0:
                    self.free.append(self.slot_of.pop(k))
                    del self.lru[k]
                    self.evictions += 1
                    break
            else:
                raise RuntimeError("no evictable slot")
        self.slot_of[key] = self.free.pop()
        self.lru[key] = None
        return

    def is_ready(self, key) -> bool:
        return key in self.slot_of

    def acquire(self, key) -> int:
        self.refs[key] = self.refs.get(key, 0) + 1
        return self.slot_of[key]

    def release(self, key) -> None:
        self.refs[key] -= 1

    def slot_rank(self, slot) -> int:
        return self.rank


class TableCost(CostModel):
    """CostModel whose step time comes from a StepTable for one policy; all fixed costs in ms."""

    def __init__(self, table: StepTable, policy: str, *, text_ms=0.0, vae_ms=0.0, encode_ms=0.0, transfer_ms=0.0,
                 swap_ms=0.0, adapter_bytes=0, h2d_gbps=25.0):
        if policy not in POLICY_MODE:
            raise ValueError(f"policy must be one of {tuple(POLICY_MODE)}")
        super().__init__(text_s=text_ms / 1e3, vae_s=vae_ms / 1e3, encode_s=encode_ms / 1e3,
                         transfer_s=transfer_ms / 1e3, swap_s=swap_ms / 1e3,
                         h2d_s=adapter_bytes / (h2d_gbps * 1e9) if h2d_gbps > 0 else 0.0)
        self.table, self.policy = table, policy

    def step(self, bs: int, k: int) -> float:
        t = self.table
        if self.policy == "P3" or k == 0:
            ms = t.T_base(bs)
        elif self.policy == "P1":
            ms = t.T_uniform(bs)
        else:
            ms = t.T_mixed(bs, k)
        return ms / 1e3

    def h2d(self, source, key) -> float:
        return 0.0 if self.policy == "P1" else super().h2d(source, key)


def simulate(trace, table: StepTable, policy: str, *, max_bs: int = 8, slots: int = 32, **cost_kw) -> ServeResult:
    """Replay `trace` (bench.trace Requests) under one policy. cost_kw: TableCost keywords."""
    loop = ServeLoop(None, SimPool(slots), policy=POLICY_MODE[policy], max_bs=max_bs, clock=VirtualClock(),
                     cost=TableCost(table, policy, **cost_kw))
    res = loop.run(trace)
    res.policy = policy
    return res


def summarize(res: ServeResult) -> dict:
    rows = res.log.to_rows()
    e2e = np.asarray([r["e2e"] for r in rows], dtype=np.float64)
    t0 = min(r["t_arrive"] for r in rows)
    t1 = max(r["t_sent"] for r in rows)
    p50, p90, p99 = np.percentile(e2e, [50, 90, 99])
    return {"policy": res.policy, "n": len(rows), "throughput_rps": len(rows) / (t1 - t0) if t1 > t0 else 0.0,
            "p50_ms": float(p50), "p90_ms": float(p90), "p99_ms": float(p99), "swaps": res.swaps}


def rate_for_rho(table: StepTable, rho: float, *, steps: int, max_bs: int) -> float:
    return rho * max_bs / (steps * table.T_base(max_bs) / 1e3)


def sweep(table: StepTable, *, policies=("P1", "P2", "P3"), rhos=(0.5, 0.7, 0.9), n_requests=500, steps=28,
          max_bs=8, seed=0, trace_kw: dict | None = None, **cost_kw) -> tuple[list[dict], list[dict]]:
    """(summary rows, timeline rows) over policies x rhos. One trace per rho (same seed), shared by policies."""
    summary, timelines = [], []
    for rho in rhos:
        rate = rate_for_rho(table, rho, steps=steps, max_bs=max_bs)
        trace = synth_trace(n_requests, rate, seed=seed, steps=steps, **(trace_kw or {}))
        for pol in policies:
            res = simulate(trace, table, pol, max_bs=max_bs, **cost_kw)
            summary.append({"rho": rho, "rate_rps": rate, **summarize(res)})
            timelines.extend(dict(r, policy=pol, rho=rho) for r in res.log.to_rows())
    return summary, timelines


def validate(sim_csv, serve_csv, *, tol: float = 0.10, policy: str | None = None) -> dict:
    """Compare P50 E2E of a simulated and a served latency CSV (L0.3 rows). `policy` filters both files on their
    policy column when given. Pass iff |p50_sim - p50_serve| / p50_serve <= tol."""
    def p50(path):
        rows = read_csv(path)[1]
        if policy is not None:
            rows = [r for r in rows if str(r.get("policy")) == policy]
        if not rows:
            raise ValueError(f"{path}: no rows" + (f" for policy {policy!r}" if policy else ""))
        return float(np.percentile([float(r["e2e"]) for r in rows], 50)), len(rows)
    s, ns = p50(sim_csv)
    v, nv = p50(serve_csv)
    err = abs(s - v) / v if v else float("inf")
    return {"p50_sim_ms": s, "p50_serve_ms": v, "p50_err": err, "n_sim": ns, "n_serve": nv, "tol": tol,
            "pass": err <= tol}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.sim.replay_sim")
    ap.add_argument("--profile", required=True, help="step-time CSV from bench.profile_steps")
    ap.add_argument("--model", default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--mode", default="eager")
    ap.add_argument("--targets", default="all")
    ap.add_argument("--cfg", action="store_true")
    ap.add_argument("--policies", default="P1,P2,P3")
    ap.add_argument("--rhos", default="0.5,0.7,0.9")
    ap.add_argument("--n-requests", type=int, default=500)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--max-bs", type=int, default=8)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-ms", type=float, default=30.0)
    ap.add_argument("--vae-ms", type=float, default=100.0)
    ap.add_argument("--encode-ms", type=float, default=10.0)
    ap.add_argument("--transfer-ms", type=float, default=0.0)
    ap.add_argument("--swap-ms", type=float, default=300.0)
    ap.add_argument("--adapter-bytes", type=int, default=150_000_000)
    ap.add_argument("--h2d-gbps", type=float, default=25.0)
    ap.add_argument("--out", default="results/sim")
    a = ap.parse_args(argv)
    table = StepTable.from_csv(a.profile, model=a.model, resolution=a.resolution, mode=a.mode, targets=a.targets,
                               rows_per_req=2 if a.cfg else 1)
    summary, timelines = sweep(table, policies=tuple(a.policies.split(",")),
                               rhos=tuple(float(r) for r in a.rhos.split(",")), n_requests=a.n_requests,
                               steps=a.steps, max_bs=a.max_bs, seed=a.seed, slots=a.slots, text_ms=a.text_ms,
                               vae_ms=a.vae_ms, encode_ms=a.encode_ms, transfer_ms=a.transfer_ms, swap_ms=a.swap_ms,
                               adapter_bytes=a.adapter_bytes, h2d_gbps=a.h2d_gbps)
    env = collect_env()
    write_csv(f"{a.out}_summary.csv", summary, env)
    write_csv(f"{a.out}_timelines.csv", timelines, env)
    return 0


__all__ = ["SimPool", "StepTable", "TableCost", "rate_for_rho", "simulate", "summarize", "sweep",
           "validate"]

if __name__ == "__main__":
    raise SystemExit(main())
