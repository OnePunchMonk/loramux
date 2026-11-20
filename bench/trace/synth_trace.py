"""Synthetic request trace matched to published aggregates of the GenTD26 trace.

Statistic definitions (used by `trace_stats` and the tests; an "adapter use" is one adapter in one request,
so a multi-adapter request contributes one use to each of its adapters):

- lora_frac:      fraction of ALL requests with at least one adapter.
- multi_frac:     fraction of ALL requests with at least two adapters (a subset of lora_frac).
- n_unique:       number of distinct adapters that appear in the trace.
- gini:           Gini coefficient of the per-adapter use counts over the adapters that appear (n_unique of
                  them; adapters never used are invisible to a trace, so they are not counted).
- top5pct_share:  share of all adapter uses held by the top ceil(5% * n_unique) most used adapters.
- median_uses:    median of the per-adapter use counts over the adapters that appear.

Popularity is Zipf(s) over `n_adapters` ranks. `fit_zipf_exponent` picks s by bisection so that the Gini of the
*expected* weights (all n_adapters, no sampling) equals the target. The Gini of a finite sample is a different
quantity (zero-use adapters drop out, small counts add noise), so the synthetic-trace tests report that error
separately.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

DEFAULT_N_ADAPTERS = 874
DEFAULT_GINI = 0.749


@dataclass
class Request:
    id: int | str
    t_arrive: float
    adapters: list[str] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    steps: int = 28
    width: int = 1024
    height: int = 1024
    prompt_id: int = 0


def gini(values) -> float:
    """Gini coefficient of non-negative values (0 = equal, ->1 = concentrated)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2.0 * (i * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def zipf_weights(n: int, s: float) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=np.float64) ** (-s)
    return w / w.sum()


def zipf_gini(n: int, s: float) -> float:
    """Gini of the expected Zipf(s) weights over n ranks."""
    return gini(zipf_weights(n, s))


def fit_zipf_exponent(n_adapters: int = DEFAULT_N_ADAPTERS, target_gini: float = DEFAULT_GINI,
                      tol: float = 1e-9) -> float:
    """Bisection on s so that zipf_gini(n_adapters, s) == target_gini. Gini is increasing in s."""
    if not 0.0 < target_gini < 1.0:
        raise ValueError("target_gini must be in (0, 1)")
    lo, hi = 0.0, 1.0
    while zipf_gini(n_adapters, hi) < target_gini:
        hi *= 2.0
        if hi > 64:
            raise ValueError(f"Gini {target_gini} unreachable with {n_adapters} adapters")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if zipf_gini(n_adapters, mid) < target_gini:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def trace_stats(adapter_ids_per_request) -> dict:
    """Stats of a trace given, per request, the list of adapter ids it uses (empty = base model)."""
    reqs = [list(a) for a in adapter_ids_per_request]
    n = len(reqs)
    if n == 0:
        raise ValueError("empty trace")
    counts = Counter(a for r in reqs for a in r)
    uses = sorted(counts.values(), reverse=True)
    total = sum(uses)
    k = max(1, math.ceil(0.05 * len(uses))) if uses else 0
    return {
        "lora_frac": sum(1 for r in reqs if len(r) >= 1) / n,
        "multi_frac": sum(1 for r in reqs if len(r) >= 2) / n,
        "n_unique": len(counts),
        "gini": gini(uses) if uses else 0.0,
        "top5pct_share": (sum(uses[:k]) / total) if total else 0.0,
        "median_uses": float(np.median(uses)) if uses else 0.0,
    }


def synth_trace(n_requests: int, rate_rps: float, *, n_adapters: int = DEFAULT_N_ADAPTERS,
                lora_frac: float = 0.168, multi_frac: float = 0.016, gini: float = DEFAULT_GINI,
                seed: int = 0, steps: int = 28, width: int = 1024, height: int = 1024,
                n_prompts: int = 1000) -> list[Request]:
    """Poisson arrivals at rate_rps; adapters ~ Zipf fitted to `gini`. Multi requests draw 2 distinct adapters.

    Adapter ids are 'ad_0000' (most popular) ... Single-adapter scale is 1.0; multi-adapter scales are drawn
    uniformly from [0.5, 1.0] (rounded to 2 decimals).
    """
    if rate_rps <= 0:
        raise ValueError("rate_rps must be > 0")
    if not 0 <= multi_frac <= lora_frac <= 1:
        raise ValueError("need 0 <= multi_frac <= lora_frac <= 1")
    rng = np.random.default_rng(seed)
    p = zipf_weights(n_adapters, fit_zipf_exponent(n_adapters, gini))
    t = np.cumsum(rng.exponential(1.0 / rate_rps, size=n_requests))
    u = rng.random(n_requests)
    out: list[Request] = []
    for i in range(n_requests):
        if u[i] < multi_frac:
            idx = rng.choice(n_adapters, size=2, replace=False, p=p)
            scales = [float(round(x, 2)) for x in rng.uniform(0.5, 1.0, size=2)]
        elif u[i] < lora_frac:
            idx = [int(rng.choice(n_adapters, p=p))]
            scales = [1.0]
        else:
            idx, scales = [], []
        out.append(Request(id=i, t_arrive=float(t[i]), adapters=[f"ad_{int(j):04d}" for j in idx], scales=scales,
                           steps=steps, width=width, height=height, prompt_id=int(rng.integers(n_prompts))))
    return out
