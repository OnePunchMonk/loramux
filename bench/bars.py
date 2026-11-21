"""Pre-registered pass/fail bars (HANDOFF.md §5 kill-test K1-K3, §11.5 Phase 2b L1-L5 / P1-P3) as typed constants.

Fixed before any GPU run. Do not edit thresholds after seeing data; a failed bar is a finding.

API (shared with the kill-test orchestrator L3.4 and the Phase 2b evaluator L3.5):
    Bar(id, metric, op, threshold, description, unit)    one comparable number
    check(bar, value) -> Verdict                          pass/fail with the number
    Group(id, mode, bars)                                 compound bar ('all' = AND, 'any' = OR)
    check_group(group, {bar_id: value}) -> GroupVerdict
    BARS[id], GROUPS[id]                                  every bar / group by id
    evaluate_k1(k1a_rel_errs, k1b_psnrs, k1c_psnrs, k1c_lpips) -> GroupVerdict

Relative bars are expressed as ratios so every bar is a plain comparison:
    K2a   value = loramux overhead / B2 overhead (same harness, 4096 tokens), <= 0.5
    K3    value = P2/P1 throughput ratio (>= 1.5) OR P2/P1 P99 ratio (<= 0.5), at >= 1 model x resolution
    L3    value = adapter_wait P90 / denoise step time, <= 1
'report' bars (P1, P2/P3 size figures) never pass or fail: passed is None and `expect` carries the expectation.

Aggregation rules (resolved here, documented once):
    K1a  worst (max) per-linear rel. error over the sweep — every linear must meet the bar.
    K1b  worst (min) PSNR over requests — batch invariance must hold for every request.
    K1c  mean PSNR and mean LPIPS, as the bar states.
A value that is None (not run) gives passed=None; NaN fails.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

Op = Literal["<=", ">=", "==", "report"]


@dataclass(frozen=True)
class Bar:
    id: str
    metric: str
    op: Op
    threshold: float | None
    description: str
    unit: str = ""
    expect: str | None = None      # for op == 'report': the stated expectation


@dataclass(frozen=True)
class Verdict:
    bar_id: str
    value: float | None
    op: Op
    threshold: float | None
    passed: bool | None            # None = not run (value None) or a report-only bar
    detail: str

    def __str__(self) -> str:
        tag = {True: "PASS", False: "FAIL", None: "N/A"}[self.passed]
        return f"{self.bar_id}: {tag} ({self.detail})"


@dataclass(frozen=True)
class Group:
    id: str
    mode: Literal["all", "any"]
    bars: tuple[Bar, ...]
    description: str = ""


@dataclass(frozen=True)
class GroupVerdict:
    group_id: str
    passed: bool | None
    verdicts: tuple[Verdict, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        tag = {True: "PASS", False: "FAIL", None: "N/A"}[self.passed]
        return f"{self.group_id}: {tag}\n" + "\n".join(f"  {v}" for v in self.verdicts)


def _fmt(x: float | None) -> str:
    return "None" if x is None else f"{x:.6g}"


def check(bar: Bar, value: float | None) -> Verdict:
    if bar.op == "report":
        exp = f"; expect {bar.expect}" if bar.expect else ""
        return Verdict(bar.id, value, bar.op, bar.threshold, None, f"{bar.metric}={_fmt(value)}{bar.unit} (report{exp})")
    if value is None:
        return Verdict(bar.id, None, bar.op, bar.threshold, None, f"{bar.metric} not measured")
    v = float(value)
    t = float(bar.threshold)
    if math.isnan(v):
        ok = False
    elif bar.op == "<=":
        ok = v <= t
    elif bar.op == ">=":
        ok = v >= t
    elif bar.op == "==":
        ok = v == t
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown op {bar.op!r}")
    return Verdict(bar.id, v, bar.op, t, ok, f"{bar.metric}={_fmt(v)}{bar.unit} {bar.op} {_fmt(t)}{bar.unit}")


def _combine(passed: list[bool | None], mode: str) -> bool | None:
    if mode == "all":
        if any(p is False for p in passed):
            return False
        return None if any(p is None for p in passed) else True
    if any(p is True for p in passed):
        return True
    return None if any(p is None for p in passed) else False


def check_group(group: Group, values: Mapping[str, float | None]) -> GroupVerdict:
    """values maps bar id -> number; a missing id counts as not measured. Report-only bars don't vote."""
    vs = tuple(check(b, values.get(b.id)) for b in group.bars)
    voting = [v.passed for v, b in zip(vs, group.bars) if b.op != "report"]
    return GroupVerdict(group.id, _combine(voting, group.mode), vs)


# ---- HANDOFF §5: kill test ----
K1A = Bar("K1a", "max per-linear rel. error vs fp32 torch_ref", "<=", 2e-3,
          "per-linear relative error vs the fp32 reference (torch_ref in fp32)")
K1B = Bar("K1b", "min PSNR mixed-batch vs solo", ">=", 45.0,
          "mixed-batch output vs the same request run alone through loramux, final image", " dB")
K1C_PSNR = Bar("K1c_psnr", "mean PSNR loramux vs diffusers merged", ">=", 35.0,
               "loramux vs diffusers merged-adapter reference, same seed, 28 steps", " dB")
K1C_LPIPS = Bar("K1c_lpips", "mean LPIPS loramux vs diffusers merged", "<=", 0.05,
                "loramux vs diffusers merged-adapter reference, same seed, 28 steps")
K2A = Bar("K2a", "linear-chain overhead / B2 overhead at 4096 tokens", "<=", 0.5,
          "linear-chain overhead vs B0 at 4096 tokens <= 1/2 of B2's overhead in the same harness")
K2B_ATTN = Bar("K2b_attn", "step overhead vs base, attention-only", "<=", 5.0,
               "end-to-end step overhead vs base, bs=8 mixed, 1024^2, compiled, attention-only", "%")
K2B_ALL = Bar("K2b_all", "step overhead vs base, all-linear", "<=", 12.0,
              "end-to-end step overhead vs base, bs=8 mixed, 1024^2, compiled, all-linear (v1)", "%")
K3_THROUGHPUT = Bar("K3_throughput", "P2/P1 throughput ratio", ">=", 1.5,
                    "simulator rho=0.7 and 0.9: mixed (P2) vs adapter-keyed (P1) throughput, best model x res")
K3_P99 = Bar("K3_p99", "P2/P1 P99 ratio", "<=", 0.5,
             "simulator rho=0.7 and 0.9: mixed (P2) vs adapter-keyed (P1) P99, best model x res")

# ---- HANDOFF §11.5: Phase 2b ----
L1 = Bar("L1", "P50 E2E LoRA / base, rho=0.7, mixed", "<=", 1.10,
         "P50 E2E of LoRA requests vs base-model requests, same run, rho=0.7, mixed policy", "x")
L2 = Bar("L2", "P99 E2E P2 / P1, rho=0.9", "<=", 0.5,
         "P99 E2E, mixed policy (P2) vs adapter-keyed (P1), rho=0.9", "x")
L3 = Bar("L3", "adapter_wait P90 (CPU hit) / denoise step time", "<=", 1.0,
         "adapter_wait P90 for CPU-hit adapters <= one denoise step (hidden by text encoding)", "x")
L4_MS = Bar("L4_ms", "adapter load pinned->GPU, rank-32 attn-only", "<=", 10.0,
            "adapter load CPU-pinned -> GPU slot, rank-32 attention-only adapter", " ms")
L4_BW = Bar("L4_bw", "load bandwidth / measured H2D bandwidth", ">=", 0.70,
            "adapter load achieves >= 70% of measured H2D bandwidth")
L5 = Bar("L5", "VAE+encode share of E2E on the critical path", "==", 0.0,
         "VAE + encode share of E2E on the critical path for requests still in the batch (fully overlapped)")
P1 = Bar("P1", "payload bytes / source .safetensors bytes", "report", None,
         "payload bytes vs the source .safetensors (exact path)", "x", expect="<= 0.5x for fp32 sources")
P2_PSNR = Bar("P2_psnr", "mean PSNR FP8 vs exact", ">=", 35.0, "FP8 storage quality vs the exact path (K1c bar)", " dB")
P2_LPIPS = Bar("P2_lpips", "mean LPIPS FP8 vs exact", "<=", 0.05, "FP8 storage quality vs the exact path (K1c bar)")
P2_SLOTS = Bar("P2_slots", "slots per GB FP8 / exact", "report", None, "slots per GB", "x", expect="~2x")
P3_PSNR = Bar("P3_psnr", "mean PSNR SVD-truncated vs exact", ">=", 35.0,
              "SVD truncation quality vs the exact path (K1c bar)", " dB")
P3_LPIPS = Bar("P3_lpips", "mean LPIPS SVD-truncated vs exact", "<=", 0.05,
               "SVD truncation quality vs the exact path (K1c bar)")
P3_BYTES = Bar("P3_bytes", "median bytes saved by SVD truncation", "report", None, "median bytes saved", " B")

ALL_BARS: tuple[Bar, ...] = (K1A, K1B, K1C_PSNR, K1C_LPIPS, K2A, K2B_ATTN, K2B_ALL, K3_THROUGHPUT, K3_P99,
                             L1, L2, L3, L4_MS, L4_BW, L5, P1, P2_PSNR, P2_LPIPS, P2_SLOTS, P3_PSNR, P3_LPIPS,
                             P3_BYTES)
BARS: dict[str, Bar] = {b.id: b for b in ALL_BARS}

K1C = Group("K1c", "all", (K1C_PSNR, K1C_LPIPS), "mean PSNR >= 35 dB and mean LPIPS <= 0.05")
K1 = Group("K1", "all", (K1A, K1B, K1C_PSNR, K1C_LPIPS), "K1a and K1b and K1c")
K2B = Group("K2b", "all", (K2B_ATTN, K2B_ALL))
K2 = Group("K2", "all", (K2A, K2B_ATTN, K2B_ALL))
K3 = Group("K3", "any", (K3_THROUGHPUT, K3_P99), "throughput >= 1.5x or P99 <= 0.5x")
L4 = Group("L4", "all", (L4_MS, L4_BW))
P2 = Group("P2", "all", (P2_PSNR, P2_LPIPS, P2_SLOTS))
P3 = Group("P3", "all", (P3_PSNR, P3_LPIPS, P3_BYTES))
GROUPS: dict[str, Group] = {g.id: g for g in (K1C, K1, K2B, K2, K3, L4, P2, P3)}


def _agg(values: Iterable[float] | None, how: str) -> float | None:
    if values is None:
        return None
    xs = [float(v) for v in values]
    if not xs:
        return None
    if any(math.isnan(x) for x in xs):
        return math.nan
    return {"max": max, "min": min, "mean": lambda v: sum(v) / len(v)}[how](xs)


def evaluate_k1(k1a_rel_errs: Iterable[float] | None = None, k1b_psnrs: Iterable[float] | None = None,
                k1c_psnrs: Iterable[float] | None = None, k1c_lpips: Iterable[float] | None = None) -> GroupVerdict:
    """K1 = K1a (max rel. err) AND K1b (min PSNR) AND K1c (mean PSNR, mean LPIPS). None/empty = not run."""
    return check_group(K1, {"K1a": _agg(k1a_rel_errs, "max"), "K1b": _agg(k1b_psnrs, "min"),
                            "K1c_psnr": _agg(k1c_psnrs, "mean"), "K1c_lpips": _agg(k1c_lpips, "mean")})
