from pathlib import Path

import pytest

from bench.common import write_csv
from bench.latency_breakdown import plot_latency_breakdown
from bench.plots import main, plot_overhead_vs_tokens, plot_sim
from loramux.latency import AdapterSource, LatencyLog, RequestTimeline

ENV = {"python": "3.13.1", "timer": "cpu"}


def _assert_outputs(prefix):
    for ext in (".png", ".svg"):
        f = Path(f"{prefix}{ext}")
        assert f.exists() and f.stat().st_size > 0, f


def test_overhead_plot(tmp_path):
    rows = [{"baseline": f"B{b}", "tokens": t, "overhead_pct": 1.0 + b + t / 4096}
            for b in range(6) for t in (1024, 4096, 16384)]
    rows.append({"baseline": "B5", "tokens": 1024, "overhead_pct": None, "status": "unavailable"})  # no vLLM
    src = tmp_path / "overhead.csv"
    write_csv(src, rows, ENV)
    out = tmp_path / "out" / "overhead"
    plot_overhead_vs_tokens(src, out)
    _assert_outputs(out)


def test_sim_plot(tmp_path):
    rows = [{"policy": p, "rho": rho, "throughput_rps": 10.0 * rho, "p50_ms": 50.0 * rho, "p99_ms": 200.0 * rho}
            for p in ("P1", "P2", "P3") for rho in (0.5, 0.7, 0.9)]
    src = tmp_path / "sim.csv"
    write_csv(src, rows, ENV)
    out = tmp_path / "sim"
    plot_sim(src, out)
    _assert_outputs(out)


def _timeline(i: int, source: AdapterSource) -> RequestTimeline:
    d = 1.0 + 0.1 * i
    return RequestTimeline(t_arrive=0.0, t_admit_ready_text=0.1, t_adapter_resident=0.3, t_first_step=0.35,
                           t_last_step=0.35 + d, t_vae_done=0.4 + d, t_encode_done=0.45 + d, t_sent=0.5 + d,
                           adapter_source=source, n_steps=28, request_id=f"r{i}",
                           adapter_key="a" if source is not AdapterSource.NONE else None)


def test_latency_breakdown_plot(tmp_path):
    log = LatencyLog()
    for i in range(6):
        log.append(_timeline(i, AdapterSource.MISS if i % 2 else AdapterSource.GPU_HIT))
    rows = [{**row, "policy": "P2" if row["request_id"] < "r3" else "P1", "regime": "1024"}
            for row in log.to_rows()]
    src = tmp_path / "latency.csv"
    write_csv(src, rows, ENV)
    out = tmp_path / "latency"
    plot_latency_breakdown(src, out)
    _assert_outputs(f"{out}_mean")
    _assert_outputs(f"{out}_p99")


def test_latency_breakdown_rejects_empty(tmp_path):
    src = tmp_path / "empty.csv"
    write_csv(src, [], ENV)
    with pytest.raises(ValueError):
        plot_latency_breakdown(src, tmp_path / "x")


def test_cli_overhead(tmp_path):
    rows = [{"baseline": "B0", "tokens": 1024, "overhead_pct": 0.5}, {"baseline": "B0", "tokens": 4096,
                                                                       "overhead_pct": 0.7}]
    src = tmp_path / "o.csv"
    write_csv(src, rows, ENV)
    assert main(["overhead", str(src), str(tmp_path / "cli")]) == 0
    _assert_outputs(tmp_path / "cli")
