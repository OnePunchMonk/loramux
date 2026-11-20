import pytest
import torch

from bench.common import collect_env, interleave, read_csv, seed_all, summarize, time_cuda, write_csv


def test_summarize_known_list():
    s = summarize([float(x) for x in range(1, 11)])
    assert s["n"] == 10
    assert s["median"] == pytest.approx(5.5)
    assert s["mean"] == pytest.approx(5.5)
    assert s["p90"] == pytest.approx(9.1)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_time_cuda_returns_one_sample_per_iter():
    ms = time_cuda(lambda: None, warmup=1, iters=3)
    assert len(ms) == 3 and all(m >= 0 for m in ms)


def test_interleave_alternates_order():
    calls = []
    configs = {"A": lambda: calls.append("A"), "B": lambda: calls.append("B")}
    out = interleave(configs, rounds=4, warmup=0, iters=1)
    assert calls == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert len(out["A"]) == 4 and len(out["B"]) == 4


def test_csv_round_trip_with_env(tmp_path):
    env = {"python": "3.13.1", "torch": "2.8.0", "gpu": None, "timestamp_utc": "2026-01-01T00:00:00+00:00"}
    rows = [{"baseline": "B0", "tokens": 1024, "overhead_pct": 0.5},
            {"baseline": "B1", "tokens": 4096, "overhead_pct": 1.25}]
    path = tmp_path / "out" / "r.csv"
    write_csv(path, rows, env)
    assert path.read_text().startswith("# python=3.13.1\n")
    env2, rows2 = read_csv(path)
    assert env2 == env
    assert rows2 == rows


def test_collect_env_keys_and_types():
    env = collect_env()
    assert {"python", "torch", "timer", "hostname", "timestamp_utc", "gpu", "driver"} <= set(env)
    assert all(v is None or isinstance(v, str) for v in env.values())


def test_seed_all_is_repeatable():
    seed_all(7)
    a = torch.rand(4)
    seed_all(7)
    assert torch.equal(a, torch.rand(4))
