import sys

import pytest

from bench import vllm_omni_swap_cost as m
from bench.common import read_csv, write_csv

CFG = {"model": "M", "steps": 4, "height": 64, "width": 64, "lora_scale": 1.0, "n_adapters": 2}


def test_import_guard(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_omni", None)
    with pytest.raises(ImportError, match="vllm-omni==0.32.0"):
        m.import_vllm_omni()


def test_parse_args_defaults_and_errors():
    a = m.parse_args(["--adapter", "a", "--adapter", "b"])
    assert a.adapters == ["a", "b"] and a.patterns == list(m.PATTERNS) and a.steps == 28
    assert m.parse_args(["--adapter", "a", "--patterns", "repeat,base_toggle"]).patterns == ["repeat", "base_toggle"]
    for bad in (["--adapter", "a"], ["--adapter", "a", "--adapter", "b", "--patterns", "nope"],
                ["--adapter", "a", "--adapter", "b", "--requests", "1"], []):
        with pytest.raises(SystemExit):
            m.parse_args(bad)


def test_schedules():
    assert m.build_schedule("repeat", ["a", "b"], 3) == ["a", "a", "a"]
    assert m.build_schedule("alternate", ["a", "b"], 4) == ["a", "b", "a", "b"]
    assert m.build_schedule("base_toggle", ["a"], 4) == ["a", m.BASE, "a", m.BASE]
    with pytest.raises(ValueError):
        m.build_schedule("alternate", ["a"], 4)


def test_unmeasurable_deltas_stay_none_without_a_repeat_baseline():
    seen: set = set()
    rows = m.request_rows("alternate", ["a", "b", "a", "b"], [130.0] * 4, seen, CFG, prev="a")
    summ = {(r["pattern"], r["metric"]): r for r in m.summary_rows(rows, CFG) if r["kind"] == "summary"}
    assert summ[("alternate", "swap_cost_ms")]["median_ms"] is None   # not the 130 ms raw latency


def test_csv_schema_and_swap_cost(tmp_path):
    seen: set = set()
    rows = m.request_rows("register", ["a", "b"], [500.0, 500.0], seen, CFG)
    rows += m.request_rows("repeat", ["a"] * 4, [100.0] * 4, seen, CFG, prev="b")
    rows += m.request_rows("alternate", ["a", "b", "a", "b"], [130.0] * 4, seen, CFG, prev="a")
    rows += m.request_rows("base_toggle", ["a", m.BASE, "a", m.BASE], [130.0, 90.0, 130.0, 90.0], seen, CFG,
                           prev="b")
    assert [r["cold"] for r in rows[:2]] == [1, 1] and not any(r["cold"] for r in rows[2:])
    allrows = rows + m.summary_rows(rows, CFG)
    p = tmp_path / "s.csv"
    write_csv(p, [{c: r.get(c) for c in m.COLUMNS} for r in allrows], {"x": "1"})
    env, back = read_csv(p)
    assert env == {"x": "1"} and tuple(back[0]) == m.COLUMNS
    summ = {(r["pattern"], r["metric"]): r for r in back if r["kind"] == "summary"}
    assert summ[("alternate", "swap_cost_ms")]["median_ms"] == pytest.approx(30.0)
    assert summ[("base_toggle", "swap_cost_ms")]["median_ms"] == pytest.approx(30.0)
    assert summ[("base_toggle", "deactivate_cost_ms")]["median_ms"] == pytest.approx(-10.0)
    assert summ[("register", "cold_load_ms")]["median_ms"] == pytest.approx(400.0)
    assert summ[("repeat", "latency_ms")]["n"] == 4
