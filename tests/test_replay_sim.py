import re

import pytest
import torch

from bench.common import write_csv
from bench.serve_loop import ServeLoop, ToyStepModel, VirtualClock
from bench.sim import SimPool, StepTable, TableCost, simulate, summarize, sweep, validate
from bench.step_bench import linear_shapes
from bench.synth_adapters import make_adapter
from bench.trace.synth_trace import Request
from loramux import AdapterPool, patch_model

BS = (1, 2, 4, 8, 16)


def _profile_rows(base=lambda b: 10 + 2 * b, uni=lambda b: 11 + 2 * b, mix=lambda b, k: 11 + 2 * b + k):
    """Rows in the bench.profile_steps CSV contract (times in ms)."""
    common = {"model": "toy", "resolution": 512, "rows_per_req": 1, "mode": "eager", "p90_ms": 0, "mean_ms": 0,
              "n": 1, "timer": "cpu"}
    rows = []
    for b in BS:
        rows.append(dict(common, bs=b, kind="base", k=0, targets="base", rank=0, median_ms=base(b)))
    for b in BS:
        rows.append(dict(common, bs=b, kind="uniform", k=1, targets="all", rank=16, median_ms=uni(b)))
        for k in (2, 4, 8, 16):
            if k <= b:
                rows.append(dict(common, bs=b, kind="mixed", k=k, targets="all", rank=16, median_ms=mix(b, k)))
    return rows


@pytest.fixture
def table(tmp_path):
    p = tmp_path / "prof.csv"
    write_csv(p, _profile_rows(), {"gpu": None})
    return StepTable.from_csv(p, model="toy", resolution=512)


FIXED = {"text_ms": 30.0, "vae_ms": 50.0, "encode_ms": 5.0, "transfer_ms": 2.0}


def _req(i, t, ad=(), steps=10):
    return Request(id=i, t_arrive=t, adapters=list(ad), scales=[1.0] * len(ad), steps=steps)


def test_table_interpolation(table):
    assert table.T_base(4) == 18 and table.T_base(3) == pytest.approx(16)
    assert table.T_uniform(12) == pytest.approx(35)
    assert table.T_mixed(8, 4) == 11 + 16 + 4
    assert table.T_mixed(6, 3) == pytest.approx(11 + 12 + 3)          # bilinear, exact for a linear table
    assert table.T_mixed(4, 1) == table.T_uniform(4) and table.T_mixed(4, 0) == table.T_base(4)
    with pytest.raises(ValueError):
        table.T_base(17)
    with pytest.raises(ValueError):
        table.T_mixed(2, 3)


@pytest.mark.parametrize("policy", ["P1", "P2", "P3"])
def test_single_request_analytic(table, policy):
    res = simulate([_req(0, 1.0, ["a"])], table, policy, **FIXED)
    t = {"P1": 11 + 2, "P2": 11 + 2, "P3": 10 + 2}[policy]
    e2e = res.log.to_rows()[0]["e2e"]
    assert e2e == pytest.approx(30 + 10 * t + 50 + 5 + 2)


def test_h2d_hidden_then_exposed(table):
    # 0.6 GB at 25 GB/s = 24 ms < 30 ms text encode: hidden; 1.5 GB = 60 ms: 30 ms exposed
    hidden = simulate([_req(0, 0, ["a"])], table, "P2", adapter_bytes=600_000_000, **FIXED).log.to_rows()[0]
    exposed = simulate([_req(0, 0, ["a"])], table, "P2", adapter_bytes=1_500_000_000, **FIXED).log.to_rows()[0]
    assert hidden["adapter_wait"] == 0 and exposed["adapter_wait"] == pytest.approx(30)
    assert exposed["e2e"] - hidden["e2e"] == pytest.approx(30)


def test_p1_batches_same_adapter(table):
    res = simulate([_req(0, 0, ["a"]), _req(1, 0, ["a"])], table, "P1", swap_ms=100, **FIXED)
    assert [r["bs"] for r in res.step_rows] == [2] * 10 and res.swaps == 1
    for r in res.log.to_rows():
        assert r["e2e"] == pytest.approx(30 + 100 + 10 * (11 + 4) + 50 + 5 * (int(r["request_id"]) + 1) + 2)


def test_p1_alternating_adapters_pay_swaps(table):
    gap = 2.0   # each request finishes before the next arrives
    alt = [_req(i, gap * i, ["ab"[i % 2]]) for i in range(4)]
    same = [_req(i, gap * i, ["a"]) for i in range(4)]
    ra = simulate(alt, table, "P1", swap_ms=100, **FIXED)
    rs = simulate(same, table, "P1", swap_ms=100, **FIXED)
    assert ra.swaps == 4 and rs.swaps == 1
    ea = [r["e2e"] for r in ra.log.to_rows()]
    es = [r["e2e"] for r in rs.log.to_rows()]
    assert all(x == pytest.approx(y + 100) for x, y in zip(ea[1:], es[1:]))
    assert ea[0] == pytest.approx(es[0])


def test_policy_order(table):
    tr = [_req(i, 0.01 * i, ["abcd"[i % 4]]) for i in range(16)]
    p = {pol: summarize(simulate(tr, table, pol, max_bs=8, swap_ms=100, **FIXED))["p50_ms"]
         for pol in ("P1", "P2", "P3")}
    assert p["P3"] <= p["P2"] <= p["P1"]
    assert p["P3"] < p["P1"]


def test_determinism_by_seed(table):
    kw = dict(policies=("P1", "P2"), rhos=(0.7,), n_requests=60, steps=8, max_bs=4, slots=8,
              trace_kw={"n_adapters": 20, "lora_frac": 0.6, "top5pct_share": None}, swap_ms=50, **FIXED)
    a = sweep(table, seed=1, **kw)
    b = sweep(table, seed=1, **kw)
    c = sweep(table, seed=2, **kw)
    assert a == b and a != c
    assert {s["policy"] for s in a[0]} == {"P1", "P2"} and all(s["n"] == 60 for s in a[0])


def test_simpool_lru_and_pins():
    p = SimPool(2)
    p.ensure_resident("a"); p.acquire("a")
    p.ensure_resident("b"); p.ensure_resident("c")      # b evicted, a pinned
    assert set(p.slot_of) == {"a", "c"}
    p.acquire("c")
    with pytest.raises(RuntimeError):
        p.ensure_resident("d")


def test_validate_against_serve_loop(table, tmp_path):
    """The serve loop on the toy model, timed by the same table, agrees with the simulator."""
    model = ToyStepModel()
    shapes = linear_shapes(model.module, "attn")
    pool = AdapterPool(shapes, slots=8, r_max=8, device="cpu", dtype=torch.float32)
    for i, k in enumerate("abcd"):
        pool.register(k, make_adapter(shapes, 8, "all", seed=i))
    patch_model(model.module, ["^(?:" + "|".join(re.escape(k) for k in shapes) + ")$"], pool)
    tr = [_req(i, 0.05 * i, ["abcd"[i % 4]] if i % 3 else [], steps=6) for i in range(20)]
    cost = TableCost(table, "P2", adapter_bytes=10**9, **FIXED)
    served = ServeLoop(model, pool, policy="mixed", max_bs=4, clock=VirtualClock(), cost=cost).run(tr)
    sim = simulate(tr, table, "P2", max_bs=4, adapter_bytes=10**9, **FIXED)
    write_csv(tmp_path / "serve.csv", served.latency_rows(), {})
    write_csv(tmp_path / "sim.csv", [dict(r, policy="mixed") for r in sim.log.to_rows()], {})
    v = validate(tmp_path / "sim.csv", tmp_path / "serve.csv", policy="mixed")
    assert v["pass"] and v["p50_err"] < 1e-9 and v["n_sim"] == v["n_serve"] == 20
    slow = [dict(r, e2e=r["e2e"] * 1.2) for r in sim.log.to_rows()]
    write_csv(tmp_path / "slow.csv", slow, {})
    v = validate(tmp_path / "slow.csv", tmp_path / "serve.csv")
    assert not v["pass"] and v["p50_err"] == pytest.approx(0.2)
