import itertools
import json

import pytest

from bench.trace import fit_zipf_exponent, load_gentd26, synth_trace, trace_stats, zipf_gini
from bench.trace.synth_trace import gini


def test_gini_known_distributions():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-12)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)  # (n-1)/n
    assert gini([1, 2, 3, 4]) == pytest.approx(0.25)


def test_trace_stats_known():
    reqs = [[], [], ["a"], ["a"], ["a", "b"], ["c"]]
    s = trace_stats(reqs)
    assert s["lora_frac"] == pytest.approx(4 / 6)
    assert s["multi_frac"] == pytest.approx(1 / 6)
    assert s["n_unique"] == 3
    assert s["median_uses"] == 1.0          # counts a=3, b=1, c=1
    assert s["top5pct_share"] == pytest.approx(3 / 5)  # top ceil(0.15)=1 adapter
    assert s["gini"] == pytest.approx(gini([3, 1, 1]))


def test_fitted_exponent_hits_gini():
    s = fit_zipf_exponent(874, 0.749)
    assert abs(zipf_gini(874, s) - 0.749) < 0.01  # tolerance 0.01; bisection is actually ~1e-9
    assert 0.5 < s < 3


def test_synth_trace_matches_targets():
    tr = synth_trace(20000, 2.0, seed=1)
    assert all(b.t_arrive > a.t_arrive for a, b in itertools.pairwise(tr))
    assert tr[-1].t_arrive == pytest.approx(20000 / 2.0, rel=0.05)  # Poisson rate
    s = trace_stats([r.adapters for r in tr])
    print("synthetic stats", s)
    # tolerances (absolute): binomial noise on 20k requests is ~0.003 for lora_frac, ~0.001 for multi_frac
    assert abs(s["lora_frac"] - 0.168) < 0.01
    assert abs(s["multi_frac"] - 0.016) < 0.004
    assert all(len(set(r.adapters)) == len(r.adapters) == len(r.scales) for r in tr)
    # top-5% share target 535 permille; the sample's error is reported, tolerance 0.06 absolute
    print("top5pct_share error:", s["top5pct_share"] - 0.535)
    assert abs(s["top5pct_share"] - 0.535) < 0.06
    # median reuse: 20k*0.168 ~ 3400 uses over the ~600 adapters that appear -> low single digits
    assert 1 <= s["median_uses"] <= 3
    assert s["n_unique"] <= 874


def test_synth_trace_deterministic():
    assert synth_trace(50, 1.0, seed=3) == synth_trace(50, 1.0, seed=3)


def test_load_with_column_map(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("ts,rid,loras,n_steps,w\n"
                 "110.5,b,x;y,20,512\n"
                 "100.0,a,,,\n"
                 "105.0,c,z,30,\n")
    cm = {"timestamp": "ts", "request_id": "rid", "adapters": "loras", "steps": "n_steps", "width": "w"}
    reqs = load_gentd26(p, cm)
    assert [r.id for r in reqs] == ["a", "c", "b"]
    assert [r.t_arrive for r in reqs] == [0.0, 5.0, 10.5]
    assert reqs[0].adapters == [] and reqs[0].steps == 28 and reqs[0].width == 1024
    assert reqs[1].adapters == ["z"] and reqs[1].steps == 30 and reqs[1].width == 1024
    assert reqs[2].adapters == ["x", "y"] and reqs[2].scales == [1.0, 1.0] and reqs[2].width == 512
    with pytest.raises(KeyError):
        load_gentd26(p)  # default headers absent
    with pytest.raises(KeyError):
        load_gentd26(p, {"bogus": "x"})


def test_load_default_columns_and_iso_time(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("timestamp,request_id,adapters,scales\n2026-01-01T00:00:02Z,1,a;b,0.5;0.7\n"
                 "2026-01-01T00:00:00Z,0,,\n")
    reqs = load_gentd26(p)
    assert [r.t_arrive for r in reqs] == [0.0, 2.0]
    assert reqs[1].scales == [0.5, 0.7]
    json.dumps([r.id for r in reqs])
