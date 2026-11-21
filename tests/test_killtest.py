import pytest

from bench.common import write_csv
from bench.killtest import Config, _any_regime, evaluate, render, run, verdict
from bench.spend import SpendCapError, append_row


@pytest.mark.parametrize("k1,k2,k3,want", [
    (True, True, True, "GO"), (True, True, False, "GREY"), (False, True, True, "NO-GO"),
    (True, False, None, "NO-GO"), (True, True, None, "INCOMPLETE"), (None, True, True, "INCOMPLETE")])
def test_verdict_table(k1, k2, k3, want):
    assert verdict(k1, k2, k3) == want


@pytest.mark.parametrize("passed,want", [([], None), ([None], None), ([False, None], None), ([False, True], True),
                                          ([False, False], False), ([None, True], True)])
def test_k3_needs_a_pass_and_stays_incomplete_while_a_regime_is_unmeasured(passed, want):
    assert _any_regime(passed) is want


def _results(d, *, k1a_err=1e-4, b4=8.0, b2=20.0, attn=3.0, all_=10.0, p99_ratio=0.4, thr_ratio=1.0, rhos=(0.7, 0.9)):
    write_csv(d / "k1a.csv", [{"check": "K1a", "status": "ok", "rel_err": k1a_err}], {})
    write_csv(d / "k1bc.csv", [{"check": "K1b", "status": "ok", "psnr": 50.0},
                               {"check": "K1c", "status": "ok", "psnr": 40.0, "lpips": 0.01}], {})
    write_csv(d / "k2a.csv", [{"baseline": b, "status": "ok", "tokens_requested": 4096, "tokens": 4096,
                               "overhead_pct": v, "note": "x"} for b, v in (("B0", 0.0), ("B2", b2), ("B4", b4))], {})
    if attn is not None:
        w = {"model": "qwen_image_1024", "bs": 8, "k": 8}
        smoke = {"model": "qwen_image_512", "bs": 1, "k": 1}   # not the preregistered workload: ignored
        write_csv(d / "k2b_qwen_image.csv", [{**w, "mode": "compile", "adapters": "attn", "overhead_pct": attn},
                                             {**w, "mode": "compile", "adapters": "all", "overhead_pct": all_},
                                             {**w, "mode": "eager", "adapters": "all", "overhead_pct": 99.0},
                                             {**smoke, "mode": "compile", "adapters": "attn", "overhead_pct": 99.0},
                                             {**smoke, "mode": "compile", "adapters": "all", "overhead_pct": 0.1}], {})
    rows = []
    for rho in rhos:
        rows += [{"rho": rho, "policy": "P1", "throughput_rps": 1.0, "p99_ms": 100.0},
                 {"rho": rho, "policy": "P2", "throughput_rps": thr_ratio, "p99_ms": 100.0 * p99_ratio}]
    write_csv(d / "sim_qwen_image_1024_summary.csv", rows, {})
    return d


@pytest.mark.parametrize("kw,want", [
    ({}, "GO"),                                  # all bars met
    ({"p99_ratio": 0.9}, "GREY"),                # K3 fails, kernel fine
    ({"k1a_err": 1e-2}, "NO-GO"),                # K1a fails
    ({"b4": 15.0}, "NO-GO"),                     # K2a: 15/20 > 0.5
    ({"all_": 13.0}, "NO-GO"),                   # K2b all-linear > 12% (eager and smoke rows ignored)
    ({"attn": None}, "INCOMPLETE"),              # K2b not measured
    ({"p99_ratio": 0.9, "thr_ratio": 1.6}, "GO"),  # K3 passes on throughput instead of P99
    ({"rhos": (0.7,)}, "INCOMPLETE"),            # a K3 rho missing -> regime not measured
])
def test_evaluate_verdicts(tmp_path, kw, want):
    ev = evaluate(_results(tmp_path, **kw), spend_path=str(tmp_path / "SPEND.md"))
    assert ev["verdict"] == want
    assert f"**Verdict: {want}.**" in render(ev)


def test_roofline_flags_overhead_far_above_prediction(tmp_path):
    ev = evaluate(_results(tmp_path, b4=60.0, b2=200.0), dim=3072, spend_path=str(tmp_path / "SPEND.md"))
    (r,) = ev["roofline"]
    assert r["predicted_pct"] == pytest.approx(230 * 3 / 3072 * 100)
    assert r["profile_with_ncu"] is True


def test_run_checks_caps_before_any_stage_and_logs(tmp_path):
    spend = tmp_path / "SPEND.md"
    append_row(spend, "2026-11-20", "earlier", "H100", 360, est_usd=24.0)
    c = Config(tmp_path / "res", adapters=["a.safetensors"])
    with pytest.raises(SpendCapError):
        run(c, dry_run=True, spend_path=str(spend))
    assert not (c.results / "runlog.csv").exists()
    log = run(c, only={"k3_512"}, dry_run=True, spend_path=str(spend))  # CPU-only stage: no cap check
    assert [r["stage"] for r in log] == ["k3_512"] and (c.results / "runlog.csv").exists()
