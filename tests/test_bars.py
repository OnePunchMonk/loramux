import math

import pytest

from bench import bars


def test_every_bar_registered_and_typed():
    ids = {"K1a", "K1b", "K1c_psnr", "K1c_lpips", "K2a", "K2b_attn", "K2b_all", "K3_throughput", "K3_p99",
           "L1", "L2", "L3", "L4_ms", "L4_bw", "L5", "P1", "P2_psnr", "P2_lpips", "P2_slots", "P3_psnr",
           "P3_lpips", "P3_bytes"}
    assert set(bars.BARS) == ids
    for b in bars.ALL_BARS:
        assert b.op in ("<=", ">=", "==", "report")
        assert (b.threshold is None) == (b.op == "report")
    assert bars.K1A.threshold == 2e-3 and bars.K1B.threshold == 45.0 and bars.K2B_ALL.threshold == 12.0


@pytest.mark.parametrize("bar,value,ok", [
    (bars.K1A, 2e-3, True), (bars.K1A, 2.1e-3, False), (bars.K1B, math.inf, True), (bars.K1B, 44.9, False),
    (bars.L5, 0.0, True), (bars.L5, 0.01, False), (bars.K1A, math.nan, False), (bars.K1A, None, None),
    (bars.P1, 0.4, None)])
def test_check(bar, value, ok):
    v = bars.check(bar, value)
    assert v.passed is ok and v.bar_id == bar.id
    assert bar.id in str(v)


def test_groups_and_or():
    k3 = bars.GROUPS["K3"]
    assert bars.check_group(k3, {"K3_throughput": 1.2, "K3_p99": 0.4}).passed is True
    assert bars.check_group(k3, {"K3_throughput": 1.2, "K3_p99": 0.6}).passed is False
    assert bars.check_group(k3, {"K3_throughput": 1.2}).passed is None
    assert bars.check_group(bars.GROUPS["P2"], {"P2_psnr": 36, "P2_lpips": 0.01}).passed is True  # report-only ignored


def test_evaluate_k1():
    v = bars.evaluate_k1([1e-4, 1e-3], [50.0, 46.0], [36.0, 34.5], [0.02, 0.04])
    assert v.passed is True                         # K1c uses the mean (35.25)
    assert [x.value for x in v.verdicts] == [1e-3, 46.0, 35.25, 0.03]
    assert bars.evaluate_k1([1e-4, 3e-3], [50], [40], [0.0]).passed is False      # K1a uses the max
    assert bars.evaluate_k1([1e-4], [50, 44], [40], [0.0]).passed is False        # K1b uses the min
    assert bars.evaluate_k1([1e-4]).passed is None
    assert bars.evaluate_k1([5e-3]).passed is False
