import pytest

from loramux.latency import AdapterSource, LatencyLog, RequestTimeline, breakdown


def _cached_timeline() -> RequestTimeline:
    # text 100 ms, adapter already resident before text finished, queue 100 ms, denoise 1000 ms,
    # vae 50 ms, image encode 50 ms, transfer 50 ms
    return RequestTimeline(t_arrive=0.0, t_admit_ready_text=0.1, t_adapter_resident=0.05, t_first_step=0.2,
                           t_last_step=1.2, t_vae_done=1.25, t_encode_done=1.3, t_sent=1.35,
                           adapter_source=AdapterSource.GPU_HIT, n_steps=28, request_id="r1", adapter_key="a")


def _miss_timeline() -> RequestTimeline:
    # text 100 ms, adapter lands 300 ms after text, queue 100 ms after adapter, denoise 1000 ms, tail 150 ms
    return RequestTimeline(t_arrive=0.0, t_admit_ready_text=0.1, t_adapter_resident=0.4, t_first_step=0.5,
                           t_last_step=1.5, t_vae_done=1.55, t_encode_done=1.6, t_sent=1.65,
                           adapter_source=AdapterSource.MISS, n_steps=28, request_id="r2", adapter_key="b")


def _base_timeline() -> RequestTimeline:
    # base model: t_adapter_resident = 99 s must be ignored
    return RequestTimeline(t_arrive=0.0, t_admit_ready_text=0.1, t_adapter_resident=99.0, t_first_step=0.2,
                           t_last_step=1.2, t_vae_done=1.25, t_encode_done=1.3, t_sent=1.35,
                           n_steps=28, request_id="r3", adapter_key=None)


def test_breakdown_cached_adapter():
    b = breakdown(_cached_timeline())
    assert b["queue_wait"] == pytest.approx(100.0)
    assert b["adapter_wait"] == pytest.approx(0.0)
    assert b["text_encode"] == pytest.approx(100.0)
    assert b["denoise"] == pytest.approx(1000.0)
    assert b["vae_decode"] == pytest.approx(50.0)
    assert b["image_encode"] == pytest.approx(50.0)
    assert b["transfer"] == pytest.approx(50.0)
    assert b["e2e"] == pytest.approx(1350.0)


def test_breakdown_adapter_miss_exposes_adapter_wait():
    b = breakdown(_miss_timeline())
    assert b["adapter_wait"] == pytest.approx(300.0)
    assert b["queue_wait"] == pytest.approx(100.0)
    assert b["e2e"] == pytest.approx(1650.0)


def test_base_request_ignores_adapter_resident():
    b = breakdown(_base_timeline())
    assert b["adapter_wait"] == 0.0
    assert b["queue_wait"] == pytest.approx(100.0)
    assert b["e2e"] == pytest.approx(1350.0)


@pytest.mark.parametrize("make", [_cached_timeline, _miss_timeline, _base_timeline])
def test_terms_sum_to_e2e(make):
    b = breakdown(make())
    parts = ("queue_wait", "adapter_wait", "text_encode", "denoise", "vae_decode", "image_encode", "transfer")
    assert sum(b[k] for k in parts) == pytest.approx(b["e2e"])


def _denoise_only(seconds: float, source: AdapterSource) -> RequestTimeline:
    return RequestTimeline(t_arrive=0.0, t_admit_ready_text=0.0, t_adapter_resident=0.0, t_first_step=0.0,
                           t_last_step=seconds, t_vae_done=seconds, t_encode_done=seconds, t_sent=seconds,
                           adapter_source=source, adapter_key=None if source is AdapterSource.NONE else "k")


def test_to_rows_has_source_timestamps_and_terms():
    log = LatencyLog()
    log.append(_cached_timeline())
    (row,) = log.to_rows()
    assert row["adapter_source"] == "gpu_hit"
    assert row["t_sent"] == 1.35
    assert row["denoise"] == pytest.approx(1000.0)
    assert row["e2e"] == pytest.approx(1350.0)


def test_percentiles_known_set():
    log = LatencyLog()
    for s in (1.0, 2.0, 3.0, 4.0, 5.0):
        log.append(_denoise_only(s, AdapterSource.GPU_HIT))
    log.append(_denoise_only(1.0, AdapterSource.NONE))
    log.append(_denoise_only(2.0, AdapterSource.NONE))
    out = log.percentiles("denoise")
    assert out["gpu_hit"] == {"p50": pytest.approx(3000.0), "p90": pytest.approx(4600.0),
                              "p99": pytest.approx(4960.0), "n": 5}
    assert out["none"]["p50"] == pytest.approx(1500.0)
    assert out["none"]["n"] == 2


def test_percentiles_rejects_unknown_term():
    with pytest.raises(ValueError):
        LatencyLog().percentiles("nope")
