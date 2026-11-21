import pytest

from bench.common import write_csv
from bench.latency_bars import evaluate, render


def _w(path, rows):
    write_csv(path, rows, {})
    return path


def _lat(d, name, lora_e2e, base_e2e, wait=1.0):
    rows = [{"e2e": e, "adapter_key": "a", "adapter_source": "cpu_hit", "adapter_wait": wait} for e in lora_e2e]
    rows += [{"e2e": e, "adapter_key": None, "adapter_source": "none", "adapter_wait": 0.0} for e in base_e2e]
    return _w(d / name, rows)


def test_all_bars_from_csvs(tmp_path):
    ev = evaluate(
        mixed07=_lat(tmp_path, "m07.csv", [105] * 9, [100] * 9, wait=40.0),          # L1 1.05, L3 40/50
        mixed09=_lat(tmp_path, "m09.csv", [200] * 9, [200] * 9),
        keyed09=_lat(tmp_path, "k09.csv", [500] * 9, [500] * 9),                    # L2 0.4
        steps=_w(tmp_path / "s.csv", [{"step_ms": 50.0, "post_ms": 0.0}] * 4),      # L5 0
        payload=_w(tmp_path / "p.csv", [{"rank": 32, "targets": "attn", "source_bytes": 100e6, "blob_bytes": 40e6,
                                         "load_ms": 2.0, "h2d_gbps": 25.0}]),        # L4 2 ms, 0.8 of H2D; P1 0.4
        lossy=_w(tmp_path / "l.csv", [{"variant": v, "adapter": "a", "prompt_idx": 0, "psnr": psnr, "lpips": 0.01,
                                       "bytes_exact": 100, "bytes_variant": 50, "slots_per_gb_exact": 10,
                                       "slots_per_gb_variant": 20} for v, psnr in (("fp8", 38.0), ("svd", 30.0))]))
    s = {k: v.value for k, v in ev["singles"].items()}
    assert s == pytest.approx({"L1": 1.05, "L2": 0.4, "L3": 0.8, "L5": 0.0, "P1": 0.4})
    assert all(ev["singles"][b].passed for b in ("L1", "L2", "L3", "L5"))
    assert ev["groups"]["L4"].passed is True
    assert ev["decisions"] == {"fp8_storage_default": True, "svd_truncation_default": False}  # SVD PSNR 30 < 35
    assert "SVD rank truncation on by default: **no**" in render(ev)


def test_missing_inputs_are_not_measured_and_lossy_stays_off(tmp_path):
    ev = evaluate(steps=_w(tmp_path / "s.csv", [{"step_ms": 50.0}]))  # pre-post_ms CSV: L5 not measured
    assert all(v.passed is None for v in ev["singles"].values())
    assert all(g.passed is None for g in ev["groups"].values())
    assert ev["decisions"] == {"fp8_storage_default": False, "svd_truncation_default": False}
