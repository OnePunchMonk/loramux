import pytest
import torch

from bench.common import write_csv
from bench.latency_bars import evaluate
from bench.lossy_quality import GB, apply_variant, size_row
from bench.payload_bench import run_payload_bench
from bench.synth_adapters import make_adapter

SHAPES = {"b.attn.to_q": (32, 32), "b.attn.to_out.0": (32, 32), "b.ff.net.0": (32, 64)}


def test_payload_rows_feed_latency_bars(tmp_path):
    rows = run_payload_bench(SHAPES, ranks=(4, 8), reps=2, workdir=tmp_path)
    assert {(r["rank"], r["targets"]) for r in rows} == {(r, t) for r in (4, 8) for t in ("attn", "all")}
    assert all(r["load_ms"] > 0 and r["h2d_gbps"] > 0 for r in rows)
    # bf16 blob at native rank is at most half the fp32 source (plus 256-byte alignment padding)
    r = next(x for x in rows if x["targets"] == "attn" and x["rank"] == 8)
    assert r["blob_bytes"] <= 0.5 * r["source_bytes"] + 2 * 256 * 2
    write_csv(tmp_path / "p.csv", rows, {})
    ev = evaluate(payload=tmp_path / "p.csv")
    assert ev["singles"]["P1"].value == pytest.approx(0.5, abs=0.02)


def test_fp8_halves_bytes_and_keeps_values():
    canon = make_adapter(SHAPES, 8, "all", 0)
    out, nb = apply_variant(canon, "fp8")
    exact = sum((a.numel() + b.numel()) * 2 for a, b in canon.values())
    assert nb < 0.75 * exact  # 1 B/elem + 4 B/row scale; scales dominate at these tiny shapes
    for k, (a, b) in canon.items():
        torch.testing.assert_close(out[k][0].float(), a, rtol=0.1, atol=1e-2 * a.abs().max().item())


def test_svd_truncation_reduces_rank_and_bytes():
    g = torch.Generator().manual_seed(0)
    A, B = torch.randn(2, 32, generator=g), torch.randn(32, 2, generator=g)
    canon = {"l": (A.repeat(4, 1), B.repeat(1, 4))}   # rank 8 holding a rank-2 delta
    out, nb = apply_variant(canon, "svd", 0.999)
    assert out["l"][0].shape[0] <= 2 and nb < 8 * 64 * 2
    torch.testing.assert_close((out["l"][1] @ out["l"][0]).float(), canon["l"][1] @ canon["l"][0], rtol=1e-2, atol=1e-2)


def test_size_row_and_unknown_variant():
    canon = make_adapter(SHAPES, 8, "all", 0)
    row = size_row(canon, "fp8")
    assert row["slots_per_gb_variant"] / row["slots_per_gb_exact"] == pytest.approx(row["bytes_exact"] / row["bytes_variant"])
    assert row["slots_per_gb_exact"] == GB / row["bytes_exact"]
    with pytest.raises(ValueError):
        apply_variant(canon, "int4")
