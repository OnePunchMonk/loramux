import torch

from bench.common import read_csv
from bench.linear_chain import BASELINES, build_chains, main, run


def test_cpu_smoke_rows_and_columns():
    rows = run(tokens=[16, 32], rank=4, layers=2, dim=16, adapters=2, device="cpu", rounds=2, warmup=1, iters=2)
    assert len(rows) == 2 * len(BASELINES)
    assert [r["baseline"] for r in rows[:6]] == list(BASELINES)
    for r in rows:
        for col in ("status", "note", "tokens", "layers", "dim", "rank", "adapters", "median_ms", "overhead_pct"):
            assert col in r
    by = {(r["tokens"], r["baseline"]): r for r in rows}
    assert by[(16, "B0")]["overhead_pct"] == 0.0
    for b in ("B0", "B1", "B2", "B3", "B4"):
        assert by[(16, b)]["status"] == "ok" and by[(16, b)]["median_ms"] > 0 and by[(16, b)]["n"] == 4
    assert by[(16, "B5")]["status"] == "unavailable" and by[(16, "B5")]["median_ms"] is None
    assert "reimplementation" in by[(16, "B2")]["note"]


def test_baselines_agree_numerically():
    with torch.no_grad():
        _x, fns = build_chains(layers=2, dim=16, rank=4, adapters=3, tokens=12, device="cpu",
                              dtype=torch.float32)
        ref = fns["B3"]()
        torch.testing.assert_close(fns["B2"](), ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(fns["B4"](), ref, atol=1e-4, rtol=1e-4)
        assert not torch.allclose(fns["B0"](), ref, atol=1e-3)
        # B1 merges adapter 0 only: equals the LoRA chain on a single-adapter batch's row 0
        row0 = fns["B1"]()[0]
        torch.testing.assert_close(row0, ref[0], atol=1e-4, rtol=1e-4)


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "k2a.csv"
    main(["--tokens", "8", "--rank", "4", "--layers", "1", "--dim", "16", "--adapters", "2", "--rounds", "1",
          "--warmup", "0", "--iters", "1", "--out", str(out)])
    env, rows = read_csv(out)
    assert "torch" in env and len(rows) == 6
