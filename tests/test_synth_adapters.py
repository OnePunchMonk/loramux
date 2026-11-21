import json

import torch

from bench.synth_adapters import ATTN_PAT, main, make_adapter, make_perf_set
from loramux.blob import read_blob

SHAPES = {
    "blocks.0.attn.to_q": (64, 64), "blocks.0.attn.to_k": (64, 64),
    "blocks.0.attn.to_v": (64, 64), "blocks.0.attn.to_out.0": (64, 64),
    "blocks.0.ff.net.0": (64, 256), "blocks.0.ff.net.2": (256, 64),
}
ATTN_KEYS = {k for k in SHAPES if ".attn." in k}


def test_determinism_by_seed():
    a = make_adapter(SHAPES, rank=8, targets="all", seed=3)
    b = make_adapter(SHAPES, rank=8, targets="all", seed=3)
    c = make_adapter(SHAPES, rank=8, targets="all", seed=4)
    for k in SHAPES:
        assert torch.equal(a[k][0], b[k][0]) and torch.equal(a[k][1], b[k][1])
    assert not torch.equal(a["blocks.0.ff.net.0"][0], c["blocks.0.ff.net.0"][0])


def test_rank_and_target_split():
    full = make_adapter(SHAPES, rank=8, targets="all", seed=0)
    attn = make_adapter(SHAPES, rank=16, targets="attn", seed=0)
    assert set(full) == set(SHAPES)
    assert set(attn) == ATTN_KEYS
    for k, (a, b) in full.items():
        d_in, d_out = SHAPES[k]
        assert a.shape == (8, d_in) and b.shape == (d_out, 8) and a.dtype == torch.float32
    assert all(attn[k][0].shape[0] == 16 for k in attn)


def test_attention_filter_pattern():
    for key in ("blocks.0.attn.to_q", "blocks.3.attn.to_k", "blocks.3.attn.to_v", "x.to_out.0", "x.to_out",
                "blocks.0.attn"):
        assert ATTN_PAT.search(key), key
    for key in ("blocks.0.ff.net.0", "blocks.0.to_q_norm", "blocks.0.to_outx", "blocks.0.attn_proj"):
        assert not ATTN_PAT.search(key), key


def test_statistics_match_init_scales():
    big = make_adapter({"big": (4096, 32)}, rank=32, targets="all", seed=0)
    a, b = big["big"]
    assert abs(a.std().item() - 1 / 4096 ** 0.5) / (1 / 4096 ** 0.5) < 0.10
    assert abs(b.std().item() - 0.1) / 0.1 < 0.10


def test_make_perf_set_names_split_and_ranks():
    perf = make_perf_set(SHAPES, n=8, seed=0)
    assert list(perf) == [
        "perf_000_attn_r8", "perf_001_attn_r16", "perf_002_attn_r32", "perf_003_attn_r64",
        "perf_004_all_r8", "perf_005_all_r16", "perf_006_all_r32", "perf_007_all_r64",
    ]
    assert all(set(perf[k]) == ATTN_KEYS for k in list(perf)[:4])
    assert all(set(perf[k]) == set(SHAPES) for k in list(perf)[4:])


def test_cli_writes_blobs_and_index(tmp_path):
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps({k: list(v) for k, v in SHAPES.items()}))
    out = tmp_path / "adapters"
    assert main(["--shapes", str(shapes), "--out", str(out), "--n", "4"]) == 0
    index = json.loads((out / "index.json").read_text())
    assert len(index["adapters"]) == 4
    name = "perf_000_attn_r8"
    assert index["adapters"][name]["targets"] == "attn" and index["adapters"][name]["rank"] == 8
    assert set(read_blob(out / f"{name}.lmx")) == ATTN_KEYS
    assert index["adapters"][name]["payload_bytes"] > 0
