import os

import pytest
import torch
from safetensors.torch import save_file

from loramux.blob import composite_from_blobs, convert, read_blob, read_header, write_blob
from loramux.formats import compose_fused
from loramux.pool import AdapterPool

SHAPES = {"blocks.0.q": (8, 6), "blocks.0.k": (8, 6), "blocks.0.v": (8, 6), "blocks.0.o": (6, 8)}


def _km(m):
    return m if m in SHAPES else None


def _peft_sd(ranks, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for (k, (d_in, d_out)), r in zip(SHAPES.items(), ranks, strict=True):
        sd[f"transformer.{k}.lora_A.weight"] = torch.randn(r, d_in, generator=g)
        sd[f"transformer.{k}.lora_B.weight"] = torch.randn(d_out, r, generator=g)
    return sd


def _bf(t):
    return t.to(torch.bfloat16)


def test_convert_register_blob_equals_register(tmp_path):
    sd = _peft_sd([2, 3, 4, 1])
    src = tmp_path / "ad.safetensors"
    save_file(sd, str(src))
    out = tmp_path / "ad.lmx"
    header = convert(src, out, _km, "toy", assume_alpha_eq_rank=True)
    assert header["base_model"] == "toy" and header["meta"]["format"] == "peft"
    assert {lay["key"] for lay in header["layers"]} == set(SHAPES)

    canon = {k: (sd[f"transformer.{k}.lora_A.weight"], sd[f"transformer.{k}.lora_B.weight"]) for k in SHAPES}
    p1 = AdapterPool(SHAPES, slots=2, r_max=4, device="cpu")
    p2 = AdapterPool(SHAPES, slots=2, r_max=4, device="cpu")
    p1.register("a", {k: (_bf(a), _bf(b)) for k, (a, b) in canon.items()})
    p2.register_blob("a", out)
    assert "a" not in p2.cpu                   # lazy: nothing loaded yet
    p1.ensure_resident("a")
    p2.ensure_resident("a")
    s1, s2 = p1.slot_of["a"], p2.slot_of["a"]
    for k in SHAPES:
        assert torch.equal(p1.A[k][s1], p2.A[k][s2])
        assert torch.equal(p1.B[k][s2], p2.B[k][s2])
    assert p1.slot_rank(s1) == p2.slot_rank(s2) == 4
    for k in SHAPES:
        assert int(p1.rank[k][s1]) == int(p2.rank[k][s2]) == canon[k][0].shape[0]   # per-layer rank
    assert p1.h2d_bytes == p2.h2d_bytes


def test_convert_uses_config_from_peft_dir(tmp_path):
    import json
    d = tmp_path / "ad"
    d.mkdir()
    save_file(_peft_sd([2, 2, 2, 2]), str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4}))
    out = tmp_path / "x.lmx"
    convert(d, out, _km, "toy")
    _, b = read_blob(out)["blocks.0.q"]
    ref = _peft_sd([2, 2, 2, 2])["transformer.blocks.0.q.lora_B.weight"] * 2.0
    torch.testing.assert_close(b.float(), _bf(ref).float())
    with pytest.raises(ValueError, match="alpha"):                     # no config -> missing alpha raises
        convert(_peft_sd([2, 2, 2, 2]), tmp_path / "y.lmx", _km, "toy")
    assert not (tmp_path / "y.lmx").exists()


def test_convert_fused_groups(tmp_path):
    sd = _peft_sd([2, 3, 4, 1])
    out = tmp_path / "f.lmx"
    convert(sd, out, _km, "toy", assume_alpha_eq_rank=True,
            fused_groups={"qkv": ["blocks.0.q", "blocks.0.k", "blocks.0.v"]})
    got = read_blob(out)
    assert set(got) == {"qkv", "blocks.0.o"}
    parts = [(sd[f"transformer.{k}.lora_A.weight"], sd[f"transformer.{k}.lora_B.weight"])
             for k in ("blocks.0.q", "blocks.0.k", "blocks.0.v")]
    A, B = compose_fused(parts)
    assert got["qkv"][0].shape == (9, 8) and got["qkv"][1].shape == (18, 9)
    x = torch.randn(3, 8)
    torch.testing.assert_close(x @ got["qkv"][0].float().T @ got["qkv"][1].float().T, x @ A.T @ B.T,
                               atol=0.5, rtol=5e-2)
    with pytest.raises(ValueError, match="missing"):
        convert({k: v for k, v in sd.items() if "blocks.0.k" not in k}, tmp_path / "g.lmx", _km, "toy",
                assume_alpha_eq_rank=True, fused_groups={"qkv": ["blocks.0.q", "blocks.0.k", "blocks.0.v"]})


def test_write_blob_replaces_atomically_and_keeps_old_mapping(tmp_path):
    path = tmp_path / "a.lmx"
    ad1 = {"l": (torch.ones(2, 8), torch.ones(6, 2))}
    ad2 = {"l": (torch.full((2, 8), 2.0), torch.full((6, 2), 2.0))}
    write_blob(path, ad1, "toy")
    old = read_blob(path)                      # mmapped view of the first file
    write_blob(path, ad2, "toy")               # replaces via rename; must not truncate the mapped inode
    assert float(old["l"][0].float().sum()) == 16.0
    assert float(read_blob(path)["l"][0].float().sum()) == 32.0
    assert [f for f in os.listdir(tmp_path) if ".tmp" in f] == []


def test_write_blob_failure_leaves_no_temp_and_keeps_target(tmp_path):
    path = tmp_path / "a.lmx"
    write_blob(path, {"l": (torch.ones(2, 8), torch.ones(6, 2))}, "toy")
    with pytest.raises(ValueError):
        write_blob(path, {"l": (torch.ones(2, 8), torch.ones(6, 3))}, "toy")   # rank mismatch
    assert os.listdir(tmp_path) == ["a.lmx"] and read_header(path)["layers"][0]["r"] == 2


def test_register_blob_validates(tmp_path):
    p = AdapterPool({"l": (8, 6)}, slots=1, r_max=2, device="cpu")
    write_blob(tmp_path / "bad_layer.lmx", {"zz": (torch.ones(1, 8), torch.ones(6, 1))}, "toy")
    with pytest.raises(KeyError):
        p.register_blob("a", tmp_path / "bad_layer.lmx")
    write_blob(tmp_path / "big.lmx", {"l": (torch.ones(3, 8), torch.ones(6, 3))}, "toy")
    with pytest.raises(ValueError, match="r_max"):
        p.register_blob("a", tmp_path / "big.lmx")
    write_blob(tmp_path / "shape.lmx", {"l": (torch.ones(1, 7), torch.ones(6, 1))}, "toy")
    with pytest.raises(ValueError, match="pool expects"):
        p.register_blob("a", tmp_path / "shape.lmx")
    with pytest.raises(KeyError):
        p.ensure_resident("never")
    assert p.free == [0]                      # failed lookup does not leak a slot


def test_register_and_register_blob_replace_each_other(tmp_path):
    p = AdapterPool({"l": (8, 6)}, slots=1, r_max=2, device="cpu")
    write_blob(tmp_path / "a.lmx", {"l": (torch.ones(1, 8), torch.ones(6, 1))}, "toy")
    p.register_blob("a", tmp_path / "a.lmx")
    p.register("a", {"l": (torch.full((1, 8), 5.0), torch.full((6, 1), 5.0))})
    assert "a" not in p.blobs
    p.ensure_resident("a")
    assert float(p.A["l"][p.slot_of["a"], 0, 0]) == 5.0


def test_composite_from_blobs_is_weighted_sum(tmp_path):
    g = torch.Generator().manual_seed(7)
    mk = lambda r, layers: {k: (torch.randn(r, SHAPES[k][0], generator=g), torch.randn(SHAPES[k][1], r, generator=g))
                            for k in layers}
    a1, a2 = mk(2, ["blocks.0.q", "blocks.0.o"]), mk(3, ["blocks.0.q", "blocks.0.k"])   # partial overlap
    write_blob(tmp_path / "1.lmx", a1, "toy")
    write_blob(tmp_path / "2.lmx", a2, "toy")
    comp = composite_from_blobs([(tmp_path / "1.lmx", 0.5), (tmp_path / "2.lmx", 2.0)])
    assert set(comp) == {"blocks.0.q", "blocks.0.o", "blocks.0.k"}
    assert comp["blocks.0.q"][0].shape[0] == 5 and comp["blocks.0.o"][0].shape[0] == 2

    r1, r2 = read_blob(tmp_path / "1.lmx"), read_blob(tmp_path / "2.lmx")     # bf16-rounded parts

    def delta(ad, k, x, s):
        return s * x @ ad[k][0].float().T @ ad[k][1].float().T

    uses = {"blocks.0.q": [(r1, .5), (r2, 2.0)], "blocks.0.o": [(r1, .5)], "blocks.0.k": [(r2, 2.0)]}
    for key, parts in uses.items():
        x = torch.randn(4, SHAPES[key][0], generator=g)
        a, b = comp[key]
        expect = sum(delta(ad, key, x, s) for ad, s in parts)
        torch.testing.assert_close(x @ a.float().T @ b.float().T, expect, atol=0.3, rtol=3e-2)
