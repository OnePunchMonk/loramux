import pytest
import torch

from loramux.formats import compose_fused, compose_multi, normalize


def _km(m):
    return m if m.startswith("blocks.") else None


def test_kohya_folds_alpha():
    r = 4
    sd = {"lora_unet_blocks.0.attn.to_q.lora_down.weight": torch.ones(r, 8),
          "lora_unet_blocks.0.attn.to_q.lora_up.weight": torch.ones(6, r),
          "lora_unet_blocks.0.attn.to_q.alpha": torch.tensor(2.0)}
    out, rep = normalize(sd, _km)
    _, B = out["blocks.0.attn.to_q"]
    assert torch.all(B.float() == 0.5)          # alpha/r = 2/4
    assert rep["matched"] == rep["total"] == 1


def _peft_sd(r=2, name="transformer.blocks.0.x"):
    return {f"{name}.lora_A.weight": torch.ones(r, 3), f"{name}.lora_B.weight": torch.ones(5, r)}


def test_peft_missing_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        normalize(_peft_sd(), _km)
    with pytest.raises(ValueError, match="alpha"):
        normalize(_peft_sd(), _km, config={"r": 2})


def test_peft_assume_alpha_eq_rank():
    out, _ = normalize(_peft_sd(), _km, assume_alpha_eq_rank=True)
    assert torch.all(out["blocks.0.x"][1].float() == 1.0)


def test_peft_alpha_from_config():
    out, rep = normalize(_peft_sd(r=2), _km, config={"lora_alpha": 8, "r": 2})
    assert torch.all(out["blocks.0.x"][1].float() == 4.0)       # 8 / 2
    assert rep["format"] == "peft" and rep["ranks"] == {"blocks.0.x": 2}


def test_peft_rslora_scale():
    out, _ = normalize(_peft_sd(r=4), _km, config={"lora_alpha": 8, "use_rslora": True})
    assert torch.all(out["blocks.0.x"][1].float() == 4.0)       # 8 / sqrt(4)


def test_peft_alpha_pattern_and_rank_pattern():
    sd = {**_peft_sd(2, "transformer.blocks.0.x"), **_peft_sd(4, "transformer.blocks.1.y")}
    cfg = {"lora_alpha": 2, "alpha_pattern": {"blocks.1.y": 16}, "rank_pattern": {"blocks.1.y": 4}}
    out, _ = normalize(sd, _km, config=cfg)
    assert torch.all(out["blocks.0.x"][1].float() == 1.0)       # 2 / 2
    assert torch.all(out["blocks.1.y"][1].float() == 4.0)       # 16 / 4
    with pytest.raises(ValueError, match="rank_pattern"):
        normalize(sd, _km, config={"lora_alpha": 2, "rank_pattern": {"blocks.1.y": 8}})


def test_alpha_tensor_wins_over_config():
    sd = _peft_sd(2)
    sd["transformer.blocks.0.x.alpha"] = torch.tensor(1.0)
    out, _ = normalize(sd, _km, config={"lora_alpha": 8})
    assert torch.all(out["blocks.0.x"][1].float() == 0.5)


def test_kohya_missing_alpha_raises():
    sd = {"lora_unet_blocks.0.x.lora_down.weight": torch.ones(2, 3),
          "lora_unet_blocks.0.x.lora_up.weight": torch.ones(5, 2)}
    with pytest.raises(ValueError, match="alpha"):
        normalize(sd, _km)
    out, _ = normalize(sd, _km, assume_alpha_eq_rank=True)
    assert "blocks.0.x" in out


@pytest.mark.parametrize("tokens", [("lora_A", "lora_B"), ("lora_down", "lora_up")])
def test_comfy_prefix_both_namings(tokens):
    a, b = tokens
    sd = {f"diffusion_model.blocks.0.x.{a}.weight": torch.ones(2, 3),
          f"diffusion_model.blocks.0.x.{b}.weight": torch.ones(5, 2),
          "diffusion_model.blocks.0.x.alpha": torch.tensor(4.0)}
    out, rep = normalize(sd, _km)
    assert rep["format"] == "comfy"
    assert torch.all(out["blocks.0.x"][1].float() == 2.0)


@pytest.mark.parametrize("key", [
    "transformer.blocks.0.x.lokr_w1", "transformer.blocks.0.x.hada_w1_a",
    "transformer.blocks.0.x.dora_scale", "transformer.blocks.0.x.lora_magnitude_vector.default.weight"])
def test_lokr_loha_dora_rejected(key):
    sd = {**_peft_sd(), key: torch.ones(2)}
    with pytest.raises(ValueError, match="unsupported adapter type"):
        normalize(sd, _km, config={"lora_alpha": 2})


def test_dora_config_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        normalize(_peft_sd(), _km, config={"lora_alpha": 2, "use_dora": True})


def test_strict_false_reports_unmatched():
    sd = {**_peft_sd(), **_peft_sd(2, "transformer.nope")}
    out, rep = normalize(sd, _km, config={"lora_alpha": 2}, strict=False)
    assert list(out) == ["blocks.0.x"]
    assert rep["matched"] == 1 and rep["total"] == 2 and rep["unmatched"] == ["nope"]


def test_split_keymap_slices_b_rows():
    sd = _peft_sd(2, "transformer.blocks.0.qkv")
    sd["transformer.blocks.0.qkv.lora_B.weight"] = torch.arange(12.0).view(6, 2)
    km = lambda m: [("blocks.0.q", slice(0, 2)), ("blocks.0.k", slice(2, 4)), ("blocks.0.v", slice(4, 6))]
    out, rep = normalize(sd, km, config={"lora_alpha": 2})
    assert rep["matched"] == 1 and set(out) == {"blocks.0.q", "blocks.0.k", "blocks.0.v"}
    full = sd["transformer.blocks.0.qkv.lora_B.weight"]
    assert torch.equal(out["blocks.0.k"][1].float(), full[2:4])
    assert torch.equal(out["blocks.0.v"][0].float(), torch.ones(2, 3))


def test_duplicate_target_raises():
    sd = {**_peft_sd(2, "transformer.blocks.0.x"), **_peft_sd(2, "transformer.blocks.0.y")}
    with pytest.raises(ValueError, match="already filled"):
        normalize(sd, lambda m: "blocks.0.x", config={"lora_alpha": 2})


def test_load_lora_dir_and_file(tmp_path):
    import json

    from safetensors.torch import save_file

    from loramux.formats import load_lora
    save_file(_peft_sd(), str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4}))
    sd, cfg = load_lora(tmp_path)
    assert cfg == {"lora_alpha": 4} and len(sd) == 2
    sd2, cfg2 = load_lora(tmp_path / "adapter_model.safetensors")
    assert cfg2 is None and set(sd2) == set(sd)


def test_unmatched_raises():
    sd = {"transformer.nope.lora_A.weight": torch.ones(2, 3), "transformer.nope.lora_B.weight": torch.ones(5, 2)}
    with pytest.raises(KeyError):
        normalize(sd, _km)


def test_fused_qkv_equals_three_loras():
    g = torch.Generator().manual_seed(1)
    d_in, ds, rs = 16, [8, 4, 4], [2, 3, 2]
    parts = [(torch.randn(r, d_in, generator=g), torch.randn(d, r, generator=g)) for d, r in zip(ds, rs)]
    A, B = compose_fused(parts)
    x = torch.randn(5, d_in, generator=g)
    sep = torch.cat([x @ a.T @ b.T for a, b in parts], dim=1)
    torch.testing.assert_close(x @ A.T @ B.T, sep, atol=1e-5, rtol=1e-5)


def test_multi_lora_composition():
    g = torch.Generator().manual_seed(2)
    a1, b1, a2, b2 = torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g), torch.randn(3, 6, generator=g), torch.randn(5, 3, generator=g)
    A, B = compose_multi([(a1, b1, 0.5), (a2, b2, 2.0)])
    x = torch.randn(4, 6, generator=g)
    exp = 0.5 * x @ a1.T @ b1.T + 2.0 * x @ a2.T @ b2.T
    torch.testing.assert_close(x @ A.T @ B.T, exp, atol=1e-5, rtol=1e-5)
