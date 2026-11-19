import pytest, torch
from loramux.formats import normalize, compose_fused, compose_multi


def _km(m):
    return m if m.startswith("blocks.") else None


def test_kohya_folds_alpha():
    r = 4
    sd = {"lora_unet_blocks.0.attn.to_q.lora_down.weight": torch.ones(r, 8),
          "lora_unet_blocks.0.attn.to_q.lora_up.weight": torch.ones(6, r),
          "lora_unet_blocks.0.attn.to_q.alpha": torch.tensor(2.0)}
    out, rep = normalize(sd, _km)
    A, B = out["blocks.0.attn.to_q"]
    assert torch.all(B.float() == 0.5)          # alpha/r = 2/4
    assert rep["matched"] == rep["total"] == 1


def test_peft_default_alpha_is_rank():
    sd = {"transformer.blocks.0.x.lora_A.weight": torch.ones(2, 3),
          "transformer.blocks.0.x.lora_B.weight": torch.ones(5, 2)}
    out, _ = normalize(sd, _km)
    assert torch.all(out["blocks.0.x"][1].float() == 1.0)


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
