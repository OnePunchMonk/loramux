import pytest
import torch

from loramux.formats import normalize
from loramux.formats.keymaps import (
    engine_linear_names,
    engine_out_features,
    make_flux2_keymap,
    make_keymap,
    make_qwen_image_keymap,
)

# ---- generic (no diffusers needed) -------------------------------------------------------------


def test_generic_exact_and_underscore():
    km = make_keymap(["a.b_c.0.d", "x.to_q", "x.to_out.0"])
    assert km("a.b_c.0.d") == "a.b_c.0.d"
    assert km("a_b_c_0_d") == "a.b_c.0.d"
    assert km("x_to_out_0") == "x.to_out.0"
    assert km("nope") is None


def test_generic_ambiguity_raises_on_lookup():
    km = make_keymap(["a.b_c", "a_b.c", "z"])
    assert km("z") == "z"
    with pytest.raises(ValueError, match="ambiguous"):
        km("a_b_c")
    assert km("a.b_c") == "a.b_c"            # the exact dotted name is never ambiguous


def test_generic_aliases_and_split():
    km = make_keymap(["q", "k"], {"qk.fused": [("q", slice(0, 2)), ("k", slice(2, 4))], "w.in": "q"})
    assert km("w_in") == "q"
    assert km("qk_fused") == [("q", slice(0, 2)), ("k", slice(2, 4))]


# ---- real diffusers models ---------------------------------------------------------------------

pytest.importorskip("diffusers")
from tiny_models import tiny_flux2, tiny_qwen_image


def _fake(names, style, r=2, seed=0):
    """Fake adapter for each engine linear: style kohya | comfy | peft. Returns (sd, {engine: dW})."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for n, (d_in, d_out) in names.items():
        sd.update(_entry(n, d_in, d_out, r, style, g))
    return sd


def _entry(src, d_in, d_out, r, style, g):
    A, B = torch.randn(r, d_in, generator=g), torch.randn(d_out, r, generator=g)
    if style == "kohya":
        p = "lora_unet_" + src.replace(".", "_")
        return {f"{p}.lora_down.weight": A, f"{p}.lora_up.weight": B, f"{p}.alpha": torch.tensor(float(r))}
    if style == "comfy":
        p = "diffusion_model." + src
        return {f"{p}.lora_A.weight": A, f"{p}.lora_B.weight": B, f"{p}.alpha": torch.tensor(float(r))}
    p = "transformer." + src
    return {f"{p}.lora_A.weight": A, f"{p}.lora_B.weight": B}


def _shapes(model):
    return {n: (m.in_features, m.out_features) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}


def test_qwen_image_linear_names():
    names = engine_linear_names(tiny_qwen_image())
    for n in ["transformer_blocks.0.attn.to_q", "transformer_blocks.1.attn.to_out.0",
              "transformer_blocks.0.img_mlp.net.0.proj", "transformer_blocks.0.txt_mod.1",
              "img_in", "txt_in", "proj_out", "norm_out.linear",
              "time_text_embed.timestep_embedder.linear_1"]:
        assert n in names


@pytest.mark.parametrize("style", ["kohya", "comfy", "peft"])
def test_qwen_image_fake_adapters_map_fully(style):
    m = tiny_qwen_image()
    km = make_qwen_image_keymap(engine_linear_names(m))
    shapes = _shapes(m)
    out, rep = normalize(_fake(shapes, style), km, assume_alpha_eq_rank=True)
    assert set(out) == set(shapes)
    assert rep["matched"] == rep["total"] == len(shapes) and rep["unmatched"] == []


def test_flux2_linear_names():
    names = engine_linear_names(tiny_flux2())
    for n in ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.add_k_proj",
              "transformer_blocks.0.ff.linear_in", "transformer_blocks.0.ff_context.linear_out",
              "single_transformer_blocks.1.attn.to_qkv_mlp_proj", "single_transformer_blocks.0.attn.to_out",
              "x_embedder", "context_embedder", "proj_out", "norm_out.linear",
              "time_guidance_embed.timestep_embedder.linear_1", "double_stream_modulation_img.linear"]:
        assert n in names


def _bfl_names(model):
    """Original (BFL / Comfy) source modules for the tiny FLUX.2: {name: (d_in, d_out)}."""
    sh = _shapes(model)
    out = {}
    for i in range(2):
        t = f"transformer_blocks.{i}"
        out[f"double_blocks.{i}.img_attn.qkv"] = (sh[f"{t}.attn.to_q"][0], 3 * sh[f"{t}.attn.to_q"][1])
        out[f"double_blocks.{i}.txt_attn.qkv"] = (sh[f"{t}.attn.add_q_proj"][0], 3 * sh[f"{t}.attn.add_q_proj"][1])
        for src, dst in [("img_attn.proj", "attn.to_out.0"), ("txt_attn.proj", "attn.to_add_out"),
                         ("img_mlp.0", "ff.linear_in"), ("img_mlp.2", "ff.linear_out"),
                         ("txt_mlp.0", "ff_context.linear_in"), ("txt_mlp.2", "ff_context.linear_out")]:
            out[f"double_blocks.{i}.{src}"] = sh[f"{t}.{dst}"]
        s = f"single_transformer_blocks.{i}.attn"
        out[f"single_blocks.{i}.linear1"] = sh[f"{s}.to_qkv_mlp_proj"]
        out[f"single_blocks.{i}.linear2"] = sh[f"{s}.to_out"]
    for src, dst in [("img_in", "x_embedder"), ("txt_in", "context_embedder"),
                     ("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1"),
                     ("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2"),
                     ("guidance_in.in_layer", "time_guidance_embed.guidance_embedder.linear_1"),
                     ("guidance_in.out_layer", "time_guidance_embed.guidance_embedder.linear_2"),
                     ("final_layer.linear", "proj_out"), ("final_layer.adaLN_modulation.1", "norm_out.linear"),
                     ("single_stream_modulation.lin", "single_stream_modulation.linear"),
                     ("double_stream_modulation_img.lin", "double_stream_modulation_img.linear"),
                     ("double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear")]:
        out[src] = sh[dst]
    return out


def _flux2_km(m):
    return make_flux2_keymap(engine_linear_names(m), engine_out_features(m))


@pytest.mark.parametrize("style", ["kohya", "comfy"])
def test_flux2_original_names_map_fully(style):
    m = tiny_flux2()
    src = _bfl_names(m)
    out, rep = normalize(_fake(src, style), _flux2_km(m), assume_alpha_eq_rank=True)
    assert rep["matched"] == rep["total"] == len(src) and rep["unmatched"] == []
    # every engine linear is covered exactly once (qkv splits fill to_q/k/v and the add_* projections)
    assert set(out) == set(_shapes(m))


@pytest.mark.parametrize("style", ["comfy", "peft"])
def test_flux2_diffusers_names_map_fully(style):
    m = tiny_flux2()
    shapes = _shapes(m)
    out, rep = normalize(_fake(shapes, style), _flux2_km(m), assume_alpha_eq_rank=True)
    assert set(out) == set(shapes) and rep["unmatched"] == []


def test_flux2_fused_qkv_split_reproduces_delta():
    m = tiny_flux2()
    d = m.transformer_blocks[0].attn.to_q.in_features
    g = torch.Generator().manual_seed(3)
    r = 4
    A, B = torch.randn(r, d, generator=g), torch.randn(3 * d, r, generator=g)
    p = "diffusion_model.double_blocks.0.img_attn.qkv"
    sd = {f"{p}.lora_A.weight": A, f"{p}.lora_B.weight": B, f"{p}.alpha": torch.tensor(2.0)}
    out, _ = normalize(sd, _flux2_km(m), rank_cap=r)
    x = torch.randn(5, d, generator=g)
    full = x @ (B @ A).T * (2.0 / r)
    for j, name in enumerate(["to_q", "to_k", "to_v"]):
        a, b = out[f"transformer_blocks.0.attn.{name}"]
        got = x @ a.float().T @ b.float().T
        torch.testing.assert_close(got, full[:, j * d:(j + 1) * d], atol=5e-2, rtol=2e-2)   # bf16 storage
