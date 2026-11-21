"""Key map for diffusers `QwenImageTransformer2DModel` (diffusers 0.43).

Source: `diffusers.loaders.lora_conversion_utils._convert_non_diffusers_qwen_lora_to_diffusers`. For Qwen-Image
the original / ComfyUI module names ARE the diffusers names (`transformer_blocks.N.attn.to_q`, `img_mlp.net.0.proj`,
`img_mod.1`, `norm_out.linear`, ...), so ComfyUI (`diffusion_model.` prefix) and PEFT keys resolve by exact match.
Kohya (`lora_unet_` + flattened names) is what that function handles with its protected-n-gram heuristic and its
hard-coded `top_level_modules` table (img_in, txt_in, proj_out, norm_out_linear, time_text_embed_timestep_embedder_*);
here the flattened names are resolved against the engine's real module names instead, which covers the same
top-level modules and block modules without a heuristic.

Real names (tiny config, block 0): transformer_blocks.0.attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,
to_out.0,to_add_out}, transformer_blocks.0.{img_mlp,txt_mlp}.net.0.proj / .net.2, transformer_blocks.0.{img_mod,
txt_mod}.1, img_in, txt_in, proj_out, norm_out.linear, time_text_embed.timestep_embedder.linear_{1,2}.
"""
from __future__ import annotations

from .generic import KeyMap, make_keymap


def make_qwen_image_keymap(engine_linear_names: list[str]) -> KeyMap:
    return make_keymap(engine_linear_names)
