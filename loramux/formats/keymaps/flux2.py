"""Key map for diffusers `Flux2Transformer2DModel` (diffusers 0.43).

Sources, all in `diffusers.loaders.lora_conversion_utils`:
- `_convert_non_diffusers_flux2_lora_to_diffusers`: original/BFL/Comfy names (`diffusion_model.` prefix,
  `double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`, ...). Also accepts files that already use the expanded
  diffusers block names, which resolve by exact match here.
- `_convert_kohya_flux2_lora_to_diffusers`: Kohya `lora_unet_double_blocks_N_img_attn_qkv` etc. (underscore form of
  the same original names, resolved by the generic keymap).

Differences from the diffusers converter: alpha is handled by `normalize` (the Comfy converter above ignores it),
and the fused qkv is split by output rows with A shared. That is exact for dense and block-sparse B alike.

Original -> engine (per block N):
  double_blocks.N.img_attn.qkv   -> transformer_blocks.N.attn.{to_q,to_k,to_v}          (split in thirds)
  double_blocks.N.txt_attn.qkv   -> transformer_blocks.N.attn.{add_q_proj,add_k_proj,add_v_proj} (split)
  double_blocks.N.img_attn.proj  -> transformer_blocks.N.attn.to_out.0
  double_blocks.N.txt_attn.proj  -> transformer_blocks.N.attn.to_add_out
  double_blocks.N.img_mlp.{0,2}  -> transformer_blocks.N.ff.{linear_in,linear_out}
  double_blocks.N.txt_mlp.{0,2}  -> transformer_blocks.N.ff_context.{linear_in,linear_out}
  single_blocks.N.linear1 / linear2 -> single_transformer_blocks.N.attn.{to_qkv_mlp_proj,to_out}
  plus the `extra_mappings` table (img_in, txt_in, time_in.*, guidance_in.*, final_layer.*, *_modulation.*).
`single_blocks.N.linear1` is NOT split: the engine keeps q, k, v and the MLP input fused in to_qkv_mlp_proj.
"""
from __future__ import annotations

import re

from .generic import KeyMap, make_keymap

_EXTRA = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
}
_DOUBLE = {
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.proj": "attn.to_add_out",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}
_QKV = {
    "img_attn.qkv": ("to_q", "to_k", "to_v"),
    "txt_attn.qkv": ("add_q_proj", "add_k_proj", "add_v_proj"),
}
_SINGLE = {"linear1": "attn.to_qkv_mlp_proj", "linear2": "attn.to_out"}


def make_flux2_keymap(engine_linear_names: list[str], out_features: dict[str, int]) -> KeyMap:
    """out_features: {engine name: out_features} (keymaps.engine_out_features(model)); used to split fused qkv."""
    names = set(engine_linear_names)
    n_double = 1 + max((int(m[1]) for n in names if (m := re.match(r"transformer_blocks\.(\d+)\.", n))), default=-1)
    n_single = 1 + max((int(m[1]) for n in names if (m := re.match(r"single_transformer_blocks\.(\d+)\.", n))),
                       default=-1)
    aliases: dict = {}
    for src, dst in _EXTRA.items():
        if dst in names:
            aliases[src] = dst
    for i in range(n_double):
        for src, dst in _DOUBLE.items():
            if f"transformer_blocks.{i}.{dst}" in names:
                aliases[f"double_blocks.{i}.{src}"] = f"transformer_blocks.{i}.{dst}"
        for src, parts in _QKV.items():
            targets = [f"transformer_blocks.{i}.attn.{p}" for p in parts]
            if all(t in names for t in targets):
                dims = [out_features[t] for t in targets]
                offs = [sum(dims[:j]) for j in range(len(dims))]
                aliases[f"double_blocks.{i}.{src}"] = [
                    (t, slice(o, o + d)) for t, o, d in zip(targets, offs, dims, strict=True)]
    for i in range(n_single):
        for src, dst in _SINGLE.items():
            if f"single_transformer_blocks.{i}.{dst}" in names:
                aliases[f"single_blocks.{i}.{src}"] = f"single_transformer_blocks.{i}.{dst}"
    return make_keymap(engine_linear_names, aliases)
