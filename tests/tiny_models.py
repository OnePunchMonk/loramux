"""Tiny diffusers transformers (CPU) for tests. Import only after pytest.importorskip("diffusers")."""
from __future__ import annotations

import torch


def tiny_qwen_image(seed: int = 0):
    from diffusers import QwenImageTransformer2DModel
    torch.manual_seed(seed)
    return QwenImageTransformer2DModel(
        patch_size=2, in_channels=16, out_channels=4, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=24, axes_dims_rope=(4, 6, 6)).eval()


def tiny_flux2(seed: int = 0):
    from diffusers import Flux2Transformer2DModel
    torch.manual_seed(seed)
    return Flux2Transformer2DModel(
        patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=16,
        axes_dims_rope=(4, 4, 4, 4), guidance_embeds=True).eval()
