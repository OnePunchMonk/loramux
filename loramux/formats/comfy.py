"""ComfyUI-style LoRA: `diffusion_model.<module>.{lora_A,lora_B | lora_down,lora_up}.weight` plus optional
`diffusion_model.<module>.alpha`. Module names are the model's own (original or diffusers) names; the keymap
renames them to engine modules."""
from __future__ import annotations

from ._common import split_entries


def matches(sd) -> bool:
    return any(k.startswith("diffusion_model.") for k in sd)


def parse(sd):
    return split_entries(sd, lambda k: True)
