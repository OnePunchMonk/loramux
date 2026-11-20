"""Kohya-style LoRA: `lora_unet_<module_with_underscores>.lora_down.weight / .lora_up.weight / .alpha`.

Underscores in the module name are ambiguous (`to_q` vs `.`), so the keymap resolves them against the engine's
module names (see keymaps.generic.make_keymap); nothing is guessed here. A dotted module name after the prefix
(as some trainers write) passes through unchanged.
"""
from __future__ import annotations

from ._common import split_entries


def matches(sd) -> bool:
    return any(k.startswith(("lora_unet_", "lora_transformer_")) for k in sd) or (
        any((".lora_down." in k or ".lora.down." in k) for k in sd) and not any(k.startswith("diffusion_model.") for k in sd))


def parse(sd):
    return split_entries(sd, lambda k: True)
