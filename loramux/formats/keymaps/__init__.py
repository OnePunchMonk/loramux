"""Key maps from adapter-file module names to engine (diffusers 0.43) module names."""
from __future__ import annotations

from torch import nn

from .flux2 import make_flux2_keymap
from .generic import make_keymap
from .qwen_image import make_qwen_image_keymap


def engine_linear_names(model: nn.Module) -> list[str]:
    """Qualified names of every nn.Linear in the model, in module order."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def engine_out_features(model: nn.Module) -> dict[str, int]:
    """{linear name: out_features}; needed by keymaps that split a fused projection by output rows."""
    return {n: m.out_features for n, m in model.named_modules() if isinstance(m, nn.Linear)}


__all__ = ["engine_linear_names", "engine_out_features", "make_flux2_keymap", "make_keymap",
           "make_qwen_image_keymap"]
