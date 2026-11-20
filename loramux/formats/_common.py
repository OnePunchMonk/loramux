"""Shared pieces of the LoRA readers: unsupported-variant detection, key splitting, prefix stripping."""
from __future__ import annotations

import re

# One entry per low-rank factor: <module>.<token>[.<adapter name>].weight
LORA_KEY = re.compile(r"^(?P<mod>.+)\.(?P<tok>lora_A|lora_B|lora_down|lora_up)(?:\.[^.]+)?\.weight$")
A_TOKENS = ("lora_A", "lora_down")

UNSUPPORTED = (
    ("lokr_", "LoKr"),
    ("hada_", "LoHa"),
    ("dora_scale", "DoRA"),
    ("lora_magnitude_vector", "DoRA"),
)

PREFIXES = ("base_model.model.", "diffusion_model.", "transformer.", "lora_unet_", "lora_transformer_")


def reject_unsupported(keys) -> None:
    """LoKr / LoHa / DoRA are not low-rank-ΔW = B@A adapters the pool can serve; refuse them by name."""
    for k in keys:
        for needle, kind in UNSUPPORTED:
            if needle in k:
                raise ValueError(
                    f"unsupported adapter type {kind} (key {k!r}); loramux serves plain LoRA "
                    f"(lora_A/lora_B or lora_down/lora_up) only. Merge or convert it to plain LoRA first.")


def strip_prefix(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if k.startswith(p):
                k, changed = k[len(p):], True
    return k


def split_entries(sd: dict, accept) -> tuple[dict[str, dict], list[str]]:
    """Group a state dict into {module: {"A", "B", "alpha"}} for keys where accept(key) is true.

    Module names are prefix-stripped. Returns (parts, unmatched_keys)."""
    parts: dict[str, dict] = {}
    unmatched: list[str] = []
    for k, v in sd.items():
        if k.endswith(".alpha"):
            parts.setdefault(strip_prefix(k[:-len(".alpha")]), {})["alpha"] = float(v)
            continue
        m = LORA_KEY.match(k)
        if not m or not accept(k):
            unmatched.append(k)
            continue
        parts.setdefault(strip_prefix(m["mod"]), {})["A" if m["tok"] in A_TOKENS else "B"] = v
    return parts, unmatched
