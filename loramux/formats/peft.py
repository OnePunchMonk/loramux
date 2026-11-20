"""PEFT / diffusers-style LoRA (`lora_A` / `lora_B`) and the loader for files and PEFT directories.

PEFT does not store alpha in the weights; it lives in adapter_config.json (`lora_alpha`, plus per-module
`alpha_pattern` and `rank_pattern`, `use_rslora`). Pattern semantics follow peft: a pattern key matches a
module when the module name equals it or ends with `.<pattern>` (the pattern is a regex).
"""
from __future__ import annotations

import json
import os
import re

from safetensors.torch import load_file

from ._common import split_entries


def load_lora(path) -> tuple[dict, dict | None]:
    """A `.safetensors` file (config None), or a directory with adapter_model.safetensors [+ adapter_config.json]."""
    path = os.fspath(path)
    if os.path.isdir(path):
        weights = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"{path}: no adapter_model.safetensors")
        cfg_path = os.path.join(path, "adapter_config.json")
        config = None
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                config = json.load(f)
        return load_file(weights), config
    return load_file(path), None


def _pattern_lookup(pattern: dict | None, module: str):
    for pat, val in (pattern or {}).items():
        if re.fullmatch(rf"(.*\.)?({pat})", module):
            return val
    return None


def config_alpha_and_rank(config: dict | None, module: str):
    """(alpha | None, rank | None) for a module from an adapter config. rank is only set via rank_pattern."""
    if not config:
        return None, None
    alpha = _pattern_lookup(config.get("alpha_pattern"), module)
    if alpha is None:
        alpha = config.get("lora_alpha")
    return (None if alpha is None else float(alpha)), _pattern_lookup(config.get("rank_pattern"), module)


def check_config(config: dict | None) -> None:
    if config and (config.get("use_dora") or config.get("peft_type", "LORA") not in ("LORA", None)):
        raise ValueError(f"unsupported adapter config (peft_type={config.get('peft_type')!r}, "
                         f"use_dora={config.get('use_dora')!r}); only plain LoRA is supported")


def parse(sd):
    return split_entries(sd, lambda k: True)
