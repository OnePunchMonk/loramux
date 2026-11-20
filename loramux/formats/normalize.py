"""Turn a LoRA state dict into canonical {engine_key: (A[r,d_in], B[d_out,r])} with alpha/r folded into B.

Supported: PEFT/diffusers (lora_A/lora_B, alpha from adapter_config.json), Kohya (lora_down/lora_up + alpha),
ComfyUI (`diffusion_model.` prefix, either factor naming). LoKr / LoHa / DoRA are rejected.
Fails loudly when a tensor maps to no engine module, or when alpha cannot be determined.

A keymap(module_name) returns an engine key, None, or a split [(engine_key, out_slice), ...] for a module that is
fused in the file but separate in the engine (e.g. FLUX qkv): A is shared, B is sliced by output rows.
"""
from __future__ import annotations

import itertools
import math

import torch

from . import comfy, kohya, peft
from ._common import reject_unsupported, strip_prefix  # noqa: F401


def detect_format(sd) -> str:
    keys = list(sd)
    reject_unsupported(keys)
    if kohya.matches(sd):
        return "kohya"
    if comfy.matches(sd):
        return "comfy"
    if any(".lora_A" in k for k in keys):
        return "peft"
    raise ValueError("unrecognized LoRA format")


def _alpha_and_scale(fmt, mod, d, r, config, assume_alpha_eq_rank):
    cfg_alpha, cfg_rank = peft.config_alpha_and_rank(config, mod)
    if cfg_rank is not None and int(cfg_rank) != r:
        raise ValueError(f"{mod}: rank_pattern says r={cfg_rank} but the weights have rank {r}")
    alpha = d.get("alpha", cfg_alpha)
    if alpha is None:
        if not assume_alpha_eq_rank:
            raise ValueError(
                f"{mod}: no alpha found ({fmt} adapter without an `alpha` tensor and no lora_alpha in "
                f"adapter_config.json). Pass config=, or assume_alpha_eq_rank=True to use alpha=r (scale 1).")
        alpha = float(r)
    rslora = bool(config and config.get("use_rslora"))
    return alpha / (math.sqrt(r) if rslora else r)


def normalize(sd: dict, keymap, *, config: dict | None = None, assume_alpha_eq_rank: bool = False,
              rank_cap: int | None = None, strict: bool = True):
    """Returns (canonical, report). report: format, matched, total, unmatched (list), ranks {engine_key: r}.

    strict=True raises KeyError if anything maps to no engine module; strict=False lists it in the report."""
    fmt = detect_format(sd)
    peft.check_config(config)
    parts, unmatched = {"kohya": kohya, "comfy": comfy, "peft": peft}[fmt].parse(sd)
    out: dict = {}
    ranks: dict = {}
    total = matched = 0
    for mod, d in parts.items():
        total += 1
        tgt = keymap(mod) if "A" in d and "B" in d else None
        if tgt is None:
            unmatched.append(mod)
            continue
        A, B = d["A"].float(), d["B"].float()
        r = A.shape[0]
        if rank_cap and r > rank_cap:
            raise ValueError(f"{mod}: rank {r} > cap {rank_cap}")
        B = B * _alpha_and_scale(fmt, mod, d, r, config, assume_alpha_eq_rank)
        pieces = [(tgt, slice(None))] if isinstance(tgt, str) else list(tgt)
        if not isinstance(tgt, str):
            covered = sum(sl.stop - sl.start for _, sl in pieces)
            if covered != B.shape[0] or pieces[0][1].start != 0 or any(
                    a[1].stop != b[1].start for a, b in itertools.pairwise(pieces)):
                raise ValueError(f"{mod}: split {[(e, (s.start, s.stop)) for e, s in pieces]} does not tile the "
                                 f"{B.shape[0]} output rows of lora_B")
        for eng, sl in pieces:
            if eng in out:
                raise ValueError(f"{mod}: engine module {eng} is already filled by another entry")
            out[eng] = (A.to(torch.bfloat16), B[sl].to(torch.bfloat16))
            ranks[eng] = r
        matched += 1
    if unmatched and strict:
        raise KeyError(f"{len(unmatched)} LoRA entries map to no engine module, e.g. {unmatched[:5]}")
    return out, {"format": fmt, "matched": matched, "total": total, "unmatched": unmatched, "ranks": ranks}


def compose_fused(parts):
    """parts: list of (A[r_i,d_in], B[d_i,r_i]) for q,k,v in output order -> fused (A, B).

    A_fused = concat along rank; B_fused = block-diagonal. Rank becomes sum of r_i.
    """
    A = torch.cat([a for a, _ in parts], dim=0)
    d_total = sum(b.shape[0] for _, b in parts)
    r_total = sum(a.shape[0] for a, _ in parts)
    B = torch.zeros(d_total, r_total, dtype=parts[0][1].dtype)
    ro = co = 0
    for a, b in parts:
        B[co:co + b.shape[0], ro:ro + a.shape[0]] = b
        ro += a.shape[0]
        co += b.shape[0]
    return A, B


def compose_multi(items):
    """items: list of (A, B, strength). One composite adapter: concat A along rank, B=[s1*B1 | s2*B2]."""
    A = torch.cat([a for a, _, _ in items], dim=0)
    B = torch.cat([b * s for _, b, s in items], dim=1)
    return A, B
