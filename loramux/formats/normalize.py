"""Turn a LoRA state dict into canonical {engine_key: (A[r,d_in], B[d_out,r])} with alpha/r folded into B.

Supported: PEFT/diffusers (lora_A/lora_B), Kohya (lora_down/lora_up + alpha).
Fails loudly when a tensor maps to no engine module.
"""
from __future__ import annotations
import re
import torch

_PEFT = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$")
_KOHYA = re.compile(r"^(?P<mod>.+)\.(?P<ab>lora_down|lora_up)\.weight$")
_PREFIXES = ("diffusion_model.", "transformer.", "base_model.model.", "lora_unet_")


def strip_prefix(k: str) -> str:
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def detect_format(sd) -> str:
    keys = list(sd)
    if any(".lora_down.weight" in k for k in keys):
        return "kohya"
    if any(".lora_A" in k for k in keys):
        return "peft"
    raise ValueError("unrecognized LoRA format")


def normalize(sd: dict, keymap, *, rank_cap: int | None = None):
    """keymap(module_name) -> engine key, or None if unknown. Returns (canonical, report)."""
    fmt = detect_format(sd)
    pat = _PEFT if fmt == "peft" else _KOHYA
    parts: dict[str, dict] = {}
    unmatched = []
    for k, v in sd.items():
        if k.endswith(".alpha"):
            parts.setdefault(strip_prefix(k[:-len(".alpha")]), {})["alpha"] = float(v)
            continue
        m = pat.match(k)
        if not m:
            unmatched.append(k)
            continue
        ab = "A" if m["ab"] in ("A", "lora_down") else "B"
        parts.setdefault(strip_prefix(m["mod"]), {})[ab] = v
    out, total = {}, 0
    for mod, d in parts.items():
        total += 1
        eng = keymap(mod)
        if eng is None or "A" not in d or "B" not in d:
            unmatched.append(mod)
            continue
        A, B = d["A"].float(), d["B"].float()
        r = A.shape[0]
        alpha = d.get("alpha", float(r))     # PEFT dumps usually carry alpha in config; default alpha=r
        B = B * (alpha / r)
        if rank_cap and r > rank_cap:
            raise ValueError(f"{mod}: rank {r} > cap {rank_cap}")
        out[eng] = (A.to(torch.bfloat16), B.to(torch.bfloat16))
    if unmatched:
        raise KeyError(f"{len(unmatched)} LoRA entries map to no engine module, e.g. {unmatched[:5]}")
    return out, dict(format=fmt, matched=len(out), total=total)


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
