"""Per-step batch metadata. Everything is built from host lists, so no GPU sync."""
from __future__ import annotations

import torch

from .context import BatchLoRAMeta, SegView, TileMeta


def expand_cfg(slots, scales, layout: str, cfg: bool):
    """Per-request (slot, scale) -> per-row lists after CFG duplication.

    layout 'concat':      [cond_0..cond_n, uncond_0..uncond_n]
    layout 'interleaved': [cond_0, uncond_0, cond_1, uncond_1, ...]
    Both halves of a request use the same adapter.
    """
    if not cfg:
        return list(slots), list(scales)
    if layout == "concat":
        return list(slots) * 2, list(scales) * 2
    if layout == "interleaved":
        return [s for s in slots for _ in (0, 1)], [c for c in scales for _ in (0, 1)]
    raise ValueError(f"unknown CFG layout {layout!r}")


def build_tiles(seg_start, seg_len, bm: int, device) -> TileMeta:
    seg, r0, r1 = [], [], []
    for i, (s, n) in enumerate(zip(seg_start, seg_len)):
        for t in range(0, n, bm):
            seg.append(i)
            r0.append(s + t)
            r1.append(s + min(t + bm, n))
    mk = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    return TileMeta(mk(seg), mk(r0), mk(r1), bm)


def _dev_tensor(vals, dtype, device) -> torch.Tensor:
    """Device tensor from a host list. On CUDA: pinned buffer + non_blocking copy, no sync."""
    t = torch.tensor(vals, dtype=dtype)
    if str(device).startswith("cuda"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def make_view(meta: BatchLoRAMeta, rows: int | None, N: int | None) -> SegView:
    """Build the SegView for (rows, N). rows=None means packed (varlen) using meta.seg_len."""
    n = meta.n_req
    if rows is None:
        slots, scales = list(meta.req_slot), list(meta.req_scale)
        lens = list(meta.seg_len)
        uniform = False
    else:
        if rows == n:
            slots, scales = list(meta.req_slot), list(meta.req_scale)
        elif rows == 2 * n:
            slots, scales = expand_cfg(meta.req_slot, meta.req_scale, meta.cfg_layout, True)
        else:
            raise ValueError(
                f"input has {rows} rows but the batch has {n} requests; expected rows == {n} "
                f"or rows == {2 * n} (CFG, layout {meta.cfg_layout!r})")
        lens = [N] * rows
        uniform = True
    starts, acc = [], 0
    for l in lens:
        starts.append(acc)
        acc += l
    dev = meta.device
    return SegView(
        rows=len(slots), N=N, uniform=uniform,
        row_slot=_dev_tensor(slots, torch.int32, dev), row_scale=_dev_tensor(scales, torch.float32, dev),
        seg_start=_dev_tensor(starts, torch.int32, dev), seg_len=_dev_tensor(lens, torch.int32, dev),
        slot_host=slots, scale_host=scales, start_host=starts, len_host=lens, bm=meta.bm, device=dev)


def build_meta(req_slots, req_scales, *, cfg_layout="concat", seg_len=None, rank_of_slot=None,
               r_max=None, device="cpu", bm=64, per_row_2d=False) -> BatchLoRAMeta:
    """req_slots[i] = pool slot of request i (-1 = base). Everything is decided on the host.

    rank_of_slot(slot) -> max rank over layers; used for r_bucket = next_pow2(max active rank)
    clamped to [16, r_max]. Without it, r_bucket = r_max or 64.
    per_row_2d: treat 2-D inputs [rows, d] as one token per row instead of raising.
    """
    slots, scales = list(req_slots), [float(s) for s in req_scales]
    if len(slots) != len(scales):
        raise ValueError(f"req_slots ({len(slots)}) and req_scales ({len(scales)}) differ in length")
    if cfg_layout not in ("concat", "interleaved"):
        raise ValueError(f"unknown CFG layout {cfg_layout!r}")
    if seg_len is not None:
        seg_len = [int(l) for l in seg_len]
        if len(seg_len) != len(slots):
            raise ValueError(f"seg_len has {len(seg_len)} entries for {len(slots)} requests")
    if rank_of_slot is None:
        r_bucket = r_max or 64
    else:
        active = [int(rank_of_slot(s)) for s in set(slots) if s >= 0]
        r_bucket = max(16, _next_pow2(max(active, default=0)))
        if r_max is not None:
            r_bucket = min(r_bucket, r_max)
    return BatchLoRAMeta(req_slot=slots, req_scale=scales, cfg_layout=cfg_layout, device=device,
                         any_lora=any(s >= 0 for s in slots), r_bucket=r_bucket, seg_len=seg_len, bm=bm,
                         per_row_2d=per_row_2d)
