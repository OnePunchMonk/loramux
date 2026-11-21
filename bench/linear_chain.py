"""K2a linear-chain microbench: LoRA overhead on a chain of L square linears (d -> d).

Baselines (one CSV row per baseline and token count):
  B0  no LoRA.
  B1  merged delta W (single adapter: every row uses adapter 0, merged into the weights ahead of time).
  B2  REIMPLEMENTATION of the vLLM-Omni diffusion LoRA wrapper pattern (vllm_omni/diffusion/lora/layers/
      base_linear.py, `apply()` fallback path): per adapter, `buffer = x @ A.T`, a FULL-WIDTH delta
      `buffer @ B.T`, then slice-assign `y_slice[:] = y_slice + delta`. The wrapper serves one adapter per
      forward, so with k adapters this loops over the k row groups. Not vLLM-Omni code; labelled in `note`.
  B3  loramux torch reference (`lora_apply_ref_`, Python loop over segments).
  B4  loramux custom op through MuxLinear (Triton on CUDA; on CPU the op falls back to the reference).
  B5  vLLM's `lora_shrink` / `lora_expand` Triton ops if importable (CUDA only), else a row with
      status=unavailable (or status=error:<reason> if the installed vLLM's API differs from the one targeted).

Input layout: x is [rows, N, d] with rows == number of distinct adapters (row i uses adapter i) and
N = tokens // rows, so the actual token count is rows * N (reported as `tokens`; `tokens_requested` is the CLI value).

CSV columns: baseline, status, note, tokens_requested, tokens, rows, layers, dim, rank, adapters, dtype, device,
median_ms, p90_ms, mean_ms, n, overhead_pct (vs B0 at the same tokens). The env header comes from collect_env().
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from torch import nn

from bench.common import collect_env, interleave, seed_all, summarize, write_csv
from bench.synth_adapters import make_adapter
from loramux import AdapterPool, build_meta, lora_batch
from loramux.layers import MuxLinear
from loramux.ops.torch_ref import lora_apply_ref_

BASELINES = ("B0", "B1", "B2", "B3", "B4", "B5")
NOTES = {
    "B0": "no LoRA",
    "B1": "merged delta W, single adapter (all rows use adapter 0)",
    "B2": "reimplementation of the vLLM-Omni wrapper pattern (full-width delta + slice-assign), not vLLM-Omni code",
    "B3": "loramux torch_ref (python loop over segments)",
    "B4": "loramux custom op via MuxLinear",
    "B5": "vLLM lora_shrink/lora_expand",
}


def _layer_keys(layers: int) -> list[str]:
    return [f"chain.{i}" for i in range(layers)]


def _make_base(layers: int, dim: int, device, dtype) -> list[nn.Linear]:
    g = torch.Generator().manual_seed(123)
    out = []
    for _ in range(layers):
        lin = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(dim, dim, generator=g) / dim ** 0.5)
        out.append(lin.to(device=device, dtype=dtype).requires_grad_(False))
    return out


def _make_pool(keys, dim, rank, adapters, device, dtype, seed):
    shapes = {k: (dim, dim) for k in keys}
    pool = AdapterPool(shapes, slots=adapters, r_max=rank, device=device, dtype=dtype)
    for a in range(adapters):
        pool.register(f"ad{a}", make_adapter(shapes, rank, "all", seed=seed + a))
        pool.ensure_resident(f"ad{a}")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return pool


def _vllm_ops():
    """(lora_shrink, lora_expand, LoRAKernelMeta) or None when vLLM's LoRA Triton ops can't be imported."""
    try:
        from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
        return lora_shrink, lora_expand, LoRAKernelMeta
    except Exception:  # noqa: BLE001 - any import problem means 'unavailable'
        return None


def build_chains(*, layers: int, dim: int, rank: int, adapters: int, tokens: int, device: str,
                 dtype: torch.dtype, seed: int = 0) -> tuple[torch.Tensor, dict[str, Callable | str]]:
    """Return (x, {baseline: callable | 'unavailable'|'error:...'}). Every callable runs the whole chain on x."""
    keys = _layer_keys(layers)
    base = _make_base(layers, dim, device, dtype)
    pool = _make_pool(keys, dim, rank, adapters, device, dtype, seed)
    rows = adapters
    N = max(1, tokens // rows)
    g = torch.Generator().manual_seed(seed + 99)
    x = torch.randn(rows, N, dim, generator=g).to(device=device, dtype=dtype)
    slots = [pool.slot_of[f"ad{a}"] for a in range(adapters)]
    scales = [1.0] * adapters
    bufs = {k: pool.buffers(k) for k in keys}  # (A [S,R,d], B [S,d,R], rank [S])
    T = rows * N

    def b0():
        h = x
        for lin in base:
            h = lin(h)
        return h

    # B1: merged weights of adapter 0 (its exact tensors, not the rank-padded slot)
    ad0 = pool.canonical("ad0")
    merged = []
    for lin, k in zip(base, keys):
        a, b = ad0[k]
        m = nn.Linear(dim, dim, bias=False).to(device=device, dtype=dtype).requires_grad_(False)
        with torch.no_grad():
            m.weight.copy_(lin.weight + (b.float() @ a.float()).to(device=device, dtype=dtype))
        merged.append(m)

    def b1():
        h = x
        for lin in merged:
            h = lin(h)
        return h

    def b2():
        h = x
        for lin, k in zip(base, keys):
            A, B, _ = bufs[k]
            y = lin(h)
            for i, s in enumerate(slots):
                y_slice = y[i]
                r = pool.layer_rank_host[k][s]
                buffer = h[i] @ A[s, :r].t()
                delta = buffer @ B[s, :, :r].t()    # full-width delta
                y_slice[:] = y_slice + delta * scales[i]   # slice-assign copy
            h = y
        return h

    meta3 = build_meta(slots, scales, device=device, r_max=rank)
    v = meta3.view_for(x)

    def b3():
        h = x
        for lin, k in zip(base, keys):
            A, B, rk = bufs[k]
            y = lin(h).view(-1, dim)
            lora_apply_ref_(y, h.reshape(-1, dim), A, B, rk, v.row_slot, v.row_scale, v.seg_start, v.seg_len)
            h = y.view(rows, N, dim)
        return h

    mux = [MuxLinear(lin, k, pool) for lin, k in zip(base, keys)]

    def b4():
        meta = build_meta(slots, scales, device=device, r_max=rank)
        with lora_batch(meta):
            h = x
            for m in mux:
                h = m(h)
        return h

    mux_f = [MuxLinear(lin, k, pool, fused=True) for lin, k in zip(base, keys)]

    def b4f():
        meta = build_meta(slots, scales, device=device, r_max=rank)
        with lora_batch(meta):
            h = x
            for m in mux_f:
                h = m(h)
        return h

    fns: dict[str, Callable | str] = {"B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B4f": b4f, "_mux": mux, "_mux_f": mux_f}
    ops = _vllm_ops() if str(device).startswith("cuda") else None
    if ops is None:
        fns["B5"] = "unavailable"
    else:
        lora_shrink, lora_expand, LoRAKernelMeta = ops
        try:
            kmeta = LoRAKernelMeta.make(max_loras=adapters, max_num_tokens=T, device=torch.device(device))
            mapping = torch.arange(adapters, device=device).repeat_interleave(N)
            kmeta.prepare_tensors(mapping)
            z = torch.empty(1, T, rank, dtype=torch.float32, device=device)

            def b5():
                h = x.reshape(T, dim)
                for lin, k in zip(base, keys):
                    A, B, _ = bufs[k]
                    y = lin(h)
                    lora_shrink(h, [A], z, *kmeta.meta_args(T), 1.0)
                    lora_expand(z, [B], y, *kmeta.meta_args(T), offset_start=0, add_inputs=True)
                    h = y
                return h

            fns["B5"] = b5
        except Exception as e:  # noqa: BLE001 - vLLM's API moves between releases
            fns["B5"] = f"error:{type(e).__name__}: {e}"
    return x, fns


def run(*, tokens: list[int], rank: int, layers: int, dim: int, adapters: int, device: str | None = None,
        dtype: torch.dtype | None = None, rounds: int = 3, warmup: int = 3, iters: int = 20,
        baselines: tuple[str, ...] = BASELINES, seed: int = 0) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    seed_all(seed)
    rows: list[dict] = []
    with torch.no_grad():
        for T_req in tokens:
            x, fns = build_chains(layers=layers, dim=dim, rank=rank, adapters=adapters, tokens=T_req,
                                  device=device, dtype=dtype, seed=seed)
            common = {"tokens_requested": T_req, "tokens": x.shape[0] * x.shape[1], "rows": x.shape[0], "layers": layers,
                          "dim": dim, "rank": rank, "adapters": adapters, "dtype": str(dtype).replace("torch.", ""),
                          "device": device}
            if callable(fns.get("B5")):            # vLLM's API moves between releases: probe once, don't crash the run
                try:
                    fns["B5"]()
                except Exception as e:  # noqa: BLE001
                    fns["B5"] = f"error:{type(e).__name__}: {e}"
            live = {b: f for b, f in fns.items() if callable(f) and b in baselines}
            times = interleave(live, rounds=rounds, warmup=warmup, iters=iters)
            base_med = summarize(times["B0"])["median"] if "B0" in times else None
            for b in baselines:
                row = dict(baseline=b, status="ok", note=NOTES[b], **common)
                if b in times:
                    s = summarize(times[b])
                    row.update(median_ms=s["median"], p90_ms=s["p90"], mean_ms=s["mean"], n=s["n"])
                    row["overhead_pct"] = ((s["median"] - base_med) / base_med * 100.0) if base_med else None
                else:
                    row["status"] = fns.get(b, "unavailable") if not callable(fns.get(b)) else "skipped"
                    row.update(median_ms=None, p90_ms=None, mean_ms=None, n=0, overhead_pct=None)
                rows.append(row)
            del x, fns
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.linear_chain")
    ap.add_argument("--tokens", default="1024,4096,16384")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--adapters", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="results/k2a.csv")
    a = ap.parse_args(argv)
    rows = run(tokens=[int(t) for t in a.tokens.split(",")], rank=a.rank, layers=a.layers, dim=a.dim,
               adapters=a.adapters, rounds=a.rounds, warmup=a.warmup, iters=a.iters)
    write_csv(a.out, rows, collect_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
