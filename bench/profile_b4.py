"""Where does B4's fixed overhead come from? Splits one 40-layer B4 chain into
meta build, host launch time, and GPU kernel time, at several token counts.

  python -m bench.profile_b4 --tokens 1024,4096,16384 --out results/profile_b4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from bench.linear_chain import build_chains
from loramux import build_meta, lora_batch


def cuda_ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in ev)
    return t[len(t) // 2]


def host_ms(fn, iters=20):
    """Wall time to enqueue the work, no sync inside: ~CPU launch cost."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1024,4096,16384")
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--adapters", type=int, default=8)
    ap.add_argument("--out", default="results/profile_b4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = []
    with torch.no_grad():
        for T in [int(t) for t in a.tokens.split(",")]:
            x, fns = build_chains(layers=a.layers, dim=a.dim, rank=a.rank, adapters=a.adapters, tokens=T,
                                  device="cuda", dtype=torch.bfloat16, seed=0)
            rows = x.shape[0]
            slots = list(range(a.adapters))[:rows]
            scales = [1.0] * rows
            meta = build_meta(slots, scales, device="cuda", r_max=a.rank)
            b0, b4 = fns["B0"], fns["B4"]

            def b4_prebuilt():
                with lora_batch(meta):
                    return fns["_mux_chain"](x) if "_mux_chain" in fns else b4()

            r = dict(tokens=T, rows=rows, N=x.shape[1])
            r["B0_gpu_ms"] = cuda_ms(b0)
            r["B4_gpu_ms"] = cuda_ms(b4)
            r["meta_build_gpu_ms"] = cuda_ms(lambda: build_meta(slots, scales, device="cuda", r_max=a.rank))
            r["meta_build_host_ms"] = host_ms(lambda: build_meta(slots, scales, device="cuda", r_max=a.rank))
            r["B0_host_ms"] = host_ms(b0)
            r["B4_host_ms"] = host_ms(b4)
            # kernel table from the torch profiler
            from torch.profiler import ProfilerActivity, profile
            b4(); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                b4(); torch.cuda.synchronize()
            ka = prof.key_averages()
            ker = sorted([(e.key, e.count, getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e3)
                          for e in ka if getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) > 0],
                         key=lambda t: -t[2])[:8]
            r["top_kernels"] = [dict(name=k[:80], count=c, total_ms=round(t, 3)) for k, c, t in ker]
            out.append(r)
            print(json.dumps(r, indent=1), flush=True)
            del x, fns
    json.dump(out, open(os.path.join(a.out, "profile_b4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
