"""Sweep expand-add configs/variants at the K2a shape to see how close to HBM bandwidth it can get.

  python -m bench.tune_expand --tokens 16384 --d_out 3072 --out results/tune_expand
Variants: base (production kernel body), evict (evict_first hints on Y), nloop (each program loops over N tiles
reusing the Z tile). Reports ms and achieved GB/s of Y read+write traffic, plus a torch copy ceiling.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _k(Y, Z, B, slot_ptr, scale_ptr, N, d_out, stride_ym, stride_zm, stride_bs, stride_bn, rank_ptr,
       RB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, EVICT: tl.constexpr, NLOOP: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    tpr = tl.cdiv(N, BM)
    row = pid // tpr
    r0 = row * N + (pid % tpr) * BM
    r1 = row * N + N
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    r = tl.load(rank_ptr + slot)
    slot64 = slot.to(tl.int64)
    scale = tl.load(scale_ptr + row)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    rows64 = rows.to(tl.int64)
    rc = tl.arange(0, RB)
    kmask = rc < r
    z = tl.load(Z + rows64[:, None] * stride_zm + rc[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
    n_iter = tl.cdiv(d_out, BN) if NLOOP else 1
    for j in range(0, n_iter):
        cols = (j if NLOOP else pn) * BN + tl.arange(0, BN)
        cmask = cols < d_out
        b = tl.load(B + slot64 * stride_bs + cols[None, :] * stride_bn + rc[:, None],
                    mask=cmask[None, :] & kmask[:, None], other=0.0)
        acc = tl.dot(z.to(b.dtype), b, out_dtype=tl.float32) * scale
        yp = Y + rows64[:, None] * stride_ym + cols[None, :]
        m = rmask[:, None] & cmask[None, :]
        if EVICT:
            y = tl.load(yp, mask=m, other=0.0, eviction_policy="evict_first")
        else:
            y = tl.load(yp, mask=m, other=0.0)
        tl.store(yp, (y.to(tl.float32) + acc).to(y.dtype), mask=m)


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in ev)
    return t[len(t) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--d_out", type=int, default=3072)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", default="results/tune_expand")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev, dt = "cuda", torch.bfloat16
    N = a.tokens // a.rows
    S = a.rows
    y = torch.randn(a.tokens, a.d_out, device=dev, dtype=dt)
    z = torch.randn(a.tokens, a.rank, device=dev, dtype=torch.float32)
    B = (torch.randn(S, a.d_out, a.rank, device=dev, dtype=dt) * 0.01)
    slot = torch.arange(S, dtype=torch.int32, device=dev)
    scale = torch.full((S,), 1e-3, dtype=torch.float32, device=dev)
    rank = torch.full((S,), a.rank, dtype=torch.int32, device=dev)
    traffic = 2 * y.numel() * y.element_size()      # Y read + write
    ceil_ms = bench(lambda: y.copy_(y.clone()))     # torch: read + write + alloc; rough ceiling reference
    buf = torch.empty_like(y)
    ceil_ms = bench(lambda: buf.copy_(y))
    ceil = 2 * y.numel() * y.element_size() / ceil_ms / 1e6
    res = [dict(variant="torch_copy_ceiling", ms=ceil_ms, gbps=ceil)]
    print(f"copy ceiling {ceil:.0f} GB/s ({ceil_ms:.3f} ms)", flush=True)
    for variant, BM, BN, w, st in itertools.product(("base", "evict", "nloop"), (32, 64, 128, 256), (64, 128, 256),
                                                  (4, 8), (2, 3)):
        if BM * BN > 128 * 256:
            continue
        evict, nloop = variant == "evict", variant == "nloop"
        tpr = triton.cdiv(N, BM)
        grid = (a.rows * tpr, 1 if nloop else triton.cdiv(a.d_out, BN))
        run = lambda: _k[grid](y, z, B, slot, scale, N, a.d_out, y.stride(0), z.stride(0), B.stride(0), B.stride(1),
                               rank, RB=a.rank, BM=BM, BN=BN, EVICT=evict, NLOOP=nloop, num_warps=w, num_stages=st)
        try:
            ms = bench(run)
        except Exception as e:  # noqa: BLE001
            print(variant, BM, BN, w, st, "fail", type(e).__name__)
            continue
        r = dict(variant=variant, BM=BM, BN=BN, warps=w, stages=st, ms=ms, gbps=traffic / ms / 1e6)
        res.append(r)
    ok = sorted([r for r in res if "BM" in r], key=lambda r: r["ms"])
    for r in ok[:12]:
        print(json.dumps(r))
    best_by_variant = {}
    for r in ok:
        best_by_variant.setdefault(r["variant"], r)
    print("best per variant:", json.dumps(best_by_variant))
    json.dump(res, open(os.path.join(a.out, f"tune_expand_{a.tokens}_{a.d_out}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
