"""Lossy-gate quality bench (HANDOFF §11.2, bars P2/P3): the CSV `bench.latency_bars --lossy` reads.

Variants are measured against the exact bf16 loramux path on the same prompts and seeds:
- `fp8`: A and B quantized to e4m3 with a per-row scale. The images use the dequantized weights in a bf16 pool,
  which is what the `*_fp8w` kernels compute up to rounding; bytes are the FP8 pool's (1 B/elem + 4 B/row).
- `svd`: every layer rank-truncated at `--energy` (loramux.compress.svd_truncate); bytes are the bf16 blob's.

CSV columns: variant, adapter, prompt_idx, psnr, lpips, bytes_exact, bytes_variant, slots_per_gb_exact,
slots_per_gb_variant. The byte/slot columns are CPU-only and tested; the image columns need a GPU, the model
weights and real adapters (same flags as bench.k1_correctness).
"""
from __future__ import annotations

import argparse

import torch

from bench.common import collect_env, write_csv
from loramux.compress import dequantize_fp8_rows, quantize_fp8_rows, svd_truncate

GB = 1 << 30
VARIANTS = ("fp8", "svd")


def _bf16_bytes(canon) -> int:
    return sum((a.numel() + b.numel()) * 2 for a, b in canon.values())


def apply_variant(canon: dict, variant: str, energy: float = 0.995) -> tuple[dict, int]:
    """(canonical adapter the variant serves as bf16 tensors, bytes the variant stores per slot)."""
    if variant == "fp8":
        out, nbytes = {}, 0
        for k, (a, b) in canon.items():
            (qa, sa), (qb, sb) = quantize_fp8_rows(a), quantize_fp8_rows(b)
            out[k] = (dequantize_fp8_rows(qa, sa), dequantize_fp8_rows(qb, sb))
            nbytes += qa.numel() + qb.numel() + 4 * (qa.shape[0] + qb.shape[0])
        return out, nbytes
    if variant == "svd":
        out = {k: svd_truncate(a, b, energy)[:2] for k, (a, b) in canon.items()}
        return out, _bf16_bytes(out)
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def size_row(canon: dict, variant: str, energy: float = 0.995) -> dict:
    exact = _bf16_bytes(canon)
    _, nb = apply_variant(canon, variant, energy)
    return {"variant": variant, "bytes_exact": exact, "bytes_variant": nb,
            "slots_per_gb_exact": GB / exact, "slots_per_gb_variant": GB / nb}


def run_lossy(a) -> list[dict]:  # pragma: no cover - needs a GPU, weights and adapters
    from bench.k1_correctness import _gen, _keymap, load_pipeline, lpips_distance, psnr
    from loramux import AdapterPool
    from loramux.formats import load_lora, normalize
    from loramux.integrations.diffusers import disable, enable, mixed_lora

    pipe = load_pipeline(a.model_id, "cuda")
    km = _keymap(a.kind, pipe.transformer)
    canon = {f"ad{i}": normalize(*load_lora(p)[:1], km, config=load_lora(p)[1])[0] for i, p in enumerate(a.adapters)}
    r_max = max(t[0].shape[0] for c in canon.values() for t in c.values())
    pool = AdapterPool({}, slots=len(canon) * (1 + len(a.variants)), r_max=r_max, device="cuda", dtype=torch.bfloat16)
    enable(pipe, pool, a.targets)
    served = {("exact", n): c for n, c in canon.items()}
    sizes = {}
    for v in a.variants:
        for n, c in canon.items():
            served[(v, n)] = apply_variant(c, v, a.energy)[0]
            sizes[(v, n)] = size_row(c, v, a.energy)
    for (v, n), c in served.items():
        pool.register(f"{v}/{n}", c)
        pool.ensure_resident(f"{v}/{n}")
        pool.wait(f"{v}/{n}")
    rows = []
    for n in canon:
        for j, prompt in enumerate(a.prompts):
            imgs = {}
            for v in ("exact", *a.variants):
                with mixed_lora(pipe, [pool.acquire(f"{v}/{n}")], [a.scale]):
                    imgs[v] = _gen(pipe, [prompt], [a.seed + j], a, "cuda")[0]
                pool.release(f"{v}/{n}")
            for v in a.variants:
                rows.append({**sizes[(v, n)], "adapter": n, "prompt_idx": j, "psnr": psnr(imgs[v], imgs["exact"]),
                             "lpips": lpips_distance(imgs[v], imgs["exact"], a.lpips_net, "cuda")})
    disable(pipe)
    return rows


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI around run_lossy
    ap = argparse.ArgumentParser(prog="python -m bench.lossy_quality")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--kind", choices=["qwen_image", "flux2"], required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--targets", default="all")
    ap.add_argument("--energy", type=float, default=0.995)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    write_csv(a.out, run_lossy(a), collect_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
