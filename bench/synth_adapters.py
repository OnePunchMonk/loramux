"""Synthetic LoRA adapters for perf benchmarks. Shapes come from the caller; weights are random.

  python -m bench.synth_adapters --shapes shapes.json --out dir/ --n 64

shapes.json maps each layer key to [d_in, d_out]. The output directory gets one .lmx per adapter
and an index.json.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import torch

from loramux.blob import blob_nbytes, write_blob

# Matches a key whose last dotted component is an attention projection: to_q, to_k, to_v, to_out (or to_out.0), attn.
ATTN_PAT = re.compile(r"(?:^|\.)(?:to_q|to_k|to_v|to_out(?:\.0)?|attn)$")
DEFAULT_RANKS = (8, 16, 32, 64)


def make_adapter(layer_shapes: dict[str, tuple[int, int]], rank: int, targets: str, seed: int) -> dict:
    """{key: (A[rank, d_in], B[d_out, rank])} in float32. A has variance 1/d_in, B has variance 1e-2."""
    if targets not in ("attn", "all"):
        raise ValueError(f"targets must be 'attn' or 'all', got {targets!r}")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key, (d_in, d_out) in layer_shapes.items():
        if targets == "attn" and not ATTN_PAT.search(key):
            continue
        a = torch.randn(rank, d_in, generator=g, dtype=torch.float32) / d_in ** 0.5
        b = torch.randn(d_out, rank, generator=g, dtype=torch.float32) * 0.1
        out[key] = (a, b)
    return out


def _iter_perf_set(layer_shapes, n: int, ranks, seed: int) -> Iterator[tuple[str, str, int, dict]]:
    for i in range(n):
        targets = "attn" if i < n // 2 else "all"
        rank = ranks[i % len(ranks)]
        name = f"perf_{i:03d}_{targets}_r{rank}"
        yield name, targets, rank, make_adapter(layer_shapes, rank, targets, seed=seed + i)


def make_perf_set(layer_shapes, n: int = 64, ranks=DEFAULT_RANKS, seed: int = 0) -> dict[str, dict]:
    """n adapters: the first half attention-only, the rest all linears, ranks cycled.

    Keys are perf_{i:03d}_{targets}_r{rank}.
    """
    return {name: adapter for name, _, _, adapter in _iter_perf_set(layer_shapes, n, ranks, seed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.synth_adapters")
    parser.add_argument("--shapes", required=True, help="JSON file: {layer_key: [d_in, d_out]}")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", default="synthetic")
    args = parser.parse_args(argv)

    shapes = {k: (int(v[0]), int(v[1])) for k, v in json.loads(Path(args.shapes).read_text()).items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"base_model": args.base_model, "adapters": {}}
    for name, targets, rank, adapter in _iter_perf_set(shapes, args.n, DEFAULT_RANKS, args.seed):
        path = out / f"{name}.lmx"
        write_blob(path, adapter, base_model=args.base_model, meta={"name": name, "targets": targets, "rank": rank})
        index["adapters"][name] = {"file": path.name, "targets": targets, "rank": rank,
                                   "layers": len(adapter), "payload_bytes": blob_nbytes(path)}
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
