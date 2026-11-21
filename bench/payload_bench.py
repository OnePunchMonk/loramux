"""Adapter payload bench (HANDOFF §11.1, bars L4 and P1): the CSV `bench.latency_bars --payload` reads.

For every (rank, targets) cell: write a synthetic adapter as an .lmx blob, then time `ensure_resident` + `wait`
for a CPU-tier blob into a GPU slot (one bulk H2D + one scatter), `--reps` times with the slot evicted between
reps. Beside it, the pinned-host to device copy bandwidth for the same byte count (`h2d_gbps`) so L4's
"≥ 70% of measured H2D bandwidth" is judged on this host. On a CPU-only host the same code runs with host
memcpy as the "bandwidth"; the env header's `timer` says which it was and those rows never count for L4.

CSV columns: rank, targets, rep, load_ms, blob_bytes, source_bytes, h2d_gbps. `source_bytes` is the size the
same tensors take as a `--source-dtype` (default fp32) safetensors file, header excluded.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from bench.common import collect_env, seed_all, timer_kind, write_csv
from bench.synth_adapters import make_adapter
from loramux.blob import blob_nbytes, write_blob
from loramux.pool import AdapterPool

_ELEM = {"fp32": 4, "bf16": 2, "fp16": 2}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_ms(fn) -> float:
    _sync()
    t = time.perf_counter()
    fn()
    _sync()
    return (time.perf_counter() - t) * 1e3


def h2d_gbps(nbytes: int, reps: int = 10) -> float:
    """Median pinned-host to device copy bandwidth (GB/s) for `nbytes`; host-to-host memcpy without CUDA."""
    cuda = torch.cuda.is_available()
    src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=cuda)
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda" if cuda else "cpu")
    dst.copy_(src)
    ms = sorted(_timed_ms(lambda: dst.copy_(src, non_blocking=cuda)) for _ in range(reps))
    return nbytes / (ms[len(ms) // 2] / 1e3) / 1e9


def _load_ms(pool: AdapterPool, reps: int) -> list[float]:
    """Time `reps` cold loads. One slot, two adapters alternating: every load evicts the other."""
    def load(i: int) -> None:
        pool.ensure_resident(f"a{i}")
        pool.wait(f"a{i}")

    load(0)  # warmup (autotune, allocator)
    return [_timed_ms(lambda i=(rep + 1) % 2: load(i)) for rep in range(reps)]


def run_payload_bench(layer_shapes: dict[str, tuple[int, int]], *, ranks=(8, 16, 32, 64), targets=("attn", "all"),
                      reps: int = 10, source_dtype: str = "fp32", workdir=None, device: str | None = None,
                      seed: int = 0) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        for tgt in targets:
            for r in ranks:
                ad = make_adapter(layer_shapes, r, tgt, seed)
                if not ad:
                    continue
                paths = [Path(tmp) / f"{tgt}_r{r}_{i}.lmx" for i in range(2)]
                for p in paths:
                    write_blob(p, ad, "bench")
                nbytes = blob_nbytes(paths[0])
                src = sum((a.numel() + b.numel()) for a, b in ad.values()) * _ELEM[source_dtype]
                pool = AdapterPool(layer_shapes, slots=1, r_max=r, device=device)
                for i, p in enumerate(paths):
                    pool.register_blob(f"a{i}", p)
                bw = h2d_gbps(nbytes)

                for rep, ms in enumerate(_load_ms(pool, reps)):
                    rows.append({"rank": r, "targets": tgt, "rep": rep, "load_ms": ms, "blob_bytes": nbytes,
                                 "source_bytes": src, "h2d_gbps": bw})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.payload_bench")
    ap.add_argument("--shapes", required=True, help="JSON file: {layer_key: [d_in, d_out]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--source-dtype", choices=sorted(_ELEM), default="fp32")
    a = ap.parse_args(argv)
    seed_all(0)
    shapes = {k: tuple(v) for k, v in json.loads(Path(a.shapes).read_text()).items()}
    rows = run_payload_bench(shapes, ranks=a.ranks, reps=a.reps, source_dtype=a.source_dtype)
    write_csv(a.out, rows, {**collect_env(), "timer": timer_kind()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
