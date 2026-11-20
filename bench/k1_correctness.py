"""K1 correctness harness (L2.8): K1a per-linear numerics, K1b batch invariance, K1c vs diffusers merged reference.

K1a  (CPU-runnable) For every shape in the sweep: random x, per-slot A/B (ranks vary, one base segment), and the
     LoRA delta computed by
       path 'op'      torch.ops.loramux.lora_apply_ (Triton on CUDA in bf16; on CPU the op falls back to
                      torch_ref, so CPU rows only check the plumbing: status=plumbing, never counted by the bar),
       path 'interp'  the Triton shrink/expand kernels called directly on CPU in fp32 under TRITON_INTERPRET=1
                      (env var must be set before the process imports triton; otherwise status=skipped),
     against `lora_apply_ref_(..., fp32=True)` on the same (dtype-rounded) inputs. y starts at zero in fp32, so the
     error is loramux's own numerics (including its bf16 cast of z before expand), not base-GEMM or output-cast
     rounding. rel_err = ||y - y_ref||_F / ||y_ref||_F. Bar K1a: max over the sweep <= 2e-3.
K1b  (GPU) n requests with different adapters (one base) in one diffusers pipeline call through loramux, vs each
     request alone; per-request PSNR of the final images. Bar: min PSNR >= 45 dB.
K1c  (GPU) per adapter and prompt: loramux solo vs diffusers `load_lora_weights` + `set_adapters([name], [scale])`
     (loramux disabled), same seed, 28 steps. PSNR and LPIPS (optional `lpips` extra; LPIPS rows are empty and the
     LPIPS bar N/A without it). Bar: mean PSNR >= 35 dB and mean LPIPS <= 0.05.

Images are float HWC in [0, 1] (`output_type="np"`); PSNR uses data_range 1. Per-request generators are seeded with
seed + request index so the mixed and solo runs draw the same noise.

Output: `--out` CSV (default results/k1.csv) via bench.common.write_csv, columns
  check, status, path, T, d_in, d_out, rank, n_seg, dtype, device, rel_err, max_abs_err,     (K1a)
  request, adapter, prompt_idx, scale, psnr, lpips                                            (K1b / K1c)
and `<out>.summary.txt` with bench.bars.evaluate_k1's verdicts (also printed).
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path

import numpy as np
import torch

from bench.bars import evaluate_k1
from bench.common import collect_env, seed_all, write_csv
from loramux.metadata import build_tiles
from loramux.ops.torch_ref import lora_apply_ref_

# (T, d_in, d_out, rank, n_seg). Ranks are r_max; per-slot ranks vary below it.
DEFAULT_SHAPES = [(t, di, do, r, s) for t, (di, do), r, s in itertools.product(
    (63, 1024, 4113), ((3072, 3072), (3072, 12288), (12288, 3072)), (16, 64), (1, 3, 8)) if s <= t]
TINY_SHAPES = [(1, 32, 48, 16, 1), (17, 32, 48, 16, 3), (40, 64, 32, 32, 4), (70, 48, 80, 16, 5)]
SHAPE_SETS = {"default": DEFAULT_SHAPES, "tiny": TINY_SHAPES}
PATHS = ("op", "interp")


# ---------------------------------------------------------------- image metrics
def _as_np(img) -> np.ndarray:
    if torch.is_tensor(img):
        img = img.detach().float().cpu().numpy()
    return np.asarray(img, dtype=np.float64)


def psnr(a, b, data_range: float = 1.0) -> float:
    """PSNR in dB between two same-shape images (any layout); inf when identical."""
    a, b = _as_np(a), _as_np(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(data_range ** 2 / mse)


def have_lpips() -> bool:
    try:
        import lpips  # noqa: F401
    except ImportError:
        return False
    return True


_LPIPS_CACHE: dict = {}


def to_lpips_tensor(img) -> torch.Tensor:
    """HWC or NHWC in [0, 1] -> NCHW in [-1, 1] (the range LPIPS expects)."""
    t = torch.as_tensor(_as_np(img), dtype=torch.float32)
    if t.ndim == 3:
        t = t[None]
    if t.ndim != 4 or t.shape[-1] not in (1, 3):
        raise ValueError(f"expected HWC/NHWC image with 1 or 3 channels, got {tuple(t.shape)}")
    return t.permute(0, 3, 1, 2) * 2.0 - 1.0


def lpips_distance(a, b, net: str = "alex", device: str = "cpu") -> float | None:
    """Mean LPIPS between images in [0, 1] (HWC/NHWC); None when the `lpips` package is not installed."""
    if not have_lpips():
        return None
    import lpips
    key = (net, device)
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    with torch.no_grad():
        d = _LPIPS_CACHE[key](to_lpips_tensor(a).to(device), to_lpips_tensor(b).to(device))
    return float(d.mean())


# ---------------------------------------------------------------- K1a
def _segments(T: int, n_seg: int, g: torch.Generator) -> tuple[list[int], list[int]]:
    cuts = sorted(torch.randperm(T - 1, generator=g)[:n_seg - 1].add(1).tolist()) if n_seg > 1 else []
    bounds = [0, *cuts, T]
    return bounds[:-1], [b - a for a, b in itertools.pairwise(bounds)]


def make_case(T, d_in, d_out, r_max, n_seg, *, dtype, device, seed=0):
    """Random inputs for one K1a case. Segment 1 (if any) is a base-model segment (slot -1)."""
    g = torch.Generator().manual_seed(seed)
    S = max(1, n_seg)
    x = torch.randn(T, d_in, generator=g)
    A = torch.randn(S, r_max, d_in, generator=g) / d_in ** 0.5
    B = torch.randn(S, d_out, r_max, generator=g) / r_max ** 0.5
    rank = [r_max if s == 0 else max(1, int(torch.randint(1, r_max + 1, (1,), generator=g))) for s in range(S)]
    starts, lens = _segments(T, n_seg, g)
    slots = [-1 if i == 1 else i for i in range(n_seg)]
    scales = [float(torch.empty(1).uniform_(0.25, 1.5, generator=g)) for _ in range(n_seg)]
    i32 = {"dtype": torch.int32, "device": device}
    return {"x": x.to(device, dtype), "A": A.to(device, dtype), "B": B.to(device, dtype),
                "rank": torch.tensor(rank, **i32), "slot": torch.tensor(slots, **i32),
                "scale": torch.tensor(scales, dtype=torch.float32, device=device),
                "seg_start": torch.tensor(starts, **i32), "seg_len": torch.tensor(lens, **i32),
                "starts": starts, "lens": lens, "T": T, "d_out": d_out}


def _reference(c) -> torch.Tensor:
    y = torch.zeros(c["T"], c["d_out"], dtype=torch.float32, device=c["x"].device)
    return lora_apply_ref_(y, c["x"], c["A"], c["B"], c["rank"], c["slot"], c["scale"], c["seg_start"],
                           c["seg_len"], fp32=True)


def _run_op(c, bm) -> torch.Tensor:
    from loramux import ops  # noqa: F401  (registers loramux::lora_apply_)
    y = torch.zeros(c["T"], c["d_out"], dtype=torch.float32, device=c["x"].device)
    t = build_tiles(c["starts"], c["lens"], bm, c["x"].device)
    torch.ops.loramux.lora_apply_(y, c["x"], c["A"], c["B"], c["rank"], c["slot"], c["scale"], 0, c["A"].shape[1],
                                  seg_start=c["seg_start"], seg_len=c["seg_len"], tile_seg=t.tile_seg,
                                  tile_row0=t.tile_row0, tile_row_end=t.tile_row_end, bm=t.bm)
    return y


def interp_available() -> str | None:
    """None if the interpreter path can run, else the reason it is skipped."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        return "skipped: set TRITON_INTERPRET=1 before starting python"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "skipped: triton not installed"
    return None


def op_runs_kernels(device: str) -> bool:
    """Whether the 'op' path on `device` runs the Triton kernels (else it is the torch_ref fallback)."""
    from loramux.ops import custom_op
    return custom_op.use_triton(torch.empty(0, device=device))


def _run_interp(c, bm) -> torch.Tensor:
    from loramux.ops.triton_expand_add import expand_add_
    from loramux.ops.triton_shrink import shrink
    y = torch.zeros(c["T"], c["d_out"], dtype=torch.float32)
    t = build_tiles(c["starts"], c["lens"], bm, "cpu")
    r_max = c["A"].shape[1]
    z = shrink(c["x"], c["A"], c["rank"], c["slot"], r_bucket=r_max, tiles=t)
    expand_add_(y, z, c["B"], c["rank"], c["slot"], c["scale"], r_bucket=r_max, tiles=t)
    return y


def run_k1a(shapes=DEFAULT_SHAPES, *, paths=PATHS, device: str | None = None, dtype: torch.dtype | None = None,
            bm: int = 16, seed: int = 0) -> list[dict]:
    """One row per (path, shape). 'op' runs on `device` (default cuda if available) in `dtype` (bf16 on cuda,
    fp32 on cpu); 'interp' always runs on CPU in fp32."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    rows = []
    skip_interp = interp_available()
    op_status = "ok" if op_runs_kernels(device) else "plumbing"
    with torch.no_grad():
        for i, (T, di, do, r, s) in enumerate(shapes):
            for p in paths:
                dev, dt = (device, dtype) if p == "op" else ("cpu", torch.float32)
                row = {"check": "K1a", "status": op_status if p == "op" else "ok", "path": p, "T": T, "d_in": di, "d_out": do, "rank": r, "n_seg": s,
                           "dtype": str(dt).replace("torch.", ""), "device": dev, "rel_err": None, "max_abs_err": None}
                if p == "interp" and skip_interp:
                    row["status"] = skip_interp
                    rows.append(row)
                    continue
                c = make_case(T, di, do, r, s, dtype=dt, device=dev, seed=seed + i)
                y = (_run_op if p == "op" else _run_interp)(c, bm)
                ref = _reference(c)
                den = float(ref.norm())
                row["rel_err"] = float((y - ref).norm()) / den if den > 0 else float((y - ref).norm())
                row["max_abs_err"] = float((y - ref).abs().max())
                rows.append(row)
    return rows


# ---------------------------------------------------------------- K1b / K1c (GPU, real pipelines)
def _keymap(kind: str, transformer):
    from loramux.formats.keymaps import (
        engine_linear_names,
        engine_out_features,
        make_flux2_keymap,
        make_qwen_image_keymap,
    )
    names = engine_linear_names(transformer)
    if kind == "qwen_image":
        return make_qwen_image_keymap(names)
    if kind == "flux2":
        return make_flux2_keymap(names, engine_out_features(transformer))
    raise ValueError(f"unknown kind {kind!r}")


def load_pipeline(model_id: str, device: str = "cuda", dtype=torch.bfloat16):  # pragma: no cover - GPU
    from diffusers import DiffusionPipeline
    return DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)


def _gen(pipe, prompts, seeds, a, device) -> np.ndarray:  # pragma: no cover - GPU
    gens = [torch.Generator(device).manual_seed(s) for s in seeds]
    out = pipe(prompt=list(prompts), num_inference_steps=a.steps, height=a.height, width=a.width,
               generator=gens, output_type="np")
    return np.asarray(out.images, dtype=np.float32)


def run_pipeline_checks(a, checks) -> list[dict]:  # pragma: no cover - needs a GPU, weights and adapters
    from loramux import AdapterPool
    from loramux.formats import load_lora, normalize
    from loramux.integrations.diffusers import disable, enable, mixed_lora

    device = "cuda"
    pipe = load_pipeline(a.model_id, device)
    km = _keymap(a.kind, pipe.transformer)
    names = [f"ad{i}" for i in range(len(a.adapters))]
    canon = {}
    for n, path in zip(names, a.adapters):
        sd, cfg = load_lora(path)
        canon[n], _ = normalize(sd, km, config=cfg)
    r_max = max(t[0].shape[0] for c in canon.values() for t in c.values())
    pool = AdapterPool({}, slots=len(names), r_max=r_max, device=device, dtype=torch.bfloat16)
    enable(pipe, pool, a.targets)
    for n in names:
        pool.register(n, canon[n])
        pool.ensure_resident(n)
        pool.wait(n)
    slot = {n: pool.acquire(n) for n in names}
    rows: list[dict] = []
    solo: dict = {}

    def solo_img(ad, prompt_idx, seed):
        key = (ad, prompt_idx, seed)
        if key not in solo:
            with mixed_lora(pipe, [slot[ad] if ad != "base" else -1], [a.scale]):
                solo[key] = _gen(pipe, [a.prompts[prompt_idx]], [seed], a, device)[0]
        return solo[key]

    if "k1b" in checks:
        reqs = [names[0], "base", *names[1:]]
        prompts = [a.prompts[i % len(a.prompts)] for i in range(len(reqs))]
        seeds = [a.seed + i for i in range(len(reqs))]
        with mixed_lora(pipe, [slot[r] if r != "base" else -1 for r in reqs], [a.scale] * len(reqs)):
            mixed = _gen(pipe, prompts, seeds, a, device)
        for i, r in enumerate(reqs):
            rows.append({"check": "K1b", "status": "ok", "request": i, "adapter": r, "prompt_idx": i % len(a.prompts),
                             "scale": a.scale, "psnr": psnr(mixed[i], solo_img(r, i % len(a.prompts), seeds[i]))})
    if "k1c" in checks:
        ours = {(n, j): solo_img(n, j, a.seed + j) for n in names for j in range(len(a.prompts))}
        disable(pipe)
        for n, path in zip(names, a.adapters):
            pipe.load_lora_weights(path, adapter_name=n)
            pipe.set_adapters([n], [a.scale])
            for j in range(len(a.prompts)):
                ref = _gen(pipe, [a.prompts[j]], [a.seed + j], a, device)[0]
                rows.append({"check": "K1c", "status": "ok", "request": j, "adapter": n, "prompt_idx": j, "scale": a.scale,
                                 "psnr": psnr(ours[(n, j)], ref),
                                 "lpips": lpips_distance(ours[(n, j)], ref, a.lpips_net, device)})
            pipe.delete_adapters(n)
    else:
        disable(pipe)
    return rows


# ---------------------------------------------------------------- CLI
def summarize_k1(rows: list[dict]):
    """GroupVerdict for K1 from CSV-style rows (only status=ok rows with a number count)."""
    def vals(check, col):
        xs = [r[col] for r in rows if r.get("check") == check and r.get("status") == "ok"
              and r.get(col) not in (None, "")]
        return xs or None
    return evaluate_k1(vals("K1a", "rel_err"), vals("K1b", "psnr"), vals("K1c", "psnr"), vals("K1c", "lpips"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m bench.k1_correctness")
    ap.add_argument("--checks", default="k1a,k1b,k1c")
    ap.add_argument("--shapes", choices=sorted(SHAPE_SETS), default="default")
    ap.add_argument("--paths", default=",".join(PATHS), help="K1a paths: op, interp")
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--model-id", default="Qwen/Qwen-Image")
    ap.add_argument("--kind", choices=("qwen_image", "flux2"), default="qwen_image")
    ap.add_argument("--adapter", action="append", default=[], dest="adapters", help="LoRA path; repeat (K1b needs 2)")
    ap.add_argument("--targets", default="all", help="loramux targets: attn | all (all matches diffusers' coverage)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prompt", action="append", default=[], dest="prompts")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--out", default="results/k1.csv")
    a = ap.parse_args(argv)
    a.checks = [c for c in a.checks.split(",") if c]
    a.paths = [p for p in a.paths.split(",") if p]
    if bad := [c for c in a.checks if c not in ("k1a", "k1b", "k1c")]:
        ap.error(f"unknown check(s) {bad}")
    if bad := [p for p in a.paths if p not in PATHS]:
        ap.error(f"unknown path(s) {bad}")
    a.prompts = a.prompts or ["a photo of a red fox in the snow", "a watercolor of a lighthouse at dusk"]
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    seed_all(a.seed)
    rows: list[dict] = []
    if "k1a" in a.checks:
        rows += run_k1a(SHAPE_SETS[a.shapes], paths=tuple(a.paths), bm=a.bm, seed=a.seed)
    gpu_checks = [c for c in a.checks if c in ("k1b", "k1c")]
    if gpu_checks:
        reason = (None if torch.cuda.is_available() else "skipped: needs CUDA") or \
                 (None if a.adapters else "skipped: no --adapter given") or \
                 (None if "k1b" not in gpu_checks or len(a.adapters) >= 2 else "skipped: K1b needs >= 2 adapters")
        if reason:
            rows += [{"check": "K1" + c[2:], "status": reason} for c in gpu_checks]
        else:
            rows += run_pipeline_checks(a, gpu_checks)
    write_csv(a.out, rows, collect_env())
    verdict = summarize_k1(rows)
    Path(str(a.out) + ".summary.txt").write_text(str(verdict) + "\n")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
