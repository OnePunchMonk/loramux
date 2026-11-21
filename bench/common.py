"""Timing, summary statistics, environment capture and CSV I/O shared by the bench scripts."""
from __future__ import annotations

import csv
import datetime as dt
import importlib.metadata as md
import io
import platform
import random
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch


def _safe(fn: Callable[[], object]):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - env capture must never raise
        return None


def _pkg_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _gpu_name() -> str | None:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _driver_version() -> str | None:
    fn = getattr(torch.cuda, "driver_version", None)
    if fn is not None and torch.cuda.is_available():
        return str(fn())
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.strip()
    return out.splitlines()[0] if out else None


def timer_kind() -> str:
    """'cuda' when CUDA events can time work, otherwise 'cpu' (perf_counter)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_env() -> dict:
    """Versions, GPU, host and UTC time for a results header. Values are str or None. Never raises."""
    return {
        "python": _safe(platform.python_version),
        "torch": _safe(lambda: torch.__version__),
        "triton": _pkg_version("triton"),
        "diffusers": _pkg_version("diffusers"),
        "safetensors": _pkg_version("safetensors"),
        "timer": _safe(timer_kind),
        "gpu": _safe(_gpu_name),
        "driver": _safe(_driver_version),
        "hostname": _safe(socket.gethostname),
        "timestamp_utc": _safe(lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")),
    }


def time_cuda(fn: Callable[[], object], warmup: int = 3, iters: int = 20) -> list[float]:
    """Milliseconds per call of fn, one sample per iteration. CUDA events on GPU, perf_counter on CPU."""
    if iters < 1:
        raise ValueError("iters must be >= 1")
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    if timer_kind() == "cuda":
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return samples
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def summarize(ms: list[float]) -> dict:
    """median, p90 (linear interpolation), mean and n of a list of samples."""
    if not ms:
        raise ValueError("summarize() needs at least one sample")
    a = np.asarray(ms, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()),
        "n": int(a.size),
    }


def interleave(configs: dict[str, Callable[[], object]], rounds: int,
               warmup: int = 3, iters: int = 20) -> dict[str, list[float]]:
    """Time every config once per round; the config order reverses each round to cancel drift."""
    names = list(configs)
    out: dict[str, list[float]] = {name: [] for name in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            out[name].extend(time_cuda(configs[name], warmup=warmup, iters=iters))
    return out


def write_csv(path, rows: list[dict], env: dict) -> None:
    """Write '# key=value' env lines, then a CSV. Columns are the union of row keys, in first-seen order.

    None is written as an empty field and reads back as None in env, '' in rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as f:
        for k, v in env.items():
            f.write(f"# {k}={'' if v is None else v}\n")
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _coerce(value: str):
    """int or float only when the text is that number's canonical form, so '007', '1e3' and '1_0' stay strings."""
    for cast in (int, float):
        try:
            number = cast(value)
        except ValueError:
            continue
        if str(number) == value:
            return number
    return value


def read_csv(path) -> tuple[dict, list[dict]]:
    """Inverse of write_csv. Env values are str (or None when empty). Row cells become int, float or str."""
    lines = Path(path).read_text().splitlines(keepends=True)
    env: dict = {}
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        key, _, value = lines[i][1:].strip().partition("=")
        env[key.strip()] = value or None
        i += 1
    reader = csv.DictReader(io.StringIO("".join(lines[i:])))
    rows = [{k: _coerce(v) for k, v in row.items()} for row in reader]
    return env, rows


def seed_all(seed: int) -> None:
    """Seed python, numpy and torch (torch seeds CUDA too)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
