"""Per-request timelines and the latency breakdown from HANDOFF §11.3.

Timestamps are seconds in the time.perf_counter domain. breakdown() returns milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

TIMESTAMPS = ("t_arrive", "t_admit_ready_text", "t_adapter_resident", "t_first_step",
              "t_last_step", "t_vae_done", "t_encode_done", "t_sent")
TERMS = ("queue_wait", "adapter_wait", "text_encode", "denoise", "vae_decode",
         "image_encode", "transfer", "e2e")


class AdapterSource(Enum):
    NONE = "none"
    GPU_HIT = "gpu_hit"
    CPU_HIT = "cpu_hit"
    DISK_HIT = "disk_hit"
    MISS = "miss"


@dataclass
class RequestTimeline:
    t_arrive: float
    t_admit_ready_text: float
    t_adapter_resident: float
    t_first_step: float
    t_last_step: float
    t_vae_done: float
    t_encode_done: float
    t_sent: float
    adapter_source: AdapterSource = AdapterSource.NONE
    n_steps: int = 0
    request_id: str = ""
    adapter_key: str | None = None


def breakdown(tl: RequestTimeline) -> dict[str, float]:
    """The §11.3 terms in ms. Base-model requests (adapter_key None) ignore t_adapter_resident."""
    base = tl.adapter_key is None
    adapter_wait = 0.0 if base else max(0.0, tl.t_adapter_resident - tl.t_admit_ready_text)
    ready = tl.t_admit_ready_text if base else max(tl.t_admit_ready_text, tl.t_adapter_resident)
    terms = {
        "queue_wait": tl.t_first_step - ready,
        "adapter_wait": adapter_wait,
        "text_encode": tl.t_admit_ready_text - tl.t_arrive,
        "denoise": tl.t_last_step - tl.t_first_step,
        "vae_decode": tl.t_vae_done - tl.t_last_step,
        "image_encode": tl.t_encode_done - tl.t_vae_done,
        "transfer": tl.t_sent - tl.t_encode_done,
        "e2e": tl.t_sent - tl.t_arrive,
    }
    return {k: 1e3 * v for k, v in terms.items()}


class LatencyLog:
    def __init__(self) -> None:
        self.timelines: list[RequestTimeline] = []

    def append(self, tl: RequestTimeline) -> None:
        self.timelines.append(tl)

    def __len__(self) -> int:
        return len(self.timelines)

    def to_rows(self) -> list[dict]:
        """One dict per request: ids, source, timestamps (s) and breakdown terms (ms)."""
        rows = []
        for tl in self.timelines:
            row = {
                "request_id": tl.request_id,
                "adapter_key": tl.adapter_key,
                "adapter_source": tl.adapter_source.value,
                "n_steps": tl.n_steps,
            }
            row.update({name: getattr(tl, name) for name in TIMESTAMPS})
            row.update(breakdown(tl))
            rows.append(row)
        return rows

    def percentiles(self, term: str, by: tuple[str, ...] = ("adapter_source",)) -> dict:
        """P50/P90/P99 (ms) of one term per group. A single 'by' field gives scalar keys, several give tuples."""
        if term not in TERMS:
            raise ValueError(f"unknown term {term!r}; expected one of {TERMS}")
        groups: dict = {}
        for row in self.to_rows():
            key = tuple(row[b] for b in by)
            groups.setdefault(key[0] if len(by) == 1 else key, []).append(row[term])
        out = {}
        for key, vals in groups.items():
            p50, p90, p99 = np.percentile(np.asarray(vals, dtype=np.float64), [50, 90, 99])
            out[key] = {"p50": float(p50), "p90": float(p90), "p99": float(p99), "n": len(vals)}
        return out
