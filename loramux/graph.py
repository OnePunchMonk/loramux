"""CUDA-graph support (L3.2): batch metadata in fixed-capacity buffers that are rewritten in place.

A captured graph replays the same kernel arguments, so per-step slots/scales cannot live in fresh tensors. `StaticMeta`
owns one int32 and one fp32 buffer sized for `capacity` requests (plain rows and CFG-duplicated rows side by side);
`update()` rewrites them from host lists with one pinned non_blocking copy each and never allocates a device tensor.
Batches smaller than `capacity` are padded with slot -1 (base GEMM only), so the graph is captured once per
(capacity, tokens per row, rank bucket, CFG layout) and the caller pads its input to `capacity` rows. Uniform-row mode only.

`GraphRunner` captures `fn(x)` under `lora_batch(static.meta)` and replays it; without CUDA it runs `fn` eagerly with the
same metadata, so everything but capture itself is testable on CPU. The returned tensor is the graph's static output:
clone it before the next call.
"""
from __future__ import annotations

import torch

from .context import BatchLoRAMeta, RowTensors, lora_batch
from .metadata import expand_cfg


class StaticMeta:
    def __init__(self, capacity: int, *, r_bucket: int, device="cpu", cfg_layout: str = "concat", bm: int = 64,
                 rank_of_slot=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity, self.r_bucket, self.cfg_layout, self.device = capacity, r_bucket, cfg_layout, device
        self.rank_of_slot = rank_of_slot
        self._cuda = str(device).startswith("cuda")
        n = 3 * capacity   # plain rows, then CFG rows
        self._slot = torch.full((n,), -1, dtype=torch.int32, device=device)
        self._scale = torch.ones(n, dtype=torch.float32, device=device)
        self._host = {k: torch.empty(n, dtype=d, pin_memory=self._cuda) for k, d in (("slot", torch.int32), ("scale", torch.float32))}
        self._copied = None   # event after the last H2D copy: the pinned host buffers are reusable once it is done
        self.meta = BatchLoRAMeta(req_slot=[-1] * capacity, req_scale=[1.0] * capacity, cfg_layout=cfg_layout, device=device,
                                  any_lora=True, r_bucket=r_bucket, bm=bm)
        for rows, lo in ((capacity, 0), (2 * capacity, capacity)):
            self.meta._rows[(rows, cfg_layout)] = RowTensors(self._slot[lo:lo + rows], self._scale[lo:lo + rows], [], [])

    def update(self, req_slots, req_scales) -> None:
        """Rewrite the buffers for this step's requests (pad to capacity with base-only rows). No device allocation."""
        n = len(req_slots)
        if n != len(req_scales) or n > self.capacity:
            raise ValueError(f"{n} slots / {len(req_scales)} scales for capacity {self.capacity}")
        if self.rank_of_slot is not None:
            bad = [s for s in set(req_slots) if s >= 0 and self.rank_of_slot(s) > self.r_bucket]
            if bad:
                raise ValueError(f"slots {bad} have rank above the captured r_bucket {self.r_bucket}")
        pad = self.capacity - n
        slots, scales = list(req_slots) + [-1] * pad, [float(s) for s in req_scales] + [1.0] * pad
        cs, cc = expand_cfg(slots, scales, self.cfg_layout, True)
        if self._copied is not None:
            self._copied.synchronize()
        self._host["slot"].copy_(torch.tensor(slots + cs, dtype=torch.int32))
        self._host["scale"].copy_(torch.tensor(scales + cc, dtype=torch.float32))
        self._slot.copy_(self._host["slot"], non_blocking=self._cuda)
        self._scale.copy_(self._host["scale"], non_blocking=self._cuda)
        if self._cuda:
            self._copied = torch.cuda.Event()
            self._copied.record()


class GraphRunner:
    def __init__(self, fn, static: StaticMeta, example_x: torch.Tensor, warmup: int = 3):
        self.fn, self.static = fn, static
        self.x = example_x.clone()
        self.graph = self.out = None
        if self.x.is_cuda:
            static.update([], [])
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), lora_batch(static.meta):   # autotune and allocator warmup happen outside capture
                for _ in range(warmup):
                    fn(self.x)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), lora_batch(static.meta):
                self.out = fn(self.x)

    def __call__(self, x: torch.Tensor, req_slots, req_scales) -> torch.Tensor:
        if x.shape != self.x.shape:
            raise ValueError(f"input {tuple(x.shape)} does not match the captured {tuple(self.x.shape)}; pad to capacity")
        self.static.update(req_slots, req_scales)
        self.x.copy_(x)
        if self.graph is None:
            with lora_batch(self.static.meta):
                return self.fn(self.x)
        self.graph.replay()
        return self.out
