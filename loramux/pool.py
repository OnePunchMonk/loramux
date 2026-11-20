from __future__ import annotations

from collections import OrderedDict

import torch

from .blob import read_blob, read_header


class AdapterPool:
    """S GPU slots of stacked (A, B) per layer, a CPU tier, LRU eviction, refcounts.

    Canonical adapter: {layer_key: (A[r, d_in], B[d_out, r])}, alpha/r already folded into B.
    """

    def __init__(self, layer_shapes: dict[str, tuple[int, int]], slots: int, r_max: int,
                 device="cuda", dtype=torch.bfloat16):
        self.layer_shapes, self.S, self.r_max = layer_shapes, slots, r_max
        self.device, self.dtype = device, dtype
        self.A = {k: torch.zeros(slots, r_max, di, dtype=dtype, device=device) for k, (di, do) in layer_shapes.items()}
        self.B = {k: torch.zeros(slots, do, r_max, dtype=dtype, device=device) for k, (di, do) in layer_shapes.items()}
        self.rank = torch.zeros(slots, dtype=torch.int32, device=device)
        self.cpu: dict[str, dict] = {}
        self.blobs: dict[str, str] = {}      # key -> .lmx path, mapped lazily in ensure_resident
        self.slot_of: dict[str, int] = {}
        self.lru: OrderedDict[str, None] = OrderedDict()   # resident keys, oldest first
        self.refs: dict[str, int] = {}
        self.free = list(range(slots))
        self.hits = self.misses = self.evictions = self.h2d_bytes = 0
        self._events: dict[str, object] = {}

    @property
    def slot_bytes(self) -> int:
        e = torch.tensor([], dtype=self.dtype).element_size()
        return sum(self.r_max * (di + do) * e for di, do in self.layer_shapes.values())

    def has_layer(self, k):
        return k in self.A

    def buffers(self, k):
        return self.A[k], self.B[k], self.rank

    def register(self, key: str, tensors: dict):
        for k, (a, b) in tensors.items():
            if k not in self.layer_shapes:
                raise KeyError(f"adapter {key}: unknown layer {k}")
            if a.shape[0] > self.r_max:
                raise ValueError(f"adapter {key}: rank {a.shape[0]} > r_max {self.r_max} at {k}")
        self.blobs.pop(key, None)
        self.cpu[key] = {k: (a.to(self.dtype).cpu().pin_memory() if torch.cuda.is_available() else a.to(self.dtype).cpu(),
                             b.to(self.dtype).cpu().pin_memory() if torch.cuda.is_available() else b.to(self.dtype).cpu())
                         for k, (a, b) in tensors.items()}

    def register_blob(self, key: str, path):
        """CPU-tier entry backed by an .lmx file. Only the header is read now (to validate); the payload is
        mmapped and copied when ensure_resident loads it."""
        for lay in read_header(path)["layers"]:
            k = lay["key"]
            if k not in self.layer_shapes:
                raise KeyError(f"adapter {key}: unknown layer {k}")
            if (lay["d_in"], lay["d_out"]) != tuple(self.layer_shapes[k]):
                raise ValueError(f"adapter {key}: layer {k} is {lay['d_in']}x{lay['d_out']}, "
                                 f"pool expects {tuple(self.layer_shapes[k])}")
            if lay["r"] > self.r_max:
                raise ValueError(f"adapter {key}: rank {lay['r']} > r_max {self.r_max} at {k}")
        self.cpu.pop(key, None)
        self.blobs[key] = str(path)

    def _payload(self, key: str) -> dict:
        if key in self.cpu:
            return self.cpu[key]
        if key in self.blobs:
            return read_blob(self.blobs[key], mmap=True)
        raise KeyError(f"adapter {key} is not registered")

    def _evict_one(self):
        for key in self.lru:
            if self.refs.get(key, 0) == 0:
                slot = self.slot_of.pop(key)
                del self.lru[key]
                self._events.pop(key, None)
                self.free.append(slot)
                self.evictions += 1
                return
        raise RuntimeError("no evictable slot: all resident adapters are in use")

    def ensure_resident(self, key: str):
        """Start loading if needed. Returns a CUDA event (or None on CPU) to poll with .query()."""
        if key in self.slot_of:
            self.hits += 1
            self.lru.move_to_end(key)
            return self._events.get(key)
        ad = self._payload(key)
        self.misses += 1
        if not self.free:
            self._evict_one()
        slot = self.free.pop()
        use_cuda = str(self.device).startswith("cuda")
        stream = torch.cuda.Stream() if use_cuda else None
        ctx = torch.cuda.stream(stream) if use_cuda else _null()
        with ctx:
            for k in self.layer_shapes:
                self.A[k][slot].zero_()
                self.B[k][slot].zero_()
            for k, (a, b) in ad.items():
                r = a.shape[0]
                self.A[k][slot, :r].copy_(a, non_blocking=True)
                self.B[k][slot, :, :r].copy_(b, non_blocking=True)
                self.h2d_bytes += (a.numel() + b.numel()) * a.element_size()
            r_adapter = max(a.shape[0] for a, _ in ad.values())
            self.rank[slot] = r_adapter
            ev = None
            if use_cuda:
                ev = torch.cuda.Event()
                ev.record(stream)
        self.slot_of[key] = slot
        self.lru[key] = None
        self._events[key] = ev
        return ev

    def acquire(self, key: str) -> int:
        if key not in self.slot_of:
            raise RuntimeError(f"{key} is not resident")
        self.refs[key] = self.refs.get(key, 0) + 1
        self.lru.move_to_end(key)
        return self.slot_of[key]

    def release(self, key: str):
        self.refs[key] -= 1
        assert self.refs[key] >= 0

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    h2d_bytes=self.h2d_bytes, slot_bytes=self.slot_bytes, resident=len(self.slot_of))


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def composite_key(items):
    """Cache key for a multi-LoRA request: sorted (id, scale) pairs."""
    return "+".join(f"{i}@{s:g}" for i, s in sorted(items))
