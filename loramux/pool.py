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
        self.layer_shapes, self.S, self.r_max = dict(layer_shapes), slots, r_max
        self.device, self.dtype = device, dtype
        self.fp8 = dtype == torch.float8_e4m3fn   # FP8 weights + fp32 per-row scales (L1.5 format)
        self.A, self.B, self.SA, self.SB = {}, {}, {}, {}
        self.layer_shapes = {}
        self.add_layers(layer_shapes)
        self.rank = torch.zeros(slots, dtype=torch.int32, device=device)
        self.rank_host = [0] * slots         # host mirror of `rank` (max over layers), for r_bucket without a sync
        self.cpu: dict[str, dict] = {}
        self.blobs: dict[str, str] = {}      # key -> .lmx path, mapped lazily in ensure_resident
        self.slot_of: dict[str, int] = {}
        self.lru: OrderedDict[str, None] = OrderedDict()   # resident keys, oldest first
        self.refs: dict[str, int] = {}
        self.free = list(range(slots))
        self.hits = self.misses = self.evictions = self.h2d_bytes = 0
        self._events: dict[str, object] = {}
        self._groups: dict[str, tuple] = {}

    @property
    def slot_bytes(self) -> int:
        e = torch.tensor([], dtype=self.dtype).element_size()
        return sum(self.r_max * (di + do) * e for di, do in self.layer_shapes.values())

    def add_layers(self, layer_shapes: dict[str, tuple[int, int]]):
        """Allocate buffers for layers not yet in the pool (e.g. targets discovered from a model)."""
        for k, (di, do) in layer_shapes.items():
            if k in self.layer_shapes:
                if tuple(self.layer_shapes[k]) != (di, do):
                    raise ValueError(f"layer {k}: pool has {self.layer_shapes[k]}, model has {(di, do)}")
                continue
            self.layer_shapes[k] = (di, do)
            self.A[k] = torch.zeros(self.S, self.r_max, di, dtype=self.dtype, device=self.device)
            self.B[k] = torch.zeros(self.S, do, self.r_max, dtype=self.dtype, device=self.device)
            if self.fp8:
                self.SA[k] = torch.ones(self.S, self.r_max, dtype=torch.float32, device=self.device)
                self.SB[k] = torch.ones(self.S, do, dtype=torch.float32, device=self.device)

    def add_group(self, name: str, keys: list[str]):
        """Linears that share their input (q/k/v): store their A stacks adjacent so one grouped shrink reads x once."""
        dims = {self.layer_shapes[k][0] for k in keys}
        if len(dims) != 1 or len(keys) < 2:
            raise ValueError(f"group {name}: needs >= 2 layers with the same d_in, got {keys}")
        if any(self.group_of(k) is not None for k in keys):
            raise ValueError(f"group {name}: a layer is already grouped")
        if self.fp8:
            raise NotImplementedError("grouped FP8 layers")
        d_in = dims.pop()
        cat = torch.zeros(self.S, len(keys) * self.r_max, d_in, dtype=self.dtype, device=self.device)
        for i, k in enumerate(keys):
            cat[:, i * self.r_max:(i + 1) * self.r_max].copy_(self.A[k])
            self.A[k] = cat[:, i * self.r_max:(i + 1) * self.r_max]
        self._groups[name] = (list(keys), cat)

    def scales(self, k):
        """(a_scale [S, R], b_scale [S, d_out]) fp32 for an FP8 pool, else None."""
        return (self.SA[k], self.SB[k]) if self.fp8 else None

    def group_of(self, k):
        for name, (keys, _) in self._groups.items():
            if k in keys:
                return name, keys.index(k), len(keys)
        return None

    def group_buffers(self, name):
        keys, cat = self._groups[name]
        return cat, self.rank.expand(len(keys), self.S), None

    def slot_rank(self, slot: int) -> int:
        return self.rank_host[slot]

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
        self.cpu[key] = {k: (self._host(a), self._host(b)) for k, (a, b) in tensors.items()}

    def _host(self, t):
        """Host copy in storage format: a tensor, or (fp8 q, fp32 row scale) for an FP8 pool."""
        pin = (lambda u: u.pin_memory()) if torch.cuda.is_available() else (lambda u: u)
        if self.fp8:
            from .compress import quantize_fp8_rows
            q, sc = quantize_fp8_rows(t.cpu())
            return pin(q), pin(sc)
        return pin(t.to(self.dtype).cpu())

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

    def _load_into(self, slot: int, ad: dict):
        use_cuda = str(self.device).startswith("cuda")
        stream = torch.cuda.Stream() if use_cuda else None
        ctx = torch.cuda.stream(stream) if use_cuda else _null()
        with ctx:
            for k in self.layer_shapes:
                self.A[k][slot].zero_()
                self.B[k][slot].zero_()
            r_adapter = 0
            for k, (a, b) in ad.items():
                if self.fp8 and not isinstance(a, tuple):       # bf16 blob payload into an FP8 pool
                    a, b = self._host(a), self._host(b)
                if self.fp8:
                    (a, sa), (b, sb) = a, b
                    self.SA[k][slot, :a.shape[0]].copy_(sa, non_blocking=True)
                    self.SB[k][slot].copy_(sb, non_blocking=True)
                    self.h2d_bytes += (sa.numel() + sb.numel()) * 4
                r = a.shape[0]
                r_adapter = max(r_adapter, r)
                self.A[k][slot, :r].copy_(a, non_blocking=True)
                self.B[k][slot, :, :r].copy_(b, non_blocking=True)
                self.h2d_bytes += (a.numel() + b.numel()) * a.element_size()
            if not ad:
                raise ValueError("adapter has no layers")
            self.rank[slot] = r_adapter
            self.rank_host[slot] = r_adapter
            ev = None
            if use_cuda:
                ev = torch.cuda.Event()
                ev.record(stream)
        return ev

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
        try:
            ev = self._load_into(slot, ad)
        except BaseException:
            self.free.append(slot)   # a failed load must not leak the slot
            raise
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
