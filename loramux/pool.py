"""Adapter pool: S GPU slots in one arena, a CPU tier, LRU eviction, refcounts, async H2D loads (HANDOFF 3.8, 11.1).

Canonical adapter: {layer_key: (A[r, d_in], B[d_out, r])}, alpha/r already folded into B.

Arena (TASKS L2.3). One tensor `arena[S, slot_elems]` of the storage dtype; one slot's whole adapter is
`arena[slot]`, contiguous. Per layer, `A[k] = arena[:, offA:offA+R*d_in].view(S, R, d_in)` and
`B[k] = arena[:, offB:offB+d_out*R].view(S, d_out, R)` are views (the kernels take strides).
FP8 pools (dtype float8_e4m3fn) keep fp32 per-row scales (L1.5 format) in `sarena[S, scale_elems]` with views
`SA[k] [S, R]` and `SB[k] [S, d_out]`.

Per-layer rank (decision 3). `rank_all [L, S]` int32 with views `rank[k] -> int32 [S]`, plus host mirrors
`layer_rank_host[k][slot]` and `rank_host[slot]` (max over layers, for r_bucket without a sync). A layer missing
from an adapter gets rank 0. Kernels read only below the layer's rank, so slots are never zeroed on load.

Load path. Payload (native rank, blob order, from the mmap or an in-memory registration) -> one memcpy into a
pinned staging buffer from a ring of `n_staging` (a buffer is reused only after its event completed) -> one
`copy_(non_blocking=True)` into a device staging buffer on a persistent side stream -> one `triton_scatter`
launch driven by a descriptor table that also updates the rank table -> event. The side stream first waits on
the current stream, so a slot evicted after its last use is not overwritten while queued kernels may still read
it. On a CPU device the same descriptor table is applied with torch copies.

Readiness. `ensure_resident(key)` returns the load's CUDA event (None on CPU). The default stream must not read
a slot before that event completes: `is_ready(key)` polls it, `wait(key)` blocks the host on it, and
`acquire(key)` raises unless the key is ready.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np
import torch

from .blob import _payload_nbytes, _read_prefix, read_blob, read_header
from .ops import custom_op
from .ops.triton_scatter import KIND_R, KIND_S, KIND_W, NCOL, desc_tensor, scatter

_EALIGN = 64      # element alignment of every region start inside a slot
_BALIGN = 256     # byte alignment of the sections of a staging buffer


def _al(n: int, a: int) -> int:
    return (n + a - 1) // a * a


@dataclass
class _Payload:
    """One adapter in staging order. w: 1-D weights (staged dtype); s: 1-D fp32 scales (FP8 pools) or None.
    layers: (key, r, offA, offB, offSA, offSB) in elements of w / s."""
    w: torch.Tensor
    s: torch.Tensor | None
    layers: list[tuple[str, int, int, int, int, int]]


class _StagingRing:
    """n pinned host buffers (+ device twins), reused round-robin; a buffer is reused only after its event."""

    def __init__(self, n: int, device, pin: bool):
        if n < 1:
            raise ValueError("n_staging must be >= 1")
        self.n, self.device, self.pin = n, device, pin
        self.host: list[torch.Tensor | None] = [None] * n
        self.dev: list[torch.Tensor | None] = [None] * n
        self.events: list = [None] * n
        self.next = 0
        self.size = 0          # bytes: largest registered payload (buffers are allocated at least this big)
        self.waits = 0         # times a take() had to block on a buffer still in flight
        self.order: deque[int] = deque(maxlen=64)   # recent buffer indices, for tests and debugging

    def take(self, nbytes: int, stream=None) -> int:
        i = self.next
        self.next = (i + 1) % self.n
        ev = self.events[i]
        if ev is not None and not ev.query():
            self.waits += 1
            ev.synchronize()
        self.events[i] = None
        need = max(nbytes, self.size)
        if self.host[i] is None or self.host[i].numel() < nbytes:
            h = torch.empty(need, dtype=torch.uint8)
            self.host[i] = h.pin_memory() if self.pin else h
        if str(self.device).startswith("cuda") and (self.dev[i] is None or self.dev[i].numel() < nbytes):
            with torch.cuda.stream(stream):
                self.dev[i] = torch.empty(need, dtype=torch.uint8, device=self.device)
        self.order.append(i)
        return i


class AdapterPool:
    """S slots in one arena, a CPU tier (in-memory or mmapped .lmx blobs), LRU eviction, refcounts."""

    def __init__(self, layer_shapes: dict[str, tuple[int, int]], slots: int, r_max: int,
                 device="cuda", dtype=torch.bfloat16, n_staging: int = 2):
        self.S, self.r_max = slots, r_max
        self.device, self.dtype = device, dtype
        self.fp8 = dtype == torch.float8_e4m3fn
        self._cuda = str(device).startswith("cuda")
        self.layer_shapes: dict[str, tuple[int, int]] = {}
        self._layout: dict[str, tuple[int, int, int, int, int]] = {}   # k -> (offA, offB, offSA, offSB, idx)
        self.slot_elems = self.scale_elems = 0
        self.arena = torch.zeros(slots, 0, dtype=dtype, device=device)
        self.sarena = torch.zeros(slots, 0, dtype=torch.float32, device=device) if self.fp8 else None
        self.rank_all = torch.zeros(0, slots, dtype=torch.int32, device=device)
        self.A: dict[str, torch.Tensor] = {}
        self.B: dict[str, torch.Tensor] = {}
        self.SA: dict[str, torch.Tensor] = {}
        self.SB: dict[str, torch.Tensor] = {}
        self.rank: dict[str, torch.Tensor] = {}
        self.layer_rank_host: dict[str, list[int]] = {}
        self.rank_host = [0] * slots          # max rank over layers per slot (host), for r_bucket
        self.cpu: dict[str, _Payload] = {}
        self.blobs: dict[str, str] = {}       # key -> .lmx path, mapped lazily in ensure_resident
        self.slot_of: dict[str, int] = {}
        self.lru: OrderedDict[str, None] = OrderedDict()   # resident keys, oldest first
        self.refs: dict[str, int] = {}
        self.free = list(range(slots))
        self.hits = self.misses = self.evictions = self.h2d_bytes = 0
        self._events: dict[str, object] = {}
        self._stream = None
        self._bound_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._ring = _StagingRing(n_staging, device, pin=torch.cuda.is_available())
        self.add_layers(layer_shapes)

    # ------------------------------------------------------------------ layout ---------------------------------

    @property
    def slot_bytes(self) -> int:
        b = self.slot_elems * self.arena.element_size()
        return b + (self.scale_elems * 4 if self.fp8 else 0)

    def add_layers(self, layer_shapes: dict[str, tuple[int, int]]):
        """Add layers not yet in the pool (e.g. targets discovered from a model). Re-lays out the arena,
        keeping resident slots' contents."""
        new = {}
        for k, (di, do) in layer_shapes.items():
            if k in self.layer_shapes:
                if tuple(self.layer_shapes[k]) != (di, do):
                    raise ValueError(f"layer {k}: pool has {self.layer_shapes[k]}, model has {(di, do)}")
                continue
            new[k] = (int(di), int(do))
        if new:
            self.layer_shapes.update(new)
            self._relayout()

    def _relayout(self):
        R = self.r_max
        order = list(self.layer_shapes)
        layout, off, soff = {}, 0, 0
        for idx, (k, (di, do)) in enumerate(self.layer_shapes.items()):   # per layer: A region, then B region
            offA, offSA = _al(off, _EALIGN), _al(soff, _EALIGN)
            offB, offSB = _al(offA + R * di, _EALIGN), _al(offSA + R, _EALIGN)
            layout[k] = (offA, offB, offSA, offSB, idx)
            off, soff = offB + do * R, offSB + do
        self._sync_side_stream()
        old = (self._layout, self.A, self.B, self.SA, self.SB, self.rank)
        S = self.S
        self.slot_elems, self.scale_elems = _al(off, _EALIGN), _al(soff, _EALIGN)
        self.arena = torch.zeros(S, self.slot_elems, dtype=self.dtype, device=self.device)
        if self.fp8:
            self.sarena = torch.ones(S, self.scale_elems, dtype=torch.float32, device=self.device)
        self.rank_all = torch.zeros(len(order), S, dtype=torch.int32, device=self.device)
        self._layout, self._order = layout, order
        self.A, self.B, self.SA, self.SB, self.rank = {}, {}, {}, {}, {}
        for k, (di, do) in self.layer_shapes.items():
            oA, oB, oSA, oSB, idx = layout[k]
            self.A[k] = self.arena[:, oA:oA + R * di].view(S, R, di)
            self.B[k] = self.arena[:, oB:oB + do * R].view(S, do, R)
            if self.fp8:
                self.SA[k] = self.sarena[:, oSA:oSA + R]
                self.SB[k] = self.sarena[:, oSB:oSB + do]
            self.rank[k] = self.rank_all[idx]
            self.layer_rank_host.setdefault(k, [0] * S)
        o_layout, oA_, oB_, oSA_, oSB_, orank = old
        for k in o_layout:                        # keep resident contents
            self.A[k].copy_(oA_[k])
            self.B[k].copy_(oB_[k])
            self.rank[k].copy_(orank[k])
            if self.fp8:
                self.SA[k].copy_(oSA_[k])
                self.SB[k].copy_(oSB_[k])
        for module, key in list(self._bound_modules.items()):
            self._rebind(module, key)

    def bind(self, module, key: str):
        """Track a MuxLinear so relayout can update its registered buffer views."""
        self._bound_modules[module] = key
        self._rebind(module, key)

    def unbind(self, module):
        """Stop tracking a module removed from a model."""
        self._bound_modules.pop(module, None)

    def _rebind(self, module, key: str):
        if key in self.A:
            A, B, rank = self.buffers(key)
            scales = self.scales(key) if self.fp8 else None
            a_scale, b_scale = scales if scales is not None else (None, None)
            has_layer = True
        else:
            A = B = rank = a_scale = b_scale = None
            has_layer = False
        module._buffers["pool_A"] = A
        module._buffers["pool_B"] = B
        module._buffers["pool_rank"] = rank
        module._buffers["pool_a_scale"] = a_scale
        module._buffers["pool_b_scale"] = b_scale
        module._has_pool_layer = has_layer

    def _sync_side_stream(self):
        if self._stream is not None:
            self._stream.synchronize()

    # ------------------------------------------------------------------ views ----------------------------------

    def has_layer(self, k):
        return k in self.A

    def buffers(self, k):
        """(A [S, R, d_in], B [S, d_out, R], rank int32 [S] of this layer)."""
        return self.A[k], self.B[k], self.rank[k]

    def scales(self, k):
        """(a_scale [S, R], b_scale [S, d_out]) fp32 for an FP8 pool, else None."""
        return (self.SA[k], self.SB[k]) if self.fp8 else None

    def slot_rank(self, slot: int) -> int:
        return self.rank_host[slot]

    # ------------------------------------------------------------------ CPU tier -------------------------------

    def _check_rank(self, key, k, r):
        if r > self.r_max:
            raise ValueError(f"adapter {key}: rank {r} > r_max {self.r_max} at {k}")

    def _pack(self, tensors: dict) -> _Payload:
        """In-memory adapter -> one contiguous host payload in storage format (FP8: quantized + scales)."""
        layers, ws, ss, off, soff = [], [], [], 0, 0
        for k, (a, b) in tensors.items():
            r = a.shape[0]
            if self.fp8:
                from .compress import quantize_fp8_rows
                (qa, sa), (qb, sb) = quantize_fp8_rows(a.detach().cpu()), quantize_fp8_rows(b.detach().cpu())
                ws += [qa.reshape(-1), qb.reshape(-1)]
                ss += [sa, sb]
                layers.append((k, r, off, off + qa.numel(), soff, soff + r))
                off += qa.numel() + qb.numel()
                soff += r + b.shape[0]
            else:
                ws += [a.detach().to(self.dtype).cpu().reshape(-1), b.detach().to(self.dtype).cpu().reshape(-1)]
                layers.append((k, r, off, off + a.numel(), 0, 0))
                off += a.numel() + b.numel()
        w = torch.cat(ws) if ws else torch.empty(0, dtype=self.dtype)
        s = (torch.cat(ss) if ss else torch.empty(0)) if self.fp8 else None
        if torch.cuda.is_available():
            w = w.pin_memory()
            s = s.pin_memory() if s is not None else None
        return _Payload(w, s, layers)

    def _staged_nbytes(self, w_bytes: int, s_elems: int, n_layers: int) -> int:
        n_desc = n_layers * (4 if self.fp8 else 2) + 1
        off_w = _al(_al(n_desc * NCOL * 8, 8) + 4 * max(len(self.layer_shapes), 1), _BALIGN)
        return _al(off_w + w_bytes, 16) + 4 * s_elems

    def _check_not_resident(self, key):
        if key in self.slot_of:            # its slot would keep the old weights while the CPU tier has new ones
            raise RuntimeError(f"adapter {key} is resident; re-register it only after it is evicted")

    def register(self, key: str, tensors: dict):
        self._check_not_resident(key)
        for k, (a, b) in tensors.items():
            if k not in self.layer_shapes:
                raise KeyError(f"adapter {key}: unknown layer {k}")
            self._check_rank(key, k, a.shape[0])
            if tuple(a.shape) != (a.shape[0], self.layer_shapes[k][0]) or \
                    tuple(b.shape) != (self.layer_shapes[k][1], a.shape[0]):
                raise ValueError(f"adapter {key}: layer {k} has A {tuple(a.shape)}, B {tuple(b.shape)}; "
                                 f"pool expects d_in x d_out = {self.layer_shapes[k]}")
        self.blobs.pop(key, None)
        p = self._pack(tensors)
        self.cpu[key] = p
        self._ring.size = max(self._ring.size, self._staged_nbytes(
            p.w.numel() * p.w.element_size(), 0 if p.s is None else p.s.numel(), len(p.layers)))

    def register_blob(self, key: str, path):
        """CPU-tier entry backed by an .lmx file. Only the header is read now (to validate); the payload is
        mmapped and copied when ensure_resident loads it."""
        self._check_not_resident(key)
        header = read_header(path)
        for lay in header["layers"]:
            k = lay["key"]
            if k not in self.layer_shapes:
                raise KeyError(f"adapter {key}: unknown layer {k}")
            if (lay["d_in"], lay["d_out"]) != tuple(self.layer_shapes[k]):
                raise ValueError(f"adapter {key}: layer {k} is {lay['d_in']}x{lay['d_out']}, "
                                 f"pool expects {tuple(self.layer_shapes[k])}")
            self._check_rank(key, k, lay["r"])
        self.cpu.pop(key, None)
        self.blobs[key] = str(path)
        nb = _payload_nbytes(header["layers"])
        s_el = sum(lay["r"] + lay["d_out"] for lay in header["layers"]) if self.fp8 else 0
        self._ring.size = max(self._ring.size, self._staged_nbytes(nb, s_el, len(header["layers"])))

    def _payload(self, key: str) -> _Payload:
        if key in self.cpu:
            return self.cpu[key]
        if key not in self.blobs:
            raise KeyError(f"adapter {key} is not registered")
        path = self.blobs[key]
        if self.fp8:                                # bf16 blob into an FP8 pool: quantize on the host
            return self._pack(read_blob(path, mmap=True))
        with open(path, "rb") as f:
            header, start = _read_prefix(f)
        nbytes = _payload_nbytes(header["layers"])
        words = np.memmap(path, dtype="<i2", mode="c", offset=start, shape=(nbytes // 2,))
        w = torch.from_numpy(np.asarray(words)).view(torch.bfloat16)   # the blob payload, padding included
        layers = [(lay["key"], lay["r"], lay["off_A"] // 2, lay["off_B"] // 2, 0, 0) for lay in header["layers"]]
        return _Payload(w, None, layers)

    def canonical(self, key: str) -> dict:
        """The registered adapter as {layer: (A [r, d_in], B [d_out, r])} on the CPU (dequantized for FP8 pools)."""
        p = self._payload(key)
        out = {}
        for k, r, oA, oB, oSA, oSB in p.layers:
            di, do = self.layer_shapes[k]
            a, b = p.w[oA:oA + r * di].view(r, di), p.w[oB:oB + do * r].view(do, r)
            if self.fp8:
                from .compress import dequantize_fp8_rows
                a, b = dequantize_fp8_rows(a, p.s[oSA:oSA + r]), dequantize_fp8_rows(b, p.s[oSB:oSB + do])
            out[k] = (a, b)
        return out

    # ------------------------------------------------------------------ loading --------------------------------

    def _descriptors(self, slot: int, p: _Payload):
        R, S = self.r_max, self.S
        rows, ranks = [], [0] * len(self._order)
        for k, r, oA, oB, oSA, oSB in p.layers:
            di, do = self.layer_shapes[k]
            dA, dB, dSA, dSB, idx = self._layout[k]
            base = slot * self.slot_elems
            rows.append((oA, base + dA, r, di, di, di, KIND_W))
            rows.append((oB, base + dB, do, r, r, R, KIND_W))
            if self.fp8:
                sbase = slot * self.scale_elems
                rows.append((oSA, sbase + dSA, 1, r, r, r, KIND_S))
                rows.append((oSB, sbase + dSB, 1, do, do, do, KIND_S))
            ranks[idx] = r
        rows.append((0, slot, len(ranks), 1, 1, S, KIND_R))
        return rows, ranks

    def _load_into(self, slot: int, key: str):
        p = self._payload(key)
        if not p.layers:
            raise ValueError(f"adapter {key} has no layers")
        rows, ranks = self._descriptors(slot, p)
        wb = p.w.numel() * p.w.element_size()
        off_r = _al(len(rows) * NCOL * 8, 8)
        off_w = _al(off_r + 4 * len(ranks), _BALIGN)
        off_s = _al(off_w + wb, 16)
        total = off_s + (0 if p.s is None else 4 * p.s.numel())
        if self._cuda and self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        i = self._ring.take(total, self._stream)
        host = self._ring.host[i][:total]
        host[:off_r].view(torch.int64).copy_(desc_tensor(rows).view(-1))
        host[off_r:off_r + 4 * len(ranks)].view(torch.int32).copy_(torch.tensor(ranks, dtype=torch.int32))
        host[off_w:off_w + wb].view(p.w.dtype).copy_(p.w)          # the one bulk memcpy of the weights
        if p.s is not None:
            host[off_s:total].view(torch.float32).copy_(p.s)

        def run(buf):
            desc = buf[:off_r].view(torch.int64).view(-1, NCOL)
            srcR = buf[off_r:off_r + 4 * len(ranks)].view(torch.int32)
            srcW = buf[off_w:off_w + wb].view(p.w.dtype)
            srcS = buf[off_s:total].view(torch.float32) if p.s is not None else None
            dstS = self.sarena.view(-1) if p.s is not None else None
            scatter(desc, rows, srcW, self.arena.view(-1), srcS, dstS, srcR, self.rank_all.view(-1),
                    use_triton=custom_op.use_triton(self.arena))

        ev = None
        if self._cuda:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                dev = self._ring.dev[i][:total]
                dev.copy_(host, non_blocking=True)                   # the one H2D copy
                run(dev)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            self._ring.events[i] = ev
        else:
            run(host)
        for k, idx in ((k, self._layout[k][4]) for k in self._order):
            self.layer_rank_host[k][slot] = ranks[idx]
        self.rank_host[slot] = max(ranks, default=0)
        self.h2d_bytes += sum(r * sum(self.layer_shapes[k]) for k, r, *_ in p.layers) * p.w.element_size()
        if p.s is not None:
            self.h2d_bytes += 4 * sum(r + self.layer_shapes[k][1] for k, r, *_ in p.layers)
        return ev

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
        """Start loading if needed. Returns the load's CUDA event (None on CPU); poll with is_ready(key)."""
        if key in self.slot_of:
            self.hits += 1
            self.lru.move_to_end(key)
            return self._events.get(key)
        if key not in self.cpu and key not in self.blobs:
            raise KeyError(f"adapter {key} is not registered")
        self.misses += 1
        if not self.free:
            self._evict_one()
        slot = self.free.pop()
        try:
            ev = self._load_into(slot, key)
        except BaseException:
            self.free.append(slot)   # a failed load must not leak the slot
            raise
        self.slot_of[key] = slot
        self.lru[key] = None
        self._events[key] = ev
        return ev

    def is_ready(self, key: str) -> bool:
        """Resident and its load has completed (event.query(); no host sync)."""
        if key not in self.slot_of:
            return False
        ev = self._events.get(key)
        return ev is None or ev.query()

    def wait(self, key: str):
        """Block the host until key's load has completed."""
        if key not in self.slot_of:
            raise RuntimeError(f"{key} is not resident")
        ev = self._events.get(key)
        if ev is not None:
            ev.synchronize()

    def acquire(self, key: str, *, loading_ok: bool = False) -> int:
        """Pin key's slot (refcount; pinned slots are never evicted) and return it. Raises unless the load has
        completed; loading_ok=True pins an in-flight load, and the caller must see is_ready(key) before the
        default stream reads the slot."""
        if key not in self.slot_of:
            raise RuntimeError(f"{key} is not resident")
        if not loading_ok and not self.is_ready(key):
            raise RuntimeError(f"{key} is still loading; admit only keys with is_ready(key) (or call wait)")
        self.refs[key] = self.refs.get(key, 0) + 1
        self.lru.move_to_end(key)
        return self.slot_of[key]

    def release(self, key: str):
        if self.refs.get(key, 0) <= 0:
            raise RuntimeError(f"release({key!r}) without a matching acquire")
        self.refs[key] -= 1

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions, "h2d_bytes": self.h2d_bytes,
                "slot_bytes": self.slot_bytes, "resident": len(self.slot_of), "staging_waits": self._ring.waits}


def composite_key(items):
    """Cache key for a multi-LoRA request: sorted (id, scale) pairs."""
    return "+".join(f"{i}@{s:g}" for i, s in sorted(items))
