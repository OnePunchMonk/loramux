from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class TileMeta:
    tile_seg: torch.Tensor      # int32 [n_tiles]
    tile_row0: torch.Tensor     # int32 [n_tiles]
    tile_row_end: torch.Tensor  # int32 [n_tiles]
    bm: int


@dataclass
class RowTensors:
    slot: torch.Tensor          # int32 [rows] (device)
    scale: torch.Tensor         # fp32 [rows] (device)
    slot_host: list[int]
    scale_host: list[float]


@dataclass
class SegView:
    """Segmentation of one linear's input. A 'row' is one segment (one row of a 3-D input, or one request of a packed input)."""
    rows: int
    N: int | None               # tokens per row, None for varlen (packed)
    uniform: bool
    row_slot: torch.Tensor      # int32 [rows], -1 = base model
    row_scale: torch.Tensor     # fp32 [rows]
    seg_start: torch.Tensor     # int32 [rows]
    seg_len: torch.Tensor       # int32 [rows]
    slot_host: list[int]
    scale_host: list[float]
    start_host: list[int]
    len_host: list[int]
    bm: int
    device: object = "cpu"
    _tiles: TileMeta | None = field(default=None, repr=False)

    @property
    def tiles(self) -> TileMeta:
        """Tile table, built lazily from host lists."""
        if self._tiles is None:
            from .metadata import build_tiles
            self._tiles = build_tiles(self.start_host, self.len_host, self.bm, self.device)
        return self._tiles


@dataclass
class BatchLoRAMeta:
    """Per-request (slot, scale). Each MuxLinear derives its own segments via view_for(x)."""
    req_slot: list[int]               # -1 = base model
    req_scale: list[float]
    cfg_layout: str = "concat"        # 'concat' or 'interleaved'
    device: object = "cpu"
    any_lora: bool = False            # host-side flag, decided when the batch is formed
    r_bucket: int = 64                # host-side rank bucket (constexpr for kernels)
    seg_len: list[int] | None = None  # packed mode only: tokens per request
    bm: int = 64
    per_row_2d: bool = False          # 2-D x [rows, d] means one token per row (e.g. temb linears in DiTs)
    _views: dict = field(default_factory=dict, repr=False)
    _rows: dict = field(default_factory=dict, repr=False)   # (rows, layout) -> RowTensors
    _zcache: dict = field(default_factory=dict, repr=False)  # grouped-shrink workspaces (eager only)

    @property
    def n_req(self) -> int:
        return len(self.req_slot)

    def rows_for(self, rows: int) -> RowTensors:
        """Per-row (slot, scale) device tensors for an input with `rows` rows (n_req or 2*n_req for CFG).
        build_meta makes both up front, so a forward pass (also a compiled one) only does a dict lookup."""
        key = (rows, self.cfg_layout)
        t = self._rows.get(key)
        if t is None:
            from .metadata import make_rows
            make_rows(self, [rows])
            t = self._rows[key]
        return t

    def uniform_shape(self, x: torch.Tensor) -> tuple[int, int] | None:
        """(rows, N) for uniform-row inputs, None for packed (varlen) metadata. Uses static shapes only."""
        if self.seg_len is not None:
            return None
        if x.dim() == 2 and self.per_row_2d:
            return x.shape[0], 1
        if x.dim() < 3:
            raise ValueError(
                f"2-D input {tuple(x.shape)} needs explicit per-request seg_len in build_meta; "
                f"otherwise pass x as [rows, N, d]")
        rows = x.shape[0]
        d = x.shape[-1]
        return rows, (x.numel() // (rows * d) if rows * d else 0)

    def view_for(self, x: torch.Tensor) -> SegView:
        from .metadata import make_view
        if x.dim() == 2 and self.per_row_2d and self.seg_len is None:
            x = x.unsqueeze(1)
        d = x.shape[-1]
        if self.seg_len is not None:
            if x.dim() != 2:
                raise ValueError(f"packed metadata (seg_len given) needs 2-D x [T, d], got shape {tuple(x.shape)}")
            key = (self.n_req, None, self.bm, self.cfg_layout)
            v = self._views.get(key)
            if v is None:
                T = x.shape[0]
                if sum(self.seg_len) != T:
                    raise ValueError(f"seg_len sums to {sum(self.seg_len)} but packed x has {T} rows")
                v = self._views[key] = make_view(self, None, None)
            elif sum(self.seg_len) != x.shape[0]:
                raise ValueError(f"seg_len sums to {sum(self.seg_len)} but packed x has {x.shape[0]} rows")
            return v
        if x.dim() < 3:
            raise ValueError(
                f"2-D input {tuple(x.shape)} needs explicit per-request seg_len in build_meta; "
                f"otherwise pass x as [rows, N, d]")
        rows = x.shape[0]
        N = x.numel() // (rows * d) if rows * d else 0
        key = (rows, N, self.bm, self.cfg_layout)
        v = self._views.get(key)
        if v is None:
            v = self._views[key] = make_view(self, rows, N)
        return v


_current: BatchLoRAMeta | None = None


@contextmanager
def lora_batch(meta: BatchLoRAMeta):
    global _current
    prev, _current = _current, meta
    try:
        yield meta
    finally:
        _current = prev


def current() -> BatchLoRAMeta | None:
    return _current
