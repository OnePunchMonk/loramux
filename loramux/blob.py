"""Packed adapter blob (.lmx), HANDOFF 11.1. Little-endian, 256-byte aligned sections.

  bytes 0..7    magic b"LMXBLOB1"
  bytes 8..15   uint64 header_len (UTF-8 JSON bytes that follow, before padding)
  header        {"version", "base_model", "dtype", "layers": [{key, d_in, d_out, r, off_A, off_B}],
                 "meta", "payload_sha256"}
  padding       zero bytes up to the next multiple of 256 -> payload_start
  payload       per layer in order: A[r, d_in] then B[d_out, r], bf16 row-major. Each array starts at a
                256-byte boundary relative to payload_start. payload_sha256 covers the payload region,
                inner padding included, and not the trailing bytes after the last array (there are none).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterator

import numpy as np
import torch

MAGIC = b"LMXBLOB1"
ALIGN = 256
VERSION = 1
_PREFIX = struct.Struct("<8sQ")
_CHUNK = 1 << 24


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _bf16_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.int16).numpy().astype("<i2", copy=False).tobytes()


def _payload_chunks(items: list[tuple[int, torch.Tensor]]) -> Iterator[bytes]:
    """Yield the payload in file order: zero padding up to each offset, then the bf16 bytes of each array."""
    pos = 0
    for off, t in items:
        if off > pos:
            yield bytes(off - pos)
        data = _bf16_bytes(t)
        yield data
        pos = off + len(data)


def _check_pair(key: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"{key}: A and B must be 2-D")
    if a.shape[0] != b.shape[1]:
        raise ValueError(f"{key}: A has rank {a.shape[0]} but B has rank {b.shape[1]}")


def _payload_nbytes(layers: list[dict]) -> int:
    return max((max(lay["off_A"] + lay["r"] * lay["d_in"] * 2, lay["off_B"] + lay["d_out"] * lay["r"] * 2)
                for lay in layers), default=0)


def _read_prefix(f) -> tuple[dict, int]:
    raw = f.read(_PREFIX.size)
    if len(raw) < _PREFIX.size:
        raise ValueError("file too short to be a blob")
    magic, header_len = _PREFIX.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
    header = json.loads(f.read(header_len).decode("utf-8"))
    if header.get("version") != VERSION:
        raise ValueError(f"unsupported blob version {header.get('version')!r}")
    return header, _align(_PREFIX.size + header_len)


def write_blob(path, adapter: dict[str, tuple[torch.Tensor, torch.Tensor]], base_model: str,
               meta: dict | None = None, order: list[str] | None = None) -> dict:
    """Write adapter {key: (A[r, d_in], B[d_out, r])} as bf16 at native rank. Returns the header written."""
    if not adapter:
        raise ValueError("cannot write an empty adapter")
    keys = list(adapter) if order is None else list(order)
    if sorted(keys) != sorted(adapter):
        raise ValueError("order must list every adapter key exactly once")
    layers, items, cursor = [], [], 0
    for key in keys:
        a, b = adapter[key]
        _check_pair(key, a, b)
        a16 = a.detach().to(torch.bfloat16).contiguous().cpu()
        b16 = b.detach().to(torch.bfloat16).contiguous().cpu()
        r, d_in = a16.shape
        d_out = b16.shape[0]
        cursor = _align(cursor)
        off_a = cursor
        cursor += a16.numel() * 2
        cursor = _align(cursor)
        off_b = cursor
        cursor += b16.numel() * 2
        layers.append({"key": key, "d_in": d_in, "d_out": d_out, "r": r, "off_A": off_a, "off_B": off_b})
        items += [(off_a, a16), (off_b, b16)]
    sha = hashlib.sha256()
    for chunk in _payload_chunks(items):
        sha.update(chunk)
    header = {"version": VERSION, "base_model": base_model, "dtype": "bfloat16", "layers": layers,
              "meta": dict(meta or {}), "payload_sha256": sha.hexdigest()}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_start = _align(_PREFIX.size + len(header_bytes))
    # Write next to the target and rename into place: a reader that has the old file mmapped keeps a valid
    # inode instead of seeing the file truncated underneath it (SIGBUS).
    path = os.fspath(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(_PREFIX.pack(MAGIC, len(header_bytes)))
            f.write(header_bytes)
            f.write(bytes(payload_start - _PREFIX.size - len(header_bytes)))
            for chunk in _payload_chunks(items):
                f.write(chunk)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return header


def read_header(path) -> dict:
    with open(path, "rb") as f:
        header, _ = _read_prefix(f)
    return header


def blob_nbytes(path) -> int:
    """Payload bytes only: from payload_start to the end of the last array."""
    return _payload_nbytes(read_header(path)["layers"])


def _verify(path, payload_start: int, nbytes: int, expected: str) -> None:
    h = hashlib.sha256()
    remaining = nbytes
    with open(path, "rb") as f:
        f.seek(payload_start)
        while remaining:
            chunk = f.read(min(remaining, _CHUNK))
            if not chunk:
                raise ValueError("payload truncated")
            h.update(chunk)
            remaining -= len(chunk)
    if h.hexdigest() != expected:
        raise ValueError("payload checksum mismatch (payload_sha256)")


def _view(words: np.ndarray, off: int, rows: int, cols: int) -> torch.Tensor:
    flat = torch.from_numpy(np.asarray(words[off // 2: off // 2 + rows * cols]))
    return flat.view(torch.bfloat16).view(rows, cols)


def read_blob(path, mmap: bool = True, verify: bool = False) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load {key: (A, B)} as bf16 tensors. mmap=True shares the file pages without copying (copy-on-write)."""
    with open(path, "rb") as f:
        header, payload_start = _read_prefix(f)
    if header.get("dtype") != "bfloat16":
        raise ValueError(f"unsupported dtype {header.get('dtype')!r}")
    nbytes = _payload_nbytes(header["layers"])
    if verify:
        _verify(path, payload_start, nbytes, header["payload_sha256"])
    if mmap:
        words = np.memmap(path, dtype="<i2", mode="c", offset=payload_start, shape=(nbytes // 2,))
    else:
        words = np.fromfile(path, dtype="<i2", count=nbytes // 2, offset=payload_start)
    out = {}
    for layer in header["layers"]:
        a = _view(words, layer["off_A"], layer["r"], layer["d_in"])
        b = _view(words, layer["off_B"], layer["d_out"], layer["r"])
        out[layer["key"]] = (a, b)
    return out


def convert(src, out_path, keymap, base_model: str, *, fused_groups: dict[str, list[str]] | None = None,
            **normalize_kwargs) -> dict:
    """Adapter file / PEFT dir / state dict -> normalize -> optional fused composition -> .lmx blob.

    fused_groups: {engine_key: [member keys in output order]}. Members are canonical (engine) keys produced by the
    keymap (e.g. to_q, to_k, to_v); they are removed and replaced by one composed layer (compose_fused: A
    concatenated along rank, B block-diagonal, rank = sum). All members must be present, or none.
    The file is written via a temp file and os.replace. Returns the blob header.
    """
    from .formats import load_lora, normalize
    from .formats.normalize import compose_fused

    config = None
    if isinstance(src, dict):
        sd = src
    else:
        sd, config = load_lora(src)
    if config is not None and "config" not in normalize_kwargs:
        normalize_kwargs["config"] = config
    canonical, report = normalize(sd, keymap, **normalize_kwargs)
    for fused_key, members in (fused_groups or {}).items():
        present = [m in canonical for m in members]
        if not any(present):
            continue
        if not all(present):
            missing = [m for m, p in zip(members, present, strict=True) if not p]
            raise ValueError(f"fused group {fused_key}: members {missing} are missing from the adapter")
        if fused_key in canonical:
            raise ValueError(f"fused group {fused_key}: key already present in the adapter")
        canonical[fused_key] = compose_fused([tuple(t.float() for t in canonical.pop(m)) for m in members])
    meta = {"format": report["format"], "source": "" if isinstance(src, dict) else os.path.basename(os.fspath(src)),
            "matched": report["matched"]}
    return write_blob(out_path, canonical, base_model, meta=meta)


def composite_from_blobs(items: list[tuple[str, float]]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """[(blob path, strength), ...] -> one canonical adapter, per layer via compose_multi.

    A layer present in only some of the adapters is composed from those alone. Computed in fp32, returned bf16."""
    from .formats.normalize import compose_multi

    per_layer: dict[str, list] = {}
    for path, strength in items:
        for key, (a, b) in read_blob(path).items():
            per_layer.setdefault(key, []).append((a.float(), b.float(), float(strength)))
    out = {}
    for key, parts in per_layer.items():
        a, b = compose_multi(parts)
        out[key] = (a.to(torch.bfloat16), b.to(torch.bfloat16))
    return out
