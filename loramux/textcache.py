"""LRU cache for prompt embeddings, bounded by entry count and total tensor bytes.

Keys are opaque and hashable; the intended key is (encoder_id, prompt, max_len).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


def nbytes(value: Any) -> int:
    """Bytes held by tensors in value: a tensor, or a tuple, list or dict of them (nested)."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(nbytes(v) for v in value)
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    raise TypeError(f"cannot size a value of type {type(value).__name__}")


def bypass(adapter_has_text_encoder: bool) -> bool:
    """True when a request carries a text-encoder LoRA. Such embeddings depend on the adapter, so skip the cache."""
    return bool(adapter_has_text_encoder)


class TextEmbedCache:
    def __init__(self, max_entries: int, max_bytes: int | None = None) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._items: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        """Value for key, or None on a miss. A hit becomes the most recently used entry."""
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return item[0]

    def put(self, key, value) -> None:
        """Insert or replace key, then evict least recently used entries until both limits hold.

        A value larger than max_bytes on its own is not cached. Any old entry for key is still removed.
        """
        self._discard(key)
        size = nbytes(value)
        if self.max_bytes is not None and size > self.max_bytes:
            return
        self._items[key] = (value, size)
        self._bytes += size
        while self._over_limits():
            self._evict_oldest()

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "entries": len(self._items), "bytes": self._bytes}

    def _over_limits(self) -> bool:
        return len(self._items) > self.max_entries or (self.max_bytes is not None and self._bytes > self.max_bytes)

    def _evict_oldest(self) -> None:
        _, (_, size) = self._items.popitem(last=False)
        self._bytes -= size
        self.evictions += 1

    def _discard(self, key) -> None:
        item = self._items.pop(key, None)
        if item is not None:
            self._bytes -= item[1]
