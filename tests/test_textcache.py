import pytest
import torch

from loramux.textcache import TextEmbedCache, bypass, nbytes


def _t(numel: int, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(numel, dtype=dtype)  # float32: 4 bytes per element


def test_lru_order_on_entry_limit():
    c = TextEmbedCache(max_entries=2)
    c.put("a", _t(1))
    c.put("b", _t(1))
    assert c.get("a") is not None          # a becomes most recent
    c.put("c", _t(1))                      # evicts b, the least recent
    assert c.get("b") is None
    assert c.get("a") is not None and c.get("c") is not None
    assert c.stats()["evictions"] == 1


def test_byte_limit_evicts_until_it_holds():
    c = TextEmbedCache(max_entries=10, max_bytes=100)
    c.put("k1", _t(10))   # 40 bytes
    c.put("k2", _t(10))   # 80 bytes
    c.put("k3", _t(10))   # 120 > 100, evict k1
    assert c.get("k1") is None
    s = c.stats()
    assert s["bytes"] == 80 and s["entries"] == 2 and s["evictions"] == 1


def test_oversize_item_is_not_cached_and_drops_old_entry():
    c = TextEmbedCache(max_entries=10, max_bytes=50)
    c.put("k", _t(10))            # 40 bytes, cached
    c.put("k", _t(100))           # 400 bytes alone: not cached
    assert c.get("k") is None
    assert c.stats()["entries"] == 0 and c.stats()["bytes"] == 0


def test_nested_value_sizes():
    value = (_t(10), {"x": _t(8, torch.bfloat16)}, [_t(2, torch.int64)])
    assert nbytes(value) == 40 + 16 + 16
    c = TextEmbedCache(max_entries=4)
    c.put(("enc", "p", 77), value)
    assert c.stats()["bytes"] == 72
    assert c.get(("enc", "p", 77)) is value


def test_nbytes_rejects_unknown_type():
    with pytest.raises(TypeError):
        nbytes("not a tensor")


def test_stats_counts_hits_and_misses():
    c = TextEmbedCache(max_entries=4)
    c.put("a", _t(1))
    c.get("a")
    c.get("a")
    c.get("zzz")
    s = c.stats()
    assert s == {"hits": 2, "misses": 1, "evictions": 0, "entries": 1, "bytes": 4}


def test_bypass_for_text_encoder_lora():
    assert bypass(True) is True
    assert bypass(False) is False


def test_constructor_validates_limits():
    with pytest.raises(ValueError):
        TextEmbedCache(max_entries=0)
