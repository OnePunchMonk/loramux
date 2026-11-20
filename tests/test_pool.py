import pytest
import torch

from loramux.pool import AdapterPool, composite_key


def _pool(S=2):
    return AdapterPool({"l": (8, 6)}, slots=S, r_max=4, device="cpu")


def _ad(r=2, v=1.0):
    return {"l": (torch.full((r, 8), v), torch.full((6, r), v))}


def test_lru_eviction_order():
    p = _pool()
    for k in "abc":
        p.register(k, _ad())
    p.ensure_resident("a"); p.ensure_resident("b")
    p.ensure_resident("a")            # a is now newest
    p.ensure_resident("c")            # evicts b
    assert "b" not in p.slot_of and {"a", "c"} == set(p.slot_of)


def test_refcount_pins():
    p = _pool()
    for k in "abc":
        p.register(k, _ad())
    p.ensure_resident("a"); p.acquire("a")
    p.ensure_resident("b"); p.acquire("b")
    with pytest.raises(RuntimeError):
        p.ensure_resident("c")
    p.release("a")
    p.ensure_resident("c")
    assert "a" not in p.slot_of


def test_slot_content_and_padding():
    p = _pool()
    p.register("a", _ad(r=2, v=3.0))
    p.ensure_resident("a")
    s = p.slot_of["a"]
    A, B, rank = p.buffers("l")
    assert int(rank[s]) == 2
    assert A[s, :2].float().eq(3).all() and A[s, 2:].eq(0).all()
    assert B[s, :, :2].float().eq(3).all() and B[s, :, 2:].eq(0).all()


def test_rank_over_rmax_rejected():
    p = _pool()
    with pytest.raises(ValueError):
        p.register("a", _ad(r=5))


def test_composite_key_order_independent():
    assert composite_key([("b", 1.0), ("a", .5)]) == composite_key([("a", .5), ("b", 1.0)])


def test_add_layers_and_host_rank():
    p = AdapterPool({}, slots=2, r_max=4, device="cpu")
    p.add_layers({"l": (8, 6)})
    p.add_layers({"l": (8, 6)})                      # idempotent
    with pytest.raises(ValueError):
        p.add_layers({"l": (8, 7)})
    p.register("a", _ad(r=3))
    p.ensure_resident("a")
    assert p.slot_rank(p.slot_of["a"]) == 3 and p.has_layer("l")


def test_failed_load_does_not_leak_slot():
    p = _pool(S=2)
    p.register("empty", {})            # an adapter with no layers: loading it fails
    p.register("ok", _ad())
    free_before = sorted(p.free)
    with pytest.raises(ValueError):
        p.ensure_resident("empty")
    assert sorted(p.free) == free_before and "empty" not in p.slot_of
    p.ensure_resident("ok")
    assert "ok" in p.slot_of
