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


def test_slot_content_and_layer_rank():
    p = _pool()
    p.register("a", _ad(r=2, v=3.0))
    p.ensure_resident("a")
    s = p.slot_of["a"]
    A, B, rank = p.buffers("l")
    assert int(rank[s]) == 2 and p.layer_rank_host["l"][s] == 2 and p.slot_rank(s) == 2
    assert A[s, :2].float().eq(3).all() and B[s, :, :2].float().eq(3).all()
    assert A[s, 2:].eq(0).all() and B[s, :, 2:].eq(0).all()      # the arena starts zeroed (loads never zero)


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


# ---------------------------------------------------------------- L2.3: arena, ranks, staging, readiness ------

SH = {"q": (8, 6), "k": (8, 6), "v": (8, 10), "o": (6, 8)}


def _rand_ad(ranks, seed=0, dt=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(r, SH[k][0], generator=g).to(dt), torch.randn(SH[k][1], r, generator=g).to(dt))
            for k, r in ranks.items()}


def _check_slot(p, key, ad):
    s = p.slot_of[key]
    for k in SH:
        A, B, rank = p.buffers(k)
        if k in ad:
            a, b = ad[k]
            r = a.shape[0]
            assert int(rank[s]) == r == p.layer_rank_host[k][s]
            torch.testing.assert_close(A[s, :r], a.to(A.dtype))
            torch.testing.assert_close(B[s, :, :r], b.to(B.dtype))
        else:
            assert int(rank[s]) == 0 == p.layer_rank_host[k][s]     # layer not in this adapter
    assert p.slot_rank(s) == max(a.shape[0] for a, _ in ad.values())


def test_arena_views_and_per_layer_ranks():
    p = AdapterPool(SH, slots=3, r_max=8, device="cpu", dtype=torch.float32)
    assert p.arena.shape == (3, p.slot_elems) and p.rank_all.shape == (4, 3)
    for k, (di, do) in SH.items():
        A, B, rank = p.buffers(k)
        assert A.shape == (3, 8, di) and B.shape == (3, do, 8) and rank.shape == (3,)
        assert A.untyped_storage().data_ptr() == p.arena.untyped_storage().data_ptr()
        assert A.stride(0) == B.stride(0) == p.slot_elems
    ad = _rand_ad({"q": 2, "v": 8, "o": 5})          # "k" missing
    p.register("a", ad)
    p.ensure_resident("a")
    _check_slot(p, "a", ad)
    assert p.stats()["h2d_bytes"] == sum((a.numel() + b.numel()) * 4 for a, b in ad.values())


def test_reload_into_used_slot_needs_no_zeroing():
    p = AdapterPool(SH, slots=1, r_max=8, device="cpu", dtype=torch.float32)
    big, small = _rand_ad({k: 8 for k in SH}, seed=1), _rand_ad({"q": 3, "o": 2}, seed=2)
    p.register("big", big)
    p.register("small", small)
    p.ensure_resident("big")
    p.ensure_resident("small")                       # evicts big, same slot
    _check_slot(p, "small", small)
    A, _, _ = p.buffers("q")
    assert torch.equal(A[0, 3:], big["q"][0][3:])     # stale rows beyond the rank stay (never read)


def test_per_layer_rank_masks_in_forward():
    from torch import nn

    from loramux import build_meta, lora_batch, patch_model

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.o = nn.Linear(8, 6), nn.Linear(6, 8)

        def forward(self, x):
            return self.o(torch.relu(self.q(x)))

    torch.manual_seed(0)
    m = M()
    p = AdapterPool({"q": (8, 6), "o": (6, 8)}, slots=1, r_max=8, device="cpu", dtype=torch.float32)
    stale = {"q": (torch.full((8, 8), 7.), torch.full((6, 8), 7.)), "o": (torch.full((8, 6), 7.), torch.full((8, 8), 7.))}
    p.register("stale", stale)
    p.ensure_resident("stale")
    g = torch.Generator().manual_seed(4)
    ad = {"q": (torch.randn(3, 8, generator=g), torch.randn(6, 3, generator=g))}     # "o" not targeted
    p.register("a", ad)
    p.ensure_resident("a")                            # reuses the slot; "o" keeps stale data but rank 0
    patch_model(m, ["^q$", "^o$"], p)
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        with lora_batch(build_meta([0, -1], [1.0, 1.0], rank_of_slot=p.slot_rank, r_max=8)):
            y = m(x)
        h = m.q.base(x)
        h[0] += x[0] @ (ad["q"][1] @ ad["q"][0]).T
        exp = m.o.base(torch.relu(h))
    torch.testing.assert_close(y, exp, atol=1e-5, rtol=1e-5)


def test_cpu_descriptor_path_equals_per_tensor_copies_from_blob(tmp_path):
    from loramux.blob import write_blob
    ad = _rand_ad({"q": 4, "k": 1, "v": 8}, seed=3, dt=torch.bfloat16)
    write_blob(tmp_path / "a.lmx", ad, "toy")
    for dt in (torch.bfloat16, torch.float32):
        p = AdapterPool(SH, slots=2, r_max=8, device="cpu", dtype=dt)
        p.register_blob("a", tmp_path / "a.lmx")
        p.ensure_resident("a")
        _check_slot(p, "a", ad)                      # bf16 payload cast into an fp32 arena by the scatter


def test_add_layers_relayout_keeps_contents():
    p = AdapterPool({"q": SH["q"], "o": SH["o"]}, slots=2, r_max=8, device="cpu", dtype=torch.float32)
    ad = _rand_ad({"q": 3, "o": 4})
    p.register("a", ad)
    p.ensure_resident("a")
    p.add_layers({"k": SH["k"], "v": SH["v"]})
    _check_slot(p, "a", ad)


def test_fp8_pool_register_and_blob_agree(tmp_path):
    from loramux.blob import write_blob
    from loramux.compress import dequantize_fp8_rows
    ad = _rand_ad({"q": 4, "v": 8}, seed=5, dt=torch.bfloat16)
    write_blob(tmp_path / "a.lmx", ad, "toy")
    pools = []
    for src in ("mem", "blob"):
        p = AdapterPool(SH, slots=2, r_max=8, device="cpu", dtype=torch.float8_e4m3fn)
        if src == "mem":
            p.register("a", ad)
        else:
            p.register_blob("a", tmp_path / "a.lmx")
        p.ensure_resident("a")
        pools.append(p)
    p1, p2 = pools
    s1, s2 = p1.slot_of["a"], p2.slot_of["a"]
    for k, (a, b) in ad.items():
        r = a.shape[0]
        assert torch.equal(p1.A[k][s1].view(torch.uint8), p2.A[k][s2].view(torch.uint8))
        sa, sb = p1.scales(k)
        torch.testing.assert_close(dequantize_fp8_rows(p1.A[k][s1, :r], sa[s1, :r]).float(), a.float(),
                                   rtol=0.07, atol=1e-3)
        torch.testing.assert_close(dequantize_fp8_rows(p1.B[k][s1, :, :r], sb[s1]).float(), b.float(),
                                   rtol=0.07, atol=1e-3)
        assert torch.equal(sb[s1], p2.scales(k)[1][s2])


class _FakeEvent:
    def __init__(self, done=False):
        self.done, self.syncs = done, 0

    def query(self):
        return self.done

    def synchronize(self):
        self.syncs += 1
        self.done = True


def test_readiness_rules():
    p = _pool()
    p.register("a", _ad())
    assert not p.is_ready("a")
    with pytest.raises(RuntimeError, match="not resident"):
        p.acquire("a")
    assert p.ensure_resident("a") is None and p.is_ready("a")      # CPU: loads are synchronous
    ev = p._events["a"] = _FakeEvent()                             # a GPU load still in flight
    assert not p.is_ready("a")
    with pytest.raises(RuntimeError, match="still loading"):
        p.acquire("a")
    assert p.ensure_resident("a") is ev                            # a hit returns the same event
    p.wait("a")
    assert ev.syncs == 1 and p.is_ready("a")
    assert p.acquire("a") == p.slot_of["a"]
    p.release("a")
    with pytest.raises(RuntimeError, match="without a matching acquire"):
        p.release("a")


def test_staging_ring_reuse_waits_for_event():
    from loramux.pool import _StagingRing
    ring = _StagingRing(2, "cpu", pin=False)
    ring.size = 64
    evs = []
    for _ in range(2):
        i = ring.take(32)
        evs.append(_FakeEvent())
        ring.events[i] = evs[-1]
    assert ring.order == [0, 1] and ring.host[0].numel() == 64 and ring.waits == 0
    evs[1].done = True
    i = ring.take(100)                     # buffer 0 is reused: must wait for its (pending) event, then grow
    assert i == 0 and evs[0].syncs == 1 and ring.waits == 1 and ring.host[0].numel() >= 100
    i = ring.take(8)                       # buffer 1's event already completed: no wait
    assert i == 1 and evs[1].syncs == 0 and ring.waits == 1 and ring.order == [0, 1, 0, 1]


def test_pool_staging_round_robin_and_size():
    p = AdapterPool(SH, slots=4, r_max=8, device="cpu", dtype=torch.float32, n_staging=3)
    for i in range(4):
        p.register(f"a{i}", _rand_ad({"q": 2 + i, "v": 8}, seed=i))
    size = p._ring.size
    assert size > 0
    for i in range(4):
        p.ensure_resident(f"a{i}")
    assert p._ring.order == [0, 1, 2, 0]
    assert all(h.numel() >= size for h in p._ring.host)


@pytest.mark.interp
def test_interp_triton_scatter_matches_torch(monkeypatch):
    from loramux.ops import custom_op
    from loramux.ops.triton_scatter import desc_tensor, scatter
    src = torch.randn(500)
    srcR = torch.tensor([3, 0, 7], dtype=torch.int32)
    rows = [(0, 5, 3, 40, 40, 40, 0), (120, 400, 37, 3, 3, 8, 0), (231, 77, 1, 200, 200, 200, 0),
            (0, 1, 3, 1, 1, 4, 2)]
    out = {}
    for tri in (False, True):
        dst = torch.zeros(1000)
        dstR = torch.zeros(12, dtype=torch.int32)
        scatter(desc_tensor(rows), rows, src.bfloat16(), dst, None, None, srcR, dstR, use_triton=tri)
        out[tri] = (dst, dstR)
    assert torch.equal(out[False][0], out[True][0]) and torch.equal(out[False][1], out[True][1])
    assert out[True][1].view(3, 4)[:, 1].tolist() == [3, 0, 7]
    assert torch.equal(out[True][0][400:400 + 37 * 8].view(37, 8)[:, :3], src[120:231].bfloat16().float().view(37, 3))
    # the pool's own load path through the kernel equals the torch path, FP8 scales included
    for dt in (torch.float32, torch.float8_e4m3fn):
        pools = []
        for tri in (False, True):
            monkeypatch.setattr(custom_op, "TRITON_ON_CPU", tri)
            p = AdapterPool(SH, slots=2, r_max=8, device="cpu", dtype=dt)
            p.register("a", _rand_ad({"q": 4, "k": 8, "o": 1}, seed=9))
            p.ensure_resident("a")
            pools.append(p)
        assert torch.equal(pools[0].arena.view(torch.uint8), pools[1].arena.view(torch.uint8))
        assert torch.equal(pools[0].rank_all, pools[1].rank_all)
        if dt == torch.float8_e4m3fn:
            assert torch.equal(pools[0].sarena, pools[1].sarena)


@pytest.mark.gpu
def test_gpu_async_load_event_and_contents():
    p = AdapterPool(SH, slots=2, r_max=8, device="cuda", dtype=torch.bfloat16)
    ad = _rand_ad({"q": 4, "v": 8, "o": 2}, seed=1, dt=torch.bfloat16)
    p.register("a", ad)
    ev = p.ensure_resident("a")
    assert isinstance(ev, torch.cuda.Event)
    p.wait("a")
    assert p.is_ready("a")
    _check_slot(p, "a", {k: (a.cuda(), b.cuda()) for k, (a, b) in ad.items()})
    stream = p._stream
    p.register("b", _rand_ad({"k": 3}, seed=2, dt=torch.bfloat16))
    p.ensure_resident("b")
    assert p._stream is stream          # persistent side stream
