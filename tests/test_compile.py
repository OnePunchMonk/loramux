import pytest
import torch
import torch._dynamo.testing
from torch import nn

from loramux import AdapterPool, build_meta, lora_batch, patch_model


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_q, self.to_out = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return x + self.to_out(torch.relu(self.to_q(x)))


class Toy(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _build(dev, dt):
    m = Toy().to(dev, dt)
    shapes = {f"blocks.{i}.{n}": (64, 64) for i in range(2) for n in ("to_q", "to_out")}
    pool = AdapterPool(shapes, slots=3, r_max=16, device=dev, dtype=dt)
    g = torch.Generator().manual_seed(0)
    for a in range(3):
        pool.register(f"ad{a}", {k: (torch.randn(8, 64, generator=g) / 8, torch.randn(64, 8, generator=g) * .1) for k in shapes})
        pool.ensure_resident(f"ad{a}")
    patch_model(m, [r"to_q$", r"to_out$"], pool)
    return m, pool


def test_mixed_batch_matches_solo_cpu():
    m, pool = _build("cpu", torch.float32)
    x = torch.randn(3, 10, 64)
    slots = [pool.slot_of["ad0"], -1, pool.slot_of["ad2"]]
    with lora_batch(build_meta(slots, [1., 1., .7])):
        y = m(x)
    for i in range(3):
        with lora_batch(build_meta([slots[i]], [[1., 1., .7][i]])):
            yi = m(x[i:i + 1])
        torch.testing.assert_close(y[i:i + 1], yi, atol=1e-4, rtol=1e-4)


def test_compile_no_graph_breaks_cpu():
    m, pool = _build("cpu", torch.float32)
    x = torch.randn(2, 10, 64)
    meta = build_meta([pool.slot_of["ad0"], pool.slot_of["ad1"]], [1., 1.])
    explain = torch._dynamo.explain
    with lora_batch(meta):
        ref = m(x)
        res = explain(m)(x)
    assert res.graph_break_count == 0
    comp = torch.compile(m, backend="eager")
    with lora_batch(meta):
        torch.testing.assert_close(comp(x), ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- L2.4: zero graph breaks, no per-value guards --


BACKENDS = ["eager", "aot_eager"]


def _batches(pool):
    s = [pool.slot_of[f"ad{i}"] for i in range(3)]
    return [([s[0], s[1]], [1., .5]), ([s[2], -1], [.7, 1.]), ([-1, s[0]], [1., 2.])]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cfg_layout", ["concat", "interleaved"])
def test_toy_compile_mixed_cfg_one_graph_no_recompiles(backend, cfg_layout):
    m, pool = _build("cpu", torch.float32)
    x = torch.randn(4, 10, 64)                       # 2 requests, CFG-duplicated rows
    torch._dynamo.reset()
    cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
    comp = torch.compile(m, backend=cnt, fullgraph=True)
    for slots, scales in _batches(pool):
        meta = build_meta(slots, scales, cfg_layout=cfg_layout, rank_of_slot=pool.slot_rank, r_max=pool.r_max)
        with torch.no_grad(), lora_batch(meta):
            ref = m(x)
            torch.testing.assert_close(comp(x), ref, atol=1e-5, rtol=1e-5)
    assert cnt.frame_count == 1                       # new slots / scales per batch: no recompiles
    with torch.no_grad(), lora_batch(build_meta(*_batches(pool)[0], cfg_layout=cfg_layout)):
        assert torch._dynamo.explain(m)(x).graph_break_count == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_diffusers_qwen_image_compile_mixed_cfg(backend):
    pytest.importorskip("diffusers")
    from test_diffusers_integration import _inputs, _run, _setup

    from loramux.integrations.diffusers import mixed_lora
    model, pool, _ = _setup("all")
    inp = _inputs(4)                                  # 2 requests x CFG (concat)
    torch._dynamo.reset()
    cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
    comp = torch.compile(model, backend=cnt)
    s0, s1 = pool.slot_of["ad0"], pool.slot_of["ad1"]
    for slots in ([s0, s1], [s1, -1], [-1, s0]):
        with mixed_lora(model, slots, [1., .7]):
            ref = _run(model, inp)
            with torch.no_grad():
                out = comp(**inp, return_dict=False)[0]
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert cnt.frame_count == 1
    with mixed_lora(model, [s0, s1], [1., .7]), torch.no_grad():
        res = torch._dynamo.explain(model)(**inp, return_dict=False)
    assert res.graph_break_count == 0 and res.graph_count == 1


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_forward_does_no_device_to_host_sync(compiled):
    m, pool = _build("cuda", torch.bfloat16)
    for k in ("ad0", "ad1", "ad2"):
        pool.wait(k)
    x = torch.randn(4, 10, 64, device="cuda", dtype=torch.bfloat16)
    fn = torch.compile(m, backend="aot_eager") if compiled else m
    slots, scales = _batches(pool)[0]
    with torch.no_grad():
        for _ in range(2):                            # warm up: autotuning and compilation may sync
            with lora_batch(build_meta(slots, scales, rank_of_slot=pool.slot_rank, r_max=16, device="cuda")):
                fn(x)
        torch.cuda.synchronize()
        meta = build_meta(slots, scales, rank_of_slot=pool.slot_rank, r_max=16, device="cuda")
        torch.cuda.set_sync_debug_mode("error")
        try:
            with lora_batch(meta):
                fn(x)
        finally:
            torch.cuda.set_sync_debug_mode("default")
