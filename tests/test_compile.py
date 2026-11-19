import pytest, torch, torch.nn as nn
from loramux import AdapterPool, patch_model, build_meta, lora_batch


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
    with lora_batch(build_meta(slots, [1., 1., .7], 10)):
        y = m(x)
    for i in range(3):
        with lora_batch(build_meta([slots[i]], [[1., 1., .7][i]], 10)):
            yi = m(x[i:i + 1])
        torch.testing.assert_close(y[i:i + 1], yi, atol=1e-4, rtol=1e-4)


def test_compile_no_graph_breaks_cpu():
    m, pool = _build("cpu", torch.float32)
    x = torch.randn(2, 10, 64)
    meta = build_meta([pool.slot_of["ad0"], pool.slot_of["ad1"]], [1., 1.], 10)
    explain = torch._dynamo.explain
    with lora_batch(meta):
        ref = m(x)
        res = explain(m)(x)
    assert res.graph_break_count == 0
    comp = torch.compile(m, backend="eager")
    with lora_batch(meta):
        torch.testing.assert_close(comp(x), ref, atol=1e-5, rtol=1e-5)
