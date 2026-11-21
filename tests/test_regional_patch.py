"""Compiled forwards must still apply MuxLinear's in-place LoRA update (regional and full compile).

aot_eager runs AOTAutograd functionalization (what inductor does to in-place custom ops) without codegen, so a
dropped or detached in-place update shows up here on CPU."""
import pytest
import torch
import torch._dynamo.testing
import torch.nn.functional as F
from torch import nn

from loramux import AdapterPool, build_meta, lora_batch, patch_model, unpatch_model
from bench.step_bench import LoRAHarness, ToyDiTTarget, _ToyBlock
from loramux.layers import MuxLinear


def _run(scope: str | None, backend: str):
    torch._dynamo.reset()
    t = ToyDiTTarget()
    inp = t.make_inputs(2)
    fwd = t.forward
    with torch.no_grad():
        if scope == "regional":
            for m in t.model.modules():
                if isinstance(m, _ToyBlock):
                    m.compile(backend=backend)
        elif scope == "full":
            fwd = torch.compile(t.forward, backend=backend)
        base = fwd(**inp).clone()
        h = LoRAHarness(t, targets="all", n_adapters=2, rank=4, seed=0)
        try:
            lora = h.step_fn(inp, 2, 2, forward=fwd)().clone()
        finally:
            h.close()
    return base, lora


@pytest.mark.parametrize("scope", ["regional", "full"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_forward_applies_lora(scope, backend):
    base_e, lora_e = _run(None, "eager")
    assert not torch.allclose(base_e, lora_e), "synthetic adapters should change the output"
    base_c, lora_c = _run(scope, backend)
    torch.testing.assert_close(base_c, base_e, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(lora_c, base_c), f"{scope}/{backend}: LoRA not applied in the compiled forward"
    torch.testing.assert_close(lora_c, lora_e, atol=1e-4, rtol=1e-4)


def test_regional_compile_reuses_identical_patched_blocks_with_mixed_adapters():
    torch.manual_seed(0)
    target = ToyDiTTarget(d=16, layers=3, n_img=4, n_txt=2)
    shapes = {name: (mod.in_features, mod.out_features)
              for name, mod in target.model.named_modules() if isinstance(mod, torch.nn.Linear)}
    pool = AdapterPool(shapes, slots=3, r_max=4, device="cpu", dtype=torch.float32)
    generator = torch.Generator().manual_seed(1)
    for adapter in range(3):
        pool.register(
            f"ad{adapter}",
            {name: (torch.randn(4, d_in, generator=generator) * 0.1,
                    torch.randn(d_out, 4, generator=generator) * 0.1)
             for name, (d_in, d_out) in shapes.items()},
        )
        pool.ensure_resident(f"ad{adapter}")
    patch_model(target.model, [r".*"], pool)

    inputs = target.make_inputs(3)
    slots = [pool.slot_of[f"ad{i}"] for i in range(3)]
    meta = build_meta(slots, [1.0, 0.5, 0.75], rank_of_slot=pool.slot_rank, r_max=4)
    with torch.no_grad(), lora_batch(meta):
        eager = target.forward(**inputs).clone()

    torch._dynamo.reset()
    counter = torch._dynamo.testing.CompileCounterWithBackend("eager")
    for block in target.model.blocks:
        block.compile(backend=counter)
    with torch.no_grad(), lora_batch(meta):
        compiled = target.forward(**inputs)

    assert counter.frame_count == 1
    torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-5)


class _TwoStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Linear(4, 5)
        self.late = nn.Linear(5, 3)

    def forward(self, x, return_hidden=False):
        hidden = torch.tanh(self.early(x))
        out = self.late(hidden)
        return (hidden, out) if return_hidden else out


def _relayout_reference(model, x, adapters, scales):
    hidden, output = [], []
    for row, adapter, scale in zip(x, adapters, scales):
        a, b = (t.float() for t in adapter["early"])
        h = F.linear(row[None], model.early.base.weight, model.early.base.bias)
        h = torch.tanh(h + scale * (row[None] @ a.T @ b.T))
        c, d = (t.float() for t in adapter["late"])
        y = F.linear(h, model.late.base.weight, model.late.base.bias)
        y = y + scale * (h @ c.T @ d.T)
        hidden.append(h)
        output.append(y)
    return torch.cat(hidden), torch.cat(output)


def _run_relayout_case(dtype):
    model = _TwoStage()
    pool = AdapterPool({"early": (4, 5)}, slots=2, r_max=4, device="cpu", dtype=dtype)
    patch_model(model, [r"^early$"], pool)
    patch_model(model, [r"^late$"], pool)
    assert model.early.pool_A is pool.A["early"]
    assert model.late.pool_A is pool.A["late"]

    generator = torch.Generator().manual_seed(13)
    adapters = []
    for key in ("ad0", "ad1"):
        adapter = {
            "early": (torch.randn(3, 4, generator=generator), torch.randn(5, 3, generator=generator)),
            "late": (torch.randn(3, 5, generator=generator), torch.randn(3, 3, generator=generator)),
        }
        pool.register(key, adapter)
        pool.ensure_resident(key)
        adapters.append(pool.canonical(key))

    x = torch.randn(2, 4, generator=torch.Generator().manual_seed(21))
    scales = [1.0, 0.5]
    slots = [pool.slot_of["ad0"], pool.slot_of["ad1"]]
    with torch.no_grad(), lora_batch(build_meta(slots, scales, rank_of_slot=pool.slot_rank,
                                                r_max=pool.r_max, per_row_2d=True)):
        hidden, output = model(x, return_hidden=True)
    hidden_ref, output_ref = _relayout_reference(model, x, adapters, scales)
    # FP8 rounds the low-rank intermediate; a stale binding would drop the LoRA term entirely (error ~1).
    tol = 5e-3 if dtype == torch.float8_e4m3fn else 2e-4
    torch.testing.assert_close(hidden, hidden_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(output, output_ref, atol=tol, rtol=tol)

    keys = unpatch_model(model)
    assert keys == ["early", "late"]
    assert not pool._bound_modules
    with torch.no_grad():
        assert isinstance(model.early, nn.Linear) and isinstance(model.late, nn.Linear)
        base = F.linear(torch.tanh(F.linear(x, model.early.weight, model.early.bias)),
                        model.late.weight, model.late.bias)
        torch.testing.assert_close(model(x), base)


def test_relayout_rebinds_early_and_late_muxlinears_with_mixed_adapters():
    _run_relayout_case(torch.float32)


def test_relayout_rebinds_fp8_muxlinears_on_cpu():
    try:
        torch.empty(1, dtype=torch.float8_e4m3fn)
    except (RuntimeError, TypeError):
        pytest.skip("CPU torch build does not support float8_e4m3fn")
    _run_relayout_case(torch.float8_e4m3fn)


def test_muxlinear_binds_when_key_is_added_after_construction():
    base = nn.Linear(4, 3)
    pool = AdapterPool({}, slots=1, r_max=4, device="cpu", dtype=torch.float32)
    mux = MuxLinear(base, "late", pool)
    assert not mux._has_pool_layer
    pool.add_layers({"late": (4, 3)})
    assert mux._has_pool_layer and mux.pool_A is pool.A["late"]

    a = torch.randn(2, 4, generator=torch.Generator().manual_seed(31))
    b = torch.randn(3, 2, generator=torch.Generator().manual_seed(32))
    pool.register("ad", {"late": (a, b)})
    pool.ensure_resident("ad")
    x = torch.randn(1, 4, generator=torch.Generator().manual_seed(33))
    with torch.no_grad(), lora_batch(build_meta([pool.slot_of["ad"]], [1.0], per_row_2d=True)):
        y = mux(x)
    ref = base(x) + x @ a.T @ b.T
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
