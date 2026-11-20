import pytest
import torch

from bench.common import read_csv, write_csv
from bench.profile_steps import profile_steps, profile_target
from bench.step_bench import LoRAHarness, StepTarget, ToyDiTTarget, linear_shapes, run_step_bench
from loramux import build_meta, lora_batch


def test_toy_is_a_step_target_and_attn_filter():
    t = ToyDiTTarget()
    assert isinstance(t, StepTarget)
    attn, full = linear_shapes(t.model, "attn"), linear_shapes(t.model, "all")
    assert len(attn) == 8 and len(full) == 12 and set(attn) < set(full)
    assert all(".attn." in k for k in attn)


def test_harness_mixed_equals_solo_and_unpatch():
    t = ToyDiTTarget()
    x = t.make_inputs(3)
    base = t.forward(**x)
    h = LoRAHarness(t, targets="all", n_adapters=3, rank=4)
    with torch.no_grad():
        mixed = h.step_fn(x, 3, 3)()
        assert not torch.allclose(mixed, base, atol=1e-4)
        for i in range(3):
            xi = {k: v[i:i + 1] for k, v in x.items()}
            with lora_batch(build_meta([h.pool.slot_of[f"ad{i}"]], [1.0])):
                solo = t.forward(**xi)
            torch.testing.assert_close(mixed[i:i + 1], solo, atol=1e-4, rtol=1e-4)
        cfg_inputs = t.make_inputs(3, 2)
        assert h.step_fn(cfg_inputs, 3, 2, rows_per_req=2)().shape[0] == 6
    h.close()
    torch.testing.assert_close(t.forward(**x), base, atol=0, rtol=0)  # unpatched: bit-exact base


def test_run_step_bench_smoke():
    rows = run_step_bench(ToyDiTTarget(), bs=4, rank=4, modes=("eager", "compile"), compile_backend="aot_eager",
                          rounds=1, warmup=1, iters=2)
    assert len(rows) == 6  # (base, attn, all) x (eager, compile)
    assert {(r["adapters"], r["mode"]) for r in rows} == {(a, m) for a in ("base", "attn", "all")
                                                         for m in ("eager", "compile")}
    assert all(r["median_ms"] > 0 and r["n"] == 2 for r in rows)
    assert all(r["overhead_pct"] == 0.0 for r in rows if r["adapters"] == "base")


def test_profile_steps_csv_contract(tmp_path):
    rows = profile_steps(lambda res: ToyDiTTarget(n_img=8), resolutions=(512, 1024), batch_sizes=(1, 2, 4),
                         k_values=(2, 4), adapter_sets=("all",), rank=4, rounds=1, warmup=0, iters=1)
    cols = ["model", "resolution", "bs", "rows_per_req", "kind", "k", "targets", "rank", "mode", "median_ms",
            "p90_ms", "mean_ms", "n", "timer"]
    assert all(list(r) == cols for r in rows)
    for res in (512, 1024):
        sel = [r for r in rows if r["resolution"] == res]
        got = {(r["bs"], r["kind"], r["k"]) for r in sel}
        want = {(1, "base", 0), (2, "base", 0), (4, "base", 0), (1, "uniform", 1), (2, "uniform", 1),
                (4, "uniform", 1), (2, "mixed", 2), (4, "mixed", 2), (4, "mixed", 4)}
        assert got == want
    p = tmp_path / "p.csv"
    write_csv(p, rows, {"x": "1"})
    _env, back = read_csv(p)
    assert len(back) == len(rows) and back[0]["kind"] == "base"


def test_profile_target_cfg_rows():
    rows = profile_target(ToyDiTTarget(), batch_sizes=(2,), k_values=(2,), rank=4, rows_per_req=2, rounds=1,
                          warmup=0, iters=1)
    assert {(r["kind"], r["k"]) for r in rows} == {("base", 0), ("uniform", 1), ("mixed", 2)}


def test_diffusers_target_tiny_qwen():
    pytest.importorskip("diffusers")
    from diffusers import QwenImageTransformer2DModel

    from bench.step_bench import DiffusersStepTarget
    model = QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4, num_layers=2,
                                        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
                                        axes_dims_rope=(2, 2, 4))
    t = DiffusersStepTarget("qwen_image", model=model, resolution=64, txt_len=6, device="cpu", dtype=torch.float32)
    x = t.make_inputs(2)
    base = t.forward(**x)
    assert base.shape[:2] == (2, 16)
    h = LoRAHarness(t, targets="all", n_adapters=2, rank=4)
    with torch.no_grad():
        out = h.step_fn(x, 2, 2)()
    h.close()
    assert out.shape == base.shape and not torch.allclose(out, base, atol=1e-5)


def test_diffusers_target_tiny_flux2():
    pytest.importorskip("diffusers")
    from diffusers import Flux2Transformer2DModel

    from bench.step_bench import DiffusersStepTarget
    model = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=1,
                                    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
                                    timestep_guidance_channels=32, axes_dims_rope=(4, 4, 4, 4))
    t = DiffusersStepTarget("flux2", model=model, resolution=64, txt_len=5, device="cpu", dtype=torch.float32)
    x = t.make_inputs(2)
    base = t.forward(**x)
    h = LoRAHarness(t, targets="all", n_adapters=2, rank=4)
    with torch.no_grad():
        out = h.step_fn(x, 2, 2)()
    h.close()
    assert out.shape == base.shape and not torch.allclose(out, base, atol=1e-5)
