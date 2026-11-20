import pytest
import torch

pytest.importorskip("diffusers")
from tiny_models import tiny_qwen_image

from loramux import AdapterPool
from loramux.integrations.diffusers import disable, enable, mixed_lora

N_IMG, N_TXT, HW = 12, 6, (3, 4)


def _inputs(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"hidden_states": torch.randn(rows, N_IMG, 16, generator=g),
            "encoder_hidden_states": torch.randn(rows, N_TXT, 24, generator=g),
            "timestep": torch.full((rows,), 0.5), "img_shapes": [[(1, *HW)]] * rows}


def _run(model, inp):
    with torch.no_grad():
        return model(**inp, return_dict=False)[0]


def _slice(inp, rows):
    out = {k: (v[rows] if torch.is_tensor(v) else [v[i] for i in range(len(v))][:len(rows)])
           for k, v in inp.items()}
    return out


def _setup(targets, n_adapters=2, r=4):
    model = tiny_qwen_image()
    pool = AdapterPool({}, slots=3, r_max=16, device="cpu", dtype=torch.float32)
    keys = enable(model, pool, targets)
    g = torch.Generator().manual_seed(11)
    for a in range(n_adapters):
        pool.register(f"ad{a}", {k: (torch.randn(r + a, pool.layer_shapes[k][0], generator=g) * .3,
                                     torch.randn(pool.layer_shapes[k][1], r + a, generator=g) * .3) for k in keys})
        pool.ensure_resident(f"ad{a}")
    return model, pool, keys


@pytest.mark.parametrize("targets", ["attn", "all"])
def test_mixed_batch_equals_solo_runs(targets):
    model, pool, _ = _setup(targets)
    slots = [pool.slot_of["ad0"], -1, pool.slot_of["ad1"]]
    scales = [1.0, 1.0, 0.7]
    inp = _inputs(3)
    with mixed_lora(model, slots, scales):
        mixed = _run(model, inp)
    for i in range(3):
        with mixed_lora(model, [slots[i]], [scales[i]]):
            solo = _run(model, _slice(inp, [i]))
        torch.testing.assert_close(mixed[i:i + 1], solo, atol=1e-4, rtol=1e-4)
    base = _run(model, inp)                         # outside the context: base model
    torch.testing.assert_close(base[1], mixed[1], atol=1e-6, rtol=0)      # the base-model row is untouched
    assert not torch.allclose(base[0], mixed[0], atol=1e-3)      # the adapter has an effect
    assert not torch.allclose(base[2], mixed[2], atol=1e-3)


def test_cfg_concatenated_batch_and_twice_called_agree():
    model, pool, _ = _setup("all")
    slots = [pool.slot_of["ad0"], pool.slot_of["ad1"]]
    scales = [1.0, 0.5]
    cond, uncond = _inputs(2, seed=1), _inputs(2, seed=2)
    cat = {k: (torch.cat([cond[k], uncond[k]]) if torch.is_tensor(cond[k]) else cond[k] + uncond[k])
           for k in cond}
    with mixed_lora(model, slots, scales):                   # concatenated: rows == 2 * n_req
        both = _run(model, cat)
        c, u = _run(model, cond), _run(model, uncond)         # twice called: rows == n_req each
    torch.testing.assert_close(both, torch.cat([c, u]), atol=1e-4, rtol=1e-4)
    # and equals solo runs of each row
    for j in range(2):
        with mixed_lora(model, [slots[j]], [scales[j]]):
            torch.testing.assert_close(c[j:j + 1], _run(model, _slice(cond, [j])), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(u[j:j + 1], _run(model, _slice(uncond, [j])), atol=1e-4, rtol=1e-4)


def test_cfg_interleaved_layout():
    model, pool, _ = _setup("attn")
    slots, scales = [pool.slot_of["ad0"], pool.slot_of["ad1"]], [1.0, 1.0]
    inp = _inputs(4, seed=3)                                  # rows: c0, u0, c1, u1
    with mixed_lora(model, slots, scales, cfg_layout="interleaved"):
        out = _run(model, inp)
    for row in range(4):
        with mixed_lora(model, [slots[row // 2]], [1.0]):
            torch.testing.assert_close(out[row:row + 1], _run(model, _slice(inp, [row])), atol=1e-4, rtol=1e-4)


def test_wrong_batch_size_raises_and_context_restored():
    from loramux.context import current
    model, pool, _ = _setup("attn")
    with mixed_lora(model, [pool.slot_of["ad0"]] * 2, [1.0, 1.0]), pytest.raises(ValueError, match="expected 2 or 4"):
        _run(model, _inputs(3))
    assert current() is None
    with mixed_lora(model, [pool.slot_of["ad0"]], [1.0]):
        _run(model, _inputs(1))
        assert current() is None                              # set only for the duration of each call
    assert not model._forward_pre_hooks and not model._forward_hooks


def test_disable_restores_outputs_bit_exactly():
    model = tiny_qwen_image()
    inp = _inputs(2)
    before = _run(model, inp)
    pool = AdapterPool({}, slots=2, r_max=8, device="cpu", dtype=torch.float32)
    keys = enable(model, pool, "all")
    assert keys and all(k in pool.layer_shapes for k in keys)
    with pytest.raises(RuntimeError, match="already enabled"):
        enable(model, pool, "attn")
    assert sorted(disable(model)) == sorted(keys)
    assert disable(model) == []
    assert torch.equal(_run(model, inp), before)
    assert not any(type(m).__name__ == "MuxLinear" for m in model.modules())
    with pytest.raises(RuntimeError, match="enable"), mixed_lora(model, [0], [1.0]):
        pass


def test_target_patterns():
    model = tiny_qwen_image()
    pool = AdapterPool({}, slots=1, r_max=8, device="cpu", dtype=torch.float32)
    keys = enable(model, pool, "attn")
    assert keys and all(".attn." in k for k in keys) and "transformer_blocks.0.attn.to_out.0" in keys
    disable(model)
    keys = enable(model, pool, [r"img_mlp\.net\.2$"])
    assert sorted(keys) == ["transformer_blocks.0.img_mlp.net.2", "transformer_blocks.1.img_mlp.net.2"]
    disable(model)
    with pytest.raises(ValueError, match="match no"):
        enable(model, pool, [r"nothing_here"])
