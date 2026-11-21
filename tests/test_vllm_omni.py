"""vLLM-Omni mixed mode against stand-ins for vLLM's linears, pipeline and worker (no vllm_omni needed)."""
import dataclasses
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from loramux import AdapterPool
from loramux.integrations.vllm_omni import MixedLoRA, drop_lora_identity, fused_groups

D = 16
BF16_ATOL = 0.15   # adapters are stored as bf16 (alpha/r folded in before the cast); the LoRA delta itself is O(1)


class FakeLinear(nn.Module):
    """vLLM LinearBase stand-in: forward returns (out, output_bias)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
        self.input_size, self.output_size = d_in, d_out

    def forward(self, x):
        return x @ self.weight.T, None


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv, self.to_out = FakeLinear(D, 3 * D), FakeLinear(D, D)

    def forward(self, x):
        qkv, _ = self.to_qkv(x)
        o, _ = self.to_out(qkv[..., :D] + qkv[..., D:2 * D] + qkv[..., 2 * D:])
        return o


class Model(nn.Module):
    packed_modules_mapping = MappingProxyType({"to_qkv": ["to_q", "to_k", "to_v"]})

    def __init__(self):
        super().__init__()
        self.attn = Attn()

    def forward(self, x):
        return self.attn(x)

    def denoise_step(self, input_batch):
        return self(input_batch.x)


def _peft(seed, members=("to_q", "to_k", "to_v", "to_out"), r=4):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for m in members:
        d_out = D
        sd[f"attn.{m}.lora_A.weight"] = torch.randn(r, D, generator=g) / D ** .5
        sd[f"attn.{m}.lora_B.weight"] = torch.randn(d_out, r, generator=g) * .3
    return sd, {"lora_alpha": r, "r": r}


def _dense(model, sd, cfg, scale):
    """Reference: the adapter merged into dense weights of a copy of the model."""
    import copy
    m = copy.deepcopy(model)
    s = cfg["lora_alpha"] / cfg["r"] * scale
    W = m.attn.to_qkv.weight.data
    for i, name in enumerate(("to_q", "to_k", "to_v")):
        if f"attn.{name}.lora_A.weight" in sd:
            W[i * D:(i + 1) * D] += s * sd[f"attn.{name}.lora_B.weight"] @ sd[f"attn.{name}.lora_A.weight"]
    if "attn.to_out.lora_A.weight" in sd:
        m.attn.to_out.weight.data += s * sd["attn.to_out.lora_B.weight"] @ sd["attn.to_out.lora_A.weight"]
    return m


def _setup(files):
    torch.manual_seed(0)
    model = Model()
    ref = Model()
    ref.load_state_dict(model.state_dict())
    pool = AdapterPool({}, slots=3, r_max=16, device="cpu", dtype=torch.float32)
    mux = MixedLoRA(model, pool, [r"attn\.(to_qkv|to_out)$"], load_fn=lambda p: files[p])
    return model, ref, pool, mux


def test_fused_groups_and_layer_shapes():
    model, _, pool, _ = _setup({})
    assert fused_groups(model.__class__()) == {"attn.to_qkv": (["attn.to_q", "attn.to_k", "attn.to_v"], [D] * 3)}
    assert pool.layer_shapes == {"attn.to_qkv": (D, 3 * D), "attn.to_out": (D, D)}


def test_mixed_batch_equals_per_request_merged_weights():
    files = {"a": _peft(1), "b": _peft(2, members=("to_q", "to_v")), "c": _peft(3, members=("to_out",))}
    model, ref, _, mux = _setup(files)
    reqs = {f"r{i}": (SimpleNamespace(lora_int_id=ord(p), lora_path=p) if p else None, sc)
            for i, (p, sc) in enumerate([("a", 1.0), ("b", .5), (None, 1.0), ("c", 2.0), ("a", 1.0)])}
    for rid, (lr, sc) in reqs.items():
        mux.note_request(rid, lr, sc)
    x = torch.randn(5, 7, D)
    with mux.batch(list(reqs)):
        y = model.denoise_step(SimpleNamespace(x=x))
    for i, (rid, (lr, sc)) in enumerate(reqs.items()):
        exp = ref(x[i:i + 1]) if lr is None else _dense(ref, *files[{0: "a", 1: "b", 3: "c", 4: "a"}[i]], sc)(x[i:i + 1])
        torch.testing.assert_close(y[i:i + 1], exp, atol=BF16_ATOL, rtol=3e-2)
        if lr is not None:   # the adapter must matter, or the tolerance above proves nothing
            assert (exp - ref(x[i:i + 1])).abs().max() > 3 * BF16_ATOL


def test_requests_pin_until_forgotten_and_outputs_repeat():
    files = {"a": _peft(1)}
    model, ref, pool, mux = _setup(files)
    lr = SimpleNamespace(lora_int_id=7, lora_path="a")
    mux.note_request("r0", lr)
    x = torch.randn(2, 3, D)   # num_outputs_per_prompt = 2 rows for one request
    with mux.batch(["r0"], repeat=2):
        y = model.denoise_step(SimpleNamespace(x=x))
    torch.testing.assert_close(y[0], y[0] * 0 + _dense(ref, *files["a"], 1.0)(x[:1])[0], atol=BF16_ATOL, rtol=3e-2)
    assert pool.refs["lora:7"] == 1
    mux.forget("r0")
    assert pool.refs["lora:7"] == 0


def test_attach_wraps_worker_step_and_request_paths():
    files = {"a": _peft(1)}
    model, ref, pool, mux = _setup(files)
    lr = SimpleNamespace(lora_int_id=1, lora_path="a")
    pipe = SimpleNamespace(denoise_step=lambda ib: model(ib.x), forward=lambda rb: model(rb.x))
    worker = SimpleNamespace(lora_manager=object(), model_runner=SimpleNamespace(pipeline=pipe))
    mux.attach(worker)
    assert worker.lora_manager is None
    new = SimpleNamespace(request_id="s0", req=SimpleNamespace(sampling_params=SimpleNamespace(lora_request=lr, lora_scale=1.0)))
    worker._activate_step_lora(SimpleNamespace(finished_req_ids=[], scheduled_new_reqs=[new]))
    x = torch.randn(1, 4, D)
    exp = _dense(ref, *files["a"], 1.0)(x)
    torch.testing.assert_close(pipe.denoise_step(SimpleNamespace(x=x, request_ids=["s0"])), exp, atol=BF16_ATOL, rtol=3e-2)
    worker._activate_step_lora(SimpleNamespace(finished_req_ids=["s0"], scheduled_new_reqs=[]))
    assert pool.refs["lora:1"] == 0
    sp = SimpleNamespace(lora_request=lr, lora_scale=1.0, num_outputs_per_prompt=1)
    rb = SimpleNamespace(x=x, request_ids=["q0"], requests=[SimpleNamespace(request_id="q0", sampling_params=sp)],
                         sampling_params_list=[sp])
    torch.testing.assert_close(pipe.forward(rb), exp, atol=BF16_ATOL, rtol=3e-2)
    assert pool.refs["lora:1"] == 0   # request-mode requests are released when the forward returns


def test_step_mode_repeats_rows_per_request_outputs():
    """Two requests with two images each: rows are [a, a, b, b], not CFG-style [a, b, a, b]."""
    files = {"a": _peft(1), "b": _peft(2)}
    model, ref, _, mux = _setup(files)
    pipe = SimpleNamespace(denoise_step=lambda ib: model(ib.x), forward=lambda rb: model(rb.x))
    worker = SimpleNamespace(lora_manager=None, model_runner=SimpleNamespace(pipeline=pipe))
    mux.attach(worker)
    new = [SimpleNamespace(request_id=rid, req=SimpleNamespace(sampling_params=SimpleNamespace(
        lora_request=SimpleNamespace(lora_int_id=i, lora_path=p), lora_scale=1.0, num_outputs_per_prompt=2)))
        for i, (rid, p) in enumerate((("s0", "a"), ("s1", "b")))]
    worker._activate_step_lora(SimpleNamespace(finished_req_ids=[], scheduled_new_reqs=new))
    x = torch.randn(4, 3, D)
    y = pipe.denoise_step(SimpleNamespace(x=x, request_ids=["s0", "s1"]))
    for row, p in enumerate("aabb"):
        torch.testing.assert_close(y[row:row + 1], _dense(ref, *files[p], 1.0)(x[row:row + 1]), atol=BF16_ATOL, rtol=3e-2)


def test_drop_lora_identity_makes_keys_equal():
    @dataclasses.dataclass(frozen=True)
    class Key:
        steps: int = 20
        lora_int_id: int | None = None
        lora_scale: float = 1.0
    assert drop_lora_identity(Key(lora_int_id=3, lora_scale=.5)) == drop_lora_identity(Key(lora_int_id=9))


def test_tensor_parallel_linear_is_rejected():
    model = Model()
    model.attn.to_out.output_size_per_partition = D // 2
    with pytest.raises(NotImplementedError):
        MixedLoRA(model, AdapterPool({}, slots=1, r_max=16, device="cpu", dtype=torch.float32), [r"to_out"])
