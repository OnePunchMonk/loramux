"""MuxLinear column windows (fused attention+MLP projections) and the FLUX.2-style harness slicing."""
import torch
from torch import nn

from loramux import AdapterPool, build_meta, lora_batch
from loramux.layers import MuxLinear


def _setup(d_in=16, d_out=24, win_in=None, win_out=None, r=4):
    torch.manual_seed(0)
    lin = nn.Linear(d_in, d_out)
    wi = (win_in[1] - win_in[0]) if win_in else d_in
    wo = (win_out[1] - win_out[0]) if win_out else d_out
    pool = AdapterPool({"l": (wi, wo)}, slots=2, r_max=r, device="cpu", dtype=torch.float32)
    a, b = torch.randn(r, wi), torch.randn(wo, r)
    pool.register("ad", {"l": (a, b)})
    pool.ensure_resident("ad")
    return lin, pool, a, b


def test_out_slice_touches_only_window():
    lin, pool, a, b = _setup(win_out=(0, 9))
    mux = MuxLinear(lin, "l", pool, out_slice=(0, 9))
    x = torch.randn(2, 5, 16)
    with lora_batch(build_meta([pool.slot_of["ad"], -1], [1.0, 1.0])):
        y = mux(x)
    ref = lin(x)
    ref[0, :, :9] += x[0] @ a.T @ b.T
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_in_slice_reads_only_window():
    lin, pool, a, b = _setup(win_in=(0, 6))
    mux = MuxLinear(lin, "l", pool, in_slice=(0, 6))
    x = torch.randn(1, 5, 16)
    with lora_batch(build_meta([pool.slot_of["ad"]], [0.5])):
        y = mux(x)
    ref = lin(x) + 0.5 * x[..., :6] @ a.T @ b.T
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_attention_slices_on_flux2_like_block():
    from bench.step_bench import attention_slices

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_dim = 8
            self.to_qkv_mlp_proj = nn.Linear(8, 3 * 8 + 32)
            self.to_out = nn.Linear(8 + 16, 8)

    m = nn.Module()
    m.attn = Attn()
    shapes = {"attn.to_qkv_mlp_proj": (8, 56), "attn.to_out": (24, 8)}
    sl, sh = attention_slices(m, shapes)
    assert sl == {"attn.to_qkv_mlp_proj": (None, (0, 24)), "attn.to_out": ((0, 8), None)}
    assert sh == {"attn.to_qkv_mlp_proj": (8, 24), "attn.to_out": (8, 8)}


def test_noncontiguous_input_is_handled():
    """A permuted input (inner stride != 1) gives the same result as its contiguous copy."""
    lin, pool, a, b = _setup()
    mux = MuxLinear(lin, "l", pool)
    x = torch.randn(2, 16, 5).transpose(1, 2)      # [2, 5, 16], inner stride 5
    assert x.stride(-1) != 1
    with lora_batch(build_meta([pool.slot_of["ad"], -1], [1.0, 1.0])):
        y = mux(x)
        y_ref = mux(x.contiguous())
    torch.testing.assert_close(y, y_ref)
