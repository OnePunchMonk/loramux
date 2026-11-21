"""Fused GEMM+LoRA epilogue (L3.1): must equal the unfused base-linear + lora_apply_ path."""
import pytest
import torch
from torch import nn

from loramux.context import lora_batch
from loramux.layers import MuxLinear
from loramux.metadata import build_meta
from tests.helpers import FixedPool
from tests.test_ops import D_IN, D_OUT, _stale_setup


def _run(fused, dev, dt, N, bias, slots, scales, z_bf16=False):
    torch.manual_seed(0)
    A, B, rank = _stale_setup(dev, dt)
    base = nn.Linear(D_IN, D_OUT, bias=bias).to(dev, dt)
    lin = MuxLinear(base, "l", FixedPool(A, B, rank), z_dtype=torch.bfloat16 if z_bf16 else torch.float32, fused=fused)
    x = torch.randn(len(slots), N, D_IN, device=dev, dtype=dt)
    with lora_batch(build_meta(slots, scales, device=dev, r_max=128)):
        return lin(x), x, lin


@pytest.mark.interp
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("N", [7, 33])
def test_interp_fused_equals_unfused(monkeypatch, bias, N):
    from loramux.ops import custom_op
    monkeypatch.setattr(custom_op, "TRITON_ON_CPU", True)
    slots, scales = [1, 4, -1, 3, 0], [1., 1., 1., .5, 2.]   # slot 4 has rank 0 in this layer, -1 is base only
    y_f, _, _ = _run(True, "cpu", torch.float32, N, bias, slots, scales)
    y_u, _, _ = _run(False, "cpu", torch.float32, N, bias, slots, scales)
    assert y_f.shape == (5, N, D_OUT) and torch.isfinite(y_f).all()   # stale NaN beyond rank must not leak
    torch.testing.assert_close(y_f, y_u, atol=1e-4, rtol=1e-4)


def test_cpu_reference_path_equals_unfused():
    slots, scales = [1, -1, 3], [1., 1., .5]
    y_f, _, _ = _run(True, "cpu", torch.float32, 9, True, slots, scales)
    y_u, _, _ = _run(False, "cpu", torch.float32, 9, True, slots, scales)
    torch.testing.assert_close(y_f, y_u, atol=1e-5, rtol=1e-5)


def test_fused_falls_back_without_batch_or_fp8():
    A, B, rank = _stale_setup("cpu", torch.float32)
    base = nn.Linear(D_IN, D_OUT)
    lin = MuxLinear(base, "l", FixedPool(A, B, rank), fused=True)
    x = torch.randn(2, 5, D_IN)
    torch.testing.assert_close(lin(x), base(x))   # no active batch: plain base


@pytest.mark.gpu
@pytest.mark.parametrize("N,z_bf16", [(1000, False), (77, True)])
def test_gpu_fused_equals_unfused(N, z_bf16):
    y_f, _, _ = _run(True, "cuda", torch.bfloat16, N, True, [0, 1, 2, 3, -1], [1., .5, 1., 1., 1.], z_bf16)
    y_u, _, _ = _run(False, "cuda", torch.bfloat16, N, True, [0, 1, 2, 3, -1], [1., .5, 1., 1., 1.], z_bf16)
    assert (y_f.float() - y_u.float()).abs().max() < 5e-2 * y_u.float().abs().max()
