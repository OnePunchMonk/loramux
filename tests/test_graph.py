"""Fixed-capacity metadata (L3.2). Everything but CUDA capture runs on CPU; capture/replay equality is gpu-marked."""
import pytest
import torch
from torch import nn

from loramux.context import lora_batch
from loramux.graph import GraphRunner, StaticMeta
from loramux.layers import MuxLinear
from loramux.metadata import build_meta
from tests.helpers import FixedPool
from tests.test_ops import D_IN, D_OUT, _stale_setup

CAP, N = 4, 6


def _layer(dev, dt, fused=False):
    torch.manual_seed(0)
    A, B, rank = _stale_setup(dev, dt)
    return MuxLinear(nn.Linear(D_IN, D_OUT).to(dev, dt), "l", FixedPool(A, B, rank), fused=fused)


def test_update_rewrites_in_place_and_pads_with_base():
    sm = StaticMeta(CAP, r_bucket=128, cfg_layout="interleaved")
    ptrs = (sm._slot.data_ptr(), sm._scale.data_ptr())
    sm.update([2, 0], [1.0, .5])
    assert sm.meta.rows_for(CAP).slot.tolist() == [2, 0, -1, -1]
    assert sm.meta.rows_for(2 * CAP).slot.tolist() == [2, 2, 0, 0, -1, -1, -1, -1]
    assert sm.meta.rows_for(CAP).scale.tolist() == [1.0, .5, 1.0, 1.0]
    sm.update([3], [2.0])
    assert sm.meta.rows_for(CAP).slot.tolist() == [3, -1, -1, -1]
    assert (sm._slot.data_ptr(), sm._scale.data_ptr()) == ptrs


def test_update_validates():
    sm = StaticMeta(2, r_bucket=16, rank_of_slot=lambda s: 64)
    with pytest.raises(ValueError, match="capacity"):
        sm.update([0, 1, 2], [1, 1, 1])
    with pytest.raises(ValueError, match="r_bucket"):
        sm.update([0], [1.0])
    sm.update([-1], [1.0])   # base-only requests never consult the rank


@pytest.mark.parametrize("fused", [False, True])
def test_runner_matches_eager_for_changing_batches(fused):
    lin = _layer("cpu", torch.float32, fused)
    runner = GraphRunner(lin, StaticMeta(CAP, r_bucket=128), torch.zeros(CAP, N, D_IN))
    for slots, scales in (([1, 3, 0, 2], [1., .5, 1., 2.]), ([3, -1], [1., 1.]), ([2], [.25])):
        x = torch.randn(CAP, N, D_IN)
        with lora_batch(build_meta(slots + [-1] * (CAP - len(slots)), scales + [1.0] * (CAP - len(scales)), r_max=128)):
            exp = lin(x)
        torch.testing.assert_close(runner(x, slots, scales), exp, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="pad to capacity"):
        runner(torch.zeros(CAP - 1, N, D_IN), [0], [1.0])


@pytest.mark.gpu
@pytest.mark.parametrize("fused", [False, True])
def test_gpu_graph_replay_matches_eager(fused):
    dev, dt = "cuda", torch.bfloat16
    lin = _layer(dev, dt, fused)
    runner = GraphRunner(lin, StaticMeta(CAP, r_bucket=128, device=dev), torch.zeros(CAP, N, D_IN, device=dev, dtype=dt))
    assert runner.graph is not None
    for slots, scales in (([1, 3, 0, 2], [1., .5, 1., 2.]), ([3, -1], [1., 1.]), ([2], [.25])):
        x = torch.randn(CAP, N, D_IN, device=dev, dtype=dt)
        with lora_batch(build_meta(slots + [-1] * (CAP - len(slots)), scales + [1.0] * (CAP - len(scales)), r_max=128, device=dev)):
            exp = lin(x)
        assert (runner(x, slots, scales).float() - exp.float()).abs().max() < 5e-2 * exp.float().abs().max()
