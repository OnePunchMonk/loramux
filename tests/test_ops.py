"""Op tests. `interp` tests run the Triton kernels in the interpreter on CPU (fp32, small shapes); `gpu` tests
run them compiled in bf16. Both compare against the torch reference with per-layer ranks."""

import pytest
import torch

from loramux.metadata import build_meta
from loramux.ops.torch_ref import lora_apply_ref_

D_IN, D_OUT = 40, 70          # d_in % BK != 0 and d_out % BN != 0 for the interpreter tiles (BK=32, BN=64)
RANKS = [8, 16, 64, 128, 0]   # slot 4: this layer is not in that adapter


def _setup(dev, dt, S=4, R=64, d_in=96, d_out=200):
    g = torch.Generator().manual_seed(0)
    A = torch.randn(S, R, d_in, generator=g).div(d_in ** .5).to(dev, dt)
    B = (torch.randn(S, d_out, R, generator=g) * 1e-1).to(dev, dt)
    rank = torch.tensor([8, 16, 64, 32], dtype=torch.int32, device=dev)
    for s in range(S):
        A[s, rank[s]:] = 0
        B[s, :, rank[s]:] = 0
    return A, B, rank


def _stale_setup(dev, dt, ranks=RANKS, R=128, d_in=D_IN, d_out=D_OUT, seed=0):
    """Slots whose data beyond the layer's rank is NaN: any stale read poisons the output."""
    g = torch.Generator().manual_seed(seed)
    S = len(ranks)
    A = torch.full((S, R, d_in), float("nan"))
    B = torch.full((S, d_out, R), float("nan"))
    for s, r in enumerate(ranks):
        A[s, :r] = torch.randn(r, d_in, generator=g) / d_in ** .5
        B[s, :, :r] = torch.randn(d_out, r, generator=g) / max(r, 1) ** .5
    return A.to(dev, dt), B.to(dev, dt), torch.tensor(ranks, dtype=torch.int32, device=dev)


def _dense_expected(y, x, A, B, rank, slots, scales, starts, lens):
    exp = y.double().clone()
    for s, c, a, n in zip(slots, scales, starts, lens):
        r = int(rank[s]) if s >= 0 else 0
        if r == 0:
            continue
        exp[a:a + n] += c * (x[a:a + n].double() @ A[s, :r].double().T) @ B[s, :, :r].double().T
    return exp


def test_ref_matches_dense_delta_cpu():
    A, B, rank = _setup("cpu", torch.float32)
    m = build_meta([2, -1, 0], [0.5, 1., 2.], seg_len=[7, 5, 9])
    x = torch.randn(21, 96)
    v = m.view_for(x)
    y = torch.randn(21, 200)
    exp = y.clone()
    exp[:7] += 0.5 * x[:7] @ (B[2] @ A[2]).T
    exp[12:] += 2.0 * x[12:] @ (B[0] @ A[0]).T
    lora_apply_ref_(y, x, A, B, rank, v.row_slot, v.row_scale, v.seg_start, v.seg_len)
    torch.testing.assert_close(y, exp, atol=1e-4, rtol=1e-4)


def test_ref_uniform_rank0_and_stale_data():
    A, B, rank = _stale_setup("cpu", torch.float32)
    slots, scales, N = [3, 4, -1, 0, 2], [1., 1., 1., .5, 2.], 6
    x, y = torch.randn(len(slots) * N, D_IN), torch.randn(len(slots) * N, D_OUT)
    exp = _dense_expected(y, x, A, B, rank, slots, scales, [i * N for i in range(5)], [N] * 5)
    lora_apply_ref_(y, x, A, B, rank, torch.tensor(slots, dtype=torch.int32), torch.tensor(scales), N=N,
                    r_bucket=128)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.double(), exp, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="rank bucket"):
        lora_apply_ref_(y, x, A, B, rank, torch.tensor(slots, dtype=torch.int32), torch.tensor(scales), N=N,
                        r_bucket=32)


# ---------------------------------------------------------------- interpreter (fp32) ---------------------------

def _run_kernels(x, A, B, rank, slots_t, scales_t, y, *, N=0, tiles=None, r_bucket=128, z=None, z_dtype=torch.float32):
    from loramux.ops.triton_expand_add import expand_add_
    from loramux.ops.triton_shrink import shrink
    z = shrink(x, A, rank, slots_t, r_bucket=r_bucket, N=N, tiles=tiles, z=z, z_dtype=z_dtype)
    expand_add_(y, z, B, rank, slots_t, scales_t, r_bucket=r_bucket, N=N, tiles=tiles)
    return y


@pytest.mark.interp
@pytest.mark.parametrize("N", [37, 1, 64])
def test_interp_uniform_mixed_ranks(N):
    A, B, rank = _stale_setup("cpu", torch.float32)
    slots = [0, 1, 2, 3, 4, -1, 3]      # ranks 8, 16, 64, 128, 0 (rank-0 layer), base, repeat
    scales = [1., .5, 2., 1., 1., 1., -.7]
    rows = len(slots)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(rows * N, D_IN, generator=g)
    y = torch.randn(rows * N, D_OUT, generator=g)
    exp = _dense_expected(y, x, A, B, rank, slots, scales, [i * N for i in range(rows)], [N] * rows)
    z = torch.full((rows * N, 128), float("nan"))       # an uninitialised workspace must be harmless
    _run_kernels(x, A, B, rank, torch.tensor(slots, dtype=torch.int32), torch.tensor(scales), y, N=N, z=z)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.double(), exp, atol=1e-4, rtol=1e-4)


@pytest.mark.interp
@pytest.mark.parametrize("lens", [[130, 64, 5], [1, 63, 65, 33, 2]])
def test_interp_varlen_ragged(lens):
    A, B, rank = _stale_setup("cpu", torch.float32)
    n = len(lens)
    slots = [[3, -1, 2, 4, 0][i] for i in range(n)]
    scales = [1., 1., .25, 1., 3.][:n]
    m = build_meta(slots, scales, seg_len=lens, bm=32)
    T = sum(lens)
    x, y = torch.randn(T, D_IN), torch.randn(T, D_OUT)
    v = m.view_for(x)
    exp = _dense_expected(y, x, A, B, rank, slots, scales, v.start_host, v.len_host)
    _run_kernels(x, A, B, rank, v.row_slot, v.row_scale, y, tiles=v.tiles)
    torch.testing.assert_close(y.double(), exp, atol=1e-4, rtol=1e-4)


@pytest.mark.interp
def test_interp_small_bucket_and_rank0_layer():
    # pool r_max 4 < 16: the kernel tile is still 16 wide, masked by the rank; a layer with rank 0 everywhere
    A, B, rank = _stale_setup("cpu", torch.float32, ranks=[4, 3, 0], R=4)
    slots, scales, N = [0, 1, 2], [1., 1., 1.], 5
    x, y = torch.randn(15, D_IN), torch.randn(15, D_OUT)
    exp = _dense_expected(y, x, A, B, rank, slots, scales, [0, 5, 10], [5] * 3)
    y0 = y.clone()
    _run_kernels(x, A, B, rank, torch.tensor(slots, dtype=torch.int32), torch.ones(3), y, N=N, r_bucket=4)
    torch.testing.assert_close(y.double(), exp, atol=1e-4, rtol=1e-4)
    assert torch.equal(y[10:], y0[10:])
    zero = torch.zeros(3, dtype=torch.int32)          # layer targeted by none of the adapters
    y1 = y0.clone()
    _run_kernels(x, A, B, zero, torch.tensor(slots, dtype=torch.int32), torch.ones(3), y1, N=N, r_bucket=4)
    assert torch.equal(y1, y0)


@pytest.mark.interp
def test_interp_z_bf16_workspace():
    A, B, rank = _stale_setup("cpu", torch.float32, ranks=[16, 64])
    slots, N = [0, 1], 9
    x, y = torch.randn(18, D_IN), torch.randn(18, D_OUT)
    y0 = y.clone()
    exp = _dense_expected(y, x, A, B, rank, slots, [1., 1.], [0, 9], [9, 9])
    _run_kernels(x, A, B, rank, torch.tensor(slots, dtype=torch.int32), torch.ones(2), y, N=N, r_bucket=64,
                 z_dtype=torch.bfloat16)
    assert 0 < _rel_err(y, y0, exp) < 1e-2          # bf16 rounding of Z only


@pytest.mark.interp
@pytest.mark.parametrize("packed", [False, True])
def test_interp_custom_op_matches_ref(monkeypatch, packed):
    from loramux.ops import custom_op
    monkeypatch.setattr(custom_op, "TRITON_ON_CPU", True)
    A, B, rank = _stale_setup("cpu", torch.float32)
    slots, scales = [1, 4, -1, 3], [1., 1., 1., .5]
    lens = [9, 3, 7, 40] if packed else [11] * 4
    T = sum(lens)
    x, y = torch.randn(T, D_IN), torch.randn(T, D_OUT)
    starts = [sum(lens[:i]) for i in range(4)]
    exp = _dense_expected(y, x, A, B, rank, slots, scales, starts, lens)
    if packed:
        v = build_meta(slots, scales, seg_len=lens, bm=32).view_for(x)
        t = v.tiles
        torch.ops.loramux.lora_apply_(y, x, A, B, rank, v.row_slot, v.row_scale, 0, 128, v.seg_start, v.seg_len,
                                      t.tile_seg, t.tile_row0, t.tile_row_end, t.bm)
    else:
        rt = build_meta(slots, scales).rows_for(4)
        torch.ops.loramux.lora_apply_(y, x, A, B, rank, rt.slot, rt.scale, 11, 128)
    torch.testing.assert_close(y.double(), exp, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- GPU (bf16) -----------------------------------

def _rel_err(y, y0, exp):
    delta_ref = exp - y0.double()
    return float((y.double() - y0.double() - delta_ref).norm() / delta_ref.norm().clamp_min(1e-9))


@pytest.mark.gpu
@pytest.mark.parametrize("lens", [[130, 64, 5], [200, 200], [1, 63, 65]])
def test_triton_matches_ref(lens):
    dev, dt = "cuda", torch.bfloat16
    A, B, rank = _setup(dev, dt)
    n = len(lens)
    slots = [(i % 4) if i != 1 else -1 for i in range(n)]
    m = build_meta(slots, [1.0] * n, seg_len=lens, device=dev)
    T = sum(lens)
    x = torch.randn(T, 96, device=dev, dtype=dt)
    v = m.view_for(x)
    y0 = torch.randn(T, 200, device=dev, dtype=dt)
    y = y0.clone()
    t = v.tiles
    torch.ops.loramux.lora_apply_(y, x, A, B, rank, v.row_slot, v.row_scale, 0, 64, v.seg_start, v.seg_len,
                                  t.tile_seg, t.tile_row0, t.tile_row_end, t.bm)
    exp = _dense_expected(y0, x, A, B, rank, slots, [1.0] * n, v.start_host, v.len_host)
    assert _rel_err(y, y0, exp) < 2e-3


@pytest.mark.gpu
@pytest.mark.parametrize("N,d_in,d_out", [(1000, 3072, 3072), (77, 200, 1000), (4096, 3072, 9216)])
@pytest.mark.parametrize("z_bf16", [False, True])
def test_triton_uniform_bf16(N, d_in, d_out, z_bf16):
    dev, dt = "cuda", torch.bfloat16
    A, B, rank = _stale_setup(dev, dt, d_in=d_in, d_out=d_out)
    slots = [0, 1, 2, 3, 4, -1]
    scales = [1., .5, 1., 1., 1., 1.]
    rt = build_meta(slots, scales, device=dev).rows_for(len(slots))
    x = torch.randn(len(slots) * N, d_in, device=dev, dtype=dt)
    y0 = torch.randn(len(slots) * N, d_out, device=dev, dtype=dt)
    y = y0.clone()
    torch.ops.loramux.lora_apply_(y, x, A, B, rank, rt.slot, rt.scale, N, 128, z_bf16=z_bf16)
    exp = _dense_expected(y0, x, A, B, rank, slots, scales, [i * N for i in range(6)], [N] * 6)
    assert torch.isfinite(y).all()
    assert _rel_err(y, y0, exp) < (4e-3 if z_bf16 else 2e-3)
