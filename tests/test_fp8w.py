"""FP8-weight path (L2.2): A/B stored float8_e4m3fn with per-row fp32 scales (L1.5 format).

The FP8 kernels pass deterministically in the Triton interpreter (fp32 activations), so they have `interp`
tests against the dequantized reference; bf16 accuracy against the bf16 path is `gpu`."""
import pytest
import torch

from loramux import AdapterPool, build_meta, lora_batch, patch_model
from loramux.compress import quantize_fp8_rows
from loramux.ops.torch_ref import lora_apply_ref_

D_IN, D_OUT = 40, 70


def _q(S, R, d_in, d_out, ranks, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(S, R, d_in, generator=g)
    B = torch.randn(S, d_out, R, generator=g)
    Aq, sA, Bq, sB = [], [], [], []
    for s in range(S):
        q, sc = quantize_fp8_rows(A[s])
        Aq.append(q), sA.append(sc)
        q, sc = quantize_fp8_rows(B[s])
        Bq.append(q), sB.append(sc)
    rank = torch.tensor(ranks, dtype=torch.int32)
    return torch.stack(Aq), torch.stack(sA), torch.stack(Bq), torch.stack(sB), rank


def _deq(q, s):
    return q.float() * s[..., None]


def _expected(y, x, Ad, Bd, rank, slots, scales, N):
    exp = y.double().clone()
    for i, (s, c) in enumerate(zip(slots, scales)):
        r = int(rank[s]) if s >= 0 else 0
        if r:
            a = slice(i * N, (i + 1) * N)
            exp[a] += c * (x[a].double() @ Ad[s, :r].double().T) @ Bd[s, :, :r].double().T
    return exp


def test_ref_fp8_matches_dequantized():
    Aq, sA, Bq, sB, rank = _q(3, 32, D_IN, D_OUT, [32, 0, 9])
    slots, scales, N = [0, 1, 2, -1], [1., 1., .5, 1.], 4
    x, y = torch.randn(16, D_IN), torch.randn(16, D_OUT)
    exp = _expected(y, x, _deq(Aq, sA), _deq(Bq, sB), rank, slots, scales, N)
    lora_apply_ref_(y, x, Aq, Bq, rank, torch.tensor(slots, dtype=torch.int32), torch.tensor(scales), N=N,
                    a_scale=sA, b_scale=sB)
    torch.testing.assert_close(y.double(), exp, atol=1e-3, rtol=1e-4)


@pytest.mark.interp
@pytest.mark.parametrize("varlen", [False, True])
def test_interp_fp8w_kernels_match_dequantized(varlen):
    from loramux.ops.triton_expand_add_fp8w import expand_add_fp8w_
    from loramux.ops.triton_shrink_fp8w import shrink_fp8w
    Aq, sA, Bq, sB, rank = _q(4, 64, D_IN, D_OUT, [64, 16, 0, 33])
    slots, scales, N = [0, 1, 2, 3, -1], [1., .5, 1., 2., 1.], 7
    x, y = torch.randn(35, D_IN), torch.randn(35, D_OUT)
    exp = _expected(y, x, _deq(Aq, sA), _deq(Bq, sB), rank, slots, scales, N)
    st = torch.tensor(slots, dtype=torch.int32)
    tiles = None
    if varlen:
        v = build_meta(slots, scales, seg_len=[N] * 5, bm=32).view_for(x)
        tiles, n = v.tiles, 0
    else:
        n = N
    for _ in range(2):          # deterministic
        yy = y.clone()
        z = shrink_fp8w(x, Aq, sA, rank, st, r_bucket=64, N=n, tiles=tiles)
        expand_add_fp8w_(yy, z, Bq, sB, rank, st, torch.tensor(scales), r_bucket=64, N=n, tiles=tiles)
        torch.testing.assert_close(yy.double(), exp, atol=1e-3, rtol=1e-4)


def test_fp8w_wrappers_reject_wrong_dtypes():
    from loramux.ops.triton_shrink_fp8w import shrink_fp8w
    A = torch.zeros(2, 16, D_IN)
    with pytest.raises(TypeError, match="float8"):
        shrink_fp8w(torch.zeros(4, D_IN), A, torch.ones(2, 16), torch.zeros(2, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32), r_bucket=16, N=4)


def _toy_pool(dtype, device="cpu"):
    shapes = {"0": (D_IN, D_OUT)}
    pool = AdapterPool(shapes, slots=3, r_max=32, device=device, dtype=dtype)
    g = torch.Generator().manual_seed(5)
    ads = {}
    for i, r in enumerate([32, 8]):
        ads[f"a{i}"] = {"0": (torch.randn(r, D_IN, generator=g) / 6, torch.randn(D_OUT, r, generator=g) / 6)}
        pool.register(f"a{i}", ads[f"a{i}"])
        pool.ensure_resident(f"a{i}")
    return pool, ads


@pytest.mark.parametrize("triton_cpu", [False, pytest.param(True, marks=pytest.mark.interp)])
def test_fp8_pool_muxlinear_close_to_full_precision(monkeypatch, triton_cpu):
    from torch import nn

    from loramux.ops import custom_op
    monkeypatch.setattr(custom_op, "TRITON_ON_CPU", triton_cpu)
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Linear(D_IN, D_OUT))
    outs = {}
    for dt in (torch.float32, torch.float8_e4m3fn):
        pool, _ = _toy_pool(dt)
        m = nn.Sequential(lin[0])
        patch_model(m, ["0"], pool)
        x = torch.randn(3, 6, D_IN, generator=torch.Generator().manual_seed(1))
        with torch.no_grad(), lora_batch(build_meta([pool.slot_of["a0"], -1, pool.slot_of["a1"]], [1., 1., .5],
                                   rank_of_slot=pool.slot_rank, r_max=32)):
            outs[dt] = m(x)
        m[0] = m[0].base
    with torch.no_grad():
        base = lin(x)
    d32, d8 = outs[torch.float32] - base, outs[torch.float8_e4m3fn] - base
    assert torch.equal(d8[1], torch.zeros_like(d8[1]))
    rel = float((d8 - d32).norm() / d32.norm())
    assert 0 < rel < 0.07          # e4m3 per-row rounding of A and B


@pytest.mark.gpu
@pytest.mark.parametrize("N,d_in,d_out", [(1024, 3072, 3072), (77, 3072, 12288)])
def test_gpu_fp8w_vs_bf16_path(N, d_in, d_out):
    dev = "cuda"
    S, R = 4, 64
    g = torch.Generator().manual_seed(0)
    A = (torch.randn(S, R, d_in, generator=g) / d_in ** .5).to(dev)
    B = (torch.randn(S, d_out, R, generator=g) / R ** .5).to(dev)
    rank = torch.tensor([64, 16, 0, 33], dtype=torch.int32, device=dev)
    qA = [quantize_fp8_rows(A[s]) for s in range(S)]
    qB = [quantize_fp8_rows(B[s]) for s in range(S)]
    Aq, sA = torch.stack([q for q, _ in qA]), torch.stack([c for _, c in qA])
    Bq, sB = torch.stack([q for q, _ in qB]), torch.stack([c for _, c in qB])
    slots, scales = [0, 1, 2, 3, -1], [1., .5, 1., 2., 1.]
    rt = build_meta(slots, scales, device=dev).rows_for(5)
    x = torch.randn(5 * N, d_in, device=dev, dtype=torch.bfloat16)
    y0 = torch.randn(5 * N, d_out, device=dev, dtype=torch.bfloat16)
    y16, y8 = y0.clone(), y0.clone()
    torch.ops.loramux.lora_apply_(y16, x, A.bfloat16(), B.bfloat16(), rank, rt.slot, rt.scale, N, 64)
    torch.ops.loramux.lora_apply_(y8, x, Aq, Bq, rank, rt.slot, rt.scale, N, 64, a_scale=sA, b_scale=sB)
    exp = _expected(y0, x, _deq(Aq, sA), _deq(Bq, sB), rank, slots, scales, N)
    d_ref = exp - y0.double()
    assert float((y8.double() - exp).norm() / d_ref.norm()) < 3e-3            # kernel vs its own dequantized math
    d16 = y16.double() - y0.double()
    assert float((y8.double() - y0.double() - d16).norm() / d16.norm()) < 0.07  # FP8 storage vs bf16 storage
