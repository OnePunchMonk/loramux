import pytest
import torch

from loramux.compress import dequantize_fp8_rows, effective_rank_report, prune_small, quantize_fp8_rows, svd_truncate


def _low_rank_plus_noise(d_out=80, d_in=60, r=16, true_rank=4, noise=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    # ΔW with `true_rank` strong directions embedded in a rank-r factorization plus small noise directions
    S = torch.tensor([10., 8., 6., 5.] + [noise * 10] * (r - true_rank))
    U = torch.linalg.qr(torch.randn(d_out, r, generator=g))[0]
    V = torch.linalg.qr(torch.randn(d_in, r, generator=g))[0]
    B = U * S.sqrt()
    A = (V * S.sqrt()).T
    return A, B, S


def test_exact_at_full_energy():
    g = torch.Generator().manual_seed(1)
    A, B = torch.randn(16, 50, generator=g), torch.randn(70, 16, generator=g)
    A2, B2, k = svd_truncate(A, B, energy=1.0)
    assert k == 16 and A2.shape == (16, 50) and B2.shape == (70, 16)
    torch.testing.assert_close(B2 @ A2, B @ A, atol=1e-4, rtol=1e-4)


def test_exact_when_rank_deficient():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 40, generator=g).repeat(4, 1)[:12]  # rank 3 inside r=12
    B = torch.randn(30, 12, generator=g)
    A2, B2, k = svd_truncate(A, B, energy=1.0)
    assert k == 3
    torch.testing.assert_close(B2 @ A2, B @ A, atol=1e-4, rtol=1e-4)


def test_energy_target_hit():
    A, B, _S = _low_rank_plus_noise()
    dW = B @ A
    for energy in (0.99, 0.995, 0.9999):
        A2, B2, k = svd_truncate(A, B, energy=energy)
        err2 = (dW - B2 @ A2).norm() ** 2 / dW.norm() ** 2
        assert err2 <= (1 - energy) + 1e-6
        # smallest k: dropping one more direction would break the target
        if k > 1:
            A3, B3, _ = svd_truncate(A, B, energy=energy, max_rank=k - 1)
            assert (dW - B3 @ A3).norm() ** 2 / dW.norm() ** 2 > (1 - energy) - 1e-6
    assert svd_truncate(A, B, energy=0.995)[2] == 4


def test_max_rank_cap_and_report_and_zero():
    A, B, _ = _low_rank_plus_noise()
    assert svd_truncate(A, B, 1.0, max_rank=2)[2] == 2
    rep = effective_rank_report({"l": (A, B)}, 0.995)
    assert rep == {"l": (16, 4)}
    A0, B0, k = svd_truncate(torch.zeros(4, 8), torch.zeros(9, 4))
    assert k == 1 and not A0.any() and not B0.any()
    with pytest.raises(ValueError):
        svd_truncate(torch.zeros(4, 8), torch.zeros(9, 5))


def test_fp8_roundtrip_rel_error_per_row():
    g = torch.Generator().manual_seed(3)
    t = torch.randn(64, 256, generator=g) * torch.logspace(-3, 2, 64)[:, None]
    t[5] = 0
    q, s = quantize_fp8_rows(t)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32 and s.shape == (64,)
    back = dequantize_fp8_rows(q, s)
    assert back.dtype == torch.bfloat16
    rel = (back.float() - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-30)
    mask = torch.arange(64) != 5
    assert rel[mask].max() < 0.07
    assert torch.equal(back[5].float(), torch.zeros(256)) and s[5] == 1.0


def test_prune_small():
    g = torch.Generator().manual_seed(4)
    W = torch.randn(32, 24, generator=g)
    big = (torch.randn(4, 24, generator=g), torch.randn(32, 4, generator=g))
    tiny = (torch.randn(4, 24, generator=g) * 1e-6, torch.randn(32, 4, generator=g) * 1e-6)
    canon = {"big": big, "tiny": tiny, "unknown": tiny}
    kept, pruned = prune_small(canon, {"big": W, "tiny": W}, rel_thresh=1e-4)
    assert pruned == ["tiny"] and set(kept) == {"big", "unknown"}
    kept, pruned = prune_small(canon, {"big": W, "tiny": W}, rel_thresh=0.0)
    assert pruned == []
