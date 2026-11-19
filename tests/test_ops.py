import pytest, torch
from loramux.metadata import build_meta
from loramux.ops.torch_ref import lora_apply_ref_


def _setup(dev, dt, S=4, R=64, d_in=96, d_out=200):
    g = torch.Generator().manual_seed(0)
    A = torch.randn(S, R, d_in, generator=g).div(d_in ** .5).to(dev, dt)
    B = (torch.randn(S, d_out, R, generator=g) * 1e-1).to(dev, dt)
    rank = torch.tensor([8, 16, 64, 32], dtype=torch.int32, device=dev)
    for s in range(S):
        A[s, rank[s]:] = 0
        B[s, :, rank[s]:] = 0
    return A, B, rank


def test_ref_matches_dense_delta_cpu():
    A, B, rank = _setup("cpu", torch.float32)
    m = build_meta([2, -1, 0], [0.5, 1., 2.], 0, seg_len=[7, 5, 9])
    x = torch.randn(21, 96)
    y = torch.randn(21, 200)
    exp = y.clone()
    exp[:7] += 0.5 * x[:7] @ (B[2] @ A[2]).T
    exp[12:] += 2.0 * x[12:] @ (B[0] @ A[0]).T
    lora_apply_ref_(y, x, A, B, rank, m.slot_id, m.scale, m.seg_start, m.seg_len)
    torch.testing.assert_close(y, exp, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("lens", [[130, 64, 5], [200, 200], [1, 63, 65]])
def test_triton_matches_ref(lens):
    from loramux.ops import custom_op  # noqa
    dev, dt = "cuda", torch.bfloat16
    A, B, rank = _setup(dev, dt)
    n = len(lens)
    slots = [(i % 4) if i != 1 else -1 for i in range(n)]
    m = build_meta(slots, [1.0] * n, 0, seg_len=lens, device=dev)
    T = sum(lens)
    x = torch.randn(T, 96, device=dev, dtype=dt)
    y0 = torch.randn(T, 200, device=dev, dtype=dt)
    y_ref = y0.float().clone()
    lora_apply_ref_(y_ref, x, A, B, rank, m.slot_id, m.scale, m.seg_start, m.seg_len, fp32=True)
    y = y0.clone()
    t = m.tiles
    torch.ops.loramux.lora_apply_(y, x, A, B, rank, m.slot_id, m.scale, t.tile_seg, t.tile_row0,
                                  t.tile_row_end, t.bm, m.seg_start, m.seg_len)
    delta_ref = y_ref - y0.float()
    err = (y.float() - y0.float() - delta_ref).norm() / delta_ref.norm().clamp_min(1e-9)
    assert err < 2e-3
