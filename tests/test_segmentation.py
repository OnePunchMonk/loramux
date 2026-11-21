import pytest
import torch
from torch import nn

from loramux import AdapterPool, build_meta, lora_batch, patch_model

D = 32


class MMBlock(nn.Module):
    """Toy MMDiT block: separate image/text linears, joint attention within each row."""

    def __init__(self):
        super().__init__()
        self.img_qkv, self.txt_qkv = nn.Linear(D, 3 * D), nn.Linear(D, 3 * D)
        self.img_out, self.txt_out = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, img, txt):
        n_img = img.shape[1]
        qi, ki, vi = self.img_qkv(img).chunk(3, -1)
        qt, kt, vt = self.txt_qkv(txt).chunk(3, -1)
        q, k, v = (torch.cat(p, 1) for p in ((qt, qi), (kt, ki), (vt, vi)))
        h = torch.softmax(q @ k.transpose(1, 2) / D ** .5, -1) @ v
        ht, hi = h[:, :txt.shape[1]], h[:, txt.shape[1]:]
        assert hi.shape[1] == n_img
        return img + self.img_out(hi), txt + self.txt_out(ht)


def _build():
    torch.manual_seed(0)
    m = MMBlock()
    shapes = {"img_qkv": (D, 3 * D), "txt_qkv": (D, 3 * D), "img_out": (D, D), "txt_out": (D, D)}
    pool = AdapterPool(shapes, slots=3, r_max=16, device="cpu", dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    for a in range(3):
        pool.register(f"ad{a}", {k: (torch.randn(8, di, generator=g) / 4, torch.randn(do, 8, generator=g) * .3)
                                 for k, (di, do) in shapes.items()})
        pool.ensure_resident(f"ad{a}")
    patch_model(m, [r"_qkv$", r"_out$"], pool)
    return m, pool


def _solo(m, slot, scale, *xs, **kw):
    with lora_batch(build_meta([slot], [scale], **kw)):
        return m(*xs)


def test_mmdit_two_streams_mixed_batch_matches_solo():
    m, pool = _build()
    slots = [pool.slot_of["ad0"], -1, pool.slot_of["ad2"]]
    scales = [1., 1., .6]
    img, txt = torch.randn(3, 10, D), torch.randn(3, 4, D)
    with lora_batch(build_meta(slots, scales)):
        yi, yt = m(img, txt)
    base_i, _ = m(img, txt)  # no context: base model everywhere
    assert not torch.allclose(yi[0], base_i[0], atol=1e-3)   # adapter really applied
    torch.testing.assert_close(yi[1], base_i[1], atol=1e-5, rtol=1e-5)
    for i in range(3):
        si, st = _solo(m, slots[i], scales[i], img[i:i + 1], txt[i:i + 1])
        torch.testing.assert_close(yi[i:i + 1], si, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(yt[i:i + 1], st, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("layout", ["concat", "interleaved"])
def test_cfg_rows_match_separate_cond_uncond(layout):
    m, pool = _build()
    slots = [pool.slot_of["ad1"], -1, pool.slot_of["ad2"]]
    scales = [.8, 1., 1.]
    n = 3
    ci, ct = torch.randn(n, 10, D), torch.randn(n, 4, D)
    ui, ut = torch.randn(n, 10, D), torch.randn(n, 4, D)
    if layout == "concat":
        img, txt = torch.cat([ci, ui]), torch.cat([ct, ut])
    else:
        img = torch.stack([ci, ui], 1).reshape(2 * n, 10, D)
        txt = torch.stack([ct, ut], 1).reshape(2 * n, 4, D)
    with lora_batch(build_meta(slots, scales, cfg_layout=layout)):
        yi, yt = m(img, txt)
    with lora_batch(build_meta(slots, scales)):
        cyi, cyt = m(ci, ct)
        uyi, uyt = m(ui, ut)
    if layout == "concat":
        ei, et = torch.cat([cyi, uyi]), torch.cat([cyt, uyt])
    else:
        ei = torch.stack([cyi, uyi], 1).reshape(2 * n, 10, D)
        et = torch.stack([cyt, uyt], 1).reshape(2 * n, 4, D)
    torch.testing.assert_close(yi, ei, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(yt, et, atol=1e-4, rtol=1e-4)


def test_rows_mismatch_raises_in_forward():
    m, pool = _build()
    with lora_batch(build_meta([pool.slot_of["ad0"], -1], [1., 1.])), pytest.raises(ValueError, match="rows"):
        m(torch.randn(3, 10, D), torch.randn(3, 4, D))


def test_packed_2d_with_seg_len():
    lin = nn.Linear(D, D)
    pool2 = AdapterPool({"l": (D, D)}, slots=3, r_max=16, device="cpu", dtype=torch.float32)
    g = torch.Generator().manual_seed(2)
    for a in range(2):
        pool2.register(f"a{a}", {"l": (torch.randn(8, D, generator=g) / 4, torch.randn(D, 8, generator=g) * .3)})
        pool2.ensure_resident(f"a{a}")
    from loramux import MuxLinear
    mux = MuxLinear(lin, "l", pool2)
    lens = [7, 3, 12]
    slots = [pool2.slot_of["a0"], -1, pool2.slot_of["a1"]]
    scales = [1., 1., .5]
    x = torch.randn(sum(lens), D)
    with lora_batch(build_meta(slots, scales, seg_len=lens)):
        y = mux(x)
    off = 0
    for l, s, sc in zip(lens, slots, scales):
        with lora_batch(build_meta([s], [sc], seg_len=[l])):
            ys = mux(x[off:off + l])
        torch.testing.assert_close(y[off:off + l], ys, atol=1e-4, rtol=1e-4)
        off += l
    assert not torch.allclose(y[:7], lin(x[:7]), atol=1e-3)
    torch.testing.assert_close(y[7:10], lin(x[7:10]))
    with lora_batch(build_meta(slots, scales)), pytest.raises(ValueError, match="seg_len"):
        mux(x)
