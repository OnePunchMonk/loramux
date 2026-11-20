import pytest
import torch

from loramux import build_meta, expand_cfg


def test_cfg_layouts():
    assert expand_cfg([1, 2], [.5, 1.], "concat", True) == ([1, 2, 1, 2], [.5, 1., .5, 1.])
    assert expand_cfg([1, 2], [.5, 1.], "interleaved", True) == ([1, 1, 2, 2], [.5, .5, 1., 1.])
    assert expand_cfg([1], [1.], "concat", False) == ([1], [1.])


def test_all_base_no_lora():
    m = build_meta([-1, -1], [1., 1.])
    assert not m.any_lora
    assert build_meta([-1, 3], [1., 1.]).any_lora


def test_tiles_cover_segments_exactly():
    m = build_meta([0, 1, -1], [1., 1., 1.], seg_len=[130, 64, 5], bm=64)
    v = m.view_for(torch.zeros(199, 8))
    t = v.tiles
    assert t.tile_seg.tolist() == [0, 0, 0, 1, 2]
    assert t.tile_row0.tolist() == [0, 64, 128, 130, 194] and t.tile_row_end.tolist() == [64, 128, 130, 194, 199]
    assert v.seg_start.tolist() == [0, 130, 194]
    assert v.seg_len.tolist() == [130, 64, 5]
    assert not v.uniform and v.N is None


def test_view_uniform_rows_and_cache():
    m = build_meta([2, -1], [.5, 1.])
    x = torch.zeros(2, 10, 8)
    v = m.view_for(x)
    assert v.uniform and v.rows == 2 and v.N == 10
    assert v.row_slot.tolist() == [2, -1] and v.row_slot.dtype == torch.int32
    assert v.row_scale.tolist() == [.5, 1.] and v.row_scale.dtype == torch.float32
    assert v.seg_start.tolist() == [0, 10] and v.seg_len.tolist() == [10, 10]
    assert v.slot_host == [2, -1] and v.scale_host == [.5, 1.]
    assert m.view_for(torch.zeros(2, 10, 8)) is v          # cached
    assert m.view_for(torch.zeros(2, 4, 8)) is not v       # different N
    assert m.view_for(torch.zeros(2, 4, 8)).N == 4


@pytest.mark.parametrize("layout,slots", [("concat", [1, 2, 1, 2]), ("interleaved", [1, 1, 2, 2])])
def test_view_cfg_rows(layout, slots):
    m = build_meta([1, 2], [.5, 1.], cfg_layout=layout)
    v = m.view_for(torch.zeros(4, 3, 8))
    assert v.rows == 4 and v.row_slot.tolist() == slots
    assert v.seg_start.tolist() == [0, 3, 6, 9]


def test_layout_part_of_cache_key():
    m = build_meta([1, 2], [1., 1.])
    v1 = m.view_for(torch.zeros(4, 3, 8))
    m.cfg_layout = "interleaved"
    v2 = m.view_for(torch.zeros(4, 3, 8))
    assert v1 is not v2 and v1.slot_host != v2.slot_host


def test_rows_mismatch_raises():
    m = build_meta([1, 2], [1., 1.])
    with pytest.raises(ValueError, match="3 rows"):
        m.view_for(torch.zeros(3, 5, 8))
    with pytest.raises(ValueError):
        m.view_for(torch.zeros(6, 5, 8))


def test_2d_needs_seg_len_and_packed_needs_2d():
    with pytest.raises(ValueError, match="seg_len"):
        build_meta([0], [1.]).view_for(torch.zeros(10, 8))
    m = build_meta([0, 1], [1., 1.], seg_len=[4, 6])
    with pytest.raises(ValueError, match="sums"):
        m.view_for(torch.zeros(11, 8))
    with pytest.raises(ValueError, match="2-D"):
        m.view_for(torch.zeros(2, 5, 8))


def test_build_meta_validation():
    with pytest.raises(ValueError):
        build_meta([0, 1], [1.])
    with pytest.raises(ValueError):
        build_meta([0], [1.], cfg_layout="bogus")
    with pytest.raises(ValueError):
        build_meta([0, 1], [1., 1.], seg_len=[3])


def test_r_bucket():
    ranks = {0: 8, 1: 20, 2: 64, 3: 128}
    rk = ranks.__getitem__
    assert build_meta([0, -1], [1., 1.], rank_of_slot=rk, r_max=128).r_bucket == 16
    assert build_meta([0, 1], [1., 1.], rank_of_slot=rk, r_max=128).r_bucket == 32
    assert build_meta([1, 2], [1., 1.], rank_of_slot=rk, r_max=128).r_bucket == 64
    assert build_meta([3], [1.], rank_of_slot=rk, r_max=128).r_bucket == 128
    assert build_meta([3], [1.], rank_of_slot=rk, r_max=64).r_bucket == 64   # clamped to r_max
    assert build_meta([-1], [1.], rank_of_slot=rk, r_max=128).r_bucket == 16  # all base
    assert build_meta([0], [1.]).r_bucket == 64
    assert build_meta([0], [1.], r_max=32).r_bucket == 32


def test_per_row_2d_inputs():
    x2 = torch.randn(3, 8)
    with pytest.raises(ValueError):
        build_meta([0, 1, -1], [1.0] * 3).view_for(x2)
    v = build_meta([0, 1, -1], [1.0] * 3, per_row_2d=True).view_for(x2)
    assert v.rows == 3 and v.N == 1 and v.uniform and v.slot_host == [0, 1, -1]
