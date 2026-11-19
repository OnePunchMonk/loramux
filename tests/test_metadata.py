from loramux.metadata import build_meta, expand_cfg


def test_cfg_layouts():
    assert expand_cfg([1, 2], [.5, 1.], "concat", True) == ([1, 2, 1, 2], [.5, 1., .5, 1.])
    assert expand_cfg([1, 2], [.5, 1.], "interleaved", True) == ([1, 1, 2, 2], [.5, .5, 1., 1.])
    assert expand_cfg([1], [1.], "concat", False) == ([1], [1.])


def test_all_base_no_lora():
    m = build_meta([-1, -1], [1., 1.], 100)
    assert not m.any_lora


def test_tiles_cover_segments_exactly():
    m = build_meta([0, 1, -1], [1., 1., 1.], 0, seg_len=[130, 64, 5], bm=64)
    seg, r0, r1 = m.tiles.tile_seg.tolist(), m.tiles.tile_row0.tolist(), m.tiles.tile_row_end.tolist()
    assert seg == [0, 0, 0, 1, 2]
    assert r0 == [0, 64, 128, 130, 194] and r1 == [64, 128, 130, 194, 199]
    assert m.seg_start.tolist() == [0, 130, 194]
