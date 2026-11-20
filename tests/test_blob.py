import struct

import pytest
import torch

from loramux.blob import ALIGN, blob_nbytes, read_blob, read_header, write_blob

LAYERS = [("blocks.0.attn.to_q", 64, 48, 8), ("blocks.0.ff.net.0", 17, 96, 3),
          ("blocks.1.to_out.0", 96, 64, 16), ("tail", 7, 5, 1)]


def _adapter(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(r, d_in, generator=g), torch.randn(d_out, r, generator=g))
            for k, d_in, d_out, r in LAYERS}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _payload_start(path) -> int:
    with open(path, "rb") as f:
        _, hlen = struct.unpack("<8sQ", f.read(16))
    return (16 + hlen + ALIGN - 1) // ALIGN * ALIGN


def test_round_trip_bitwise_bf16_mixed_ranks(tmp_path):
    src = _adapter()
    path = tmp_path / "a.lmx"
    write_blob(path, src, base_model="toy", meta={"name": "x"})
    for mmap in (True, False):
        out = read_blob(path, mmap=mmap)
        assert list(out) == [k for k, *_ in LAYERS]
        for k, (a, b) in src.items():
            oa, ob = out[k]
            assert oa.dtype == torch.bfloat16 and ob.dtype == torch.bfloat16
            assert oa.shape == a.shape and ob.shape == b.shape
            assert torch.equal(_bits(oa), _bits(a.to(torch.bfloat16)))
            assert torch.equal(_bits(ob), _bits(b.to(torch.bfloat16)))


def test_mmap_tensors_stay_valid_after_return(tmp_path):
    import gc
    src = _adapter(2)
    path = tmp_path / "m.lmx"
    write_blob(path, src, base_model="toy")
    out = read_blob(path, mmap=True)  # the function has returned; only the tensors keep the mapping alive
    gc.collect()
    for k, (a, b) in src.items():
        assert torch.equal(_bits(out[k][0]), _bits(a.to(torch.bfloat16)))
        assert torch.equal(_bits(out[k][1]), _bits(b.to(torch.bfloat16)))


def test_offsets_are_256_aligned(tmp_path):
    path = tmp_path / "a.lmx"
    header = write_blob(path, _adapter(), base_model="toy")
    assert _payload_start(path) % ALIGN == 0
    for layer in header["layers"]:
        assert layer["off_A"] % ALIGN == 0
        assert layer["off_B"] % ALIGN == 0
        assert layer["off_B"] >= layer["off_A"] + layer["r"] * layer["d_in"] * 2
    assert read_header(path) == header


def test_order_controls_layout_and_round_trips(tmp_path):
    src = _adapter(1)
    order = list(reversed(list(src)))
    path = tmp_path / "rev.lmx"
    header = write_blob(path, src, base_model="toy", order=order)
    assert [layer["key"] for layer in header["layers"]] == order
    out = read_blob(path)
    assert torch.equal(_bits(out["tail"][0]), _bits(src["tail"][0].to(torch.bfloat16)))


def test_blob_nbytes_is_payload_only(tmp_path):
    path = tmp_path / "a.lmx"
    header = write_blob(path, _adapter(), base_model="toy")
    last = header["layers"][-1]
    assert blob_nbytes(path) == last["off_B"] + last["d_out"] * last["r"] * 2
    raw = sum(r * d_in * 2 + d_out * r * 2 for _, d_in, d_out, r in LAYERS)
    assert blob_nbytes(path) >= raw


def test_header_carries_base_model_and_meta(tmp_path):
    path = tmp_path / "a.lmx"
    write_blob(path, _adapter(), base_model="toy-dit", meta={"scale_convention": "alpha/r folded"})
    header = read_header(path)
    assert header["base_model"] == "toy-dit" and header["dtype"] == "bfloat16" and header["version"] == 1
    assert header["meta"] == {"scale_convention": "alpha/r folded"}
    assert len(header["payload_sha256"]) == 64


def test_verify_accepts_clean_and_rejects_corrupt_payload(tmp_path):
    path = tmp_path / "a.lmx"
    header = write_blob(path, _adapter(), base_model="toy")
    read_blob(path, verify=True)
    off = _payload_start(path) + header["layers"][0]["off_A"] + 3
    with open(path, "r+b") as f:
        f.seek(off)
        byte = f.read(1)
        f.seek(off)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="checksum"):
        read_blob(path, verify=True)
    read_blob(path, verify=False)  # unverified read still succeeds


def test_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.lmx"
    path.write_bytes(b"NOTBLOB!" + struct.pack("<Q", 2) + b"{}" + bytes(300))
    with pytest.raises(ValueError, match="magic"):
        read_header(path)
    with pytest.raises(ValueError, match="magic"):
        read_blob(path)


def test_empty_adapter_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_blob(tmp_path / "e.lmx", {}, base_model="toy")


def test_rank_mismatch_raises(tmp_path):
    bad = {"x": (torch.zeros(4, 8), torch.zeros(8, 3))}
    with pytest.raises(ValueError, match="rank"):
        write_blob(tmp_path / "m.lmx", bad, base_model="toy")


def test_order_must_list_every_key(tmp_path):
    with pytest.raises(ValueError, match="order"):
        write_blob(tmp_path / "o.lmx", _adapter(), base_model="toy", order=["tail"])
