import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bench import k1_correctness as k1
from bench.common import read_csv

ROOT = Path(__file__).resolve().parents[1]


def test_psnr_math():
    a = np.zeros((4, 4, 3))
    assert k1.psnr(a, a) == math.inf
    b = a + 0.1                                   # mse = 0.01 -> 20 dB
    assert k1.psnr(a, b) == pytest.approx(20.0)
    assert k1.psnr(a * 255, b * 255, data_range=255.0) == pytest.approx(20.0)
    assert k1.psnr(torch.zeros(2, 2), torch.full((2, 2), 0.01)) == pytest.approx(40.0)
    with pytest.raises(ValueError):
        k1.psnr(np.zeros((2, 2)), np.zeros((2, 3)))


def test_lpips_tensor_layout_and_guard():
    img = np.random.default_rng(0).random((8, 6, 3))
    t = k1.to_lpips_tensor(img)
    assert t.shape == (1, 3, 8, 6) and t.min() >= -1 and t.max() <= 1
    assert torch.allclose(t[0, :, 2, 3], torch.tensor(img[2, 3] * 2 - 1, dtype=torch.float32))
    with pytest.raises(ValueError):
        k1.to_lpips_tensor(np.zeros((8, 6, 5)))
    d = k1.lpips_distance(img, img)
    if k1.have_lpips():
        assert d == pytest.approx(0.0, abs=1e-6)
    else:
        assert d is None


def test_k1a_cpu_tiny_passes():
    rows = k1.run_k1a(k1.TINY_SHAPES, paths=("op",), device="cpu")
    assert len(rows) == len(k1.TINY_SHAPES)
    assert all(r["status"] == "ok" and r["rel_err"] <= 2e-3 for r in rows)
    assert k1.summarize_k1(rows).verdicts[0].passed is True


def test_k1a_detects_wrong_answer(monkeypatch):
    monkeypatch.setattr(k1, "_run_op", lambda c, bm: k1._reference(c) * 1.01)
    rows = k1.run_k1a(k1.TINY_SHAPES[1:2], paths=("op",), device="cpu")
    assert rows[0]["rel_err"] == pytest.approx(0.01, rel=1e-3)
    assert k1.summarize_k1(rows).verdicts[0].passed is False


def test_case_has_base_segment_untouched():
    c = k1.make_case(40, 16, 24, 16, 3, dtype=torch.float32, device="cpu")
    y = k1._reference(c)
    a, n = c["starts"][1], c["lens"][1]
    assert c["slot"][1] == -1 and torch.count_nonzero(y[a:a + n]) == 0
    assert sum(c["lens"]) == 40


def test_interp_skipped_without_env(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    rows = k1.run_k1a(k1.TINY_SHAPES[:1], paths=("interp",), device="cpu")
    assert rows[0]["status"].startswith("skipped") and rows[0]["rel_err"] is None


def test_cli_k1a_with_triton_interpreter(tmp_path):
    pytest.importorskip("triton")
    out = tmp_path / "k1.csv"
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-m", "bench.k1_correctness", "--checks", "k1a,k1b,k1c", "--shapes", "tiny",
                    "--out", str(out)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=600)
    _, rows = read_csv(out)
    interp = [r for r in rows if r["check"] == "K1a" and r["path"] == "interp"]
    assert len(interp) == len(k1.TINY_SHAPES) and all(r["status"] == "ok" for r in interp)
    assert max(r["rel_err"] for r in interp) <= 2e-3
    assert {r["check"] for r in rows if str(r["status"]).startswith("skipped")} == {"K1b", "K1c"}
    summary = Path(str(out) + ".summary.txt").read_text()
    assert "K1a: PASS" in summary and "K1: N/A" in summary


def test_parse_args():
    a = k1.parse_args(["--checks", "k1a", "--shapes", "tiny"])
    assert a.checks == ["k1a"] and a.paths == ["op", "interp"] and a.steps == 28 and len(a.prompts) == 2
    with pytest.raises(SystemExit):
        k1.parse_args(["--checks", "k9"])
