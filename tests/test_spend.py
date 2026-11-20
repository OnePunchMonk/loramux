import pytest

from bench.spend import SpendCapError, append_row, check_caps, estimate, read_rows

OLD_TABLE = "# GPU spend log\n\n| date | run | GPU | minutes | est. USD | actual USD | cumulative |\n|---|---|---|---|---|---|---|\n"


def test_estimate_math():
    assert estimate("H100", 30, 4.0) == pytest.approx(2.0)
    assert estimate("H100", 90, 3.95) == pytest.approx(5.925)
    assert estimate("A10G", 0, 1.1) == 0.0


def test_append_recomputes_cumulative_and_keeps_preamble(tmp_path):
    path = tmp_path / "SPEND.md"
    path.write_text(OLD_TABLE)
    append_row(path, date="2026-11-20", run="smoke", gpu="H100", minutes=10, est_usd=0.66, actual_usd=None)
    append_row(path, date="2026-11-21", run="bench", gpu="H100", minutes=60, est_usd=4.0, actual_usd=3.5)
    rows = read_rows(path)
    assert [r["run"] for r in rows] == ["smoke", "bench"]
    assert rows[0]["cumulative"] == pytest.approx(0.66)   # no actual yet, so the estimate counts
    assert rows[1]["cumulative"] == pytest.approx(4.16)   # 0.66 + actual 3.5
    assert rows[1]["phase"] == "1"
    text = path.read_text()
    assert text.startswith("# GPU spend log\n")
    assert "| cumulative | phase |" in text


def test_append_with_phase_and_missing_file(tmp_path):
    path = tmp_path / "new.md"
    append_row(path, date="d", run="r", gpu="H100", minutes=5, est_usd=1.0, actual_usd=None, phase="2")
    (row,) = read_rows(path)
    assert row["phase"] == "2" and row["est. USD"] == pytest.approx(1.0)


def test_cap_error_per_phase_and_total(tmp_path):
    path = tmp_path / "SPEND.md"
    path.write_text(OLD_TABLE)
    append_row(path, date="d", run="a", gpu="H100", minutes=60, est_usd=24.0, actual_usd=24.0)
    check_caps(path, phase_cap=25.0, new_est=0.5)
    with pytest.raises(SpendCapError, match="phase 1"):
        check_caps(path, phase_cap=25.0, new_est=2.0)
    with pytest.raises(SpendCapError, match="total"):
        check_caps(path, phase_cap=100.0, total_cap=30.0, new_est=7.0)


def test_other_phase_does_not_count_toward_phase_cap(tmp_path):
    path = tmp_path / "SPEND.md"
    path.write_text(OLD_TABLE)
    append_row(path, date="d", run="a", gpu="H100", minutes=60, est_usd=20.0, actual_usd=20.0, phase="2")
    check_caps(path, phase_cap=25.0, phase="1", new_est=10.0)
    with pytest.raises(SpendCapError):
        check_caps(path, phase_cap=25.0, phase="2", new_est=10.0)


def test_header_only_table_has_no_rows(tmp_path):
    path = tmp_path / "SPEND.md"
    path.write_text(OLD_TABLE)
    assert read_rows(path) == []
    check_caps(path, new_est=25.0)
    assert read_rows(tmp_path / "missing.md") == []
