"""GPU spend log (SPEND.md) and cap checks. All amounts are USD.

A row's cost is its actual cost when recorded, otherwise its estimate. 'cumulative' is the running sum of
row costs across all phases. Caps are checked against the same sums, per phase and in total.
"""
from __future__ import annotations

from pathlib import Path

HEADER = ["date", "run", "GPU", "minutes", "est. USD", "actual USD", "cumulative", "phase"]
DEFAULT_PHASE = "1"
_NUMERIC = ("minutes", "est. USD", "actual USD", "cumulative")


class SpendCapError(RuntimeError):
    pass


def estimate(gpu: str, minutes: float, usd_per_hour: float) -> float:
    """Expected cost in USD. gpu is a label for the log only."""
    return minutes / 60.0 * usd_per_hour


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _find_header(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|") and _cells(line)[:1] == ["date"]:
            return i
    return None


def _is_separator(cells: list[str]) -> bool:
    return all(c and set(c) <= set("-: ") for c in cells)


def _num(text: str) -> float | None:
    return float(text) if text else None


def _typed(raw: dict) -> dict:
    row = {k: raw.get(k, "") for k in HEADER}
    for k in _NUMERIC:
        row[k] = _num(row[k])
    row["phase"] = row["phase"] or DEFAULT_PHASE
    return row


def _cost(row: dict) -> float:
    if row["actual USD"] is not None:
        return row["actual USD"]
    return row["est. USD"] or 0.0


def _spent(rows: list[dict], phase: str | None = None) -> float:
    return sum(_cost(r) for r in rows if phase is None or r["phase"] == phase)


def read_rows(path: str | Path = "SPEND.md") -> list[dict]:
    """Table rows keyed by header name. Numbers are float or None. A missing file or no table gives []."""
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    start = _find_header(lines)
    if start is None:
        return []
    names = _cells(lines[start])
    rows = []
    for line in lines[start + 1:]:
        if not line.lstrip().startswith("|"):
            break
        cells = _cells(line)
        if _is_separator(cells):
            continue
        rows.append(_typed(dict(zip(names, cells))))
    return rows


def _render(rows: list[dict]) -> list[str]:
    table = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    running = 0.0
    for r in rows:
        running += _cost(r)
        cells = [
            r["date"], r["run"], r["GPU"],
            "" if r["minutes"] is None else f"{r['minutes']:g}",
            "" if r["est. USD"] is None else f"{r['est. USD']:.4f}",
            "" if r["actual USD"] is None else f"{r['actual USD']:.4f}",
            f"{running:.4f}",
            r["phase"],
        ]
        table.append("| " + " | ".join(cells) + " |")
    return table


def _write_table(path: Path, rows: list[dict]) -> None:
    """Rewrite the table in place, keeping any text before and after it. Adds the table if absent."""
    lines = path.read_text().splitlines() if path.exists() else []
    start = _find_header(lines)
    if start is None:
        before = lines + ([""] if lines and lines[-1].strip() else [])
        after: list[str] = []
    else:
        end = start + 1
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        before, after = lines[:start], lines[end:]
    path.write_text("\n".join(before + _render(rows) + after) + "\n")


def append_row(path: str | Path = "SPEND.md", date: str = "", run: str = "", gpu: str = "",
               minutes: float = 0.0, est_usd: float | None = None, actual_usd: float | None = None,
               phase: str = DEFAULT_PHASE) -> None:
    """Append one run and recompute every 'cumulative' value. Adds the phase column if the header lacks it."""
    p = Path(path)
    rows = read_rows(p)
    rows.append({"date": date, "run": run, "GPU": gpu, "minutes": float(minutes),
                 "est. USD": est_usd, "actual USD": actual_usd, "cumulative": None, "phase": phase})
    _write_table(p, rows)


def check_caps(path: str | Path = "SPEND.md", phase_cap: float = 25.0, total_cap: float = 200.0,
               phase: str = DEFAULT_PHASE, *, new_est: float) -> None:
    """Raise SpendCapError if adding new_est would take the phase past phase_cap or the total past total_cap."""
    rows = read_rows(path)
    phase_total = _spent(rows, phase) + new_est
    total = _spent(rows) + new_est
    if phase_total > phase_cap:
        raise SpendCapError(f"phase {phase} would reach ${phase_total:.2f}, cap is ${phase_cap:.2f}")
    if total > total_cap:
        raise SpendCapError(f"total would reach ${total:.2f}, cap is ${total_cap:.2f}")
