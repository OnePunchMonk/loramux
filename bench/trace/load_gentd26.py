"""Loader for a GenTD26-style request trace stored as CSV.

The trace's real schema is not known when this was written, so the loader is driven by a column map.

Default columns (header names):
    timestamp    arrival time, seconds (float) or ISO-8601 text; rebased so the first request is at t=0
    request_id   any string
    adapters     adapter ids separated by ';' (empty = base model)
    steps, width, height   optional ints (defaults 28, 1024, 1024)
    scales       optional ';'-separated floats, one per adapter (default 1.0 each)
    prompt_id    optional int (default: row index)

Map real column names with a JSON object from canonical name to the file's column name, either inline or as a file:
    python -m bench.trace.load_gentd26 trace.csv --column-map '{"timestamp": "ts", "adapters": "lora_names"}'
Canonical names not mentioned keep their default header. A canonical column whose header is absent from the
file is skipped if optional; a missing required one (timestamp, request_id, adapters) raises KeyError.
Options: `adapter_sep` (default ';'). Rows are returned sorted by arrival time.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from .synth_trace import Request, trace_stats

REQUIRED = ("timestamp", "request_id", "adapters")
OPTIONAL = ("steps", "width", "height", "scales", "prompt_id")
DEFAULTS = {"steps": 28, "width": 1024, "height": 1024}


def _parse_time(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return dt.datetime.fromisoformat(s.strip().replace("Z", "+00:00")).timestamp()


def load_gentd26(path, column_map: dict[str, str] | None = None, adapter_sep: str = ";") -> list[Request]:
    cmap = {c: c for c in REQUIRED + OPTIONAL}
    unknown = set(column_map or {}) - set(cmap)
    if unknown:
        raise KeyError(f"unknown canonical columns in column_map: {sorted(unknown)}")
    cmap.update(column_map or {})
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        for c in REQUIRED:
            if cmap[c] not in header:
                raise KeyError(f"required column {c!r} (file header {cmap[c]!r}) not in {sorted(header)}")
        has = {c: cmap[c] in header for c in OPTIONAL}
        rows = list(reader)
    reqs: list[Request] = []
    for i, row in enumerate(rows):
        ads = [a.strip() for a in row[cmap["adapters"]].split(adapter_sep) if a.strip()]
        scales = [1.0] * len(ads)
        if has["scales"] and row[cmap["scales"]].strip():
            scales = [float(x) for x in row[cmap["scales"]].split(adapter_sep)]
            if len(scales) != len(ads):
                raise ValueError(f"row {i}: {len(scales)} scales for {len(ads)} adapters")
        kw = {c: int(row[cmap[c]]) if has[c] and row[cmap[c]].strip() else DEFAULTS[c] for c in DEFAULTS}
        pid = int(row[cmap["prompt_id"]]) if has["prompt_id"] and row[cmap["prompt_id"]].strip() else i
        reqs.append(Request(id=row[cmap["request_id"]], t_arrive=_parse_time(row[cmap["timestamp"]]),
                            adapters=ads, scales=scales, prompt_id=pid, **kw))
    reqs.sort(key=lambda r: r.t_arrive)
    if reqs:
        t0 = reqs[0].t_arrive
        for r in reqs:
            r.t_arrive -= t0
    return reqs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--column-map", default=None, help="JSON object or path to a JSON file")
    args = ap.parse_args(argv)
    cm = None
    if args.column_map:
        p = Path(args.column_map)
        cm = json.loads(p.read_text() if p.exists() else args.column_map)
    reqs = load_gentd26(args.csv, cm)
    print(json.dumps(trace_stats([r.adapters for r in reqs]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
