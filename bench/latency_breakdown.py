"""Stacked bars of the §11.3 latency terms, grouped by policy x regime.

Input: a CSV with the LatencyLog.to_rows() columns plus 'policy' and 'regime'.
Writes <out>_mean and <out>_p99 (each as .png and .svg). The P99 bars stack per-term P99s, so their total
is not the P99 of E2E.

  python -m bench.latency_breakdown in.csv out_prefix
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from bench.common import read_csv
from bench.plots import new_figure, save_figure

TERMS = ("queue_wait", "adapter_wait", "text_encode", "denoise", "vae_decode", "image_encode", "transfer")


def _stat(values: np.ndarray, stat: str) -> float:
    return float(values.mean()) if stat == "mean" else float(np.percentile(values, 99))


def plot_latency_breakdown(csv, out) -> None:
    _, rows = read_csv(csv)
    if not rows:
        raise ValueError("no rows to plot")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(str(r["policy"]), str(r["regime"]))].append(r)
    keys = sorted(groups)
    labels = [f"{policy} / {regime}" for policy, regime in keys]
    x = np.arange(len(keys))
    for stat in ("mean", "p99"):
        values = {t: np.asarray([_stat(np.asarray([float(r[t]) for r in groups[k]]), stat) for k in keys])
                  for t in TERMS}
        fig = new_figure(max(7.0, 1.2 * len(keys)), 4.8)
        ax = fig.add_subplot(1, 1, 1)
        bottom = np.zeros(len(keys))
        for t in TERMS:
            ax.bar(x, values[t], bottom=bottom, label=t)
            bottom += values[t]
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("ms")
        ax.set_title(f"Latency breakdown, {'mean' if stat == 'mean' else 'P99 per term'}")
        ax.legend(fontsize="small", bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        save_figure(fig, f"{out}_{stat}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.latency_breakdown")
    parser.add_argument("csv")
    parser.add_argument("out_prefix")
    args = parser.parse_args(argv)
    plot_latency_breakdown(args.csv, args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
