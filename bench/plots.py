"""Plots from benchmark CSVs written by bench.common.write_csv. Pure matplotlib on the Agg canvas.

Every function writes <out>.png and <out>.svg.

  python -m bench.plots overhead in.csv out_prefix
  python -m bench.plots sim in.csv out_prefix
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bench.common import read_csv


def new_figure(width: float = 7.0, height: float = 4.5) -> Figure:
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def save_figure(fig: Figure, out) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=150)
    fig.savefig(f"{out}.svg")


def plot_overhead_vs_tokens(csv, out) -> None:
    """One line per baseline: overhead_pct against tokens on a log-2 x axis."""
    _, rows = read_csv(csv)
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if isinstance(r.get("overhead_pct"), (int, float)):   # unavailable/error baselines have an empty overhead
            series[str(r["baseline"])].append((float(r["tokens"]), float(r["overhead_pct"])))
    fig = new_figure()
    ax = fig.add_subplot(1, 1, 1)
    for name in sorted(series):
        pts = sorted(series[name])
        ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens (log scale)")
    ax.set_ylabel("overhead vs base (%)")
    ax.set_title("Unmerged LoRA path overhead vs tokens")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="baseline")
    fig.tight_layout()
    save_figure(fig, out)


def plot_sim(csv, out) -> None:
    """Two panels, one line per policy: throughput vs rho, and P99 latency vs rho."""
    _, rows = read_csv(csv)
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_policy[str(r["policy"])].append(r)
    fig = new_figure(10.0, 4.5)
    ax_t, ax_p = fig.subplots(1, 2)
    for policy in sorted(by_policy):
        pts = sorted(by_policy[policy], key=lambda r: float(r["rho"]))
        rho = [float(r["rho"]) for r in pts]
        ax_t.plot(rho, [float(r["throughput_rps"]) for r in pts], marker="o", label=policy)
        ax_p.plot(rho, [float(r["p99_ms"]) for r in pts], marker="o", label=policy)
    for ax, title, ylabel in ((ax_t, "Throughput vs load", "throughput"),
                              (ax_p, "P99 latency vs load", "P99 (ms)")):
        ax.set_title(title)
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(title="policy")
    fig.tight_layout()
    save_figure(fig, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.plots")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("overhead", plot_overhead_vs_tokens), ("sim", plot_sim)):
        p = sub.add_parser(name)
        p.add_argument("csv")
        p.add_argument("out_prefix")
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    args.func(args.csv, args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
