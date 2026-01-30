import argparse
import csv
import os
from typing import List, Tuple

import matplotlib


def read_summary(path: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    dims: List[int] = []
    mins: List[float] = []
    maxs: List[float] = []
    means: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dims.append(int(float(row["dimension"])))
            mins.append(float(row["min"]))
            maxs.append(float(row["max"]))
            means.append(float(row["mean"]))
    order = sorted(range(len(dims)), key=lambda i: dims[i])
    dims = [dims[i] for i in order]
    mins = [mins[i] for i in order]
    maxs = [maxs[i] for i in order]
    means = [means[i] for i in order]
    return dims, mins, maxs, means


def with_log_suffix(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext:
        return f"{base}-log{ext}"
    return f"{path}-log"


def write_plot(
    dims: List[int],
    mins: List[float],
    maxs: List[float],
    means: List[float],
    path: str,
    title: str,
    subtitle: str,
    logy: bool,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(dims, means, marker="o", linewidth=2.0, label="mean")
    ax.fill_between(dims, mins, maxs, alpha=0.18, label="min-max band")
    ax.set_xlabel("dimension n")
    ax.set_ylabel("set size")
    ax.set_title(title, fontsize=13, pad=10)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    if logy:
        ax.set_yscale("log")
        ax.set_ylabel("set size (log scale)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate blog-ready plots from summary.csv output."
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="summary.csv",
        help="Path to summary CSV (default: summary.csv)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="plot.png",
        help="Output path for the linear plot (default: plot.png)",
    )
    parser.add_argument(
        "--log-plot",
        type=str,
        default="",
        help="Output path for the log plot (default: <plot>-log.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Greedy almost-orthogonal vectors",
        help="Plot title (default: Greedy almost-orthogonal vectors)",
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default="",
        help="Optional subtitle line shown above the axes",
    )
    args = parser.parse_args()

    dims, mins, maxs, means = read_summary(args.summary)
    log_plot = args.log_plot or with_log_suffix(args.plot)

    write_plot(
        dims,
        mins,
        maxs,
        means,
        args.plot,
        args.title,
        args.subtitle,
        logy=False,
    )
    write_plot(
        dims,
        mins,
        maxs,
        means,
        log_plot,
        args.title,
        args.subtitle,
        logy=True,
    )


if __name__ == "__main__":
    main()
