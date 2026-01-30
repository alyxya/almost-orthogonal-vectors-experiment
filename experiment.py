import argparse
import math
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_LOGSPACE_MIN = 10
DEFAULT_LOGSPACE_MAX = 10000
DEFAULT_LOGSPACE_POINTS = 13


def logspace_dims(start: int, stop: int, points: int) -> List[int]:
    if start <= 0 or stop <= 0:
        raise ValueError("logspace bounds must be positive")
    if start >= stop:
        raise ValueError("logspace min must be smaller than max")
    if points < 2:
        raise ValueError("logspace points must be at least 2")
    values = np.logspace(math.log10(start), math.log10(stop), num=points)
    dims = sorted({int(round(v)) for v in values})
    dims = [d for d in dims if start <= d <= stop]
    if start not in dims:
        dims.insert(0, start)
    if stop not in dims:
        dims.append(stop)
    return dims


def parse_dims(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    if ":" in raw:
        parts = [p for p in raw.split(":") if p != ""]
        if len(parts) not in (2, 3):
            raise ValueError("range must be start:stop or start:stop:step")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0:
            raise ValueError("step must be positive")
        return list(range(start, stop + 1, step))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def grow_set(
    n: int,
    rng: np.random.Generator,
    epsilon: float,
    max_fails: int,
    candidates: int,
) -> Tuple[int, int]:
    vectors = np.empty((0, n), dtype=np.float64)
    fails = 0
    steps = 0
    while fails < max_fails:
        steps += 1
        cand = rng.standard_normal((candidates, n))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if vectors.shape[0] == 0:
            vectors = np.vstack([vectors, cand[0]])
            fails = 0
            continue
        dots = vectors @ cand.T
        valid = np.all(np.abs(dots) <= epsilon, axis=0)
        if not np.any(valid):
            fails += 1
            continue
        valid_idx = np.where(valid)[0]
        # Prefer candidates that are as aligned as possible while still valid.
        scores = np.sum(np.abs(dots[:, valid_idx]), axis=0)
        best_idx = valid_idx[int(np.argmax(scores))]
        vectors = np.vstack([vectors, cand[best_idx]])
        fails = 0
    return vectors.shape[0], steps


def summarize(values: Iterable[int]) -> Tuple[int, int, float]:
    arr = np.array(list(values), dtype=np.float64)
    return int(arr.min()), int(arr.max()), float(arr.mean())


def write_plot(
    rows: Sequence[Tuple[int, int, int, float]],
    path: str,
    logy: bool,
    epsilon: float,
    trials: int,
    max_fails: int,
    candidates: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [r[0] for r in rows]
    mins = [r[1] for r in rows]
    maxs = [r[2] for r in rows]
    means = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(dims, means, marker="o", linewidth=1.8, label="mean")
    ax.fill_between(dims, mins, maxs, alpha=0.2, label="min-max band")
    ax.set_xlabel("dimension n")
    ax.set_ylabel("set size")
    ax.set_title(
        "Greedy epsilon-almost-orthogonal vectors "
        f"(epsilon={epsilon}, trials={trials}, max_fails={max_fails}, candidates={candidates})"
    )
    if logy:
        ax.set_yscale("log")
        ax.set_ylabel("set size (log scale)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def with_log_suffix(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext:
        return f"{base}-log{ext}"
    return f"{path}-log"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Greedy experiment for epsilon-almost-orthogonal vectors. "
            "Tries to grow a set by sampling batches of random unit vectors, "
            "picking the most aligned valid candidate, and stopping after a "
            "fixed number of consecutive failed batches."
        )
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Almost-orthogonal threshold (default: 0.1)",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="",
        help=(
            "Comma-separated list (e.g. 10,20,50) or range start:stop:step "
            "(inclusive). If omitted, uses a log-spaced range."
        ),
    )
    parser.add_argument(
        "--logspace-min",
        type=int,
        default=DEFAULT_LOGSPACE_MIN,
        help=f"Logspace min dimension (default: {DEFAULT_LOGSPACE_MIN})",
    )
    parser.add_argument(
        "--logspace-max",
        type=int,
        default=DEFAULT_LOGSPACE_MAX,
        help=f"Logspace max dimension (default: {DEFAULT_LOGSPACE_MAX})",
    )
    parser.add_argument(
        "--logspace-points",
        type=int,
        default=DEFAULT_LOGSPACE_POINTS,
        help=f"Number of logspace points (default: {DEFAULT_LOGSPACE_POINTS})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of independent trials per dimension (default: 5)",
    )
    parser.add_argument(
        "--max-fails",
        type=int,
        default=100,
        help="Stop after this many consecutive failed batches (default: 100)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="Number of candidates sampled per batch (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base RNG seed (default: 12345)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results.csv",
        help="Path to write per-trial results as CSV (default: results.csv)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="summary.csv",
        help="Path to write summary results as CSV (default: summary.csv)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="plot.png",
        help="Output path for the linear PNG plot (default: plot.png); also writes a -log variant",
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable per-trial CSV output")
    parser.add_argument("--no-summary", action="store_true", help="Disable summary CSV output")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot output")
    args = parser.parse_args()

    dims = parse_dims(args.dims)
    if not dims:
        dims = logspace_dims(args.logspace_min, args.logspace_max, args.logspace_points)
    epsilon = float(args.epsilon)

    header = "dimension | min | max | mean"
    print(header)
    print("-" * len(header))

    csv_rows = []
    summary_rows: List[Tuple[int, int, int, float]] = []
    for n in dims:
        sizes: List[int] = []
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + n * 1000 + t)
            size, steps = grow_set(n, rng, epsilon, args.max_fails, args.candidates)
            sizes.append(size)
            csv_rows.append((n, t, size, steps))
        min_v, max_v, mean_v = summarize(sizes)
        summary_rows.append((n, min_v, max_v, mean_v))
        mean_display = f"{mean_v:.1f}" if not math.isclose(mean_v, round(mean_v)) else str(int(round(mean_v)))
        print(f"{n} | {min_v} | {max_v} | {mean_display}")

    if args.csv and not args.no_csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("dimension,trial,size,steps\n")
            for n, t, size, steps in csv_rows:
                f.write(f"{n},{t},{size},{steps}\n")

    if args.summary and not args.no_summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            f.write("dimension,min,max,mean\n")
            for n, min_v, max_v, mean_v in summary_rows:
                f.write(f"{n},{min_v},{max_v},{mean_v:.6f}\n")

    if args.plot and not args.no_plot:
        write_plot(
            summary_rows,
            args.plot,
            False,
            epsilon,
            args.trials,
            args.max_fails,
            args.candidates,
        )
        write_plot(
            summary_rows,
            with_log_suffix(args.plot),
            True,
            epsilon,
            args.trials,
            args.max_fails,
            args.candidates,
        )


if __name__ == "__main__":
    main()
