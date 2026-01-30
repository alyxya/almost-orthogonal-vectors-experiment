# Almost-Orthogonal Vectors — Experiments

Small utilities for running the greedy almost-orthogonal vector experiment and generating blog-ready plots.

## Requirements

```bash
uv pip install numpy matplotlib
```

## Run the experiment

```bash
python experiment.py
```

Defaults:
- dimensions: linear range from 0 (exclusive) to 1000 with 20 points
- epsilon: 0.1
- candidates per step: 100
- stop after 100 failed batches
- outputs: `results.csv`, `summary.csv`, `plot.png`, `plot-log.png`

Override examples:

```bash
python experiment.py --dims 10:1000:50 --trials 3
python experiment.py --linear-points 30
```

## Generate plots from existing data

```bash
python plot_from_summary.py \
  --summary summary.csv \
  --plot almost-orthogonal.png \
  --title "Almost-orthogonal vectors" \
  --subtitle "ε=0.1 · candidates=100 · trials=5 · max_fails=100"
```

This writes:
- `almost-orthogonal.png`
- `almost-orthogonal-log.png`
