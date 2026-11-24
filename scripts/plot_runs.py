"""Compare training runs on one figure (for the report's comparison section). Owner: Abdallah Afifi

    python scripts/plot_runs.py runs/A_centernet runs/B_noaug runs/C_grid --out docs/figures/runs.png

Plots validation loss and validation mAP@0.5 per epoch for each run, and prints a table of each
run's best epoch. Loss values are only comparable between runs that use the same head.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(run: Path):
    with open(run / "history.csv") as f:
        return list(csv.DictReader(f))


def series(rows, key):
    pts = [(int(r["epoch"]), float(r[key])) for r in rows if r.get(key) not in (None, "")]
    return list(zip(*pts)) if pts else ([], [])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs/figures/runs.png"))
    args = p.parse_args(argv)

    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(11, 4))
    print(f"{'run':<24}{'epochs':>7}{'best val loss':>15}{'best mAP@0.5':>14}{'at epoch':>10}")
    for run in args.runs:
        rows = read(run)
        ax_loss.plot(*series(rows, "train_total"), "--", alpha=0.5, label=f"{run.name} train")
        ax_loss.plot(*series(rows, "val_total"), label=f"{run.name} val")
        ep, maps = series(rows, "val_mAP")
        if maps:
            ax_map.plot(ep, maps, marker="o", label=run.name)
        _, vloss = series(rows, "val_total")
        best_map = max(maps) if maps else float("nan")
        best_ep = ep[maps.index(best_map)] if maps else "-"
        print(f"{run.name:<24}{len(rows):>7}{min(vloss) if vloss else float('nan'):>15.4f}"
              f"{best_map:>14.4f}{best_ep:>10}")
    ax_loss.set(title="loss (compare same-head runs only)", xlabel="epoch")
    ax_map.set(title="validation mAP@0.5", xlabel="epoch", ylim=(0, 1))
    for ax in (ax_loss, ax_map):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
