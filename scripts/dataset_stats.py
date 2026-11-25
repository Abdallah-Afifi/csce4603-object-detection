"""Images and objects per class in each split -> docs/figures/. Owner: Mariam Abdelaal

    python scripts/dataset_stats.py --config configs/centernet_voc5.yaml
    python scripts/dataset_stats.py --config configs/centernet_voc5.yaml --set data.voc_root=/kaggle/input/voc
    python scripts/dataset_stats.py --fake      # no VOC needed: runs on a generated VOC-format folder

Writes class_counts.csv, class_counts.md (table for the README / report) and class_counts.png.
Counts are before augmentation. "objects" for train excludes difficult objects (they are
dropped); for val and test, difficult objects are counted and also shown separately, because
mAP ignores them.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import dataset as D  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/centernet_voc5.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.voc_root=...")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--fake", action="store_true", help="use a generated VOC-format folder")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(ROOT / args.config), args.set)
    data = cfg["data"]
    if args.fake:
        from tests.fake_voc import make_fake_voc
        root = make_fake_voc(Path(tempfile.mkdtemp()) / "VOCdevkit")
    else:
        root = Path(data["voc_root"])
        root = root if root.is_absolute() else ROOT / root
    kw = dict(input_size=cfg["input_size"], val_fraction=data.get("val_fraction", 0.1),
              seed=data.get("seed", 0))
    sets = [tuple(s) for s in data["train_sets"]]
    splits = {"train": D.VOC5Dataset(root, sets=sets, split="train", **kw),
              "val": D.VOC5Dataset(root, sets=sets, split="val", **kw)}
    test_set = data.get("test_set")
    if test_set:
        try:
            splits["test"] = D.VOC5Dataset(root, sets=[tuple(test_set)], split="test", **kw)
        except FileNotFoundError as e:
            print(f"(skipping test split: {e})")

    counts = {name: ds.class_counts() for name, ds in splits.items()}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "images", "objects", "difficult_objects"])
        for name, c in counts.items():
            for cls in D.CLASSES:
                w.writerow([name, cls, c[cls]["images"], c[cls]["objects"], c[cls]["difficult"]])

    lines = ["| Class | " + " | ".join(f"{s} images | {s} objects" for s in counts) + " |",
             "|---|" + "---:|---:|" * len(counts)]
    for cls in D.CLASSES:
        cells = [f"{counts[s][cls]['images']} | {counts[s][cls]['objects']}" for s in counts]
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    totals = [f"{len(splits[s])} | {sum(counts[s][c]['objects'] for c in D.CLASSES)}" for s in counts]
    lines.append("| **all (images, objects)** | " + " | ".join(totals) + " |")
    lines.append("")
    for s, ds in splits.items():
        objs = sum(counts[s][c]["objects"] for c in D.CLASSES)
        share = 100 * counts[s]["person"]["objects"] / max(1, objs)
        hard = sum(counts[s][c]["difficult"] for c in D.CLASSES)
        extra = {k: v for k, v in ds.info.items() if k != "images"}
        lines.append(f"- {s}: person is {share:.1f}% of objects; difficult objects: {hard}; "
                     f"bookkeeping: {dict(sorted(extra.items())) or '-'}")
    table = "\n".join(lines) + "\n"
    (out / "class_counts.md").write_text(table)
    print(table)

    x = np.arange(len(D.CLASSES))
    width = 0.8 / len(counts)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, (s, c) in enumerate(counts.items()):
        vals = [c[cls]["objects"] for cls in D.CLASSES]
        bars = ax.bar(x + (k - (len(counts) - 1) / 2) * width, vals, width, label=s)
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x, D.CLASSES)
    ax.set_ylabel("objects")
    ax.set_title("Objects per class (PASCAL VOC, our 5 classes)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=150)
    print(f"saved {out / 'class_counts.csv'}, class_counts.md, class_counts.png")


if __name__ == "__main__":
    main()
