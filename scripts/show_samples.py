"""Draw ground-truth boxes on (augmented) dataset samples -> docs/figures/. Owner: Mariam Abdelaal

    python scripts/show_samples.py --config configs/centernet_voc5.yaml
    python scripts/show_samples.py --config configs/centernet_voc5.yaml --set data.voc_root=/path/to/VOCdevkit
    python scripts/show_samples.py --fake       # no VOC needed: runs on a generated VOC-format folder

Each row is one training image: the first panel is the letterboxed image without augmentation,
the others are random augmentations of it (crop, flip, colour). Boxes are drawn exactly as the
dataset returns them, so the figure is also a visual check that boxes follow the augmentation.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import dataset as D  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402

COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]  # one per class


def draw(ax, image, target, title):
    ax.imshow(D.denormalize(image)[..., ::-1])
    for box, lab in zip(target["boxes"].tolist(), target["labels"].tolist()):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.8,
                                       edgecolor=COLOURS[lab]))
        ax.text(x1, y1 - 1, D.CLASSES[lab], fontsize=7, color="white", va="bottom",  # above the box
                bbox=dict(facecolor=COLOURS[lab], edgecolor="none", pad=1))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/centernet_voc5.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.voc_root=...")
    ap.add_argument("--images", type=int, default=4, help="rows (different images)")
    ap.add_argument("--augs", type=int, default=3, help="augmented versions per image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/figures/augmented_samples.png")
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
    kw = dict(sets=[tuple(s) for s in data["train_sets"]], split="train", input_size=cfg["input_size"],
              val_fraction=data.get("val_fraction", 0.1), seed=data.get("seed", 0))
    plain, augmented = D.VOC5Dataset(root, augment=False, **kw), D.VOC5Dataset(root, augment=True, **kw)

    torch.manual_seed(args.seed)
    picks = np.random.default_rng(args.seed).choice(len(plain), size=min(args.images, len(plain)),
                                                    replace=False)
    cols = 1 + args.augs
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.6 * cols, 2.6 * len(picks)), squeeze=False)
    for row, idx in enumerate(picks):
        image, target = plain[int(idx)]
        draw(axes[row, 0], image, target, f"{target['image_id']} (no augmentation)")
        for c in range(1, cols):
            image, target = augmented[int(idx)]
            draw(axes[row, c], image, target, f"augmented {c}")
    handles = [patches.Patch(color=COLOURS[i], label=n) for i, n in enumerate(D.CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(D.CLASSES), frameon=False, fontsize=9)
    fig.suptitle("Training samples after letterbox to 384x384: original and random crop / flip / colour",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
