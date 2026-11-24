"""Sanity check: overfit a few training images and draw the predictions. Owner: Abdallah Afifi

    python scripts/overfit_check.py --config configs/centernet_voc5.yaml      # real VOC images
    python scripts/overfit_check.py --config configs/smoke_synthetic.yaml     # no data needed

A from-scratch detector should memorise a handful of images within a few hundred steps. If it
cannot, the bug is in the data, target encoding, loss or decoding, not in the training schedule.
Saves docs/figures/overfit_check.png (ground truth green, predictions red) for the checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from scripts.train import build_datasets  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train_utils import EncodedDataset, detection_collate, pick_device  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def to_rgb(image: torch.Tensor) -> np.ndarray:
    return (np.clip(image.permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1) * 255).astype(np.uint8).copy()


def draw(img, boxes, labels, names, color, scores=None):
    for i, (box, label) in enumerate(zip(boxes.tolist(), labels.tolist())):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = names[label] + (f" {float(scores[i]):.2f}" if scores is not None else "")
        cv2.putText(img, text, (x1 + 2, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/centernet_voc5.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=ROOT / "docs/figures/overfit_check.png")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), ["augment=false", *args.set])
    device = pick_device(args.device)
    torch.manual_seed(0)
    train_ds, _ = build_datasets(cfg)
    model = build_model(cfg).to(device)
    data = EncodedDataset(Subset(train_ds, range(min(args.images, len(train_ds)))), model.target_encoder())
    images, enc, targets = next(iter(DataLoader(data, batch_size=args.images, collate_fn=detection_collate)))
    images = images.to(device)
    enc = {k: v.to(device) for k, v in enc.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for step in range(1, args.steps + 1):
        loss, parts = model.compute_loss(model(images), enc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} loss {float(loss):.4f} " + " ".join(f"{k} {float(v):.3f}" for k, v in parts.items()))

    model.eval()
    with torch.no_grad():
        dets = model.decode(model(images), score_thresh=0.3)
    names = cfg["classes"]
    cols = min(4, len(dets))
    rows = (len(dets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax, img, det, gt in zip(axes.ravel(), images, dets, targets):
        canvas = draw(to_rgb(img), gt["boxes"], gt["labels"], names, (0, 200, 0))
        canvas = draw(canvas, det["boxes"].cpu(), det["labels"].cpu(), names, (230, 0, 0), det["scores"].cpu())
        ax.imshow(canvas)
        ax.set_title(f"{len(gt['boxes'])} true / {len(det['boxes'])} predicted", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Overfit check after {args.steps} steps (green = ground truth, red = prediction)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
