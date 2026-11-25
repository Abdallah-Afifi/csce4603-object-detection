"""Score trained models on the VOC 2007 test set (or val) and write the report numbers. Owner: Youssef Badawy

    python scripts/evaluate.py runs/A_centernet runs/B_noaug runs/C_grid --out docs/results/comparison
    python scripts/evaluate.py runs/checkpoint_voc07/best.pt --max-images 500      # quick look

For every run (a run folder means its best.pt):
  1. Validation set: find the score threshold with the best F1. The operating point is chosen on
     val, never on test, so the test numbers are not tuned on the test set.
  2. Test set (data.test_set in the run's config, VOC 2007 test by default): mAP@0.5 (all-point and
     the official VOC2007 11-point), per-class AP, and precision / recall / F1 / error rate at the val
     threshold. Also the detection confusion matrix, the precision-recall curve, and FPS (batch 1,
     model + decode).
Predictions and ground truth are compared in the 384x384 network-input space, as in training.

Writes <out>/<run>/metrics.json, confusion_matrix.png, pr_curve.png and <out>/summary.md, a table
with one row per run (the comparison table for the report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src import dataset as D  # noqa: E402
from src.evaluate import (best_f1_threshold, detection_confusion_matrix, error_rate,  # noqa: E402
                          pr_curves, precision_recall, voc_map)
from src.model import load_model  # noqa: E402
from src.train_utils import pick_device  # noqa: E402


def _collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def predict_dataset(model, dataset, device, batch_size=16, workers=2):
    preds, gts = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=_collate)
    for images, targets in loader:
        out = model(images.to(device))
        for det in model.decode(out, score_thresh=0.01, max_det=100):
            preds.append({k: v.cpu() for k, v in det.items()})
        for t in targets:
            gts.append({k: t[k] for k in ("boxes", "labels", "difficult") if k in t})
    return preds, gts


@torch.no_grad()
def measure_fps(model, dataset, device, n=50):
    sync = {"cuda": torch.cuda.synchronize, "mps": getattr(torch.mps, "synchronize", lambda: None)}.get(
        device.type, lambda: None)
    images = [dataset[i][0] for i in range(min(n, len(dataset)))]
    for img in images[:5]:
        model.decode(model(img[None].to(device)))
    sync()
    t0 = time.perf_counter()
    for img in images:
        model.decode(model(img[None].to(device)))
    sync()
    return len(images) / (time.perf_counter() - t0)


def plot_confusion(matrix, names, path, title):
    labels = list(names) + ["background"]
    norm = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), [*names, "missed"], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), [*names, "false pos."])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr(curves, names, threshold_point, path, title):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for name, (_, precision, recall) in zip(names, curves["per_class"]):
        ax.plot(recall, precision, lw=1.2, label=name)
    _, precision, recall = curves["overall"]
    ax.plot(recall, precision, "k", lw=2, label="all classes")
    ax.plot(threshold_point["recall"], threshold_point["precision"], "ro",
            label=f"val-chosen threshold {threshold_point['score_thresh']:.2f}")
    ax.axhline(0.85, color="grey", ls=":", lw=1)
    ax.axvline(0.85, color="grey", ls=":", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", ylabel="precision", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_run(weights, args, device):
    model, cfg = load_model(weights, device=device)
    names, n = cfg["classes"], len(cfg["classes"])
    data = cfg["data"]
    root = Path(args.voc_root or data["voc_root"])
    root = root if root.is_absolute() else ROOT / root
    size = cfg["input_size"]
    common = dict(input_size=size, val_fraction=data.get("val_fraction", 0.1), seed=data.get("seed", 0))
    val = D.VOC5Dataset(root, sets=[tuple(s) for s in data["train_sets"]], split="val", **common)
    test = D.VOC5Dataset(root, sets=[tuple(args.test_set or data["test_set"])], split="test", **common)
    if args.max_images:
        val = Subset(val, range(min(args.max_images, len(val))))
        test = Subset(test, range(min(args.max_images, len(test))))

    vp, vg = predict_dataset(model, val, device, args.batch_size, args.workers)
    chosen = best_f1_threshold(vp, vg, n)
    thresh = args.thresh if args.thresh is not None else chosen["score_thresh"]

    tp, tg = predict_dataset(model, test, device, args.batch_size, args.workers)
    m_all, m_07 = voc_map(tp, tg, n), voc_map(tp, tg, n, use_07_metric=True)
    pr = precision_recall(tp, tg, n, score_thresh=thresh)
    result = {
        "weights": str(weights), "epoch": torch.load(weights, map_location="cpu", weights_only=False).get("epoch"),
        "test_images": len(test), "val_images": len(val),
        "mAP": m_all["mAP"], "mAP_voc07_11pt": m_07["mAP"],
        "AP": dict(zip(names, m_all["AP"])),
        "threshold": thresh, "threshold_from": "fixed" if args.thresh is not None else "best F1 on val",
        "val_at_threshold": chosen,
        "precision": pr["overall"]["precision"], "recall": pr["overall"]["recall"], "f1": pr["overall"]["f1"],
        "error_rate": error_rate(tp, tg, n, score_thresh=thresh),
        "per_class": {name: {k: c[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_gt")}
                      for name, c in zip(names, pr["per_class"])},
        "fps": measure_fps(model, test, device),
        "device": str(device),
        "bonus_met": pr["overall"]["precision"] > 0.85 and pr["overall"]["recall"] > 0.85,
    }
    matrix = detection_confusion_matrix(tp, tg, n, score_thresh=thresh)
    result["confusion_matrix"] = matrix.tolist()

    name = Path(weights).parent.name if Path(weights).name in ("best.pt", "last.pt") else Path(weights).stem
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    plot_confusion(matrix, names, out / "confusion_matrix.png",
                   f"{name}: test, score >= {thresh:.2f}, IoU >= 0.5")
    test_pr_at = {"score_thresh": thresh, "precision": result["precision"], "recall": result["recall"]}
    plot_pr(pr_curves(tp, tg, n), names, test_pr_at, out / "pr_curve.png", f"{name}: precision-recall on test")
    return name, result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run folders (uses best.pt) or .pt files")
    p.add_argument("--out", type=Path, default=ROOT / "docs/results/evaluation")
    p.add_argument("--voc-root", default=None, help="override data.voc_root from the run's config")
    p.add_argument("--test-set", nargs=2, default=None, metavar=("YEAR", "SET"), help="default: data.test_set")
    p.add_argument("--thresh", type=float, default=None, help="fixed score threshold instead of best-F1 on val")
    p.add_argument("--max-images", type=int, default=None, help="evaluate only the first N images (quick look)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.test_set:
        args.test_set = (int(args.test_set[0]), args.test_set[1])
    device = pick_device(args.device)

    rows = []
    for run in args.runs:
        weights = run / "best.pt" if run.is_dir() else run
        name, r = evaluate_run(weights, args, device)
        rows.append((name, r))
        print(f"{name}: mAP {r['mAP']:.3f} (VOC07 11-pt {r['mAP_voc07_11pt']:.3f}) | P {r['precision']:.3f} "
              f"R {r['recall']:.3f} F1 {r['f1']:.3f} at {r['threshold']:.2f} | error {r['error_rate']:.3f} | "
              f"{r['fps']:.0f} FPS")

    names = list(rows[0][1]["AP"])
    lines = ["| Run | epoch | mAP@0.5 | mAP (VOC07 11-pt) | " + " | ".join(f"AP {c}" for c in names)
             + " | threshold | precision | recall | F1 | error rate | FPS |",
             "|---" * (11 + len(names)) + "|"]
    for name, r in rows:
        lines.append(f"| {name} | {r['epoch']} | {r['mAP']:.3f} | {r['mAP_voc07_11pt']:.3f} | "
                     + " | ".join(f"{r['AP'][c]:.3f}" for c in names)
                     + f" | {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                       f"{r['error_rate']:.3f} | {r['fps']:.0f} |")
    lines.append("")
    lines.append(f"Test set: {rows[0][1]['test_images']} images. Threshold = best F1 on the validation split "
                 "(not tuned on test). Error rate = (FP + FN) / (TP + FP + FN). FPS = batch 1, model + decode, "
                 f"on {rows[0][1]['device']}.")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
