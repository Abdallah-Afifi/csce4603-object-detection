"""Metrics. Owner: Youssef Badawy

Interface used by scripts/train.py to pick best.pt by validation mAP (until this exists, training
selects best.pt by validation loss and still works):

    voc_map(predictions, ground_truths, num_classes, iou_thresh=0.5) -> {"mAP": float, "AP": [float] * num_classes}
        predictions  : list (one per image) of {"boxes": [M,4] xyxy, "scores": [M], "labels": [M]}
                       (exactly what model.decode returns)
        ground_truths: list (one per image) of {"boxes": [N,4], "labels": [N], optional "difficult": [N] bool}
    A perfect prediction must give mAP == 1.0 (train.py checks this before trusting it).
    src.boxes.box_iou is available for matching.

Also planned here (after the checkpoint):
- precision / recall at a chosen score threshold (bonus target: both > 0.85), and the full P-R curve
- detection confusion matrix (matched boxes -> class confusion, plus missed / false-positive rows)
- FPS on video
"""
import numpy as np
import torch

from src.boxes import box_iou


def average_precision(recall, precision):
    """VOC 2010+ AP: area under the precision-recall curve after making precision non-increasing."""
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):          # precision envelope (max to the right)
        p[i] = max(p[i], p[i + 1])
    changes = np.where(r[1:] != r[:-1])[0]        # points where recall changes
    return float(np.sum((r[changes + 1] - r[changes]) * p[changes + 1]))


def voc_map(predictions, ground_truths, num_classes, iou_thresh=0.5):
    aps = []
    for c in range(num_classes):
        # all detections of class c over all images: (score, image index, box)
        dets = []
        for i, pred in enumerate(predictions):
            mask = pred["labels"] == c
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                dets.append((float(score), i, box))
        dets.sort(key=lambda d: -d[0])           # highest score first

        # ground-truth boxes of class c per image; "difficult" boxes are neither counted nor punished
        gt_boxes, gt_difficult, used = [], [], []
        n_gt = 0
        for gt in ground_truths:
            mask = gt["labels"] == c
            boxes = gt["boxes"][mask]
            # as bool: with 0/1 integer flags, ~difficult would flip bits (~1 = -2) instead of negating
            difficult = (torch.as_tensor(gt["difficult"], dtype=torch.bool)[mask] if "difficult" in gt
                         else torch.zeros(len(boxes), dtype=torch.bool))
            gt_boxes.append(boxes)
            gt_difficult.append(difficult)
            used.append(np.zeros(len(boxes), dtype=bool))
            n_gt += int((~difficult).sum())

        if n_gt == 0:                            # class not in this set: leave it out of the mean
            aps.append(float('nan'))
            continue

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for k, (score, i, box) in enumerate(dets):
            if len(gt_boxes[i]) == 0:
                fp[k] = 1
                continue
            ious = box_iou(box.reshape(1, 4).float(), gt_boxes[i].float())[0]
            j = int(ious.argmax())
            if ious[j] >= iou_thresh:
                if gt_difficult[i][j]:
                    continue                     # matched a difficult box: ignore this detection
                if not used[i][j]:
                    tp[k] = 1
                    used[i][j] = True
                else:
                    fp[k] = 1                    # duplicate detection of the same object
            else:
                fp[k] = 1

        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        aps.append(average_precision(recall, precision))

    valid = [a for a in aps if not np.isnan(a)]
    return {"mAP": float(np.mean(valid)) if valid else 0.0, "AP": aps}
