"""Metrics. Owner: Youssef Badawy

All functions take the same inputs:
    predictions  : list (one per image) of {"boxes": [M,4] xyxy, "scores": [M], "labels": [M]}
                   (exactly what model.decode returns)
    ground_truths: list (one per image) of {"boxes": [N,4], "labels": [N], optional "difficult": [N] bool}
Boxes of a prediction and its ground truth must be in the same coordinates (training and
scripts/evaluate.py both use the 384x384 network-input space).

Matching rule (PASCAL VOC): detections are taken from highest to lowest score; each is matched to
the ground-truth box of the same class with the highest IoU; IoU >= iou_thresh and not yet used ->
true positive; already used (duplicate) or IoU too low -> false positive; matched to a "difficult"
box -> ignored (neither TP nor FP). Difficult boxes are not counted as objects to find.

    voc_map(...)                    mAP and per-class AP. Used by scripts/train.py to pick best.pt.
                                    use_07_metric=False: all-point AP (VOC 2010+), the default
                                    use_07_metric=True : 11-point AP (official VOC 2007 test metric)
    precision_recall(..., score_thresh)   P, R, F1 per class and overall at one score threshold
    pr_curves(...)                  precision/recall at every threshold, per class and overall
    best_f1_threshold(...)          the score threshold with the highest overall F1 (choose it on val!)
    detection_confusion_matrix(...) (C+1)x(C+1) counts, last row/column = background:
                                    row "background" = false positives, column "background" = missed
    error_rate(...)                 (FP + FN) / (TP + FP + FN) at one threshold
"""
import numpy as np
import torch

from src.boxes import box_iou


def average_precision(recall, precision, use_07_metric=False):
    """Area under the precision-recall curve.

    use_07_metric=False: VOC 2010+ all-point AP (precision made non-increasing, exact area).
    use_07_metric=True : VOC 2007 11-point AP (mean of the best precision at recall 0, 0.1, ..., 1).
    """
    recall, precision = np.asarray(recall, float), np.asarray(precision, float)
    if use_07_metric:
        return float(np.mean([precision[recall >= t].max() if (recall >= t).any() else 0.0
                              for t in np.linspace(0, 1, 11)]))
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):          # precision envelope (max to the right)
        p[i] = max(p[i], p[i + 1])
    changes = np.where(r[1:] != r[:-1])[0]        # points where recall changes
    return float(np.sum((r[changes + 1] - r[changes]) * p[changes + 1]))


def _difficult(gt, n):
    # as bool: with 0/1 integer flags, ~difficult would flip bits (~1 = -2) instead of negating
    if "difficult" in gt:
        return torch.as_tensor(gt["difficult"], dtype=torch.bool).reshape(-1)
    return torch.zeros(n, dtype=torch.bool)


def match_class(predictions, ground_truths, c, iou_thresh=0.5):
    """VOC matching for class c over all images.

    Returns (scores, tp, fp, n_gt): per detection of class c sorted by score (high to low), whether
    it is a true or false positive (both 0 if it matched a difficult box), and the number of
    non-difficult ground-truth objects of class c.
    """
    dets = []                                     # (score, image index, box)
    for i, pred in enumerate(predictions):
        mask = pred["labels"] == c
        for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
            dets.append((float(score), i, box))
    dets.sort(key=lambda d: -d[0])               # highest score first

    gt_boxes, gt_difficult, used, n_gt = [], [], [], 0
    for gt in ground_truths:
        mask = gt["labels"] == c
        boxes = gt["boxes"][mask]
        difficult = _difficult(gt, len(gt["labels"]))[mask]
        gt_boxes.append(boxes)
        gt_difficult.append(difficult)
        used.append(np.zeros(len(boxes), dtype=bool))
        n_gt += int((~difficult).sum())

    tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
    for k, (score, i, box) in enumerate(dets):
        if len(gt_boxes[i]) == 0:
            fp[k] = 1
            continue
        ious = box_iou(box.reshape(1, 4).float(), gt_boxes[i].float())[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thresh:
            if gt_difficult[i][j]:
                continue                         # matched a difficult box: ignore this detection
            if not used[i][j]:
                tp[k] = 1
                used[i][j] = True
            else:
                fp[k] = 1                        # duplicate detection of the same object
        else:
            fp[k] = 1
    scores = np.array([d[0] for d in dets], dtype=float)
    return scores, tp, fp, n_gt


def voc_map(predictions, ground_truths, num_classes, iou_thresh=0.5, use_07_metric=False):
    aps = []
    for c in range(num_classes):
        _, tp, fp, n_gt = match_class(predictions, ground_truths, c, iou_thresh)
        if n_gt == 0:                            # class not in this set: leave it out of the mean
            aps.append(float("nan"))
            continue
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        aps.append(average_precision(recall, precision, use_07_metric))
    valid = [a for a in aps if not np.isnan(a)]
    return {"mAP": float(np.mean(valid)) if valid else 0.0, "AP": aps}


def _counts(predictions, ground_truths, num_classes, score_thresh, iou_thresh):
    """TP / FP / FN per class at one score threshold."""
    out = []
    for c in range(num_classes):
        scores, tp, fp, n_gt = match_class(predictions, ground_truths, c, iou_thresh)
        keep = scores >= score_thresh
        n_tp, n_fp = int(tp[keep].sum()), int(fp[keep].sum())
        out.append({"tp": n_tp, "fp": n_fp, "fn": n_gt - n_tp, "n_gt": n_gt})
    return out


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def precision_recall(predictions, ground_truths, num_classes, score_thresh=0.3, iou_thresh=0.5):
    """Precision, recall and F1 at one score threshold: per class and pooled over all classes."""
    counts = _counts(predictions, ground_truths, num_classes, score_thresh, iou_thresh)
    per_class = [{**c, **_prf(c["tp"], c["fp"], c["fn"])} for c in counts]
    total = {k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn", "n_gt")}
    return {"overall": {**total, **_prf(total["tp"], total["fp"], total["fn"])},
            "per_class": per_class, "score_thresh": score_thresh}


def error_rate(predictions, ground_truths, num_classes, score_thresh=0.3, iou_thresh=0.5):
    """Share of detection outcomes that are wrong: (FP + FN) / (TP + FP + FN)."""
    o = precision_recall(predictions, ground_truths, num_classes, score_thresh, iou_thresh)["overall"]
    total = o["tp"] + o["fp"] + o["fn"]
    return (o["fp"] + o["fn"]) / total if total else 0.0


def pr_curves(predictions, ground_truths, num_classes, iou_thresh=0.5):
    """Precision and recall as the score threshold is lowered through every detection.

    Returns {"overall": (thresholds, precision, recall), "per_class": [...]} with NumPy arrays.
    """
    per_class, all_scores, all_tp, all_fp, total_gt = [], [], [], [], 0
    for c in range(num_classes):
        scores, tp, fp, n_gt = match_class(predictions, ground_truths, c, iou_thresh)
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        per_class.append((scores, ctp / np.maximum(ctp + cfp, 1e-9), ctp / max(n_gt, 1)))
        all_scores.append(scores); all_tp.append(tp); all_fp.append(fp); total_gt += n_gt
    scores, tp, fp = (np.concatenate(a) if a else np.zeros(0) for a in (all_scores, all_tp, all_fp))
    order = np.argsort(-scores, kind="stable")
    ctp, cfp = np.cumsum(tp[order]), np.cumsum(fp[order])
    overall = (scores[order], ctp / np.maximum(ctp + cfp, 1e-9), ctp / max(total_gt, 1))
    return {"overall": overall, "per_class": per_class}


def best_f1_threshold(predictions, ground_truths, num_classes, iou_thresh=0.5):
    """Score threshold with the highest pooled F1, and the P/R/F1 there.

    Choose it on the validation set and then apply it unchanged to the test set; picking it on
    test would overstate the test numbers.
    """
    thresholds, precision, recall = pr_curves(predictions, ground_truths, num_classes, iou_thresh)["overall"]
    if len(thresholds) == 0:
        return {"score_thresh": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    # Only thresholds where the next detection has a lower score are real operating points.
    last_of_score = np.r_[thresholds[1:] != thresholds[:-1], True]
    k = int(np.argmax(np.where(last_of_score, f1, -1)))
    return {"score_thresh": float(thresholds[k]), "precision": float(precision[k]),
            "recall": float(recall[k]), "f1": float(f1[k])}


def detection_confusion_matrix(predictions, ground_truths, num_classes, score_thresh=0.3,
                               iou_thresh=0.5):
    """Which class each real object was detected as. Shape (C+1, C+1), index C = background.

    Detections above ``score_thresh`` are matched to ground truth regardless of class (highest
    score first, best IoU >= iou_thresh, each object at most once):
        matched           -> [true class, predicted class]  (off-diagonal = wrong class)
        unmatched object  -> [true class, background]       (missed)
        unmatched det.    -> [background, predicted class]  (false positive)
    Difficult objects and detections that land on them are left out, as in mAP.
    """
    y_true, y_pred = [], []
    bg = num_classes
    for pred, gt in zip(predictions, ground_truths):
        keep = pred["scores"] >= score_thresh
        boxes, labels, scores = pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep]
        order = torch.as_tensor(scores).argsort(descending=True)
        g_boxes, g_labels = gt["boxes"], gt["labels"]
        difficult = _difficult(gt, len(g_labels))
        used = np.zeros(len(g_labels), dtype=bool)
        ious = box_iou(torch.as_tensor(boxes).float().reshape(-1, 4), torch.as_tensor(g_boxes).float().reshape(-1, 4))
        for k in order.tolist():
            cand = ious[k].clone()
            if len(cand):
                cand[torch.from_numpy(used)] = -1
            j = int(cand.argmax()) if len(cand) else -1
            if j >= 0 and cand[j] >= iou_thresh:
                used[j] = True
                if not difficult[j]:
                    y_true.append(int(g_labels[j])); y_pred.append(int(labels[k]))
            else:
                y_true.append(bg); y_pred.append(int(labels[k]))
        for j in np.where(~used)[0]:
            if not difficult[j]:
                y_true.append(int(g_labels[j])); y_pred.append(bg)
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    np.add.at(matrix, (np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)), 1)
    return matrix
