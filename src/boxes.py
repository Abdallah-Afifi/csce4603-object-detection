"""Box utilities shared by the detectors, training and evaluation.

All boxes are ``xyxy`` in pixels: ``[x1, y1, x2, y2]`` with ``x2 > x1`` and ``y2 > y1``.
"""
from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``a`` [N, 4] and ``b`` [M, 4] -> [N, M]."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    top_left = torch.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / union.clamp(min=1e-9)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Greedy non-maximum suppression. Returns indices of kept boxes, highest score first."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        best = order[0]
        keep.append(best)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return torch.stack(keep)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                iou_thresh: float) -> torch.Tensor:
    """NMS applied separately per class (boxes of different classes never suppress each other)."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    # Shift each class into its own coordinate range so one NMS call handles all classes.
    shift = labels.to(boxes.dtype)[:, None] * (boxes.max() + 1)
    return nms(boxes + shift, scores, iou_thresh)


def clip_boxes(boxes: torch.Tensor, height: float, width: float) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes
