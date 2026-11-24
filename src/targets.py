"""Target encoding: ground-truth boxes -> the maps each detection head is trained to predict.

Encoders run inside DataLoader worker processes (see ``EncodedDataset`` in ``src/train_utils.py``),
so they use NumPy and work on one image at a time.

Input to every encoder call:
    boxes  : float32 [N, 4] xyxy in input-image pixels (the 384x384 network input)
    labels : int64 [N], 0..num_classes-1
    input_hw : (height, width) of the network input
"""
from __future__ import annotations

import math

import numpy as np


def _sanitize(boxes: np.ndarray, labels: np.ndarray, input_hw: tuple[int, int],
              num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip boxes to the image and drop empty/invalid ones (augmentation can create them)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError(f"{boxes.shape[0]} boxes but {labels.shape[0]} labels")
    h, w = input_hw
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & np.isfinite(boxes).all(1)
    keep &= (labels >= 0) & (labels < num_classes)
    return boxes[keep], labels[keep]


# ----------------------------------------------------------------------------------------------
# CenterNet (heatmap) targets
# ----------------------------------------------------------------------------------------------

def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Largest centre shift that still keeps IoU >= ``min_overlap`` with the true box.

    Three cases (both corners inside / both outside / one of each) from CornerNet; we take the
    smallest radius. This is the formula used by the original CenterNet code.
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Splat a 2-D Gaussian (peak exactly 1.0) at integer ``center`` = (x, y), in place, by max."""
    diameter = 2 * radius + 1
    sigma = diameter / 6
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    gaussian = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32)

    x, y = center
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    region = heatmap[y - top:y + bottom, x - left:x + right]
    patch = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(region, patch, out=region)


class CenterNetTargets:
    """Encodes boxes for the CenterNet head.

    Returns a dict of NumPy arrays (K = max_objects):
        heatmap : float32 [C, H/s, W/s]  Gaussian per object centre, exactly 1.0 at the centre cell
        wh      : float32 [K, 2]         box width/height in output-grid units
        offset  : float32 [K, 2]         centre minus its integer cell, in [0, 1)
        index   : int64   [K]            flattened centre cell index y * W_out + x
        mask    : float32 [K]            1 for real objects, 0 for padding
    """

    def __init__(self, num_classes: int, stride: int = 4, max_objects: int = 128,
                 min_overlap: float = 0.7):
        self.num_classes = num_classes
        self.stride = stride
        self.max_objects = max_objects
        self.min_overlap = min_overlap

    def __call__(self, boxes, labels, input_hw):
        in_h, in_w = int(input_hw[0]), int(input_hw[1])
        if in_h % self.stride or in_w % self.stride:
            raise ValueError(f"Input size {input_hw} must be divisible by stride {self.stride}")
        out_h, out_w = in_h // self.stride, in_w // self.stride
        boxes, labels = _sanitize(boxes, labels, (in_h, in_w), self.num_classes)

        heatmap = np.zeros((self.num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        index = np.zeros((self.max_objects,), dtype=np.int64)
        mask = np.zeros((self.max_objects,), dtype=np.float32)

        # Largest objects first, so if an image has more than K objects we drop the tiny ones.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas, kind="stable")[: self.max_objects]
        for k, i in enumerate(order):
            box = boxes[i] / self.stride
            bw, bh = box[2] - box[0], box[3] - box[1]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            ix = int(min(max(math.floor(cx), 0), out_w - 1))
            iy = int(min(max(math.floor(cy), 0), out_h - 1))
            radius = max(0, int(gaussian_radius(math.ceil(bh), math.ceil(bw), self.min_overlap)))
            draw_gaussian(heatmap[labels[i]], (ix, iy), radius)
            wh[k] = (bw, bh)
            offset[k] = (cx - ix, cy - iy)
            index[k] = iy * out_w + ix
            mask[k] = 1.0
        return {"heatmap": heatmap, "wh": wh, "offset": offset, "index": index, "mask": mask}


# ----------------------------------------------------------------------------------------------
# Grid (YOLO-v1-style) targets: the comparison baseline
# ----------------------------------------------------------------------------------------------

class GridTargets:
    """Encodes boxes for the grid head: the cell that contains a box centre is responsible for it.

    Returns:
        obj   : float32 [G_h, G_w]     1 where a cell owns an object
        xy    : float32 [2, G_h, G_w]  centre position inside the owning cell, in [0, 1)
        wh    : float32 [2, G_h, G_w]  sqrt(box size / image size), as in YOLO v1
        cls   : int64   [G_h, G_w]     class of the owned object, -1 elsewhere
    If two objects fall in the same cell, the larger one wins. This collision is a known weakness of
    the grid head; ``dropped`` counts how often it happens, for the report.
    """

    def __init__(self, num_classes: int, stride: int = 16):
        self.num_classes = num_classes
        self.stride = stride

    def __call__(self, boxes, labels, input_hw):
        in_h, in_w = int(input_hw[0]), int(input_hw[1])
        if in_h % self.stride or in_w % self.stride:
            raise ValueError(f"Input size {input_hw} must be divisible by stride {self.stride}")
        gh, gw = in_h // self.stride, in_w // self.stride
        boxes, labels = _sanitize(boxes, labels, (in_h, in_w), self.num_classes)

        obj = np.zeros((gh, gw), dtype=np.float32)
        xy = np.zeros((2, gh, gw), dtype=np.float32)
        wh = np.zeros((2, gh, gw), dtype=np.float32)
        cls = np.full((gh, gw), -1, dtype=np.int64)
        dropped = 0

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for i in np.argsort(-areas, kind="stable"):  # larger boxes claim cells first
            x1, y1, x2, y2 = boxes[i]
            cx, cy = (x1 + x2) / 2 / self.stride, (y1 + y2) / 2 / self.stride
            ix = int(min(max(math.floor(cx), 0), gw - 1))
            iy = int(min(max(math.floor(cy), 0), gh - 1))
            if obj[iy, ix] > 0:
                dropped += 1
                continue
            obj[iy, ix] = 1.0
            xy[:, iy, ix] = (cx - ix, cy - iy)
            wh[:, iy, ix] = (math.sqrt((x2 - x1) / in_w), math.sqrt((y2 - y1) / in_h))
            cls[iy, ix] = labels[i]
        return {"obj": obj, "xy": xy, "wh": wh, "cls": cls,
                "dropped": np.array(dropped, dtype=np.int64)}
