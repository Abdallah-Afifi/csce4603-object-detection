"""Synthetic shapes dataset that follows the same interface as ``src/dataset.py``.

Used for unit tests, CI and smoke runs of the training script before (or without) PASCAL VOC.
Each class is a different shape drawn in a random colour on a noisy background, so the model has
to learn shape rather than colour:
    0 filled circle, 1 filled rectangle, 2 filled triangle, 3 thick cross, 4 rectangle outline

Images are generated deterministically from (seed, index), so a dataset is reproducible and needs
no files on disk.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SyntheticShapes(Dataset):
    def __init__(self, length: int = 256, input_size: int = 384, num_classes: int = 5,
                 max_objects: int = 4, min_size: float = 0.08, max_size: float = 0.4, seed: int = 0,
                 classes: list[str] | None = None):
        if num_classes > 5:
            raise ValueError("SyntheticShapes has 5 shape types")
        self.length = length
        self.size = input_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.min_size, self.max_size = min_size, max_size
        self.seed = seed
        self.CLASSES = classes or [f"shape{i}" for i in range(num_classes)]

    def __len__(self):
        return self.length

    def render(self, idx: int):
        """Returns (uint8 BGR image [S, S, 3], boxes float32 [N, 4], labels int64 [N])."""
        rng = np.random.default_rng((self.seed, idx))
        s = self.size
        base = rng.integers(40, 200, size=3)
        img = np.clip(base + rng.normal(0, 18, size=(s, s, 3)), 0, 255).astype(np.uint8)
        boxes, labels = [], []
        for _ in range(rng.integers(1, self.max_objects + 1)):
            side = int(rng.uniform(self.min_size, self.max_size) * s)
            bw = max(8, int(side * rng.uniform(0.7, 1.3)))
            bh = max(8, int(side * rng.uniform(0.7, 1.3)))
            x1, y1 = int(rng.integers(0, s - bw)), int(rng.integers(0, s - bh))
            x2, y2 = x1 + bw, y1 + bh
            label = int(rng.integers(0, self.num_classes))
            color = tuple(int(c) for c in rng.integers(0, 256, size=3))
            self._draw(img, label, x1, y1, x2, y2, color, max(2, side // 10))
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
        return img, np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64)

    @staticmethod
    def _draw(img, label, x1, y1, x2, y2, color, thick):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if label == 0:
            cv2.ellipse(img, (cx, cy), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, color, -1)
        elif label == 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif label == 2:
            pts = np.array([[cx, y1], [x1, y2], [x2, y2]], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif label == 3:
            cv2.rectangle(img, (cx - thick, y1), (cx + thick, y2), color, -1)
            cv2.rectangle(img, (x1, cy - thick), (x2, cy + thick), color, -1)
        else:
            cv2.rectangle(img, (x1 + thick // 2, y1 + thick // 2), (x2 - thick // 2, y2 - thick // 2),
                          color, thick)

    def __getitem__(self, idx):
        img, boxes, labels = self.render(idx)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())
        return tensor, {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels),
                        "image_id": f"synthetic_{self.seed}_{idx}"}
