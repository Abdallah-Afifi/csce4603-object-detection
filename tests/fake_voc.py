"""Writes a tiny PASCAL VOC-format folder for testing src/dataset.py without downloading VOC.

It reproduces the edge cases of the real data:
  * VOC2007 ids ("000005") and VOC2012 ids ("2008_000003")
  * a VOC2012 image that re-uses a VOC2007 trainval image ("2007_000005") -> must be deduplicated
  * a VOC2012 image that is also in VOC2007 test ("2007_000033") -> must stay out of train/val
  * images with only other classes (dog) and images whose only objects are difficult
Each object is a filled rectangle in its class colour on a grey background, so a test can check
that boxes still sit on their rectangle after augmentation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASS_BGR = {"person": (0, 0, 255), "car": (0, 255, 0), "bicycle": (255, 0, 0),
             "motorbike": (0, 255, 255), "bus": (255, 0, 255), "dog": (255, 255, 0)}
BACKGROUND = 128


def _write_image(base: Path, file_id: str, objects, size):
    w, h = size
    img = np.full((h, w, 3), BACKGROUND, np.uint8)
    xml = [f"<annotation><filename>{file_id}.jpg</filename>",
           f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"]
    for name, (x1, y1, x2, y2), difficult in objects:
        img[y1 - 1:y2, x1 - 1:x2] = CLASS_BGR[name]  # VOC coords are 1-based, inclusive
        xml.append(f"<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated>"
                   f"<difficult>{int(difficult)}</difficult><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                   f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")
    xml.append("</annotation>")
    cv2.imwrite(str(base / "JPEGImages" / f"{file_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    (base / "Annotations" / f"{file_id}.xml").write_text("".join(xml))


def _random_objects(rng, size, n_max=3):
    w, h = size
    names = ["person", "person", "car", "bicycle", "motorbike", "bus"]
    objs = []
    for _ in range(rng.integers(1, n_max + 1)):
        bw, bh = int(rng.integers(30, w // 2)), int(rng.integers(30, h // 2))
        x1, y1 = int(rng.integers(1, w - bw)), int(rng.integers(1, h - bh))
        objs.append((names[rng.integers(len(names))], (x1, y1, x1 + bw, y1 + bh), False))
    return objs


def make_fake_voc(root: Path) -> Path:
    """Creates root/VOC2007 and root/VOC2012; returns root."""
    rng = np.random.default_rng(0)
    sizes = [(500, 375), (375, 500), (333, 250)]
    plan = {
        (2007, "trainval"): [f"{i:06d}" for i in range(1, 31)],
        (2007, "test"): [f"{i:06d}" for i in range(31, 41)],
        (2012, "trainval"): [f"2008_{i:06d}" for i in range(1, 21)] + ["2007_000005", "2007_000033"],
    }
    special = {
        "000002": [("dog", (10, 10, 100, 100), False)],                      # none of the 5 classes
        "000003": [("car", (10, 10, 120, 90), True)],                        # only a difficult object
        "000004": [("person", (20, 20, 80, 200), False), ("bus", (100, 50, 300, 200), True)],
    }
    for year in (2007, 2012):
        base = root / f"VOC{year}"
        for sub in ("Annotations", "JPEGImages", "ImageSets/Main"):
            (base / sub).mkdir(parents=True, exist_ok=True)
    for (year, image_set), ids in plan.items():
        base = root / f"VOC{year}"
        (base / "ImageSets" / "Main" / f"{image_set}.txt").write_text("\n".join(ids) + "\n")
        for k, file_id in enumerate(ids):
            size = sizes[k % len(sizes)]
            objects = special.get(file_id) or _random_objects(rng, size)
            _write_image(base, file_id, objects, size)
    return root
