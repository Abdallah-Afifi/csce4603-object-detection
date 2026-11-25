"""PASCAL VOC loader filtered to 5 classes + augmentation.

Owner: Mariam Abdelaal

Interface contract (scripts/train.py and tests/test_dataset_contract.py depend on it; tell the
team before changing it):

    ds = VOC5Dataset(root, sets=[(2007, "trainval"), (2012, "trainval")], split="train",
                     input_size=384, augment=True, val_fraction=0.1, seed=0)
        root   : folder containing VOC2007/ and VOC2012/ (e.g. data/VOCdevkit)
        split  : "train" (90% of the given sets), "val" (the other 10%), or "test" (whole set,
                 e.g. sets=[(2007, "test")]). The train/val split must be fixed by ``seed`` and
                 must not overlap.
        augment: only ever True for split="train"

    image, target = ds[i]
        image  : float32 tensor [3, 384, 384], RGB, normalised with MEAN/STD below
        target : dict
            "boxes"    : float32 tensor [N, 4], xyxy in pixels of the 384x384 image (after resize
                         and augmentation); N may be 0 after cropping
            "labels"   : int64 tensor [N], values 0..4 in CLASSES order
            "image_id" : str, e.g. "2007_000027"
            "difficult": bool tensor [N] (optional; val/test only, used by mAP)

    The training script adds its own collate function and target encoding; the dataset only
    returns images and boxes.

Rules: for split="train" drop VOC "difficult" objects and skip images that contain none of the 5
classes. val/test keep every image, including ones with none of our classes (N = 0), so false
positives on them are counted by mAP. predict.py must preprocess frames exactly the same way (resize method + MEAN/STD), so
the resize is exposed as ``preprocess(bgr_image) -> (tensor, scale_info)``.

Implementation notes
--------------------
Resize: letterbox. The longer side is scaled to ``input_size`` (aspect ratio kept) and the rest is
padded with the MEAN colour, so padding is exactly 0 after normalisation. ``preprocess`` and
``boxes_to_original`` give predict.py the same resize and its inverse.

Image ids: every image gets a canonical id "<year>_<number>". VOC2007 files are named "000032",
so they become "2007_000032"; VOC2012 files already look like "2007_000032" or "2010_001234".
VOC2012 trainval re-uses some VOC2007 images under the same number, so with 07+12 the same photo
would appear twice and could land in both train and val. Duplicates are dropped (the first set
listed wins), and any image that is in VOC2007 test is removed from train/val, so the test set
stays unseen.

Train/val split: all images of the given sets are sorted by id, shuffled with ``seed`` and cut
``val_fraction`` / rest, so train and val never share an image. Train then drops difficult objects
and any image left with no objects (including images with none of our classes); val keeps them all.

Augmentation (train only, applied to the original image before the letterbox; boxes follow):
    random crop  p=0.5: crop covering 30-100% of each side, aspect 0.5-2; an object is kept if
                 its centre lies inside the crop, then its box is clipped to the crop
    flip         p=0.5: horizontal
    colour       always: hue +-10 deg, saturation and brightness x0.6-1.4 (HSV)
Randomness comes from torch's RNG, which the DataLoader seeds per worker, so workers do not
repeat each other and a seeded run is reproducible.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["person", "car", "bicycle", "motorbike", "bus"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
_MEAN = np.array(MEAN, np.float32)
_STD = np.array(STD, np.float32)
PAD_BGR = tuple(int(round(m * 255)) for m in MEAN[::-1])  # MEAN colour in BGR -> 0 after normalising

# Augmentation settings (quoted in the report; change here, not inline).
AUG = {
    "crop_p": 0.5,
    "crop_min_side": 0.3,
    "crop_aspect": (0.5, 2.0),
    "crop_tries": 20,
    "flip_p": 0.5,
    "hue_deg": 10,           # OpenCV hue runs 0..180, so +-10 deg is +-5 units
    "sat": (0.6, 1.4),
    "val": (0.6, 1.4),
}
MIN_BOX_PX = 2.0            # train only: boxes smaller than this (in 384-px input) are dropped


# ----------------------------------------------------------------------------------------------
# Resize shared with predict.py
# ----------------------------------------------------------------------------------------------

def letterbox(bgr: np.ndarray, size: int = 384):
    """Resize keeping aspect ratio and pad to size x size. Returns (image, scale_info).

    scale_info = {"scale", "pad_x", "pad_y", "orig_w", "orig_h"}; a point (x, y) in the original
    image maps to (x * scale + pad_x, y * scale + pad_y) in the letterboxed image.
    """
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.empty((size, size, 3), np.uint8)
    out[:] = PAD_BGR
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_w": w, "orig_h": h}


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """uint8 BGR [H, W, 3] -> float32 RGB tensor [3, H, W] normalised with MEAN/STD."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())


def preprocess(bgr: np.ndarray, input_size: int = 384):
    """What predict.py should call on every frame: (tensor [3, S, S], scale_info)."""
    img, info = letterbox(bgr, input_size)
    return to_tensor(img), info


def boxes_to_original(boxes, info: dict) -> np.ndarray:
    """Map xyxy boxes from network-input pixels back to the original image (undo letterbox)."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - info["pad_x"]) / info["scale"]
    b[:, [1, 3]] = (b[:, [1, 3]] - info["pad_y"]) / info["scale"]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, info["orig_w"])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, info["orig_h"])
    return b


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Inverse of to_tensor, for drawing: float tensor [3, H, W] -> uint8 BGR [H, W, 3]."""
    rgb = image.detach().cpu().numpy().transpose(1, 2, 0) * _STD + _MEAN
    return cv2.cvtColor((rgb.clip(0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------------------------
# VOC annotations
# ----------------------------------------------------------------------------------------------

def canonical_id(year: int, file_id: str) -> str:
    """'000032' in VOC2007 -> '2007_000032'; VOC2012 ids ('2008_000123') are already canonical."""
    return file_id if "_" in file_id else f"{year}_{file_id}"


def parse_voc_xml(path: Path):
    """Returns (boxes float32 [N, 4] xyxy, labels int64 [N], difficult bool [N]) for the 5 classes.

    VOC pixel coordinates are 1-based and inclusive, so xmin/ymin are shifted by -1 to get
    continuous xyxy boxes whose width is xmax - xmin + 1.
    """
    root = ET.parse(path).getroot()
    boxes, labels, difficult = [], [], []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in CLASS_TO_ID:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x1 - 1, y1 - 1, x2, y2])
        labels.append(CLASS_TO_ID[name])
        difficult.append(obj.findtext("difficult", "0").strip() == "1")
    return (np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64),
            np.array(difficult, bool))


def _read_ids(root: Path, year: int, image_set: str) -> list[str]:
    path = root / f"VOC{year}" / "ImageSets" / "Main" / f"{image_set}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Expected PASCAL VOC under {root} "
                                f"(VOC{year}/Annotations, JPEGImages, ImageSets/Main).")
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


# ----------------------------------------------------------------------------------------------
# Augmentation (numpy / OpenCV, on the original image before the letterbox)
# ----------------------------------------------------------------------------------------------

def random_crop(img, boxes, labels, rng):
    h, w = img.shape[:2]
    lo, hi = AUG["crop_aspect"]
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    for _ in range(AUG["crop_tries"]):
        side = rng.uniform(AUG["crop_min_side"], 1.0)
        ar = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        cw = min(w, w * side * math.sqrt(ar))
        ch = min(h, h * side / math.sqrt(ar))
        if cw < AUG["crop_min_side"] * w or ch < AUG["crop_min_side"] * h:
            continue
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
        keep = ((centres[:, 0] > x0) & (centres[:, 0] < x0 + cw)
                & (centres[:, 1] > y0) & (centres[:, 1] < y0 + ch))
        if len(boxes) and not keep.any():
            continue  # try again rather than return an image with every object cut away
        ix0, iy0, ix1, iy1 = int(x0), int(y0), int(math.ceil(x0 + cw)), int(math.ceil(y0 + ch))
        b = boxes[keep].copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(ix0, ix1) - ix0
        b[:, [1, 3]] = b[:, [1, 3]].clip(iy0, iy1) - iy0
        return img[iy0:iy1, ix0:ix1], b, labels[keep]
    return img, boxes, labels


def hflip(img, boxes):
    w = img.shape[1]
    b = boxes.copy()
    b[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img[:, ::-1], b


def colour_jitter(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dh = rng.uniform(-AUG["hue_deg"], AUG["hue_deg"]) / 2  # degrees -> OpenCV units (0..180)
    s, v = rng.uniform(*AUG["sat"]), rng.uniform(*AUG["val"])
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x + dh) % 180).astype(np.uint8)
    lut_s = (x * s).clip(0, 255).astype(np.uint8)
    lut_v = (x * v).clip(0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hsv[..., 0], lut_h), cv2.LUT(hsv[..., 1], lut_s),
                     cv2.LUT(hsv[..., 2], lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_sample(img, boxes, labels, rng):
    """Random crop -> horizontal flip -> colour jitter; boxes and labels follow the image."""
    if rng.random() < AUG["crop_p"]:
        img, boxes, labels = random_crop(img, boxes, labels, rng)
    if rng.random() < AUG["flip_p"]:
        img, boxes = hflip(img, boxes)
    img = colour_jitter(np.ascontiguousarray(img), rng)
    return img, boxes, labels


# ----------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------

class VOC5Dataset(Dataset):
    def __init__(self, root, sets=((2007, "trainval"), (2012, "trainval")), split: str = "train",
                 input_size: int = 384, augment: bool = False, val_fraction: float = 0.1,
                 seed: int = 0, require_test_list: bool = True):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        if augment and split != "train":
            raise ValueError("augment=True is only allowed for split='train'")
        self.root = Path(root)
        self.sets = [(int(y), str(s)) for y, s in sets]
        self.split, self.size, self.augment = split, input_size, augment
        self.val_fraction, self.seed = val_fraction, seed
        self.require_test_list = require_test_list
        self.info = Counter()  # bookkeeping for the README / report

        records = self._load_records()
        if split == "test":
            self.records = records
        else:
            # Split over every image (with or without our classes) so train and val never share one.
            order = np.random.default_rng(seed).permutation(len(records))
            n_val = int(round(len(records) * val_fraction))
            chosen = order[:n_val] if split == "val" else order[n_val:]
            self.records = [records[i] for i in sorted(chosen)]
            if split == "train":  # drop difficult objects, then images left with no objects
                kept = []
                for r in self.records:
                    m = ~r["difficult"]
                    if m.any():
                        kept.append({**r, "boxes": r["boxes"][m], "labels": r["labels"][m],
                                     "difficult": r["difficult"][m]})
                    elif len(r["labels"]) == 0:
                        self.info["images_without_5_classes"] += 1
                    else:
                        self.info["images_only_difficult"] += 1
                self.records = kept
        self.info["images"] = len(self.records)

    def _load_records(self):
        seen, records = set(), []
        test_ids = set()
        if self.split != "test":
            test_txt = self.root / "VOC2007" / "ImageSets" / "Main" / "test.txt"
            if test_txt.exists():
                test_ids = {canonical_id(2007, i) for i in _read_ids(self.root, 2007, "test")}
            elif self.require_test_list and any(year != 2007 for year, _ in self.sets):
                # Only other years' lists (VOC2012 re-uses 2007 ids) can contain 2007 test images.
                # Without the list we cannot keep VOC2007 test images out of train/val.
                raise FileNotFoundError(
                    f"{test_txt} not found: extract VOCtest_06-Nov-2007.tar into the same VOC2007/ "
                    "folder, or pass require_test_list=False (then VOC2007 test is not protected).")
        for year, image_set in self.sets:
            base = self.root / f"VOC{year}"
            for file_id in _read_ids(self.root, year, image_set):
                cid = canonical_id(year, file_id)
                if cid in seen:
                    self.info["duplicates_dropped"] += 1
                    continue
                if cid in test_ids:
                    self.info["test_images_excluded"] += 1
                    continue
                seen.add(cid)
                boxes, labels, difficult = parse_voc_xml(base / "Annotations" / f"{file_id}.xml")
                # Images with none of our classes are kept here: val/test need them, because a false
                # positive on a horse-only image must count against precision. Train drops them later.
                records.append({"path": str(base / "JPEGImages" / f"{file_id}.jpg"), "id": cid,
                                "boxes": boxes, "labels": labels, "difficult": difficult})
        records.sort(key=lambda r: r["id"])
        return records

    def __len__(self):
        return len(self.records)

    def load_raw(self, idx: int):
        """Original BGR image and its annotations, before resize or augmentation."""
        r = self.records[idx]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        return img, r["boxes"].copy(), r["labels"].copy(), r["difficult"].copy()

    def __getitem__(self, idx):
        img, boxes, labels, difficult = self.load_raw(idx)
        if self.augment:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())  # per-worker torch RNG
            img, boxes, labels = augment_sample(img, boxes, labels, np.random.default_rng(seed))
            difficult = np.zeros(len(labels), bool)  # train has no difficult objects
        img, info = letterbox(img, self.size)
        boxes = boxes * info["scale"]
        boxes[:, [0, 2]] += info["pad_x"]
        boxes[:, [1, 3]] += info["pad_y"]
        boxes = boxes.clip(0, self.size)
        # Training drops slivers left by cropping; val/test keep every real box so mAP is not inflated.
        min_px = MIN_BOX_PX if self.split == "train" else 0.0
        keep = ((boxes[:, 2] - boxes[:, 0]) > min_px) & ((boxes[:, 3] - boxes[:, 1]) > min_px)
        target = {"boxes": torch.from_numpy(boxes[keep].astype(np.float32)),
                  "labels": torch.from_numpy(labels[keep].astype(np.int64)),
                  "image_id": self.records[idx]["id"]}
        if self.split != "train":
            target["difficult"] = torch.from_numpy(difficult[keep])
        return to_tensor(img), target

    def class_counts(self) -> dict:
        """Objects and images per class in this split (before augmentation)."""
        objects, images, hard = Counter(), Counter(), Counter()
        for r in self.records:
            for lab, diff in zip(r["labels"], r["difficult"]):
                objects[CLASSES[lab]] += 1
                hard[CLASSES[lab]] += int(diff)
            for lab in set(r["labels"].tolist()):
                images[CLASSES[lab]] += 1
        return {c: {"objects": objects[c], "difficult": hard[c], "images": images[c]} for c in CLASSES}

    def __repr__(self):
        return (f"VOC5Dataset(split={self.split}, sets={self.sets}, images={len(self)}, "
                f"augment={self.augment})")
