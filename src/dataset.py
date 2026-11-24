"""PASCAL VOC loader filtered to 5 classes + augmentation.

Owner: Mariam Abdelaal

Interface contract (other modules depend on this, change it only after telling the team):
    ds = VOC5Dataset(root, sets=[(2007, "trainval"), (2012, "trainval")], split="train",
                     input_size=384, augment=True)
    image, target = ds[i]
        image  : float32 tensor [3, 384, 384], normalised (mean/std below)
        target : dict
            "boxes"  : float32 tensor [N, 4], xyxy in pixels of the 384x384 image
            "labels" : int64 tensor [N], values 0..4 in CLASSES order
            "image_id": str (e.g. "2007_000027")
    collate_fn(batch) -> (images [B,3,H,W], list[target])

Rules: drop VOC "difficult" objects from training targets (keep them flagged for eval);
skip images that contain none of the 5 classes; 90/10 train/val split with a fixed seed.
"""

CLASSES = ["person", "car", "bicycle", "motorbike", "bus"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# TODO(Mariam): parse VOC XML annotations, filter classes, letterbox/resize to 384
# TODO(Mariam): augmentation: random horizontal flip, random crop, colour jitter (boxes must follow)
# TODO(Mariam): scripts/show_samples.py: draw boxes on augmented samples (checkpoint figure)
