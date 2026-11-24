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

Rules: for split="train" drop VOC "difficult" objects; skip images that contain none of the 5
classes. predict.py must preprocess frames exactly the same way (resize method + MEAN/STD), so
expose the resize as a function, e.g. ``preprocess(bgr_image) -> (tensor, scale_info)``.

Bonus: an optional ``COCO5Dataset(root, split, input_size, augment)`` with the same output lets
us add COCO images of the same 5 classes via ``data.extra_train`` in the config.
"""

CLASSES = ["person", "car", "bicycle", "motorbike", "bus"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# TODO(Mariam): parse VOC XML annotations, filter classes, resize to 384
# TODO(Mariam): augmentation: random horizontal flip, random crop, colour jitter (boxes must follow)
# TODO(Mariam): scripts/show_samples.py: draw boxes on augmented samples (checkpoint figure)
