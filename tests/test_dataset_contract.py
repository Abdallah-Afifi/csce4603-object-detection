"""Checks src/dataset.py against the interface the rest of the code relies on.

Skips until VOC5Dataset exists and the data is downloaded; then it guards against interface drift.
"""
from pathlib import Path

import pytest
import torch

from src.config import load_config

cfg = load_config(Path(__file__).resolve().parents[1] / "configs/centernet_voc5.yaml")
ds_mod = pytest.importorskip("src.dataset")
root = Path(__file__).resolve().parents[1] / cfg["data"]["voc_root"]
# Use whichever of the configured VOC years are downloaded (e.g. only 2007 while 2012 is still coming).
SETS = [tuple(s) for s in cfg["data"]["train_sets"]
        if (root / f"VOC{s[0]}" / "ImageSets" / "Main" / f"{s[1]}.txt").exists()]
pytestmark = pytest.mark.skipif(not hasattr(ds_mod, "VOC5Dataset") or not SETS,
                                reason="VOC5Dataset not implemented or VOC not downloaded")


@pytest.mark.parametrize("split,augment", [("train", True), ("val", False)])
def test_voc5_dataset_contract(split, augment):
    assert list(ds_mod.CLASSES) == cfg["classes"]
    sets = SETS
    ds = ds_mod.VOC5Dataset(root, sets=sets, split=split, input_size=384, augment=augment,
                            val_fraction=0.1, seed=0)
    assert len(ds) > 0
    for i in range(0, len(ds), max(1, len(ds) // 20)):
        image, target = ds[i]
        assert image.dtype == torch.float32 and tuple(image.shape) == (3, 384, 384)
        boxes, labels = target["boxes"], target["labels"]
        assert boxes.dtype == torch.float32 and boxes.ndim == 2 and boxes.shape[1] == 4
        assert labels.dtype == torch.int64 and labels.shape == (boxes.shape[0],)
        assert ((labels >= 0) & (labels < 5)).all()
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert (boxes >= -1).all() and (boxes <= 385).all()
        assert isinstance(target["image_id"], str)


def test_train_and_val_do_not_overlap():
    sets = SETS
    kw = dict(sets=SETS, input_size=384, augment=False, val_fraction=0.1, seed=0)
    train_ids = {ds_mod.VOC5Dataset(root, split="train", **kw)[i][1]["image_id"] for i in range(50)}
    val = ds_mod.VOC5Dataset(root, split="val", **kw)
    val_ids = {val[i][1]["image_id"] for i in range(min(50, len(val)))}
    assert not train_ids & val_ids
