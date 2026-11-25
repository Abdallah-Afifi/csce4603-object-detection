"""Tests for src/dataset.py on a tiny fake VOC folder (tests/fake_voc.py), so they run in CI
without downloading PASCAL VOC. tests/test_dataset_contract.py runs the same contract on real VOC."""
import cv2
import numpy as np
import pytest
import torch

from src import dataset as D
from src.model import build_model
from src.train_utils import EncodedDataset, detection_collate
from tests.fake_voc import CLASS_BGR, make_fake_voc

SETS = [(2007, "trainval"), (2012, "trainval")]


@pytest.fixture(scope="module")
def voc_root(tmp_path_factory):
    return make_fake_voc(tmp_path_factory.mktemp("VOCdevkit"))


def make(root, split, augment=False, seed=0, sets=SETS):
    return D.VOC5Dataset(root, sets=sets, split=split, input_size=384, augment=augment,
                         val_fraction=0.1, seed=seed)


def all_ids(ds):
    return [r["id"] for r in ds.records]


@pytest.mark.parametrize("split,augment", [("train", True), ("train", False), ("val", False)])
def test_contract(voc_root, split, augment):
    ds = make(voc_root, split, augment)
    assert len(ds) > 0
    for i in range(len(ds)):
        image, target = ds[i]
        assert image.dtype == torch.float32 and tuple(image.shape) == (3, 384, 384)
        boxes, labels = target["boxes"], target["labels"]
        assert boxes.dtype == torch.float32 and boxes.ndim == 2 and boxes.shape[1] == 4
        assert labels.dtype == torch.int64 and labels.shape == (boxes.shape[0],)
        assert ((labels >= 0) & (labels < 5)).all()
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert (boxes >= 0).all() and (boxes <= 384).all()
        assert isinstance(target["image_id"], str)
        assert ("difficult" in target) == (split != "train")


def test_split_is_disjoint_complete_and_seeded(voc_root):
    ds_train = make(voc_root, "train")
    train, val = all_ids(ds_train), all_ids(make(voc_root, "val"))
    assert not set(train) & set(val)
    n_split = len(train) + ds_train.info["images_only_difficult"] + len(val)
    assert len(val) == round(0.1 * n_split)
    assert all_ids(make(voc_root, "val")) == val                  # same seed, same split
    assert all_ids(make(voc_root, "val", seed=1)) != val


def test_duplicates_and_test_images_are_removed(voc_root):
    ds_train, ds_val = make(voc_root, "train"), make(voc_root, "val")
    ids = all_ids(ds_train) + all_ids(ds_val)
    assert len(ids) == len(set(ids))                  # 2007_000005 appears once, not twice
    assert "2007_000005" in ids
    assert "2007_000033" not in ids                   # in VOC2007 test, so kept out of train/val
    assert ds_train.info["duplicates_dropped"] == 1 and ds_train.info["test_images_excluded"] == 1
    test = make(voc_root, "test", sets=[(2007, "test")])
    assert "2007_000033" in all_ids(test) and not set(all_ids(test)) & set(ids)


def test_class_filter_and_difficult_handling(voc_root):
    train, val = make(voc_root, "train"), make(voc_root, "val")
    ids = all_ids(train) + all_ids(val)
    assert "2007_000002" not in ids                   # only a dog
    for r in train.records:
        assert not r["difficult"].any()
    if "2007_000003" in all_ids(val):                 # only-difficult image: allowed in val
        assert val.records[all_ids(val).index("2007_000003")]["difficult"].all()
    else:
        assert "2007_000003" not in all_ids(train)
    if "2007_000004" in all_ids(train):               # difficult bus dropped, person kept
        r = train.records[all_ids(train).index("2007_000004")]
        assert r["labels"].tolist() == [D.CLASS_TO_ID["person"]]


def test_voc_coordinates_are_converted_to_continuous_xyxy(voc_root):
    boxes, labels, difficult = D.parse_voc_xml(voc_root / "VOC2007" / "Annotations" / "000004.xml")
    assert boxes[0].tolist() == [19, 19, 80, 200]     # xmin=20 -> 19, xmax stays 80
    assert labels.tolist() == [0, 4] and difficult.tolist() == [False, True]


def _saturated_fraction(bgr, box, margin=3):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    patch = bgr[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
    if patch.size == 0:
        return 1.0
    sat = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 1]
    return float((sat > 60).mean())


@pytest.mark.parametrize("augment", [False, True])
def test_boxes_sit_on_their_objects(voc_root, augment):
    """Every object is a saturated rectangle on grey; after resize (and augmentation) the inside
    of each returned box must still be that rectangle, and the class colour must match."""
    ds = make(voc_root, "train", augment)
    hue = {n: cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0, 0, 0] for n, c in CLASS_BGR.items()}
    for rep in range(3 if augment else 1):
        for i in range(len(ds)):
            image, target = ds[i]
            bgr = D.denormalize(image)
            for box, lab in zip(target["boxes"].tolist(), target["labels"].tolist()):
                if min(box[2] - box[0], box[3] - box[1]) < 10:
                    continue
                assert _saturated_fraction(bgr, box) > 0.9, (target["image_id"], box)
                if not augment:  # hue is jittered under augmentation, so check it only without
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    h = int(cv2.cvtColor(bgr[cy:cy + 1, cx:cx + 1], cv2.COLOR_BGR2HSV)[0, 0, 0])
                    want = int(hue[D.CLASSES[lab]])
                    assert min(abs(h - want), 180 - abs(h - want)) <= 4 or _overlapped(target, box)


def _overlapped(target, box):
    b = torch.tensor(box)
    others = target["boxes"]
    ix = (torch.minimum(others[:, 2], b[2]) - torch.maximum(others[:, 0], b[0])).clamp(min=0)
    iy = (torch.minimum(others[:, 3], b[3]) - torch.maximum(others[:, 1], b[1])).clamp(min=0)
    return int(((ix * iy) > 0).sum()) > 1


def test_hflip_and_crop_move_boxes_with_the_image():
    img = np.zeros((100, 200, 3), np.uint8)
    boxes = np.array([[10, 20, 50, 60]], np.float32)
    flipped, fb = D.hflip(img, boxes)
    assert fb.tolist() == [[150, 20, 190, 60]]
    rng = np.random.default_rng(0)
    for _ in range(50):
        crop, cb, cl = D.random_crop(img, boxes, np.array([1]), rng)
        assert len(cb) == 1                            # the only object is always kept
        assert (cb[:, [0, 2]] >= 0).all() and (cb[:, [0, 2]] <= crop.shape[1]).all()
        assert (cb[:, [1, 3]] >= 0).all() and (cb[:, [1, 3]] <= crop.shape[0]).all()


def test_preprocess_round_trip_for_predict():
    frame = np.zeros((480, 640, 3), np.uint8)
    tensor, info = D.preprocess(frame, 384)
    assert tuple(tensor.shape) == (3, 384, 384)
    assert info["scale"] == pytest.approx(0.6) and info["pad_x"] == 0 and info["pad_y"] == 48
    box = np.array([[100, 50, 300, 400]], np.float32)
    in_net = box * info["scale"] + [info["pad_x"], info["pad_y"]] * 2
    assert np.allclose(D.boxes_to_original(in_net, info), box, atol=1e-4)
    pad = tensor[:, :48, :]                            # letterbox padding is 0 after normalising
    assert pad.abs().max() < 0.03


def test_augment_only_allowed_for_train(voc_root):
    with pytest.raises(ValueError):
        make(voc_root, "val", augment=True)


def test_works_with_training_encoder(voc_root):
    """The dataset plugs into the training pipeline (EncodedDataset + collate) unchanged."""
    model = build_model({"classes": D.CLASSES, "model": {
        "head": "centernet", "backbone_widths": [8, 16, 32, 64], "backbone_depths": [1, 1, 1, 1],
        "neck_channels": 16, "head_channels": 16}})
    data = EncodedDataset(make(voc_root, "train", augment=True), model.target_encoder())
    images, enc, targets = detection_collate([data[i] for i in range(4)])
    assert images.shape == (4, 3, 384, 384) and len(targets) == 4
    assert enc["heatmap"].shape[0] == 4 if "heatmap" in enc else len(enc) > 0
    with torch.no_grad():
        model.eval()(images)


def test_tiny_boxes_dropped_in_train_but_kept_for_evaluation(tmp_path):
    """The 2-px minimum is a training rule; mAP must be computed on every real box."""
    from tests.fake_voc import _write_image
    root = tmp_path / "VOCdevkit"
    for year in (2007, 2012):
        for sub in ("Annotations", "JPEGImages", "ImageSets/Main"):
            (root / f"VOC{year}" / sub).mkdir(parents=True, exist_ok=True)
    ids = [f"{i:06d}" for i in range(1, 11)]
    (root / "VOC2007/ImageSets/Main/trainval.txt").write_text("\n".join(ids) + "\n")
    (root / "VOC2012/ImageSets/Main/trainval.txt").write_text("")
    for i in ids:   # 1000x500 image -> scale 0.384: the 3-px-wide person becomes ~1.2 px
        _write_image(root / "VOC2007", i, [("person", (10, 10, 12, 60), False),
                                           ("car", (100, 100, 400, 300), False)], (1000, 500))
    for split, n_expected in (("train", 1), ("val", 2)):
        ds = D.VOC5Dataset(root, sets=[(2007, "trainval")], split=split, input_size=384, val_fraction=0.5)
        boxes = ds[0][1]["boxes"]
        assert len(boxes) == n_expected, split
        assert ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()
