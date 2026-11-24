"""End-to-end tests of the training pipeline on synthetic data (no dataset download needed)."""
import csv

import torch
from torch.utils.data import DataLoader

from conftest import SMALL
from scripts.train import main, train
from src.boxes import box_iou
from src.config import apply_overrides, load_config
from src.model import CenterNet, GridDetector, load_model
from src.synthetic import SyntheticShapes
from src.train_utils import EncodedDataset, ModelEMA, detection_collate, warmup_cosine

TINY = ["input_size=64", "data.train_length=24", "data.val_length=8", "train.batch_size=8",
        "train.epochs=2", "train.workers=0", "train.log_interval=0",
        "model.backbone_widths=[8,16,32,64]", "model.neck_channels=16", "model.head_channels=16"]


def _cfg(*extra):
    return apply_overrides(load_config("configs/smoke_synthetic.yaml"), TINY + list(extra))


def test_train_writes_outputs_and_resumes(tmp_path):
    out = tmp_path / "run"
    history = train(_cfg(), out, device="cpu", log=lambda *_: None)
    assert [r["epoch"] for r in history] == [1, 2]
    for name in ("last.pt", "best.pt", "history.csv", "loss_curve.png", "config.yaml"):
        assert (out / name).exists(), name
    rows = list(csv.DictReader(open(out / "history.csv")))
    assert len(rows) == 2 and {"train_total", "train_hm", "val_total", "lr"} <= set(rows[0])
    assert all(torch.isfinite(torch.tensor(float(r["train_total"]))) for r in rows)

    # Resume: raising epochs continues from epoch 3, keeping the earlier history.
    history = train(_cfg("train.epochs=3"), out, device="cpu", log=lambda *_: None)
    assert [r["epoch"] for r in history] == [1, 2, 3]
    ckpt = torch.load(out / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 3 and ckpt["global_step"] == 3 * 3

    # Already finished: nothing more happens.
    assert len(train(_cfg("train.epochs=3"), out, device="cpu", log=lambda *_: None)) == 3

    model, cfg = load_model(out / "best.pt")
    assert isinstance(model, CenterNet) and cfg["input_size"] == 64


def test_train_grid_head_and_cli(tmp_path):
    out = tmp_path / "grid"
    main(["--config", "configs/smoke_synthetic.yaml", "--out", str(out), "--device", "cpu",
          "--max-iters", "2", "--set", *TINY, "model.head=grid", "model.grid_stride=16", "train.epochs=1"])
    model, _ = load_model(out / "best.pt")
    assert isinstance(model, GridDetector)
    row = next(csv.DictReader(open(out / "history.csv")))
    assert {"train_obj", "train_box", "train_cls"} <= set(row)


def test_warmup_cosine_schedule():
    f = warmup_cosine(total_iters=100, warmup_iters=10, min_ratio=0.01)
    assert f(0) == 0.1 and f(9) == 1.0
    assert abs(f(10) - 1.0) < 1e-9 and abs(f(99) - 0.01) < 1e-3
    assert all(f(i) >= f(i + 1) for i in range(10, 99))


def test_ema_tracks_model():
    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.5, tau=1e-9)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    torch.testing.assert_close(ema.ema.weight, model.weight - 0.5)


def test_overfits_small_synthetic_set():
    """The decisive sanity check: a correct detector must memorise a handful of images.
    If encoding, loss, or decoding were inconsistent, the boxes would not come back."""
    torch.manual_seed(0)
    data = SyntheticShapes(length=8, input_size=64, num_classes=3, max_objects=2, min_size=0.25,
                           max_size=0.5, seed=3)
    model = CenterNet(num_classes=3, **SMALL)
    loader = DataLoader(EncodedDataset(data, model.target_encoder()), batch_size=8,
                        collate_fn=detection_collate)
    images, enc, targets = next(iter(loader))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(400):
        loss, _ = model.compute_loss(model(images), enc)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        dets = model.decode(model(images), score_thresh=0.3)
    found = total = 0
    for det, gt in zip(dets, targets):
        for box, label in zip(gt["boxes"], gt["labels"]):
            total += 1
            same = det["labels"] == label
            if same.any() and box_iou(box[None], det["boxes"][same]).max() >= 0.5:
                found += 1
    assert found / total >= 0.9, f"recovered {found}/{total} boxes after overfitting"
