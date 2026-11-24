import numpy as np
import pytest
import torch

from conftest import SMALL
from src.boxes import batched_nms, box_iou, nms
from src.config import apply_overrides, load_config
from src.model import CenterNet, GridDetector, build_model, count_parameters, load_model
from src.train_utils import detection_collate


def test_centernet_output_shapes_and_prior():
    model = CenterNet(num_classes=5, **SMALL).eval()
    out = model(torch.randn(2, 3, 128, 96))
    assert out["heatmap"].shape == (2, 5, 32, 24)
    assert out["wh"].shape == out["offset"].shape == (2, 2, 32, 24)
    p = torch.sigmoid(out["heatmap"]).mean()
    assert 0.05 < p < 0.2          # bias prior makes the start p ~ 0.1, not 0.5


@pytest.mark.parametrize("stride,cells", [(8, 16), (16, 8), (32, 4)])
def test_grid_output_shapes(stride, cells):
    model = GridDetector(num_classes=3, grid_stride=stride, **SMALL).eval()
    out = model(torch.randn(1, 3, 128, 128))
    assert out["obj"].shape == (1, 1, cells, cells)
    assert out["box"].shape == (1, 4, cells, cells)
    assert out["cls"].shape == (1, 3, cells, cells)


def test_rejects_input_not_divisible_by_32():
    with pytest.raises(ValueError):
        CenterNet(**SMALL)(torch.randn(1, 3, 100, 128))


def test_full_size_model_matches_plan():
    cfg = load_config("configs/centernet_voc5.yaml")
    model = build_model(cfg).eval()
    params = count_parameters(model)
    assert 10e6 < params < 16e6, params    # ResNet-18-sized
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 384, 384))
    assert out["heatmap"].shape == (1, 5, 96, 96)


def test_ablation_configs_inherit_from_main():
    a = load_config("configs/centernet_voc5.yaml")
    b = load_config("configs/centernet_voc5_noaug.yaml")
    c = load_config("configs/grid_voc5.yaml")
    assert b["augment"] is False and a["augment"] is True
    assert {k: v for k, v in b.items() if k != "augment"} == {k: v for k, v in a.items() if k != "augment"}
    assert c["model"]["head"] == "grid" and c["data"] == a["data"] and c["train"]["epochs"] == a["train"]["epochs"]
    assert isinstance(build_model(c), GridDetector)
    assert apply_overrides(a, ["train.epochs=7", "model.head=grid"])["train"]["epochs"] == 7


def test_gradients_reach_every_parameter():
    for model in (CenterNet(num_classes=2, **SMALL), GridDetector(num_classes=2, **SMALL)):
        enc = model.target_encoder()
        items = []
        for _ in range(2):
            t = enc(np.array([[10, 10, 60, 50]], np.float32), np.array([1]), (64, 64))
            items.append((torch.randn(3, 64, 64), {k: torch.from_numpy(np.asarray(v)) for k, v in t.items()}, {}))
        images, targets, _ = detection_collate(items)
        loss, _ = model.compute_loss(model(images), targets)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, missing
        assert torch.isfinite(loss)


def _perfect_centernet_output(model, boxes, labels, hw):
    t = model.target_encoder()(np.array(boxes, np.float32), np.array(labels), hw)
    c, h, w = t["heatmap"].shape
    out = {"heatmap": torch.from_numpy(np.where(t["heatmap"] == 1, 10.0, -10.0)).float()[None],
           "wh": torch.zeros(1, 2, h, w), "offset": torch.zeros(1, 2, h, w)}
    for k in range(int(t["mask"].sum())):
        i = int(t["index"][k])
        out["wh"].view(1, 2, -1)[0, :, i] = torch.from_numpy(t["wh"][k])
        out["offset"].view(1, 2, -1)[0, :, i] = torch.from_numpy(t["offset"][k])
    return out


def test_centernet_encode_decode_round_trip():
    model = CenterNet(num_classes=3, **SMALL)
    boxes = [[10.0, 12.0, 50.0, 70.0], [70.0, 20.0, 120.0, 60.0], [3.0, 90.0, 30.0, 125.0]]
    labels = [0, 2, 1]
    det = model.decode(_perfect_centernet_output(model, boxes, labels, (128, 128)), score_thresh=0.5)[0]
    assert len(det["boxes"]) == 3
    order = det["boxes"][:, 0].argsort()
    expected = torch.tensor(boxes)[torch.tensor(boxes)[:, 0].argsort()]
    torch.testing.assert_close(det["boxes"][order], expected, atol=1e-4, rtol=0)
    assert sorted(det["labels"].tolist()) == sorted(labels)


def test_grid_encode_decode_round_trip():
    model = GridDetector(num_classes=3, grid_stride=16, **SMALL)
    boxes, labels = [[10.0, 12.0, 50.0, 70.0], [70.0, 20.0, 120.0, 60.0]], [0, 2]
    t = model.target_encoder()(np.array(boxes, np.float32), np.array(labels), (128, 128))
    logit = lambda x: torch.logit(torch.from_numpy(x).clamp(1e-6, 1 - 1e-6))
    out = {"obj": logit(t["obj"])[None, None],
           "box": logit(np.concatenate([t["xy"], t["wh"]]))[None],
           "cls": torch.nn.functional.one_hot(torch.from_numpy(t["cls"]).clamp(min=0), 3)
           .permute(2, 0, 1).float()[None] * 20}
    det = model.decode(out, score_thresh=0.5)[0]
    assert len(det["boxes"]) == 2
    order = det["boxes"][:, 0].argsort()
    torch.testing.assert_close(det["boxes"][order], torch.tensor(boxes), atol=1e-2, rtol=0)
    assert det["labels"][order].tolist() == labels


def test_decode_handles_nothing_detected_and_batches():
    model = CenterNet(num_classes=2, **SMALL).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 3, 64, 64))
    dets = model.decode(out, score_thresh=0.99)
    assert len(dets) == 3
    for d in dets:
        assert d["boxes"].shape[1] == 4 and len(d["boxes"]) == len(d["scores"]) == len(d["labels"])
    dets = model.decode(out, score_thresh=0.0, max_det=7)
    assert all(len(d["boxes"]) <= 7 for d in dets)
    assert all((d["boxes"][:, 2:] <= 64).all() and (d["boxes"] >= 0).all() for d in dets)


def test_box_iou_and_nms():
    a = torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    iou = box_iou(a, a)
    assert iou[0, 1] == 1 and abs(iou[0, 2] - 25 / 175) < 1e-6 and iou[0, 3] == 0
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    assert nms(a, scores, 0.5).tolist() == [0, 2, 3]
    labels = torch.tensor([0, 1, 0, 0])
    assert sorted(batched_nms(a, scores, labels, 0.5).tolist()) == [0, 1, 2, 3]
    assert nms(torch.zeros(0, 4), torch.zeros(0), 0.5).numel() == 0


def test_load_model_round_trip(tmp_path):
    cfg = load_config("configs/smoke_synthetic.yaml")
    model = build_model(cfg).eval()
    torch.save({"model": model.state_dict(), "config": cfg}, tmp_path / "w.pt")
    loaded, loaded_cfg = load_model(tmp_path / "w.pt")
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(x)["heatmap"], loaded(x)["heatmap"])
    assert loaded_cfg["classes"] == cfg["classes"] and not loaded.training
