import numpy as np
import torch

from src.losses import centernet_loss, focal_loss, gather_at, grid_loss, masked_l1
from src.targets import CenterNetTargets, GridTargets


def _logit(p):
    return torch.log(p / (1 - p))


def test_focal_loss_near_zero_for_near_perfect_prediction():
    gt = torch.zeros(1, 1, 8, 8)
    gt[0, 0, 3, 4] = 1.0
    perfect = torch.where(gt == 1, torch.tensor(12.0), torch.tensor(-12.0))
    assert focal_loss(perfect, gt) < 1e-6
    assert focal_loss(torch.zeros_like(gt), gt) > 0.1


def test_focal_loss_is_finite_with_no_objects_and_extreme_logits():
    gt = torch.zeros(2, 3, 8, 8)
    for value in (-1e4, 0.0, 1e4):
        loss = focal_loss(torch.full_like(gt, value), gt)
        assert torch.isfinite(loss)


def test_focal_loss_gradient_points_the_right_way():
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 1, 1] = 1.0
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    focal_loss(logits, gt).backward()
    assert logits.grad[0, 0, 1, 1] < 0          # raise the score at the centre
    assert logits.grad[0, 0, 3, 3] > 0          # lower it in the background


def test_gather_and_masked_l1():
    feat = torch.arange(2 * 2 * 3 * 3, dtype=torch.float32).reshape(2, 2, 3, 3)
    idx = torch.tensor([[0, 4], [8, 0]])
    got = gather_at(feat, idx)
    assert got.shape == (2, 2, 2)
    assert got[0, 1].tolist() == [4.0, 13.0] and got[1, 0].tolist() == [26.0, 35.0]
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert masked_l1(got, got + 3, mask) == 3.0
    assert masked_l1(got, got + 3, torch.zeros(2, 2)) == 0.0


def _batch(encoder, boxes, labels, hw):
    enc = encoder(np.array(boxes, np.float32), np.array(labels), hw)
    return {k: torch.from_numpy(np.asarray(v))[None] for k, v in enc.items()}


def test_centernet_loss_parts_and_perfect_regression():
    t = _batch(CenterNetTargets(2, 4, 4), [[8, 8, 24, 40]], [1], (32, 32))
    out = {"heatmap": torch.where(t["heatmap"] == 1, 12.0, -12.0),
           "wh": torch.zeros(1, 2, 8, 8), "offset": torch.zeros(1, 2, 8, 8)}
    i = int(t["index"][0, 0])
    out["wh"].view(1, 2, -1)[0, :, i] = t["wh"][0, 0]
    out["offset"].view(1, 2, -1)[0, :, i] = t["offset"][0, 0]
    total, parts = centernet_loss(out, t)
    assert set(parts) == {"hm", "wh", "off"}
    assert total < 1e-5


def test_grid_loss_lower_for_correct_prediction():
    t = _batch(GridTargets(3, 16), [[0, 0, 32, 32]], [2], (64, 64))
    good = {"obj": _logit(t["obj"].clamp(0.001, 0.999))[:, None],
            "box": _logit(torch.cat([t["xy"], t["wh"]], 1).clamp(0.001, 0.999)),
            "cls": torch.nn.functional.one_hot(t["cls"].clamp(min=0), 3).permute(0, 3, 1, 2).float() * 20}
    bad = {k: torch.zeros_like(v) for k, v in good.items()}
    assert grid_loss(good, t)[0] < 0.05 < grid_loss(bad, t)[0]
