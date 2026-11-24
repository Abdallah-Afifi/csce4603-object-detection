import numpy as np
import pytest

from src.targets import CenterNetTargets, GridTargets, draw_gaussian, gaussian_radius


def test_gaussian_peak_is_exactly_one_and_clipped_at_border():
    hm = np.zeros((10, 10), np.float32)
    draw_gaussian(hm, (0, 9), 3)
    assert hm[9, 0] == 1.0
    assert hm.max() == 1.0 and (hm >= 0).all()
    assert hm[9, 1] < 1.0 and hm[9, 1] > 0


def test_gaussian_overlap_takes_maximum():
    hm = np.zeros((10, 10), np.float32)
    draw_gaussian(hm, (4, 4), 2)
    draw_gaussian(hm, (5, 4), 2)
    assert hm[4, 4] == 1.0 and hm[4, 5] == 1.0


def test_radius_grows_with_box_and_is_non_negative():
    assert gaussian_radius(1, 1) >= 0
    assert gaussian_radius(40, 40) > gaussian_radius(10, 10)


def test_centernet_encoding_values():
    enc = CenterNetTargets(num_classes=3, stride=4, max_objects=8)
    boxes = np.array([[10, 20, 50, 60]], np.float32)   # centre (30, 40) -> (7.5, 10) in output units
    out = enc(boxes, np.array([2]), (64, 64))
    assert out["heatmap"].shape == (3, 16, 16)
    assert out["heatmap"][2, 10, 7] == 1.0
    assert out["heatmap"][:2].max() == 0.0
    np.testing.assert_allclose(out["wh"][0], [10, 10])
    np.testing.assert_allclose(out["offset"][0], [0.5, 0.0])
    assert out["index"][0] == 10 * 16 + 7
    assert out["mask"].tolist() == [1] + [0] * 7


def test_centernet_handles_empty_invalid_and_out_of_image_boxes():
    enc = CenterNetTargets(num_classes=2, stride=4, max_objects=4)
    out = enc(np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), (32, 32))
    assert out["mask"].sum() == 0 and out["heatmap"].sum() == 0
    boxes = np.array([[5, 5, 5, 9],            # zero width -> dropped
                      [-10, -10, 8, 8],        # partly outside -> clipped
                      [40, 40, 50, 50],        # fully outside -> dropped
                      [1, 1, 3, 3]], np.float32)
    labels = np.array([0, 1, 0, 7])           # label 7 is out of range -> dropped
    out = enc(boxes, labels, (32, 32))
    assert out["mask"].sum() == 1
    np.testing.assert_allclose(out["wh"][0], [2, 2])   # clipped to [0,0,8,8]


def test_centernet_keeps_largest_objects_when_too_many():
    enc = CenterNetTargets(num_classes=1, stride=4, max_objects=2)
    boxes = np.array([[0, 0, 4, 4], [0, 0, 20, 20], [8, 8, 20, 20]], np.float32)
    out = enc(boxes, np.zeros(3, np.int64), (32, 32))
    np.testing.assert_allclose(out["wh"], [[5, 5], [3, 3]])


def test_centernet_rejects_bad_input_size():
    with pytest.raises(ValueError):
        CenterNetTargets(1, stride=4)(np.zeros((0, 4)), np.zeros(0), (30, 32))


def test_grid_encoding_and_collisions():
    enc = GridTargets(num_classes=3, stride=16)
    boxes = np.array([[0, 0, 32, 32], [4, 4, 12, 12], [40, 8, 56, 24]], np.float32)
    out = enc(boxes, np.array([0, 1, 2]), (64, 64))
    assert out["obj"].shape == (4, 4)
    assert out["obj"][1, 1] == 1 and out["cls"][1, 1] == 0
    assert out["obj"][0, 0] == 1 and out["cls"][0, 0] == 1
    assert out["obj"][1, 3] == 1 and out["cls"][1, 3] == 2
    np.testing.assert_allclose(out["wh"][:, 1, 1], [np.sqrt(0.5), np.sqrt(0.5)])
    assert int(out["dropped"]) == 0
    clash = enc(np.array([[0, 0, 30, 30], [2, 2, 26, 26]], np.float32), np.array([0, 1]), (64, 64))
    assert int(clash["dropped"]) == 1 and clash["cls"][0, 0] == 0   # larger box wins
