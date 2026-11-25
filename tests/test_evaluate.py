import torch

from src.evaluate import voc_map


def pred(boxes, scores, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32), "scores": torch.tensor(scores),
            "labels": torch.tensor(labels)}


def gt(boxes, labels, difficult=None):
    g = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels)}
    if difficult is not None:
        g["difficult"] = torch.tensor(difficult)
    return g


def test_perfect_prediction_gives_one():
    r = voc_map([pred([[0, 0, 10, 10]], [0.9], [0])], [gt([[0, 0, 10, 10]], [0])], num_classes=1)
    assert abs(r["mAP"] - 1.0) < 1e-6


def test_missed_object_halves_recall():
    # 2 objects, only one found -> AP = 0.5
    r = voc_map([pred([[0, 0, 10, 10]], [0.9], [0])],
                [gt([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 0])], num_classes=1)
    assert abs(r["mAP"] - 0.5) < 1e-6


def test_false_positive_ranked_first():
    # wrong box with the higher score: precision at the true hit is 1/2 -> AP = 0.5
    r = voc_map([pred([[100, 100, 110, 110], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])],
                [gt([[0, 0, 10, 10]], [0])], num_classes=1)
    assert abs(r["mAP"] - 0.5) < 1e-6


def test_duplicate_detection_counts_as_false_positive():
    r = voc_map([pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])],
                [gt([[0, 0, 10, 10]], [0])], num_classes=1)
    assert abs(r["mAP"] - 1.0) < 1e-6      # the duplicate comes after full recall, AP stays 1


def test_wrong_class_and_difficult():
    # class 1 has no ground truth -> NaN and left out of the mean
    r = voc_map([pred([[0, 0, 10, 10]], [0.9], [0])], [gt([[0, 0, 10, 10]], [0])], num_classes=2)
    assert abs(r["mAP"] - 1.0) < 1e-6 and r["AP"][1] != r["AP"][1]
    # a detection on a difficult box is ignored, not a false positive
    r = voc_map([pred([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])],
                [gt([[50, 50, 60, 60], [0, 0, 10, 10]], [0, 0], [True, False])], num_classes=1)
    assert abs(r["mAP"] - 1.0) < 1e-6


def test_difficult_flags_given_as_integers():
    # VOC XML stores difficult as 0/1; integer flags must behave exactly like booleans
    preds = [pred([[50, 50, 60, 60], [0, 0, 10, 10], [80, 80, 90, 90]], [0.9, 0.8, 0.7], [0, 0, 0])]
    boxes = [[50, 50, 60, 60], [0, 0, 10, 10], [80, 80, 90, 90]]
    as_bool = voc_map(preds, [gt(boxes, [0, 0, 0], [True, False, False])], num_classes=1)
    as_int = voc_map(preds, [gt(boxes, [0, 0, 0], [1, 0, 0])], num_classes=1)
    assert as_bool["mAP"] == as_int["mAP"] == 1.0


def test_multiple_images_pool_detections_by_score():
    # image 0: hit (0.9); image 1: false positive (0.8) then hit (0.7) -> P = 1, 1/2, 2/3; AP = 0.5*1 + 0.5*2/3
    preds = [pred([[0, 0, 10, 10]], [0.9], [0]),
             pred([[30, 30, 40, 40], [0, 0, 10, 10]], [0.8, 0.7], [0, 0])]
    gts = [gt([[0, 0, 10, 10]], [0]), gt([[0, 0, 10, 10]], [0])]
    assert abs(voc_map(preds, gts, num_classes=1)["mAP"] - (0.5 + 1 / 3)) < 1e-6
