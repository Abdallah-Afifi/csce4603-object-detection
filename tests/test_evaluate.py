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


# --- metrics beyond mAP ------------------------------------------------------------------------

from src.evaluate import (average_precision, best_f1_threshold, detection_confusion_matrix,  # noqa: E402
                          error_rate, precision_recall, pr_curves)


def _scene():
    """2 images, 3 real objects (one difficult) and 5 detections with known outcomes."""
    preds = [pred([[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]], [0.9, 0.6, 0.4], [0, 1, 0]),
             pred([[0, 0, 10, 10], [30, 30, 40, 40]], [0.8, 0.7], [1, 0])]
    gts = [gt([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 0]),           # 2nd object detected as class 1
           gt([[0, 0, 10, 10], [30, 30, 40, 40]], [1, 0], [False, True])]
    return preds, gts


def test_eleven_point_ap_matches_the_voc2007_definition():
    recall, precision = [0.25, 0.5, 0.5, 1.0], [1.0, 1.0, 0.67, 0.5]
    expected = (6 * 1.0 + 5 * 0.5) / 11       # recall 0..0.5 -> 1.0 ; 0.6..1.0 -> 0.5
    assert abs(average_precision(recall, precision, use_07_metric=True) - expected) < 1e-9
    assert abs(average_precision(recall, precision) - 0.75) < 1e-9


def test_precision_recall_and_error_rate_at_a_threshold():
    preds, gts = _scene()
    r = precision_recall(preds, gts, num_classes=2, score_thresh=0.5)
    # >= 0.5: (0.9, c0) TP ; (0.6, c1) FP: wrong class ; (0.8, c1) TP ; (0.7, c0) on a difficult box: ignored
    assert r["overall"]["tp"] == 2 and r["overall"]["fp"] == 1 and r["overall"]["fn"] == 1
    assert r["overall"]["precision"] == 2 / 3 and r["overall"]["recall"] == 2 / 3
    assert r["per_class"][0]["n_gt"] == 2 and r["per_class"][1]["n_gt"] == 1
    assert error_rate(preds, gts, num_classes=2, score_thresh=0.5) == 2 / 4


def test_best_f1_threshold_and_curves():
    preds, gts = _scene()
    best = best_f1_threshold(preds, gts, num_classes=2)
    assert abs(best["score_thresh"] - 0.8) < 1e-6 and best["precision"] == 1.0 and abs(best["recall"] - 2 / 3) < 1e-9
    thresholds, precision, recall = pr_curves(preds, gts, num_classes=2)["overall"]
    assert list(thresholds) == sorted(thresholds, reverse=True) and recall[-1] == 2 / 3


def test_detection_confusion_matrix():
    preds, gts = _scene()
    m = detection_confusion_matrix(preds, gts, num_classes=2, score_thresh=0.3)
    # rows = true class (2 = background), columns = predicted class (2 = missed)
    expected = [[1, 1, 0],    # class 0: one found as 0, one found as 1 (wrong class)
                [0, 1, 0],    # class 1: found
                [1, 0, 0]]    # background: one false positive of class 0 (0.4 at [80,80,90,90])
    assert m.tolist() == expected            # the detection on the difficult box is left out
    m = detection_confusion_matrix(preds, gts, num_classes=2, score_thresh=0.95)
    assert m.tolist() == [[0, 0, 2], [0, 0, 1], [0, 0, 0]]   # nothing detected: all missed
