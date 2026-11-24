"""Metrics. Owner: Youssef Badawy

Interface used by scripts/train.py to pick best.pt by validation mAP (until this exists, training
selects best.pt by validation loss and still works):

    voc_map(predictions, ground_truths, num_classes, iou_thresh=0.5) -> {"mAP": float, "AP": [float] * num_classes}
        predictions  : list (one per image) of {"boxes": [M,4] xyxy, "scores": [M], "labels": [M]}
                       (exactly what model.decode returns)
        ground_truths: list (one per image) of {"boxes": [N,4], "labels": [N], optional "difficult": [N] bool}
    A perfect prediction must give mAP == 1.0 (train.py checks this before trusting it).
    src.boxes.box_iou is available for matching.

Also planned here:
- precision / recall at a chosen score threshold (bonus target: both > 0.85), and the full P-R curve
- detection confusion matrix (matched boxes -> class confusion, plus missed / false-positive rows)
- FPS on video
"""


def voc_map(predictions, ground_truths, num_classes, iou_thresh=0.5):
    raise NotImplementedError("voc_map is Youssef's part")
