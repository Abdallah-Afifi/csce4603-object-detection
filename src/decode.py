"""Turn raw model outputs into boxes. Shared by train-time validation and predict.py.

decode(out, k=100, score_thresh=0.3) -> list (per image) of dict:
    "boxes": [M, 4] xyxy in 384x384 input pixels, "scores": [M], "labels": [M]
Peak picking = 3x3 max-pool on sigmoid(heatmap) (acts as NMS), then top-k.
"""
