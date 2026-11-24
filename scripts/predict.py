"""Prediction on an image. Owner: Youssef Badawy

Usage:
    python scripts/predict.py --weights runs/A_centernet/best.pt --source path/to/image.jpg
    python scripts/predict.py --weights runs/A_centernet/best.pt --source image.jpg --save out.jpg --thresh 0.3

Reads with OpenCV, letterboxes to the model input size, runs the model, decodes boxes, maps them back
to the original resolution and draws boxes + class + confidence.
Video (box overlay + FPS) comes after the checkpoint.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import MEAN, STD          # noqa: E402
from src.model import load_model           # noqa: E402


def letterbox(bgr, size):
    """Resize keeping the aspect ratio and pad to size x size. Returns the padded image and how to undo it."""
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[top:top + nh, left:left + nw] = resized
    return padded, scale, left, top


def to_tensor(padded_bgr):
    # same preprocessing as the dataset: RGB, 0..1, normalised with MEAN / STD
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def predict_image(model, bgr, size, thresh):
    padded, scale, left, top = letterbox(bgr, size)
    with torch.no_grad():
        det = model.decode(model(to_tensor(padded).to(next(model.parameters()).device)), score_thresh=thresh)[0]
    boxes = det["boxes"].cpu().numpy()
    # undo the letterbox: remove the padding, then the resize
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    h, w = bgr.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes, det["scores"].cpu().numpy(), det["labels"].cpu().numpy()


def draw(bgr, boxes, scores, labels, names):
    out = bgr.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{names[c]} {s:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image file")
    parser.add_argument("--save", default=None, help="output image (default: <source>_pred.jpg)")
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, cfg = load_model(args.weights, device=args.device)
    size, names = cfg.get("input_size", 384), cfg["classes"]

    bgr = cv2.imread(args.source)
    if bgr is None:
        sys.exit(f"Could not read image: {args.source}")

    start = time.time()
    boxes, scores, labels = predict_image(model, bgr, size, args.thresh)
    print(f"{len(boxes)} detections in {1000 * (time.time() - start):.0f} ms")
    for b, s, c in zip(boxes, scores, labels):
        print(f"  {names[c]:10s} {s:.2f}  box {np.round(b).astype(int).tolist()}")

    save = args.save or str(Path(args.source).with_name(Path(args.source).stem + "_pred.jpg"))
    cv2.imwrite(save, draw(bgr, boxes, scores, labels, names))
    print("saved", save)


if __name__ == "__main__":
    main()
