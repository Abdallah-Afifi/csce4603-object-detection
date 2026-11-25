"""Prediction on an image, a video file or a webcam, with the results drawn on top. Owner: Youssef Badawy

Usage:
    python scripts/predict.py --weights runs/A_centernet/best.pt --source photo.jpg
    python scripts/predict.py --weights runs/A_centernet/best.pt --source clip.mp4 --save clip_pred.mp4
    python scripts/predict.py --weights runs/A_centernet/best.pt --source clip.mp4 --show     # live window
    python scripts/predict.py --weights runs/A_centernet/best.pt --source 0 --show            # webcam 0

Every frame is preprocessed exactly like training (src.dataset.preprocess: letterbox + MEAN/STD).
The model runs, heatmap peaks are decoded into boxes, the boxes are mapped back to the frame, and
boxes, class, confidence and (for video) FPS are drawn. --show displays the result in a window
(press q to stop a video). Without --save, the output goes next to the input as <name>_pred.<ext>.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import boxes_to_original, preprocess  # noqa: E402
from src.model import load_model           # noqa: E402
from src.train_utils import pick_device     # noqa: E402


def predict_image(model, bgr, size, thresh):
    """Run the detector on one BGR frame. Returns boxes (xyxy in the frame's pixels), scores, labels.

    Uses the dataset's own preprocess() (letterbox with mean-colour padding + MEAN/STD), so a frame is
    prepared exactly like the training images, and boxes_to_original() to undo it.
    """
    tensor, info = preprocess(bgr, size)
    with torch.no_grad():
        det = model.decode(model(tensor.unsqueeze(0).to(next(model.parameters()).device)), score_thresh=thresh)[0]
    boxes = boxes_to_original(det["boxes"].cpu().numpy(), info)
    # a box that lay entirely in the padding collapses to zero width/height after clipping: drop it
    keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
    return boxes[keep], det["scores"].cpu().numpy()[keep], det["labels"].cpu().numpy()[keep]


def draw(bgr, boxes, scores, labels, names):
    out = bgr.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{names[c]} {s:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def annotate_fps(frame, fps):
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def run_image(model, path, size, names, args):
    bgr = cv2.imread(str(path))
    if bgr is None:
        sys.exit(f"Could not read image: {path}")
    start = time.time()
    boxes, scores, labels = predict_image(model, bgr, size, args.thresh)
    print(f"{len(boxes)} detections in {1000 * (time.time() - start):.0f} ms")
    for b, s, c in zip(boxes, scores, labels):
        print(f"  {names[c]:10s} {s:.2f}  box {np.round(b).astype(int).tolist()}")
    out = draw(bgr, boxes, scores, labels, names)
    save = Path(args.save or Path(path).with_name(Path(path).stem + "_pred.jpg"))
    save.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(save), out)  # may return False silently
    except cv2.error:
        ok = False
    if not ok:
        sys.exit(f"Could not write {save} (use a .jpg or .png path)")
    print("saved", save)
    if args.show:
        cv2.imshow("detections (press any key)", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(model, source, size, names, args):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, save = None, None
    if args.save or not str(source).isdigit():
        save = Path(args.save or Path(source).with_name(Path(source).stem + "_pred.mp4"))
        save.parent.mkdir(parents=True, exist_ok=True)
    frames, detections, smoothed, t_start = 0, 0, None, time.time()
    try:
        while args.max_frames is None or frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            boxes, scores, labels = predict_image(model, frame, size, args.thresh)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            smoothed = fps if smoothed is None else 0.9 * smoothed + 0.1 * fps
            out = annotate_fps(draw(frame, boxes, scores, labels, names), smoothed)
            if save is not None:
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                    if not writer.isOpened():
                        sys.exit(f"Could not write {save} (use a .mp4 path)")
                writer.write(out)
            if args.show:
                cv2.imshow("detections (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frames += 1
            detections += len(boxes)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if frames == 0:
        sys.exit(f"No frames could be read from {source}")
    total = time.time() - t_start
    print(f"{frames} frames, {detections / frames:.1f} detections per frame, "
          f"{frames / total:.1f} FPS end to end (model + drawing{' + writing' if save else ''})")
    if save is not None:
        print("saved", save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="best.pt / last.pt written by scripts/train.py")
    parser.add_argument("--source", required=True, help="image file, video file, or webcam index (0)")
    parser.add_argument("--save", default=None, help="output image/video path (default: <source>_pred.*)")
    parser.add_argument("--thresh", type=float, default=0.3, help="minimum confidence to draw a box")
    parser.add_argument("--show", action="store_true", help="display the result in a window")
    parser.add_argument("--max-frames", type=int, default=None, help="stop a video after this many frames")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    args = parser.parse_args(argv)

    model, cfg = load_model(args.weights, device=pick_device(args.device))
    size, names = cfg.get("input_size", 384), cfg["classes"]
    source = args.source
    if not source.isdigit() and not Path(source).exists():
        sys.exit(f"Source not found: {source}")
    if Path(source).suffix.lower() in IMAGE_EXTS:
        run_image(model, source, size, names, args)
    else:
        run_video(model, source, size, names, args)


if __name__ == "__main__":
    main()
