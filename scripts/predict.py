"""Prediction on an image or video. Owner: Youssef Badawy

Usage:
    python scripts/predict.py --weights runs/exp1/best.pt --source path/to/image.jpg
    python scripts/predict.py --weights runs/exp1/best.pt --source path/to/video.mp4 --save out.mp4

Reads with OpenCV, letterboxes to 384, runs the model, decodes boxes, maps them back to the
original resolution and draws boxes + class + confidence + FPS.
"""
