# Project Plan: Object Detection using Deep Learning

CSCE 4603 Fundamentals of Computer Vision · Fall 2026 · Option 4
Team: Mariam Abdelaal, Abdallah Afifi, Youssef Badawy
(Submitted 15 Nov. Student IDs are left out of this public copy.)

## 1. Goal

- Detect **person, car, bicycle, motorbike and bus** in video frames.
- Build our own CNN detector (a convolutional neural network learns image filters for edges, shapes and objects) in PyTorch.
- Train it from scratch: the weights start random, and we use no pretrained models such as YOLO.

## 2. Pipeline

| Stage | What happens |
|---|---|
| Data | Load PASCAL VOC (a public dataset of labelled photos), keep the 5 classes, resize to 384×384, and augment (random flip, crop, colour change) so the model sees more variety. |
| Model | A ResNet-style backbone extracts image features. An upsampling neck enlarges them so small objects can be found. CenterNet-style heads predict object centres (a heatmap), box width/height and a small position offset. |
| Training | Focal loss on the centres, which focuses on hard cases, plus L1 loss on box size. The AdamW optimiser updates the weights over about 120 epochs (full passes over the data) on a free Kaggle/Colab GPU. |
| Prediction | `predict.py` loads the trained model, reads an image or video, and decodes the output into boxes. |
| Output | Boxes, class labels and confidence scores drawn on the image or video. |

## 3. Data plan

- **Train/validation:** VOC 2007 + 2012 trainval. 90% is used for training and 10% is held back to check progress.
- **Test:** the VOC 2007 test set, plus 3–5 phone videos we film on and around campus.
- **Labels:** we draw boxes on about 150 frames of our videos to score the model. These frames are never used for training.

## 4. Evaluation

- **Precision / recall:** the share of predicted boxes that are correct, and the share of real objects found.
- **mAP@0.5:** the main detection score (0–1). A box counts as correct if it overlaps the true box by at least 50%.
- **Confusion matrix, loss curves, FPS:** which classes get mixed up, how the error drops during training, and the speed on video.
- **Comparison:** a simple grid-based head (each grid cell predicts a box) against our heatmap model, each with and without augmentation.
- **Failure cases:** at least 3 wrong predictions, explained.

## 5. Who does what

| Member | Responsibility |
|---|---|
| Mariam Abdelaal | Data: dataset loader, augmentation, filming and labelling videos, dataset README |
| Abdallah Afifi | Model and training: architecture, loss, training script, training results |
| Youssef Badawy | Prediction and evaluation: predict.py, video overlay, metrics, demo video |

Each member writes their own report section and presents their part in the demo. The code is shared on GitHub.

## 6. Timeline

| Dates | Work |
|---|---|
| 15–19 Nov | Data loader, model skeleton, GitHub repo |
| 20–25 Nov | First training runs → **Checkpoint 25 Nov** |
| 26 Nov–2 Dec | Comparison runs, our own videos, prediction on video |
| 3–6 Dec | Evaluation, report, demo video → **Submit 6 Dec**; **Demo 9 Dec** |
