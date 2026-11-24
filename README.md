# Object Detection from Scratch (CSCE 4603, Fall 2026, Option 4)

A CNN detector for **person, car, bicycle, motorbike and bus**, built in PyTorch and trained
from random initialisation on PASCAL VOC. It does not use pretrained weights or ready-made detectors.
The model is a ResNet-style backbone with an upsampling neck and CenterNet-style heads (heatmap, box size, offset).

**Team:** Mariam Abdelaal (data), Abdallah Afifi (model & training), Youssef Badawy (prediction & evaluation)

## Layout

```
configs/   training configs (one YAML per experiment / ablation)
src/       dataset.py, model.py, losses.py, decode.py, evaluate.py
scripts/   train.py, predict.py, show_samples.py
notebooks/ end-to-end notebook for the final submission
docs/      figures and report material
data/      VOCdevkit/ (not committed)
runs/      checkpoints, history.csv, curves (not committed)
videos/    our phone videos + labels (not committed; zipped for submission)
```

Each module's docstring sets out the interface the other modules rely on. Tell the team before you change one.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# VOC 2007 trainval+test and VOC 2012 trainval -> data/VOCdevkit/VOC2007, VOC2012
```

## Experiments (comparison section)

| Run | Head | Augmentation | Purpose |
|---|---|---|---|
| A | CenterNet heatmap | yes | main model |
| B | CenterNet heatmap | no | effect of augmentation |
| C | Grid (YOLO-v1-style) | yes | effect of head design |

We report mAP@0.5, per-class AP, precision/recall, the confusion matrix and FPS on the VOC 2007 test set and on our own labelled video frames.

## Milestones

- [x] 15 Nov: project plan
- [ ] 25 Nov: checkpoint (data loader, model and training running, early results)
- [ ] 6 Dec: code, report (8 pages max), dataset zip and README, 3-minute demo video
- [ ] 9 Dec: live demo

## Use of AI tools

We state any use of AI tools in the report, following the Lecture 01 policy.
