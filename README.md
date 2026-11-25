# Object Detection from Scratch (CSCE 4603, Fall 2026, Option 4)

[![tests](https://github.com/Abdallah-Afifi/csce4603-object-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Abdallah-Afifi/csce4603-object-detection/actions/workflows/tests.yml)

A CNN detector for **person, car, bicycle, motorbike and bus**, built in PyTorch and trained
from random initialisation on PASCAL VOC. It does not use pretrained weights or ready-made detectors.

**Team:** Mariam Abdelaal (data), Abdallah Afifi (model & training), Youssef Badawy (prediction & evaluation)

- [Project plan](docs/project_plan.md): goal, pipeline, data, evaluation, who does what
- [Tasks and checkpoint checklist](docs/tasks.md)
- [Training guide](docs/training.md): how the model works, how to train, how to load weights
- [Data](docs/data.md): download, folder layout, loader rules, augmentation, class counts

## Pipeline

```
image 384x384 ─► Backbone (ResNet-18-style, from scratch) ─► Upsampling neck (FPN-style) ─► Heads
                  C2..C5 at strides 4..32                     P2 at stride 4 (96x96)        heatmap 5x96x96  (object centres)
                                                                                            wh      2x96x96  (box size)
                                                                                            offset  2x96x96  (sub-cell centre)
decode: 3x3 max-pool peaks ─► top-100 ─► boxes ─► draw on image / video
```

Comparison baseline: the same backbone and neck with a **grid head** (24×24 cells, one box per cell). It is our own implementation of the idea from YOLO v1 (Redmon et al. 2016), with no YOLO code or weights.

## Layout

```
configs/    centernet_voc5.yaml (run A), centernet_voc5_noaug.yaml (B), grid_voc5.yaml (C), smoke_synthetic.yaml
src/        dataset.py       VOC loader + augmentation                         (Mariam)
            model.py         backbone, neck, CenterNet + grid detectors        (Abdallah)
            targets.py       boxes -> heatmap / grid training targets          (Abdallah)
            losses.py        focal loss, L1, grid loss                         (Abdallah)
            train_utils.py   EMA, LR schedule, checkpoints, plots              (Abdallah)
            boxes.py         IoU, NMS
            synthetic.py     generated shapes for tests and smoke runs
            evaluate.py      mAP (all-point + VOC07), P/R/F1, confusion matrix (Youssef)
scripts/    train.py, overfit_check.py, model_summary.py, plot_runs.py         (Abdallah)
            predict.py (image / video / webcam, --show), evaluate.py (test-set metrics)  (Youssef)
            show_samples.py, dataset_stats.py                                  (Mariam)
notebooks/  train_kaggle.ipynb: training on a free Kaggle GPU
tests/      pytest suite (runs on every push via GitHub Actions)
docs/       plan, tasks, training guide, figures
data/ runs/ videos/   not committed (datasets, checkpoints, our videos)
```

Each module's docstring sets out the interface the other modules rely on. Tell the team before you change one.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # a few seconds on CPU; no dataset needed
```

PASCAL VOC (2007 trainval + test, 2012 trainval) goes in `data/VOCdevkit/VOC2007` and `data/VOCdevkit/VOC2012`.

## Training

```bash
python scripts/train.py --config configs/smoke_synthetic.yaml --out runs/smoke     # pipeline check, no data
python scripts/train.py --config configs/centernet_voc5.yaml  --out runs/A_centernet
python scripts/evaluate.py runs/A_centernet runs/B_noaug runs/C_grid --out docs/results/comparison
python scripts/predict.py --weights runs/A_centernet/best.pt --source clip.mp4 --show
```

Re-running the same command resumes from `last.pt`. Each run writes `best.pt`, `last.pt`, `history.csv` and `loss_curve.png`. See the [training guide](docs/training.md) for Kaggle and the comparison runs.

## Experiments (comparison section)

| Run | Config | Head | Augmentation | Question it answers |
|---|---|---|---|---|
| A | `centernet_voc5.yaml` | CenterNet heatmap | yes | main model |
| B | `centernet_voc5_noaug.yaml` | CenterNet heatmap | no | what does augmentation contribute? |
| C | `grid_voc5.yaml` | Grid head (own code, YOLO-v1 idea) | yes | what does the head design contribute? |

We report mAP@0.5, per-class AP, precision/recall, the confusion matrix and FPS on the VOC 2007 test set and on our own labelled video frames.

## Milestones

- [x] 15 Nov: project plan
- [ ] 25 Nov: checkpoint (data loader, model and training running, early results)
- [ ] 6 Dec: code, report (8 pages max), dataset zip and README, 3-minute demo video
- [ ] 9 Dec: live demo

## Use of AI tools

We state any use of AI tools in the report, following the Lecture 01 policy.
