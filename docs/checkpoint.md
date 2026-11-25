# Checkpoint run sheet (25 Nov)

The handout asks us to "show the first half of your pipeline working in the lab". Our first half is **data → model → training**. Prediction on video, the comparison runs and the evaluation are the second half.

Open `notebooks/checkpoint_demo.ipynb` and run it from top to bottom. Each section matches one person.

| # | Presenter | What to show (notebook section) | Points to make |
|---|---|---|---|
| 1 | Mariam | Class counts table and bar chart | 5 classes kept from VOC 07+12; person dominates (class imbalance, in the report); difficult objects dropped from train, ignored by mAP |
| 1 | Mariam | Augmented samples figure | letterbox keeps the aspect ratio; crop / flip / colour; boxes follow the image (tested in `tests/test_dataset_fake_voc.py`) |
| 1 | Mariam | Split bookkeeping | 90/10 seeded split, no shared images; 07/12 duplicates and VOC 2007 test images removed from train/val |
| 2 | Abdallah | `model_summary.py` output | ResNet-18-style backbone (11.2M parameters), FPN-style neck to 96×96, three heads; random init, no pretrained weights |
| 2 | Abdallah | Heatmap target figure | how a box becomes a Gaussian centre target; focal loss + L1 |
| 3 | Abdallah | Overfit check figure | the full loop (data → targets → loss → decode) is consistent: it memorises 8 real images |
| 3 | Abdallah | Loss curve and val mAP of the checkpoint run | model A, full recipe, on VOC 2007 trainval for 40 epochs (the real run A is 07+12 for 120 epochs); val mAP rose from 0.007 (epoch 5) to 0.31 (epoch 40) |
| 4 | Youssef | Predictions on validation images (epoch 40, val-chosen threshold 0.22) and test-set numbers | `predict.py` uses the same `preprocess` as training; boxes mapped back to the original image |
| 4 | Youssef | `voc_map` | VOC mAP@0.5, checked against the official devkit logic; picks `best.pt` during training |

## Before the lab

```bash
source .venv/bin/activate
python -m pytest -q                       # all tests pass
jupyter / VS Code: open notebooks/checkpoint_demo.ipynb, kernel = .venv
```

Run the notebook once before the lab so the figures in `docs/figures/` already exist. If the lab has no time for the overfit check (about 2 minutes on a laptop GPU), show the saved figure.

## Questions to be ready for

- **Why train from scratch, and how?** The handout requires it. We use a deep stem, zero-initialised residual branches, warm-up, BatchNorm, a long schedule and EMA.
- **Why CenterNet, not YOLO?** YOLO is explicitly not allowed as a pre-existing model. CenterNet has no anchors, a simple decode (3×3 max-pool instead of NMS) and a published design. We wrote every layer ourselves.
- **What is the comparison?** Run A (heatmap head) vs run C (our own grid head, idea from YOLO v1, same backbone) vs run B (A without augmentation). All three use the same data and epoch budget.
- **What does mAP@0.5 mean?** It is the area under each class's precision–recall curve (a box counts if IoU ≥ 0.5), averaged over the 5 classes.
- **What is next?** Finish the runs, video prediction with overlay and FPS, precision/recall and the confusion matrix, our own campus videos as the test video, and failure cases.

## Results

Checkpoint run, VOC 2007 test (4,952 images never used in training):
- **mAP@0.5:** 0.266 (official VOC2007 11-point: 0.291).
- **AP per class:** person 0.38, car 0.39, bicycle 0.20, motorbike 0.27, bus 0.10.
- **At the val-chosen threshold (0.22):** precision 0.41, recall 0.42.
- **Speed:** 141 FPS on the laptop GPU.

The model is trained from scratch on VOC 2007 only for 40 epochs. The real run A adds VOC 2012 and trains for 120 epochs.

Known failure cases to explain:
- **Missed objects** are the largest error; bus has the fewest training examples.
- **Duplicate boxes** appear on large objects (several heatmap peaks for one car).
- **Bus is sometimes called car.**

`docs/results/checkpoint_voc07/` holds the config, `history.csv`, loss curve, and `test_eval/` (metrics.json, confusion matrix, P–R curve). The figures are in `docs/figures/`: class counts, augmented samples and the overfit check. The notebook is saved with its outputs, so it can be shown without re-running.
