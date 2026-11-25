# Tasks

Tick items off in pull requests. **Checkpoint: Wed 25 Nov** ("show the first half of the pipeline working in the lab").

## Checkpoint (25 Nov)

What we show: the data loads correctly, the model builds, training runs, and the loss goes down.

### Abdallah: model and training
- [x] Backbone + upsampling neck + CenterNet heads (`src/model.py`)
- [x] Grid-head baseline (own code, idea from YOLO v1) for the comparison
- [x] Target encoding, focal + L1 losses (`src/targets.py`, `src/losses.py`)
- [x] Training script: AMP, EMA, warm-up + cosine, resume, history.csv, loss curve (`scripts/train.py`)
- [x] Tests: encode→decode round trip, overfit check, train + resume (`tests/`)
- [x] Kaggle training notebook (`notebooks/train_kaggle.ipynb`)
- [x] Add Mariam and Youssef as collaborators
- [ ] When `VOC5Dataset` lands: `overfit_check.py` on real images, then start run A on Kaggle
- [ ] Checkpoint material: `model_summary.py` output, overfit figure, loss curve so far

### Mariam: data
- [ ] VOC 2007 trainval + test and VOC 2012 trainval downloaded; shared as one Kaggle dataset
- [x] `VOC5Dataset` following the interface in `src/dataset.py`; tested on a generated VOC-format folder; run `pytest tests/test_dataset_contract.py` once real VOC is downloaded
- [x] Augmentation (flip, crop, colour jitter), with boxes kept correct
- [x] `scripts/show_samples.py` figure of augmented samples with boxes (checkpoint figure)
- [x] Object counts per class script (`scripts/dataset_stats.py`); run it on real VOC for the report table

### Youssef: prediction and evaluation
- [x] `scripts/predict.py` on one image using `load_model` + `model.decode` (see `docs/training.md`)
- [x] `voc_map` in `src/evaluate.py` (once it exists, training uses it to pick `best.pt`)
- [x] Agree with Mariam on one shared resize function (letterbox, `src.dataset.preprocess`) that both the dataset and predict.py import

## After the checkpoint (26 Nov – 6 Dec)

- [ ] Runs A, B, C trained; `scripts/plot_runs.py` figure + table (Abdallah)
- [ ] Bonus: COCO 5-class subset (Mariam) + longer run A (Abdallah)
- [ ] Film 3–5 campus videos; label ~150 frames, never used for training (Mariam)
- [ ] predict.py on video with box overlay + FPS (Youssef)
- [ ] Precision/recall at the best-F1 threshold, P–R curve, confusion matrix, per-class AP on VOC 2007 test and on our frames (Youssef)
- [ ] At least 3 failure cases with explanations (all)
- [ ] Report, max 8 pages, with each member writing their own section; AI-tool use stated (all)
- [ ] Dataset zip + README (Mariam); 3-minute demo video on new input (Youssef)
- [ ] Final notebook running end to end with labelled figures (all)
