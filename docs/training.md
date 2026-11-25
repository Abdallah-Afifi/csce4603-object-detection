# Training guide

Owner: Abdallah Afifi. Code: `src/model.py`, `src/targets.py`, `src/losses.py`, `src/train_utils.py`, `scripts/train.py`.

## How the model works (what to say in the demo)

1. **Backbone (ResNet-18-sized, about 11M parameters, random initialisation).** A stem of three 3×3 convolutions brings the image to 1/4 resolution, then four stages of residual blocks produce feature maps at 1/4, 1/8, 1/16 and 1/32 resolution. Deeper maps "know" more about what is in the image but are coarse.
   - We use three 3×3 convs instead of ResNet's 7×7 + max-pool stem because small stacked convs train better from scratch (DSOD).
   - The last BatchNorm of each residual block starts at zero, so every block starts as an identity mapping. This makes training from scratch more stable.
2. **Upsampling neck (FPN-style top-down path).** It starts from the 1/32 map, upsamples ×2, adds a 1×1-projected copy of the backbone map at that scale, and smooths with a 3×3 conv. Repeating this down to 1/4 resolution gives a 96×96 map that is both high-resolution (good for small, far-away people and bikes) and semantically strong.
3. **CenterNet heads (objects as points).** Three small conv heads run on the 96×96 map:
   - `heatmap` [5×96×96]: one channel per class, with a peak at each object's centre.
   - `wh` [2×96×96]: box width and height, read at the peak.
   - `offset` [2×96×96]: where the true centre falls inside its cell, which recovers the precision lost by downsampling ×4.
4. **Targets.** For each ground-truth box we draw a 2-D Gaussian on its class channel, with value 1 at the centre cell. Its radius grows with the box size, using the heuristic from CornerNet that the original CenterNet code reuses. Cells near a centre are therefore penalised less for firing.
5. **Losses.**
   - Focal loss on the heatmap. Background cells vastly outnumber centres; focal loss down-weights easy cells, and the `(1 - gt)^4` term penalises cells next to a centre less.
   - L1 loss on `wh` and `offset`, at centre cells only.
   - Total = 1.0·heatmap + 0.1·wh + 1.0·offset.
6. **Decoding.** Take the sigmoid of the heatmap. A cell is a detection if it is the maximum of its 3×3 neighbourhood, which replaces NMS. Keep the top 100 peaks, then build boxes from `wh` and `offset`.
7. **Comparison baseline: grid head (YOLO v1 style).** This uses the same backbone and the same neck design (stopping at stride 16), with a 24×24 grid. The cell that contains an object's centre predicts one box (centre in the cell, sqrt of the size), an objectness score and the class; outputs then go through NMS. If two objects fall in the same cell, only one can be learned. The training log counts these as `grid_collisions`, a concrete number for the report.

## Training recipe

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 (not on BN/bias) | stable from scratch |
| Schedule | 3-epoch linear warm-up, then cosine to 1e-5 | big early steps on random weights destabilise BatchNorm |
| Epochs | 120 | from-scratch training needs long schedules (He et al., "Rethinking ImageNet Pre-training") |
| Batch | 32 at 384×384 | |
| EMA | exponential moving average of the weights (decay 0.9998); validation and `best.pt` use the EMA weights | smoother and usually more accurate than the raw weights |
| Mixed precision | float16 on CUDA, losses in float32 | about 2× faster on a T4 |
| Safety | gradient clipping at 10, non-finite steps skipped, atomic checkpoint writes, automatic resume | multi-hour runs on preemptible notebooks |

## Running

```bash
# 0. Pipeline check without any dataset (about 1 minute on a laptop)
python scripts/train.py --config configs/smoke_synthetic.yaml --out runs/smoke

# 1. Can the model memorise 8 real images? Saves docs/figures/overfit_check.png
python scripts/overfit_check.py --config configs/centernet_voc5.yaml

# 2. The three runs for the comparison section
python scripts/train.py --config configs/centernet_voc5.yaml       --out runs/A_centernet
python scripts/train.py --config configs/centernet_voc5_noaug.yaml --out runs/B_noaug
python scripts/train.py --config configs/grid_voc5.yaml            --out runs/C_grid

# 3. Report figure + table
python scripts/plot_runs.py runs/A_centernet runs/B_noaug runs/C_grid --out docs/figures/runs.png
python scripts/model_summary.py --config configs/centernet_voc5.yaml
```

Any config value can be overridden: `--set train.epochs=60 train.batch_size=16`.
Re-running the same command, with the same `--set` values, resumes from `runs/<name>/last.pt`. A changed schedule (epochs, batch size, learning rate) is refused unless you pass `--allow-config-change`.

## On Kaggle (free GPU)

Use `notebooks/train_kaggle.ipynb`. Set the accelerator to GPU T4 ×2 or P100 and turn the internet on. Sessions last up to 12 h and the GPU quota is about 30 h a week, so the three runs need planning:

- Run A first. It is the main model and the source of the demo weights.
- **All three runs must use the same epoch count**, because the comparison is only fair at the same budget. If time is short, train A, B and C for 60 epochs each and give A a longer run separately for the final demo weights. Don't compare a 120-epoch A against 60-epoch B and C.
- To continue after a timeout: open the notebook's saved output version, add it as an input, copy its `runs/<name>` folder back to `/kaggle/working/runs/`, and run the same command again.

## For Youssef: using the trained model

```python
from src.model import load_model
model, cfg = load_model("runs/A_centernet/best.pt", device="cuda")   # eval mode, EMA weights
out = model(batch)                                   # batch: [B, 3, 384, 384], same preprocessing as the dataset
dets = model.decode(out, score_thresh=0.3)           # list of {"boxes" xyxy in 384-px input coords, "scores", "labels"}
names = cfg["classes"]                               # label id -> name
```

Use the same call for the grid model; `load_model` reads the head type from the checkpoint. Prepare frames with `src.dataset.preprocess(frame)` and map boxes back with `src.dataset.boxes_to_original(boxes, info)`; `scripts/predict.py` already does both.

## Bonus (precision and recall > 85%): levers in this code

- **More data:** add COCO images of the same 5 classes through `data.extra_train` in the config (Mariam's `COCO5Dataset`). This is the biggest gain when training from scratch.
- **Longer training:** set a larger `train.epochs` from the start. Resuming with a different `epochs`, batch size or learning rate is refused, because the cosine schedule would jump; `--allow-config-change` overrides this.
- **Larger input:** `--set input_size=512` helps small objects but is about 1.8× slower.
- **Operating point:** report precision and recall at the score threshold that maximises F1, and show the full P–R curve on the VOC test set and on our own video frames.
