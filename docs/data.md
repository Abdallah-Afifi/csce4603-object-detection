# Data (owner: Mariam Abdelaal)

How the dataset is obtained, laid out and loaded. Code: `src/dataset.py`. Figures and counts: `scripts/show_samples.py`, `scripts/dataset_stats.py`.

## 1. Download

PASCAL VOC, three archives from the official site (about 2.8 GB in total):

| Archive | Contents | Used for |
|---|---|---|
| `VOCtrainval_06-Nov-2007.tar` | VOC 2007 trainval, 5,011 images | train / val |
| `VOCtest_06-Nov-2007.tar` | VOC 2007 test, 4,952 images | test only |
| `VOCtrainval_11-May-2012.tar` | VOC 2012 trainval, 11,540 images | train / val |

Extract all three into the same folder. The 2007 test archive unpacks into the same `VOC2007/` folder as the 2007 trainval archive, which is what we want:

```
data/VOCdevkit/
  VOC2007/  Annotations/  JPEGImages/  ImageSets/Main/{trainval,test}.txt
  VOC2012/  Annotations/  JPEGImages/  ImageSets/Main/trainval.txt
```

```bash
mkdir -p data && cd data
tar -xf VOCtrainval_06-Nov-2007.tar && tar -xf VOCtest_06-Nov-2007.tar && tar -xf VOCtrainval_11-May-2012.tar
```

## 2. Sharing on Kaggle

Upload the `VOCdevkit` folder once as a private Kaggle dataset and share it with the team (Datasets → New Dataset → upload, then Settings → Sharing). The training notebook finds the folder that contains both `VOC2007/` and `VOC2012/` under `/kaggle/input` by itself. Scripts take the path with `--set data.voc_root=/kaggle/input/<dataset>/VOCdevkit`.

## 3. What the loader does

| Step | Rule | Why |
|---|---|---|
| Classes | keep person, car, bicycle, motorbike, bus (ids 0-4 in that order); every other object is ignored | the 5 classes in our plan |
| Images | train uses an image only if it has at least one of the 5 classes; val and test keep every image | images with only other classes teach little in training, but evaluation must count false positives on them |
| Duplicates | VOC 2012 re-uses some VOC 2007 images under the same number (`2007_000032`). Each image is used once | otherwise one photo can land in both train and val |
| Test images | any image in the VOC 2007 test list is removed from train/val | the test set stays unseen |
| Train / val | images sorted by id, shuffled with `seed`, 10% to val. Fixed by the seed; train and val never share an image | repeatable model selection |
| Difficult objects | dropped from train (and images left with no objects are skipped); kept in val/test and flagged, so mAP can ignore them | the VOC rule for training and scoring |
| Coordinates | VOC boxes are 1-based and inclusive; converted to continuous `xyxy` (`xmin - 1, ymin - 1, xmax, ymax`) | correct box sizes |
| Resize | letterbox: longer side to 384, aspect ratio kept, padded with the mean colour (exactly 0 after normalising) | objects are not stretched; `predict.py` uses the same `preprocess()` |
| Normalise | RGB, ImageNet mean/std (just a fixed scaling; no pretrained weights are used) | stable training |

`predict.py` should call `preprocess(frame)` to get the network input and `boxes_to_original(boxes, info)` to map predicted boxes back to the frame.

## 4. Augmentation (training only)

Applied to the original image before the letterbox; boxes are transformed with the image. Settings live in `AUG` in `src/dataset.py`.

| Augmentation | Setting | What happens to the boxes |
|---|---|---|
| Random crop | probability 0.5; the crop covers 30-100% of each side, aspect 0.5-2 | an object is kept if its centre is inside the crop, then clipped to the crop; at least one object always survives |
| Horizontal flip | probability 0.5 | x coordinates mirrored |
| Colour | always: hue ±10°, saturation and brightness ×0.6-1.4 (HSV) | unchanged |

After augmentation, training boxes smaller than 2 px in the 384×384 input are dropped. Validation and test keep every box, so mAP is never computed on a filtered ground truth. Run B (`centernet_voc5_noaug.yaml`) turns all of this off, which is how the report measures what augmentation contributes.

Randomness comes from PyTorch's random generator, which the DataLoader seeds separately for each worker, so workers do not produce identical augmentations and a seeded run is repeatable.

## 5. Figures and counts for the checkpoint and report

```bash
python scripts/show_samples.py  --set data.voc_root=<path to VOCdevkit>   # docs/figures/augmented_samples.png
python scripts/dataset_stats.py --set data.voc_root=<path to VOCdevkit>   # docs/figures/class_counts.{csv,md,png}
```

`dataset_stats.py` prints the table of images and objects per class for train, val and test, the share of objects that are person, and how many images were removed as duplicates, as test images or for having none of our classes. Paste `class_counts.md` into the report; the person-class imbalance is discussed there.

Both scripts also take `--fake`, which runs them on a small generated VOC-format folder with no download.

## 6. Tests

- `tests/test_dataset_fake_voc.py` runs on every push. It builds a tiny VOC-format folder (`tests/fake_voc.py`) with the same edge cases as the real data (07/12 duplicates, a 2007 test image inside 2012, difficult-only images, images with only other classes) and checks the interface, the split, the filtering, and that boxes still sit on their objects after resize and augmentation.
- `tests/test_dataset_contract.py` runs the same interface checks on the real VOC once it is in `data/VOCdevkit`.

## 7. Our own videos (after the checkpoint)

3-5 phone videos filmed on and around campus; about 150 frames labelled by hand in VOC format for testing only, never for training. Folder layout and labelling instructions will be added here with the dataset zip.
