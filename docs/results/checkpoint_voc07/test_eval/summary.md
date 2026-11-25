| Run | epoch | mAP@0.5 | mAP (VOC07 11-pt) | AP person | AP car | AP bicycle | AP motorbike | AP bus | threshold | precision | recall | F1 | error rate | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| checkpoint_voc07 | 40 | 0.266 | 0.291 | 0.377 | 0.388 | 0.198 | 0.265 | 0.102 | 0.22 | 0.410 | 0.418 | 0.414 | 0.739 | 141 |

Test set: 4952 images. Threshold = best F1 on the validation split (not tuned on test). Error rate = (FP + FN) / (TP + FP + FN). FPS = batch 1, model + decode, on mps.
