"""Target encoding + losses. Owner: Abdallah Afifi

- encode_targets(targets, num_classes, out_size, stride): Gaussian heatmaps (CenterNet radius rule),
  wh/offset regression maps, index + mask tensors.
- focal_loss(pred_heatmap_logits, gt_heatmap): penalty-reduced focal loss (alpha=2, beta=4).
- l1 regression loss on wh and offset, only at object centres.
"""
