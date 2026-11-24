"""Detector built and trained from scratch (no pretrained weights, no YOLO/torchvision detectors).

Owner: Abdallah Afifi

Interface contract:
    model = CenterNet(num_classes=5)
    out = model(images)                   # images [B, 3, 384, 384]
        out["heatmap"] : [B, 5, 96, 96]   # logits; sigmoid applied in loss/decode
        out["wh"]      : [B, 2, 96, 96]   # box width/height in output-grid units
        out["offset"]  : [B, 2, 96, 96]   # sub-pixel centre offset in [0, 1)

    GridDetector(num_classes=5) is the comparison baseline (YOLO-v1-style grid head),
    sharing the same backbone so the comparison isolates the head.
"""

# TODO(Abdallah): ResNet-style backbone (BasicBlocks, stride 32)
# TODO(Abdallah): upsampling neck (3x transposed conv / upsample+conv) back to stride 4
# TODO(Abdallah): heads; init heatmap bias to -2.19 (prior p=0.1) so focal loss starts stable
# TODO(Abdallah): GridDetector baseline for the comparison section
