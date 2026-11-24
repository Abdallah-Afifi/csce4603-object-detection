"""Detectors built and trained from scratch (random init; no pretrained weights, no YOLO).

Owner: Abdallah Afifi

Architecture (both detectors share the same backbone and neck):

    image [B,3,384,384]
      -> Backbone: 3-conv stem (stride 4) + 4 ResNet stages of BasicBlocks -> C2..C5 (stride 4..32)
      -> Neck: top-down upsampling path with lateral connections (FPN-style), down to the head's stride
      -> Head:
           CenterNet : stride 4 (96x96)  -> heatmap [B,5,96,96], wh [B,2,..], offset [B,2,..]
           Grid      : stride 16 (24x24) -> obj [B,1,..], box [B,4,..], cls [B,5,..]

Every detector exposes the same API, so training and prediction code works with either:
    out = model(images)                                  # dict of raw head outputs
    enc = model.target_encoder()(boxes, labels, (H, W))  # NumPy targets for one image
    loss, parts = model.compute_loss(out, batched_targets, weights)
    dets = model.decode(out, score_thresh=0.3, max_det=100)
        -> list (one per image) of {"boxes": [M,4] xyxy input pixels, "scores": [M], "labels": [M]}

Use ``build_model(cfg)`` to construct from a config and ``load_model(path)`` to load a checkpoint.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.boxes import batched_nms, clip_boxes
from src.losses import centernet_loss, gather_at, grid_loss
from src.targets import CenterNetTargets, GridTargets


def conv_bn_relu(cin: int, cout: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs plus a shortcut (1x1 conv when the shape changes)."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class Backbone(nn.Module):
    """ResNet-style feature extractor. Returns features at strides 4, 8, 16 and 32.

    The stem is three 3x3 convs instead of ResNet's single 7x7 + max-pool. Deep stems train
    better from scratch (DSOD, Shen et al. 2017), and we have no pretrained weights.
    """

    def __init__(self, widths=(64, 128, 256, 512), depths=(2, 2, 2, 2)):
        super().__init__()
        if len(widths) != 4 or len(depths) != 4:
            raise ValueError("backbone needs 4 stage widths and 4 depths")
        stem = widths[0] // 2
        self.stem = nn.Sequential(
            conv_bn_relu(3, stem, 3, 2),
            conv_bn_relu(stem, stem, 3, 1),
            conv_bn_relu(stem, widths[0], 3, 2),
        )
        stages, cin = [], widths[0]
        for i, (width, depth) in enumerate(zip(widths, depths)):
            blocks = [BasicBlock(cin, width, stride=1 if i == 0 else 2)]
            blocks += [BasicBlock(width, width) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = width
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(widths)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # strides 4, 8, 16, 32


class UpsamplingNeck(nn.Module):
    """Top-down path: start from the deepest (most semantic) map, upsample x2, add the lateral
    projection of the backbone map at that scale, smooth with a 3x3 conv. The stride-4 output is
    both semantically strong and high-resolution, which helps small, distant objects."""

    def __init__(self, in_channels, channels: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(conv_bn_relu(c, channels, 1) for c in in_channels)
        self.smooth = nn.ModuleList(conv_bn_relu(channels, channels, 3) for _ in in_channels[:-1])

    def forward(self, feats):
        """feats: backbone maps from finest to coarsest. Returns the finest fused map."""
        x = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            lateral = self.lateral[i](feats[i])
            x = F.interpolate(x, size=lateral.shape[-2:], mode="nearest") + lateral
            x = self.smooth[i](x)
        return x


def make_head(cin: int, cout: int, mid: int, bias: float = 0.0) -> nn.Sequential:
    head = nn.Sequential(nn.Conv2d(cin, mid, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, cout, 1))
    nn.init.normal_(head[-1].weight, std=0.01)
    nn.init.constant_(head[-1].bias, bias)
    return head


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    # Zero the last BN of each residual branch so every block starts as an identity mapping
    # (Goyal et al. 2017); this makes training from scratch noticeably more stable.
    for m in module.modules():
        if isinstance(m, BasicBlock):
            nn.init.zeros_(m.bn2.weight)


def prior_bias(p: float) -> float:
    """Bias so that sigmoid(bias) = p, i.e. the head starts by predicting 'object' with prob p."""
    return -math.log((1 - p) / p)


class _Detector(nn.Module):
    stride: int
    LEVEL = {4: 0, 8: 1, 16: 2, 32: 3}   # output stride -> backbone level

    def __init__(self, num_classes, widths, depths, neck_channels, stride):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.level = self.LEVEL[stride]
        self.backbone = Backbone(widths, depths)
        # The neck only goes down to the resolution the head needs (stride 4 for CenterNet).
        self.neck = UpsamplingNeck(self.backbone.out_channels[self.level:], neck_channels)
        _init_weights(self.backbone)
        _init_weights(self.neck)

    def features(self, x):
        """Fused feature map at ``self.stride``."""
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(f"input size {tuple(x.shape[-2:])} must be divisible by 32")
        return self.neck(self.backbone(x)[self.level:])


class CenterNet(_Detector):
    """Objects as points (Zhou et al. 2019): predict a heatmap peak at each object centre, plus
    the box size and a sub-cell offset at that peak. No anchors; NMS is optional."""

    def __init__(self, num_classes=5, widths=(64, 128, 256, 512), depths=(2, 2, 2, 2),
                 neck_channels=128, head_channels=64, max_objects=128):
        super().__init__(num_classes, widths, depths, neck_channels, stride=4)
        self.max_objects = max_objects
        # Start with p(object) = 0.1 everywhere; with bias 0 (p = 0.5) the first focal-loss steps
        # are dominated by thousands of background cells and training can diverge.
        self.heatmap = make_head(neck_channels, num_classes, head_channels, prior_bias(0.1))
        self.wh = make_head(neck_channels, 2, head_channels)
        self.offset = make_head(neck_channels, 2, head_channels)

    def forward(self, x):
        p2 = self.features(x)
        return {"heatmap": self.heatmap(p2), "wh": self.wh(p2), "offset": self.offset(p2)}

    def target_encoder(self):
        return CenterNetTargets(self.num_classes, self.stride, self.max_objects)

    def compute_loss(self, out, targets, weights=None):
        return centernet_loss(out, targets, weights)

    @torch.no_grad()
    def decode(self, out, score_thresh=0.3, max_det=100, nms_iou=None):
        heat = torch.sigmoid(out["heatmap"].float())
        b, c, h, w = heat.shape
        # A cell is a detection only if it is the maximum of its 3x3 neighbourhood. This replaces
        # NMS: one object makes one peak.
        peaks = heat * (F.max_pool2d(heat, 3, stride=1, padding=1) == heat)
        k = min(max_det, c * h * w)
        scores, idx = peaks.reshape(b, -1).topk(k)
        labels = idx // (h * w)
        cell = idx % (h * w)
        ys, xs = (cell // w).float(), (cell % w).float()
        wh = gather_at(out["wh"].float(), cell).clamp(min=0)
        off = gather_at(out["offset"].float(), cell)
        cx, cy = (xs + off[..., 0]) * self.stride, (ys + off[..., 1]) * self.stride
        bw, bh = wh[..., 0] * self.stride, wh[..., 1] * self.stride
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
        return _finalize(boxes, scores, labels, score_thresh, nms_iou, h * self.stride, w * self.stride)


class GridDetector(_Detector):
    """Comparison baseline in the style of YOLO v1 (Redmon et al. 2016): split the image into a
    grid and let the cell containing an object's centre predict one box, an objectness score and
    the class. Uses the same backbone and neck as CenterNet, so the comparison isolates the head."""

    def __init__(self, num_classes=5, widths=(64, 128, 256, 512), depths=(2, 2, 2, 2),
                 neck_channels=128, head_channels=64, grid_stride=16):
        if grid_stride not in self.LEVEL:
            raise ValueError("grid_stride must be 4, 8, 16 or 32")
        super().__init__(num_classes, widths, depths, neck_channels, stride=grid_stride)
        self.obj = make_head(neck_channels, 1, head_channels, prior_bias(0.01))
        self.box = make_head(neck_channels, 4, head_channels)
        self.cls = make_head(neck_channels, num_classes, head_channels)

    def forward(self, x):
        p = self.features(x)
        return {"obj": self.obj(p), "box": self.box(p), "cls": self.cls(p)}

    def target_encoder(self):
        return GridTargets(self.num_classes, self.stride)

    def compute_loss(self, out, targets, weights=None):
        return grid_loss(out, targets, weights)

    @torch.no_grad()
    def decode(self, out, score_thresh=0.3, max_det=100, nms_iou=0.5):
        obj = torch.sigmoid(out["obj"].float())[:, 0]               # [B, G, G]
        cls_prob = torch.softmax(out["cls"].float(), dim=1)         # [B, C, G, G]
        cls_score, labels = cls_prob.max(dim=1)
        scores = obj * cls_score
        box = torch.sigmoid(out["box"].float())
        b, gh, gw = scores.shape
        in_h, in_w = gh * self.stride, gw * self.stride
        ys, xs = torch.meshgrid(torch.arange(gh, device=box.device), torch.arange(gw, device=box.device),
                                indexing="ij")
        cx = (xs + box[:, 0]) * self.stride
        cy = (ys + box[:, 1]) * self.stride
        bw, bh = box[:, 2] ** 2 * in_w, box[:, 3] ** 2 * in_h
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
        boxes = boxes.reshape(b, -1, 4)
        scores, labels = scores.reshape(b, -1), labels.reshape(b, -1)
        k = min(max_det * 4, scores.shape[1])  # keep extra candidates for NMS to choose from
        scores, idx = scores.topk(k)
        labels = labels.gather(1, idx)
        boxes = boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
        return _finalize(boxes, scores, labels, score_thresh, nms_iou, in_h, in_w, max_det)


def _finalize(boxes, scores, labels, score_thresh, nms_iou, in_h, in_w, max_det=None):
    results = []
    for bx, sc, lb in zip(boxes, scores, labels):
        keep = sc >= score_thresh
        bx, sc, lb = clip_boxes(bx[keep], in_h, in_w), sc[keep], lb[keep]
        valid = (bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1])
        bx, sc, lb = bx[valid], sc[valid], lb[valid]
        if nms_iou is not None and bx.shape[0] > 1:
            kept = batched_nms(bx, sc, lb, nms_iou)
            bx, sc, lb = bx[kept], sc[kept], lb[kept]
        if max_det is not None:
            bx, sc, lb = bx[:max_det], sc[:max_det], lb[:max_det]
        results.append({"boxes": bx, "scores": sc, "labels": lb})
    return results


# ----------------------------------------------------------------------------------------------
# Construction and loading
# ----------------------------------------------------------------------------------------------

def build_model(cfg: dict) -> _Detector:
    m = cfg.get("model", {})
    common = dict(
        num_classes=len(cfg["classes"]),
        widths=tuple(m.get("backbone_widths", (64, 128, 256, 512))),
        depths=tuple(m.get("backbone_depths", (2, 2, 2, 2))),
        neck_channels=m.get("neck_channels", 128),
        head_channels=m.get("head_channels", 64),
    )
    head = m.get("head", "centernet")
    if head == "centernet":
        return CenterNet(max_objects=m.get("max_objects", 128), **common)
    if head == "grid":
        return GridDetector(grid_stride=m.get("grid_stride", 16), **common)
    raise ValueError(f"Unknown model.head '{head}' (expected 'centernet' or 'grid')")


def load_model(path: str | Path, device: str | torch.device = "cpu", use_ema: bool = True):
    """Load a checkpoint written by scripts/train.py for inference.

    Returns (model in eval mode on ``device``, config dict). Works with best.pt and last.pt.
    The config includes "classes", the class names in label order.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    state = ckpt.get("ema") if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
