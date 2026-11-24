"""Loss functions for both detection heads. Owner: Abdallah Afifi

All losses are computed in float32 even when the forward pass ran in mixed precision, because
log() of small probabilities is not reliable in float16.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0,
               beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss from CenterNet / CornerNet.

    logits : [B, C, H, W] raw heatmap scores
    gt     : [B, C, H, W] Gaussian targets in [0, 1], exactly 1 at object centres

    At a centre the loss is -(1-p)^alpha log(p): confident correct cells contribute almost nothing,
    so training focuses on hard cells. Elsewhere it is -(1-gt)^beta p^alpha log(1-p): cells
    near a centre (gt close to 1) are penalised less for firing, since a box there is still good.
    Normalised by the number of objects so the scale does not depend on how crowded an image is.
    """
    logits = logits.float()
    gt = gt.float()
    prob = torch.sigmoid(logits)
    log_p = F.logsigmoid(logits)            # log(p), numerically stable
    log_1mp = F.logsigmoid(-logits)         # log(1 - p)

    pos = gt.eq(1).float()
    neg = 1.0 - pos
    pos_loss = (log_p * (1 - prob).pow(alpha) * pos).sum()
    neg_loss = (log_1mp * prob.pow(alpha) * (1 - gt).pow(beta) * neg).sum()
    num_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss + neg_loss) / num_pos


def gather_at(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Pick feature vectors at flattened positions. feature [B, C, H, W], index [B, K] -> [B, K, C]."""
    b, c, h, w = feature.shape
    flat = feature.reshape(b, c, h * w).permute(0, 2, 1)
    return flat.gather(1, index.long().unsqueeze(-1).expand(-1, -1, c))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over real objects only. pred/target [B, K, D], mask [B, K]."""
    m = mask.float().unsqueeze(-1)
    total = ((pred.float() - target.float()).abs() * m).sum()
    return total / (m.sum() * pred.shape[-1]).clamp(min=1.0)


def centernet_loss(out: dict, targets: dict, weights: dict | None = None):
    """Returns (total loss, dict of detached parts for logging)."""
    w = {"heatmap": 1.0, "wh": 0.1, "offset": 1.0, **(weights or {})}
    hm = focal_loss(out["heatmap"], targets["heatmap"])
    wh = masked_l1(gather_at(out["wh"], targets["index"]), targets["wh"], targets["mask"])
    off = masked_l1(gather_at(out["offset"], targets["index"]), targets["offset"], targets["mask"])
    total = w["heatmap"] * hm + w["wh"] * wh + w["offset"] * off
    return total, {"hm": hm.detach(), "wh": wh.detach(), "off": off.detach()}


def grid_loss(out: dict, targets: dict, weights: dict | None = None):
    """YOLO-v1-style loss, with BCE/CE in place of squared error for objectness and class.

    obj  : BCE on every cell; empty cells down-weighted by ``noobj`` (most cells are empty)
    box  : squared error on centre (in-cell) and sqrt-size, owning cells only, weighted by ``coord``
    cls  : cross-entropy on owning cells only
    Summed over cells and averaged over images, as in YOLO v1.
    """
    w = {"coord": 5.0, "noobj": 0.5, "obj": 1.0, "cls": 1.0, **(weights or {})}
    batch = out["obj"].shape[0]
    obj_t = targets["obj"].float()                       # [B, G, G]
    obj_logit = out["obj"].float()[:, 0]                 # [B, G, G]
    bce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    obj_loss = (bce * (obj_t + w["noobj"] * (1 - obj_t))).sum() / batch

    pos = obj_t.unsqueeze(1)                             # [B, 1, G, G]
    box = torch.sigmoid(out["box"].float())              # [B, 4, G, G]: x, y, sqrt(w), sqrt(h)
    box_t = torch.cat([targets["xy"], targets["wh"]], dim=1).float()
    box_loss = (((box - box_t) ** 2) * pos).sum() / batch

    cls_t = targets["cls"].long()
    cls_loss = F.cross_entropy(out["cls"].float(), cls_t.clamp(min=0), reduction="none")
    cls_loss = (cls_loss * obj_t).sum() / batch

    total = w["obj"] * obj_loss + w["coord"] * box_loss + w["cls"] * cls_loss
    return total, {"obj": obj_loss.detach(), "box": box_loss.detach(), "cls": cls_loss.detach()}
