"""Building blocks for scripts/train.py. Owner: Abdallah Afifi"""
from __future__ import annotations

import copy
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

class EncodedDataset(Dataset):
    """Wraps a dataset that follows the src/dataset.py interface and adds encoded training targets.

    Encoding runs here (inside DataLoader workers), not in the training loop, so the GPU never
    waits for it. Returns (image, encoded targets as tensors, raw target dict).
    """

    def __init__(self, base: Dataset, encoder):
        self.base = base
        self.encoder = encoder

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(target["labels"], dtype=torch.int64).reshape(-1)
        encoded = self.encoder(boxes.numpy(), labels.numpy(), tuple(image.shape[-2:]))
        encoded = {k: torch.from_numpy(np.asarray(v)) for k, v in encoded.items()}
        return image, encoded, {**target, "boxes": boxes, "labels": labels}


def detection_collate(batch):
    images = torch.stack([b[0] for b in batch])
    encoded = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    targets = [b[2] for b in batch]
    return images, encoded, targets


# ----------------------------------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------------------------------

def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on biases and BatchNorm parameters (decaying them only hurts)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_iters: int, warmup_iters: int, min_ratio: float = 0.01):
    """LR multiplier per iteration: linear warm-up from ~0, then cosine decay to ``min_ratio``.
    Warm-up matters from scratch: big early updates on random weights destabilise BatchNorm."""
    warmup_iters = max(0, min(warmup_iters, total_iters - 1))

    def factor(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        progress = min(1.0, progress)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


class ModelEMA:
    """Exponential moving average of the weights (Polyak averaging). The averaged model is smoother
    and usually scores better than the raw weights, so we validate and save it. The decay ramps
    up over the first ~``tau`` updates so early random weights are forgotten quickly."""

    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: float = 2000.0):
        self.ema = copy.deepcopy(unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# ----------------------------------------------------------------------------------------------
# Environment and I/O
# ----------------------------------------------------------------------------------------------

def pick_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own NumPy/random seed (otherwise forked workers repeat the
    same 'random' augmentations)."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    try:  # each worker is already a process; OpenCV's own thread pool would oversubscribe the CPUs
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a crash or notebook timeout mid-save never corrupts the
    only checkpoint we have."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    keys: list[str] = []
    for row in history:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in keys})


def plot_history(history: list[dict], path: Path, title: str = "") -> None:
    """Loss curves (total and each part, train vs val) plus validation mAP when available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [r["epoch"] for r in history]
    parts = [k[len("train_"):] for k in history[0] if k.startswith("train_")]
    has_map = any(r.get("val_mAP") not in (None, "") for r in history)
    n = len(parts) + int(has_map)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, part in zip(axes, parts):
        ax.plot(epochs, [r.get(f"train_{part}") for r in history], label="train")
        val = [(r["epoch"], r.get(f"val_{part}")) for r in history if r.get(f"val_{part}") not in (None, "")]
        if val:
            ax.plot(*zip(*val), label="val")
        ax.set_title(f"{part} loss")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    if has_map:
        ax = axes[len(parts)]
        pts = [(r["epoch"], r["val_mAP"]) for r in history if r.get("val_mAP") not in (None, "")]
        ax.plot(*zip(*pts), marker="o", color="tab:green")
        ax.set_title("val mAP@0.5")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
