"""Print the architecture summary used in the report and at the checkpoint. Owner: Abdallah Afifi

    python scripts/model_summary.py --config configs/centernet_voc5.yaml

Shows parameters per part, output shapes, multiply-accumulates (MACs) of the conv layers, and
forward-pass speed on the available device.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.config import apply_overrides, load_config  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.train_utils import pick_device  # noqa: E402


def conv_macs(model: nn.Module, x: torch.Tensor) -> int:
    total = 0

    def hook(m, _inp, out):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1]
        total += out.numel() // out.shape[0] * (m.in_channels // m.groups) * k

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/centernet_voc5.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    model = build_model(cfg).eval()
    size = cfg["input_size"]
    x = torch.zeros(1, 3, size, size)

    print(f"{type(model).__name__} for {len(cfg['classes'])} classes {cfg['classes']}, input {size}x{size}")
    print(f"{'part':<12}{'params':>12}")
    heads = [n for n, _ in model.named_children() if n not in ("backbone", "neck")]
    for name in ["backbone", "neck", *heads]:
        print(f"{name:<12}{count_parameters(getattr(model, name)) / 1e6:>11.2f}M")
    print(f"{'total':<12}{count_parameters(model) / 1e6:>11.2f}M")
    print(f"conv MACs per image: {conv_macs(model, x) / 1e9:.2f} G")

    with torch.no_grad():
        feats = model.backbone(x)
        print("backbone features:", ", ".join(f"s{size // f.shape[-1]} {tuple(f.shape[1:])}" for f in feats))
        out = model(x)
    print("head outputs:     ", ", ".join(f"{k} {tuple(v.shape[1:])}" for k, v in out.items()))

    device = pick_device(args.device)
    model = model.to(device)
    xb = torch.zeros(1, 3, size, size, device=device)
    sync = {"cuda": torch.cuda.synchronize, "mps": getattr(torch.mps, "synchronize", lambda: None)}.get(
        device.type, lambda: None)
    with torch.no_grad():
        for _ in range(3):
            model(xb)
        sync()
        t0, n = time.perf_counter(), 20
        for _ in range(n):
            model.decode(model(xb))
        sync()
    print(f"forward + decode on {device}: {(time.perf_counter() - t0) / n * 1000:.1f} ms/image "
          f"({n / (time.perf_counter() - t0):.1f} FPS, batch 1)")


if __name__ == "__main__":
    main()
