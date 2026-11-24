"""Train a detector from scratch. Owner: Abdallah Afifi

Usage (from the repo root):
    python scripts/train.py --config configs/centernet_voc5.yaml --out runs/A_centernet
    python scripts/train.py --config configs/smoke_synthetic.yaml --out runs/smoke   # no data needed
    python scripts/train.py --config ... --out ... --set train.epochs=60 train.batch_size=16

Re-running the same command resumes from <out>/last.pt (Kaggle/Colab sessions time out), unless
--no-resume is given.

Writes to <out>/:
    config.yaml     the exact config used (after --set overrides)
    last.pt         full training state (model, EMA, optimiser, AMP scaler, history), every epoch
    best.pt         EMA weights + config of the best epoch (by val mAP if src/evaluate.py provides
                    voc_map, otherwise by val loss); load with src.model.load_model
    history.csv     one row per epoch: lr, train/val loss parts, val mAP, time
    loss_curve.png  loss curves (+ val mAP), redrawn every epoch
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from src.config import apply_overrides, get, load_config  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.train_utils import (EncodedDataset, ModelEMA, atomic_save, detection_collate,  # noqa: E402
                             param_groups, pick_device, plot_history, seed_everything, unwrap,
                             warmup_cosine, worker_init, write_history_csv)


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def build_datasets(cfg: dict):
    """Returns (train_dataset, val_dataset) following the src/dataset.py interface."""
    data = cfg.get("data", {})
    size = cfg["input_size"]
    kind = data.get("type", "voc")
    seed = data.get("seed", get(cfg, "train.seed", 0))

    if kind == "synthetic":
        from src.synthetic import SyntheticShapes
        n = len(cfg["classes"])
        train = SyntheticShapes(data.get("train_length", 512), size, n, seed=seed, classes=cfg["classes"])
        val = SyntheticShapes(data.get("val_length", 128), size, n, seed=seed + 1, classes=cfg["classes"])
        return train, val

    if kind != "voc":
        raise ValueError(f"Unknown data.type '{kind}' (expected 'voc' or 'synthetic')")

    from src import dataset as ds_mod
    if not hasattr(ds_mod, "VOC5Dataset"):
        raise SystemExit(
            "src/dataset.py does not define VOC5Dataset yet (data loader is Mariam's part).\n"
            "To test the training pipeline without it, run:\n"
            "  python scripts/train.py --config configs/smoke_synthetic.yaml --out runs/smoke")
    if list(getattr(ds_mod, "CLASSES", cfg["classes"])) != list(cfg["classes"]):
        raise SystemExit(f"Class order mismatch: dataset {ds_mod.CLASSES} vs config {cfg['classes']}")

    root = Path(data["voc_root"])
    if not root.is_absolute():
        root = ROOT / root
    sets = [tuple(s) for s in data["train_sets"]]
    common = dict(input_size=size, val_fraction=data.get("val_fraction", 0.1), seed=seed)
    train = ds_mod.VOC5Dataset(root, sets=sets, split="train", augment=cfg.get("augment", True), **common)
    val = ds_mod.VOC5Dataset(root, sets=sets, split="val", augment=False, **common)

    # Extra training sources (e.g. COCO restricted to our 5 classes), defined in src/dataset.py:
    #   data.extra_train: [{class: COCO5Dataset, kwargs: {root: data/coco, ...}}]
    extras = []
    for spec in data.get("extra_train") or []:
        cls = getattr(ds_mod, spec["class"])
        extras.append(cls(input_size=size, augment=cfg.get("augment", True), **spec.get("kwargs", {})))
    if extras:
        train = ConcatDataset([train, *extras])
    return train, val


def make_loader(dataset, encoder, batch_size, workers, shuffle, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EncodedDataset(dataset, encoder),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(dataset) > batch_size,  # avoid a tiny last batch upsetting BN
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
        generator=generator,
    )


# ----------------------------------------------------------------------------------------------
# Evaluation hook
# ----------------------------------------------------------------------------------------------

def get_map_fn():
    """Returns src.evaluate.voc_map if it is implemented and works, otherwise None.

    Contract (implemented by Youssef in src/evaluate.py):
        voc_map(predictions, ground_truths, num_classes, iou_thresh=0.5) -> {"mAP": float, "AP": [...]}
    """
    try:
        from src.evaluate import voc_map
    except ImportError:
        return None
    probe_pred = [{"boxes": torch.tensor([[0., 0., 10., 10.]]), "scores": torch.tensor([0.9]),
                   "labels": torch.tensor([0])}]
    probe_gt = [{"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([0])}]
    try:
        result = voc_map(probe_pred, probe_gt, num_classes=1)
        if abs(float(result["mAP"]) - 1.0) > 1e-6:
            print(f"[warn] voc_map gives {result['mAP']} for a perfect prediction; not using it")
            return None
    except NotImplementedError:
        return None
    except Exception as exc:  # a broken metric must never kill a multi-hour training run
        print(f"[warn] src.evaluate.voc_map failed a self-check ({exc!r}); selecting best.pt by val loss")
        return None
    return voc_map


@torch.no_grad()
def validate(model, loader, device, loss_weights, map_fn, class_names, decode_kwargs, amp):
    model.eval()
    sums, count = defaultdict(float), 0
    preds, gts = [], []
    for images, enc, targets in loader:
        images = images.to(device, non_blocking=True)
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(images)
        total, parts = model.compute_loss(out, enc, loss_weights)
        n = images.shape[0]
        sums["total"] += float(total) * n
        for k, v in parts.items():
            sums[k] += float(v) * n
        count += n
        if map_fn is not None:
            for det in model.decode(out, **decode_kwargs):
                preds.append({k: v.cpu() for k, v in det.items()})
            for t in targets:
                gts.append({k: t[k] for k in ("boxes", "labels", "difficult") if k in t})
    metrics = {f"val_{k}": v / max(count, 1) for k, v in sums.items()}
    if map_fn is not None:
        result = map_fn(preds, gts, num_classes=len(class_names))
        metrics["val_mAP"] = float(result["mAP"])
        for name, ap in zip(class_names, result.get("AP", [])):
            metrics[f"val_AP_{name}"] = float(ap)
    return metrics


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def train(cfg: dict, out_dir: str | Path, resume: bool = True, device: str | None = None,
          workers: int | None = None, max_iters: int | None = None, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = cfg.get("train", {})
    seed = t.get("seed", 0)
    seed_everything(seed)
    dev = pick_device(device or t.get("device"))
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = dev.type == "cuda" and t.get("amp", True)
    workers = t.get("workers", 4) if workers is None else workers
    batch_size = t.get("batch_size", 32)
    epochs = t.get("epochs", 120)
    loss_weights = t.get("loss_weights")

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Model, EMA, optimiser
    core = build_model(cfg).to(dev)
    encoder = core.target_encoder()
    model: nn.Module = core
    if dev.type == "cuda" and torch.cuda.device_count() > 1 and t.get("multi_gpu", True):
        model = nn.DataParallel(core)
        log(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
    ema = ModelEMA(core, decay=t.get("ema_decay", 0.9998), tau=t.get("ema_tau", 2000))
    base_lr = t.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(param_groups(core, t.get("weight_decay", 1e-4)), lr=base_lr,
                                  betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # Data
    train_ds, val_ds = build_datasets(cfg)
    if len(train_ds) == 0:
        raise SystemExit("Training dataset is empty")
    train_loader = make_loader(train_ds, encoder, batch_size, workers, True, dev, seed)
    val_loader = make_loader(val_ds, encoder, batch_size, workers, False, dev, seed)
    iters_per_epoch = len(train_loader) if max_iters is None else min(max_iters, len(train_loader))
    total_iters = epochs * iters_per_epoch
    lr_factor = warmup_cosine(total_iters, t.get("warmup_epochs", 3) * iters_per_epoch,
                              t.get("min_lr_ratio", 0.01))

    map_fn = get_map_fn()
    eval_cfg = cfg.get("eval", {})
    decode_kwargs = {"score_thresh": eval_cfg.get("score_thresh", 0.01),
                     "max_det": eval_cfg.get("max_det", 100)}
    select_by = "val_mAP" if map_fn else "val_total"

    # Resume
    start_epoch, global_step, history, best = 0, 0, [], None
    last_path = out_dir / "last.pt"
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        core.load_state_dict(ckpt["model"])
        ema.ema.load_state_dict(ckpt["ema"])
        ema.updates = ckpt.get("ema_updates", 0)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]
        history, best = ckpt.get("history", []), ckpt.get("best")
        log(f"Resumed from {last_path} at epoch {start_epoch}/{epochs}")
        if start_epoch >= epochs:
            log("Training already finished; raise train.epochs to continue.")
            return history

    log(f"Device {dev}{' + AMP' if amp else ''} | {type(core).__name__} "
        f"{count_parameters(core) / 1e6:.2f}M params | train {len(train_ds)} imgs, val {len(val_ds)} imgs | "
        f"{iters_per_epoch} it/epoch x {epochs} epochs | best.pt by {select_by}")

    clip = t.get("grad_clip", 10.0)
    log_every = t.get("log_interval", 50)
    val_every, map_every = t.get("val_interval", 1), t.get("eval_interval", 5)
    skipped_total = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        sums, seen, skipped, collisions = defaultdict(float), 0, 0, 0
        for it, (images, enc, _) in enumerate(train_loader):
            if it >= iters_per_epoch:
                break
            lr = base_lr * lr_factor(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = images.to(dev, non_blocking=True)
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            if "dropped" in enc:
                collisions += int(enc["dropped"].sum())

            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                out = model(images)
            loss, parts = core.compute_loss(out, enc, loss_weights)

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                skipped_total += 1
                log(f"[warn] non-finite loss at epoch {epoch + 1} iter {it}; step skipped")
                if skipped > max(10, iters_per_epoch // 20):
                    raise RuntimeError("Too many non-finite losses; lower train.lr or check the data")
                global_step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(core)
            global_step += 1

            n = images.shape[0]
            seen += n
            sums["total"] += float(loss.detach()) * n
            for k, v in parts.items():
                sums[k] += float(v) * n
            if log_every and (it + 1) % log_every == 0:
                rate = seen / (time.time() - t0)
                eta = (iters_per_epoch - it - 1) * n / max(rate, 1e-9)
                part_str = " ".join(f"{k} {float(v):.3f}" for k, v in parts.items())
                log(f"  ep {epoch + 1} [{it + 1}/{iters_per_epoch}] loss {float(loss.detach()):.3f} ({part_str}) "
                    f"lr {lr:.2e} |g| {float(grad_norm):.1f} {rate:.0f} img/s eta {eta / 60:.1f}m")

        row = {"epoch": epoch + 1, "lr": lr}
        row.update({f"train_{k}": v / max(seen, 1) for k, v in sums.items()})
        last_epoch = epoch + 1 == epochs
        if (epoch + 1) % val_every == 0 or last_epoch:
            want_map = map_fn if ((epoch + 1) % map_every == 0 or last_epoch) else None
            row.update(validate(ema.ema, val_loader, dev, loss_weights, want_map, cfg["classes"],
                                decode_kwargs, amp))
        if collisions:
            row["grid_collisions"] = collisions
        row["skipped_steps"] = skipped
        row["epoch_minutes"] = (time.time() - t0) / 60
        history.append(row)

        improved = False
        if select_by in row:
            score = row[select_by]
            improved = best is None or (score > best if select_by == "val_mAP" else score < best)
            if improved:
                best = score
                atomic_save({"model": ema.ema.state_dict(), "config": cfg, "classes": cfg["classes"],
                             "epoch": epoch + 1, "metrics": row}, out_dir / "best.pt")

        atomic_save({"model": core.state_dict(), "ema": ema.ema.state_dict(), "ema_updates": ema.updates,
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if amp else None,
                     "epoch": epoch + 1, "global_step": global_step, "history": history, "best": best,
                     "config": cfg, "classes": cfg["classes"]}, last_path)
        write_history_csv(history, out_dir / "history.csv")
        plot_history(history, out_dir / "loss_curve.png", title=out_dir.name)

        msg = f"epoch {epoch + 1}/{epochs} train {row.get('train_total', float('nan')):.4f}"
        if "val_total" in row:
            msg += f" val {row['val_total']:.4f}"
        if "val_mAP" in row:
            msg += f" mAP@0.5 {row['val_mAP']:.4f}"
        log(msg + f" ({row['epoch_minutes']:.1f} min){'  * best' if improved else ''}")

    if skipped_total:
        log(f"[warn] {skipped_total} steps were skipped because of non-finite losses")
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="run directory, e.g. runs/A_centernet")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    p.add_argument("--max-iters", type=int, default=None, help="cap iterations per epoch (smoke tests)")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    train(cfg, args.out, resume=not args.no_resume, device=args.device, workers=args.workers,
          max_iters=args.max_iters)


if __name__ == "__main__":
    main()
