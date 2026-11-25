"""scripts/evaluate.py end to end on the fake VOC folder (no real data needed)."""
import json

import torch

from scripts.evaluate import main
from src.config import apply_overrides, load_config
from src.model import build_model
from tests.fake_voc import make_fake_voc


def test_evaluate_script_writes_metrics_figures_and_summary(tmp_path):
    root = make_fake_voc(tmp_path / "VOCdevkit")
    cfg = apply_overrides(load_config("configs/smoke_synthetic.yaml"),
                          [f"data.voc_root={root}", "data.type=voc", "input_size=128"])
    cfg["data"]["train_sets"], cfg["data"]["test_set"] = [[2007, "trainval"], [2012, "trainval"]], [2007, "test"]
    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True)
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg, "epoch": 3}, run / "best.pt")

    out = tmp_path / "eval"
    main([str(run), str(run / "best.pt"), "--out", str(out), "--workers", "0", "--device", "cpu"])
    metrics = json.loads((out / "tiny" / "metrics.json").read_text())
    assert metrics["test_images"] == 10 and metrics["epoch"] == 3
    assert 0 <= metrics["mAP"] <= 1 and 0 <= metrics["precision"] <= 1 and 0 <= metrics["error_rate"] <= 1
    assert len(metrics["confusion_matrix"]) == 6 and metrics["threshold_from"] == "best F1 on val"
    assert (out / "tiny" / "confusion_matrix.png").exists() and (out / "tiny" / "pr_curve.png").exists()
    summary = (out / "summary.md").read_text()
    assert summary.count("| tiny |") == 2 and "AP person" in summary
