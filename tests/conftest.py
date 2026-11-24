import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMALL = dict(widths=(8, 16, 32, 64), depths=(1, 1, 1, 1), neck_channels=16, head_channels=16)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
