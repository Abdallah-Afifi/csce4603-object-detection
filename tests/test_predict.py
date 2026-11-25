import numpy as np
import torch

from scripts.predict import predict_image
from src.dataset import preprocess


class StubDetector(torch.nn.Module):
    """Returns fixed boxes in letterboxed-input coordinates, to test the geometry of predict.py."""

    def __init__(self, boxes):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.boxes = torch.tensor(boxes, dtype=torch.float32)
        self.input_shape = None
        self.seen = None

    def forward(self, x):
        self.input_shape = tuple(x.shape)
        self.seen = x.detach().cpu()
        return x

    def decode(self, out, score_thresh):
        n = len(self.boxes)
        return [{"boxes": self.boxes.clone(), "scores": torch.full((n,), 0.9), "labels": torch.zeros(n, dtype=torch.long)}]


def test_predict_feeds_the_model_exactly_what_training_sees():
    """predict.py must use the dataset's preprocessing (same letterbox, padding colour, MEAN/STD)."""
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    model = StubDetector([[100, 150, 200, 250]])
    predict_image(model, frame, 384, 0.3)
    expected, _ = preprocess(frame, 384)
    assert torch.equal(model.seen[0], expected)
    assert model.seen[0, :, :48].abs().max() < 0.03       # padding is the mean colour -> ~0


def test_boxes_map_back_to_original_image_and_padding_boxes_are_dropped():
    # 640x480 frame -> scale 0.6, 48 px of padding above and below
    model = StubDetector([[100, 150, 200, 250],     # inside the image area
                          [0, 0, 50, 40]])          # entirely in the top padding
    boxes, scores, labels = predict_image(model, np.zeros((480, 640, 3), np.uint8), 384, 0.3)
    assert model.input_shape == (1, 3, 384, 384)
    assert len(boxes) == len(scores) == len(labels) == 1
    np.testing.assert_allclose(boxes[0], [100 / 0.6, 102 / 0.6, 200 / 0.6, 202 / 0.6], atol=0.01)
