import numpy as np
import torch

from scripts.predict import letterbox, predict_image


class StubDetector(torch.nn.Module):
    """Returns fixed boxes in letterboxed-input coordinates, to test the geometry of predict.py."""

    def __init__(self, boxes):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.boxes = torch.tensor(boxes, dtype=torch.float32)
        self.input_shape = None

    def forward(self, x):
        self.input_shape = tuple(x.shape)
        return x

    def decode(self, out, score_thresh):
        n = len(self.boxes)
        return [{"boxes": self.boxes.clone(), "scores": torch.full((n,), 0.9), "labels": torch.zeros(n, dtype=torch.long)}]


def test_letterbox_keeps_aspect_ratio_and_centres():
    padded, scale, left, top = letterbox(np.full((480, 640, 3), 255, np.uint8), 384)
    assert padded.shape == (384, 384, 3) and scale == 0.6 and (left, top) == (0, 48)
    assert padded[:48].max() == 0 and padded[48:336].min() == 255 and padded[336:].max() == 0


def test_boxes_map_back_to_original_image_and_padding_boxes_are_dropped():
    # 640x480 frame -> scale 0.6, 48 px of padding above and below
    model = StubDetector([[100, 150, 200, 250],     # inside the image area
                          [0, 0, 50, 40]])          # entirely in the top padding
    boxes, scores, labels = predict_image(model, np.zeros((480, 640, 3), np.uint8), 384, 0.3)
    assert model.input_shape == (1, 3, 384, 384)
    assert len(boxes) == len(scores) == len(labels) == 1
    np.testing.assert_allclose(boxes[0], [100 / 0.6, 102 / 0.6, 200 / 0.6, 202 / 0.6], atol=0.01)
