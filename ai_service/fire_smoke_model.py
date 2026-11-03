"""
Fire/Smoke Neural Classifier — Model Definition

A dedicated, per-frame image classifier for Fire/Smoke/Normal, completely
separate from both the suspicious-behaviour CNN (ai_service/activity_model.py
— video-temporal, different classes, different checkpoints, different
training data) and the classical HSV+optical-flow FireSmokeDetector
(ai_service/detector.py, kept as-is and unaffected by this file).

Deliberately a single-frame (not video-temporal) classifier: fire and smoke
have strong per-frame visual signatures (color, texture, plume shape) that
don't require multi-frame motion context the way a behaviour like "fighting"
does, and a single ResNet-18 forward pass per frame is far cheaper than the
16-frame TimeDistributedCNN + LSTM used for behaviour — appropriate for a
CPU-friendly, real-time-per-camera detector. Pretrained ImageNet backbone for
the same transfer-learning reason activity_model.py uses one: a few thousand
images is not enough to learn good visual features from scratch.
"""
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class DummyModule:
        def __init__(self, *args, **kwargs): pass
        def eval(self): return self
    nn = type("nn", (), {"Module": DummyModule})()
    torch = None
from typing import Any


class FireSmokeCNN(nn.Module):
    """
    ResNet-18 backbone (ImageNet-pretrained) + a small classifier head.
    Input: a single (3, 224, 224) BGR->RGB-normalized frame.
    Output: logits over `num_classes` (default 3: Normal/Fire/Smoke — see
    ai_service/config.py's FIRE_SMOKE_MODEL_CLASSES).
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        import torchvision.models as models
        try:
            resnet = models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Could not download pretrained ResNet-18 weights; falling back "
                "to random initialization. Accuracy will suffer."
            )
            resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Any) -> Any:
        """
        Args:
            x: Tensor of shape (batch, 3, 224, 224)
        Returns:
            Tensor of shape (batch, num_classes) — class logits
        """
        features = self.feature_extractor(x)          # (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 512)
        return self.classifier(features)


def create_fire_smoke_model(num_classes: int = 3, **kwargs) -> Any:
    """Factory function, mirroring ai_service/activity_model.py's create_model."""
    if not HAS_TORCH:
        return DummyModule()
    return FireSmokeCNN(num_classes=num_classes, **kwargs)
