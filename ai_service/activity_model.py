"""
Activity Recognition Model Definition

Implements Time-Distributed CNN and Conv3D architectures for video activity recognition.
These models classify short video clips into predefined activity categories.

Architecture Options:
1. Time-Distributed CNN: Applies a 2D CNN (ResNet/EfficientNet) to each frame,
   then passes the sequence of features through an LSTM/GRU for temporal modeling.
2. Conv3D: Uses 3D convolutions to jointly learn spatial and temporal features.

The actual activity classes are loaded from the trained model configuration,
NOT hard-coded. Default classes are provided only for initial training.
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
from typing import List, Optional, Any


class TimeDistributedCNN(nn.Module):
    """
    Time-Distributed CNN for video activity recognition.
    
    Architecture:
    - Backbone: ResNet-18 (or configurable) applied to each frame
    - Temporal: LSTM for sequence modeling
    - Classifier: Fully connected layers
    """
    
    def __init__(
        self,
        num_classes: int,
        sequence_length: int = 16,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Spatial feature extractor (ResNet-18 backbone). Pretrained ImageNet
        # weights give the model real visual features (edges, textures,
        # body/object shapes) from day one instead of starting from random
        # noise — critical for transfer learning on a dataset this small
        # (thousands, not millions, of samples). Training from scratch
        # (weights=None) on a few hundred samples per class is why several
        # classes previously never learned anything (near-zero recall).
        import torchvision.models as models
        try:
            resnet = models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            # Offline or download blocked — fall back rather than crash,
            # but this will train much worse; log so it's not silently weak.
            import logging
            logging.getLogger(__name__).warning(
                "Could not download pretrained ResNet-18 weights; falling back "
                "to random initialization. Accuracy will suffer."
            )
            resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: Any) -> Any:
        """
        Args:
            x: Tensor of shape (batch, sequence_length, channels, height, width)
        Returns:
            Tensor of shape (batch, num_classes) - class logits
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape for batch processing: (batch * seq, c, h, w)
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract spatial features
        features = self.feature_extractor(x)  # (batch * seq, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq, 512)
        
        # Temporal modeling
        lstm_out, _ = self.lstm(features)  # (batch, seq, hidden*2)
        
        # Use the last timestep output
        out = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Classify
        logits = self.classifier(out)  # (batch, num_classes)
        return logits


class Conv3DModel(nn.Module):
    """
    3D Convolutional Network for video activity recognition.
    
    Uses 3D convolutions to jointly capture spatial and temporal patterns.
    """
    
    def __init__(
        self,
        num_classes: int,
        sequence_length: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # Block 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Block 3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Block 4
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: Any) -> Any:
        """
        Args:
            x: Tensor of shape (batch, sequence_length, channels, height, width)
        Returns:
            Tensor of shape (batch, num_classes) - class logits
        """
        # Conv3D expects (batch, channels, depth, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


def create_model(
    architecture: str = "TimeDist-CNN",
    num_classes: int = 8,
    sequence_length: int = 16,
    **kwargs,
):
    """Factory function to create the appropriate model."""
    if not HAS_TORCH:
        return DummyModule()
    if architecture.lower() in ("timedist-cnn", "time-distributed cnn", "td-cnn"):
        return TimeDistributedCNN(num_classes=num_classes, sequence_length=sequence_length, **kwargs)
    elif architecture.lower() in ("conv3d", "3dcnn", "c3d"):
        return Conv3DModel(num_classes=num_classes, sequence_length=sequence_length, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
