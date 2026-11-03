"""
Fire/Smoke Image Dataset Loader.

Single-frame image dataset for the dedicated Fire/Smoke neural classifier
(ai_service/fire_smoke_model.py) — separate from ai_service/training/
dataset.py (ActivityVideoDataset), which is video-temporal and serves the
unrelated behaviour CNN.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass
    torch = None


class FireSmokeImageDataset(Dataset):
    """
    Loads single images (or, if given a video file, its middle frame) and
    resizes to `image_size` for the Fire/Smoke classifier.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: Tuple[int, int] = (224, 224),
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            samples: List of {"path": str, "label": str} dicts.
            image_size: Target (width, height).
            class_to_idx: Mapping of label -> integer index.
        """
        self.samples = samples
        self.image_size = image_size
        if class_to_idx is None:
            labels = sorted({s["label"] for s in samples})
            self.class_to_idx = {label: i for i, label in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mkv", ".webm"):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            return cv2.resize(frame, self.image_size)

        frame = cv2.imread(path)
        if frame is None:
            return np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        return cv2.resize(frame, self.image_size)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        label_idx = self.class_to_idx.get(sample["label"], 0)
        frame = self._load_frame(sample["path"])

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        chw = np.transpose(norm, (2, 0, 1))
        tensor = torch.from_numpy(chw)
        return tensor, label_idx
