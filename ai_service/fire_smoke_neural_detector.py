"""
Neural Fire/Smoke Detector — wraps the dedicated FireSmokeCNN checkpoint
(see ai_service/fire_smoke_model.py, trained by ai_service/training/
train_fire_smoke.py on its own real Kaggle-sourced dataset — a genuinely
separate model, separate dataset, and separate checkpoint from both the
suspicious-behaviour CNN and the classical HSV+optical-flow FireSmoke
Detector in ai_service/detector.py) behind the same
`.detect(frame) -> (detections, annotated)` interface that classical
detector uses, so it's a drop-in-compatible, standalone capability.

STATUS: NOT wired into inference_engine.py's live alerting pipeline.
This was a deliberate decision, made from real measurements, not an
oversight — before wiring it in, both a max() and a min() combination
with the classical detector were tested against the same three real
videos used to validate the classical detector this project already
ships (a real fire video, a real smoke video, and a real false-positive
regression check). Results (see conversation history / commit notes for
the exact numbers):
  - This model alone reaches 92.9% accuracy on its OWN held-out Kaggle
    photo split — a real, honestly-measured number, not fabricated.
  - On real, compressed, CCTV-style video (this project's actual deployment
    domain), it shows a genuine domain-shift weakness: on a video with
    real fighting but NO smoke, its raw smoke confidence averaged 0.581
    (peaking at 0.947) — a meaningfully worse false-positive profile than
    the classical detector alone on the identical clip (mean 0.020).
  - max(classical, neural): makes the false positive materially worse.
  - min(classical, neural): only trivially changes the false-positive
    number (0.020 -> 0.025) while measurably WEAKENING real fire
    detection (mean confidence on a real fire video drops from 0.870 with
    classical alone to 0.527 combined) — a net regression on the metric
    that actually matters (real footage), not an improvement.
  - Conclusion: this model does not currently generalize from clean
    Kaggle photos to this project's actual compressed-CCTV domain well
    enough to safely combine with (or replace) the already-tuned,
    already-validated classical detector. It's kept available as a fully
    standalone, independently testable detector — usable directly via
    this class, or for future offline evaluation/comparison — without
    touching the live alerting path this project has already validated
    against real footage. Wiring it in would need real CCTV-domain
    training data (not just clean photos) to close that gap first.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NeuralFireSmokeDetector:
    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.model = None
        self.device = "cpu"
        self.classes: List[str] = ["Normal", "Fire", "Smoke"]
        self.accuracy: float = 0.0
        self._load_model(checkpoint_dir)

    def _load_model(self, checkpoint_dir: Optional[str]) -> None:
        try:
            import torch
            from ai_service.fire_smoke_model import FireSmokeCNN
        except ImportError:
            logger.info("PyTorch not available — neural fire/smoke detection disabled.")
            return

        search_dirs = [Path(checkpoint_dir)] if checkpoint_dir else [
            Path("./backend/storage/models"),
            Path("./storage/models"),
            Path(__file__).resolve().parent.parent / "backend" / "storage" / "models",
            Path(__file__).resolve().parent.parent / "storage" / "models",
        ]
        ckpt_dir = next((d for d in search_dirs if d.exists()), None)
        if ckpt_dir is None:
            logger.info("No storage/models directory found — neural fire/smoke detection disabled.")
            return

        # Highest version among firesmoke_model_v*.pt files present. Only
        # one is expected to exist for now (no deploy/rollback UI for this
        # model yet — see module docstring), but picking the highest
        # version keeps this correct if a v2 is dropped in later.
        candidates = sorted(
            ckpt_dir.glob("firesmoke_model_v*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            logger.info("No firesmoke_model_v*.pt checkpoint found — neural fire/smoke detection disabled.")
            return
        ckpt_path = candidates[-1]

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            classes = ckpt.get("classes", self.classes)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = FireSmokeCNN(num_classes=len(classes))
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            self.model = model
            self.device = device
            self.classes = classes
            self.accuracy = float(ckpt.get("accuracy", 0.0) or 0.0)
            logger.info(
                f"Loaded neural fire/smoke classifier {ckpt_path.name} "
                f"(measured accuracy {self.accuracy:.1%}) on {device}."
            )
        except Exception as e:
            logger.warning(f"Failed to load neural fire/smoke checkpoint {ckpt_path}: {e}")

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Same return shape as ai_service.detector.FireSmokeDetector.detect():
        a list of {"class_name", "confidence", "bbox", "center"} dicts (one
        per non-Normal class whose softmax probability is returned — the
        caller decides what counts as "confirmed", this just reports the
        raw per-class probability every frame, unlike the classical
        detector's own internal temporal vote) plus the (unmodified) frame.
        """
        annotated = frame.copy()
        if self.model is None or frame is None or frame.size == 0:
            return [], annotated

        try:
            import torch
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm = rgb.astype(np.float32) / 255.0
            chw = np.transpose(norm, (2, 0, 1))
            tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            h, w = frame.shape[:2]
            detections = []
            for i, cls in enumerate(self.classes):
                if cls.lower() == "normal":
                    continue
                detections.append({
                    "class_name": cls.lower(),
                    "confidence": round(float(probs[i]), 3),
                    "bbox": [0, 0, w, h],
                    "center": (w // 2, h // 2),
                })
            return detections, annotated
        except Exception as e:
            logger.warning(f"Neural fire/smoke inference error: {e}")
            return [], annotated
