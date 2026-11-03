"""
YOLOv8 Fire/Smoke Detector — a genuine object-detection replacement for
the classical HSV+optical-flow `PracticalFireSmokeDetector` in
ai_service/detector.py. Trained by
ai_service/training/train_fire_smoke_yolo.py on a combined, isolated
dataset (see ai_service/training/prepare_fire_smoke_yolo_dataset.py) —
own checkpoint directory (backend/storage/models/fire_smoke_yolov8/),
never touches suspicious_activity_*.pt or firesmoke_model_v1.0.0.pt.

Same `.detect(frame) -> (detections, annotated)` interface as both the
classical detector and the (unused) NeuralFireSmokeDetector, so it is a
drop-in replacement at the one call site that matters —
InferenceEngine.__init__'s `self.fire_smoke_detector = ...` in
ai_service/inference_engine.py. `annotated` is returned unmodified
(same convention as the classical detector); overlay drawing for
fire/smoke happens later in inference_engine.py's own render path using
the returned `detections` list, not here.

No internal temporal voting is done here (unlike the classical
detector's own 6-of-8-frame vote) — this detector emits the model's raw
per-frame, per-box confidence directly. Temporal stability comes
entirely from the existing, already-tuned, UNCHANGED
ai_service/fire_smoke_state.py hysteresis state machine (EMA smoothing +
12-window confirm + grace period) that every caller already runs this
output through — that module needed zero changes for this swap.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model's own per-box confidence floor before a detection is even
# reported upstream. Deliberately low — real filtering/debounce is the
# state machine's job (FIRE_ALERT_THRESHOLD=0.70 / SMOKE_ALERT_THRESHOLD
# =0.60 after EMA + 12-window confirm), not this detector's. Only guards
# against passing near-zero noise through at all.
_MIN_BOX_CONF = 0.20

_CLASS_NAMES = {0: "fire", 1: "smoke"}


class YOLOFireSmokeDetector:
    def __init__(self, weights_path: Optional[str] = None):
        self.model = None
        self._load_model(weights_path)

    def _load_model(self, weights_path: Optional[str]) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not available — YOLO fire/smoke detection disabled.")
            return

        if weights_path is None:
            search_dirs = [
                Path("./backend/storage/models/fire_smoke_yolov8"),
                Path("./storage/models/fire_smoke_yolov8"),
                Path(__file__).resolve().parent.parent / "backend" / "storage" / "models" / "fire_smoke_yolov8",
                Path(__file__).resolve().parent.parent / "storage" / "models" / "fire_smoke_yolov8",
            ]
            candidates = []
            for d in search_dirs:
                if d.exists():
                    candidates.extend(d.glob("*/weights/best.pt"))
            if not candidates:
                logger.warning("No fire_smoke_yolov8/*/weights/best.pt checkpoint found — YOLO fire/smoke detection disabled.")
                return
            weights_path = str(max(candidates, key=lambda p: p.stat().st_mtime))

        try:
            self.model = YOLO(weights_path)
            logger.info(f"Loaded YOLO fire/smoke detector from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO fire/smoke checkpoint {weights_path}: {e}")
            self.model = None

    def reset_temporal_state(self) -> None:
        """No-op — kept for interface parity with PracticalFireSmokeDetector,
        which callers (e.g. video_stream_service.py's loop-restart handling)
        call unconditionally. This detector has no internal temporal state
        to reset (see module docstring)."""
        return

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        annotated = frame.copy() if frame is not None else np.zeros((1, 1, 3), np.uint8)
        if self.model is None or frame is None or frame.size == 0:
            return [], annotated

        try:
            results = self.model.predict(frame, verbose=False, conf=_MIN_BOX_CONF)
            detections: List[Dict[str, Any]] = []
            if not results:
                return [], annotated

            boxes = results[0].boxes
            if boxes is None:
                return [], annotated

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = _CLASS_NAMES.get(cls_id)
                if class_name is None:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append({
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                })
            return detections, annotated
        except Exception as e:
            logger.warning(f"YOLO fire/smoke inference error: {e}")
            return [], annotated
