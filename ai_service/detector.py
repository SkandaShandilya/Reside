"""
YOLO Person Detector — Phase 3

Provides real-time person detection in video frames using Ultralytics YOLOv8.
Falls back gracefully to OpenCV DNN / Haar Cascade if PyTorch/YOLO model weights
are not available locally.

Output Format:
  List of Detection Object dicts:
  [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "bbox": [x1, y1, x2, y2],  # Normalized or pixel coordinates
      "center": (cx, cy)
    },
    ...
  ]
"""
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# COCO Class ID for person is 0
PERSON_CLASS_ID = 0


class PersonDetector:
    """
    YOLOv8 Person Detector with OpenCV fallback.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model_name = model_name
        self._model = None
        self._backend = "none"
        # Ultralytics' own internal per-stage timing (ms) from the most
        # recent predict() call — {"preprocess", "inference",
        # "postprocess"} — exposed for callers that want a real
        # frame-scaling-vs-GPU-inference breakdown (see
        # inference_engine.py's analyze() timing instrumentation) instead
        # of hand-rolled wall-clock timing around the whole detect() call,
        # which can't see inside Ultralytics' own preprocess/postprocess
        # split.
        self.last_yolo_speed_ms: Dict[str, float] = {}

        self._initialize_model()

    def _initialize_model(self):
        """
        Attempt to load Ultralytics YOLOv8 on CUDA. Falls back to OpenCV
        HOG only when torch/ultralytics themselves aren't importable
        (package not installed) — that is a genuinely different failure
        mode from "torch is installed but CUDA isn't available", which is
        treated as a hard configuration error below rather than a silent
        CPU fallback: CPU-only YOLO inference is an order of magnitude
        slower than this pipeline's real-time latency budget, so silently
        degrading to it would hide a real deployment problem (wrong
        driver, CPU-only torch build, etc.) behind what looks like normal
        operation.
        """
        try:
            from ultralytics import YOLO
            import torch
        except Exception as e:
            logger.warning(f"Could not import torch/ultralytics ({e}). Falling back to OpenCV HOG Person Detector.")
            self._init_hog_fallback()
            return

        logger.info(
            f"CUDA check — torch.cuda.is_available()={torch.cuda.is_available()}, "
            f"torch.version.cuda={torch.version.cuda}"
        )
        if not torch.cuda.is_available():
            # Deliberately NOT caught/degraded to HOG: CPU-only YOLO
            # inference is an order of magnitude slower than this
            # pipeline's real-time latency budget, so silently falling
            # back would hide a real deployment problem (wrong driver, a
            # CPU-only torch build, etc.) behind what looks like normal
            # operation. This must be fixed at the environment level, not
            # papered over here.
            raise RuntimeError(
                "torch.cuda.is_available() is False — a CUDA-enabled build of "
                "PyTorch must be installed for this pipeline to meet its "
                "real-time latency budget. Install a CUDA build of torch "
                "(see https://pytorch.org/get-started/locally/) matching this "
                "machine's NVIDIA driver."
            )
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)} (torch.version.cuda={torch.version.cuda})")

        try:
            self._model = YOLO(self.model_name)
            self._model.to("cuda:0")
            self._backend = "yolov8"
            logger.info(f"Loaded YOLOv8 model ({self.model_name}) on {torch.cuda.get_device_name(0)}.")
            self._log_device_diagnostic(torch)
        except Exception as e:
            # CUDA itself is fine here — this is a model-loading problem
            # (bad checkpoint path, corrupt weights file, etc.), which HOG
            # can reasonably stand in for rather than crashing the service.
            logger.warning(f"Could not load YOLOv8 model ({e}). Falling back to OpenCV HOG Person Detector.")
            self._init_hog_fallback()

    def _log_device_diagnostic(self, torch) -> None:
        """
        Hard diagnostic confirming YOLO is ACTUALLY running on GPU with
        FP16, not just that we asked it to. self._model.device is valid
        immediately after .to("cuda:0"), but FP16 status only exists on
        self._model.predictor (None until the first real predict() call —
        Ultralytics builds it lazily) — so this runs one tiny real
        inference on a blank frame to force that predictor into
        existence, then reads back the CONFIRMED device/fp16 state from
        it, rather than printing what was merely requested. This also
        absorbs the one-time CUDA kernel/cuDNN autotune cold-start cost
        (previously measured at ~400-600ms on the first real frame) here
        at startup instead of on the first live frame.
        """
        try:
            with torch.inference_mode():
                self._model.predict(
                    source=np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
                    imgsz=self.imgsz, device="cuda:0", half=True, verbose=False,
                )
            device = self._model.predictor.device if self._model.predictor is not None else self._model.device
            fp16 = getattr(self._model.predictor.model, "fp16", None) if self._model.predictor is not None else None
            print(f"[YOLO ENGINE] Running on device: {device}, FP16: {fp16}")
            logger.info(f"[YOLO ENGINE] Running on device: {device}, FP16: {fp16}")
        except Exception as e:
            logger.warning(f"YOLO device diagnostic warm-up failed (non-fatal): {e}")

    def _init_hog_fallback(self) -> None:
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._model = hog
            self._backend = "hog"
            logger.info("Loaded OpenCV HOG Person Detector fallback.")
        except Exception as hog_err:
            logger.error(f"HOG initialization failed: {hog_err}")
            self._backend = "none"

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run person detection on a single frame (BGR numpy array).

        Returns:
            Tuple of (detections_list, annotated_frame)
        """
        if frame is None or frame.size == 0:
            return [], frame

        annotated_frame = frame.copy()
        detections = []

        if self._backend == "yolov8" and self._model is not None:
            detections, annotated_frame = self._detect_yolo(frame, annotated_frame)
        elif self._backend == "hog" and self._model is not None:
            detections, annotated_frame = self._detect_hog(frame, annotated_frame)
        else:
            # Basic fallback: return empty detections
            pass

        return detections, annotated_frame

    def _detect_yolo(
        self, frame: np.ndarray, annotated_frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run Ultralytics YOLOv8 detection, always on cuda:0 — CUDA
        availability was already verified as a hard requirement in
        _initialize_model(), so there is no per-call CPU fallback here
        (that would reintroduce the same silent-degradation problem for a
        different code path). half=True enables FP16 inference on the
        RTX 5070's Tensor Cores; predict() is wrapped in
        torch.inference_mode() to strip autograd bookkeeping overhead on
        top of Ultralytics' own internal no_grad wrapping.

        Real measurement on this GPU (60 real frames, yolov8n vs yolov8s,
        half on/off): all four combinations landed within ~8-9.5ms median
        of each other — this model is already small enough that FP16
        casting overhead roughly cancels its Tensor Core benefit at batch
        size 1. half=True is kept on because it's free (no measured
        regression) and matches the RTX-specific directive, not because
        it measurably reduced latency in this case.
        """
        detections = []
        try:
            import torch
            with torch.inference_mode():
                results = self._model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    device="cuda:0",
                    half=True,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=[PERSON_CLASS_ID],
                    verbose=False,
                )

            if results and len(results) > 0:
                self.last_yolo_speed_ms = dict(results[0].speed)
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    det = {
                        "class_id": cls_id,
                        "class_name": "person",
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy),
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                    detections.append(det)

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")

        return detections, annotated_frame

    def _detect_hog(
        self, frame: np.ndarray, annotated_frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Run OpenCV HOG Default People Detector fallback."""
        detections = []
        try:
            # Resize frame for faster HOG processing
            h, w = frame.shape[:2]
            boxes, weights = self._model.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )

            for (x, y, bw, bh), weight in zip(boxes, weights):
                conf = float(weight)
                # HOG weights are not 0-1 like YOLO; valid detections are often 0.3+
                if conf < 0.3:
                    continue


                x1, y1, x2, y2 = x, y, x + bw, y + bh
                cx, cy = int(x + bw / 2), int(y + bh / 2)

                det = {
                    "class_id": PERSON_CLASS_ID,
                    "class_name": "person",
                    "confidence": round(min(conf, 0.99), 2),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "width": bw,
                    "height": bh,
                }
                detections.append(det)
        except Exception as e:
            logger.error(f"HOG detection error: {e}")

        return detections, annotated_frame

    @property
    def backend(self) -> str:
        return self._backend


# ---------------------------------------------------------------------------
# Fire & Smoke Detector — HSV Color Segmentation + Temporal Consistency
# ---------------------------------------------------------------------------

class PracticalFireSmokeDetector:
    """
    Detects fire and smoke in video frames using HSV color segmentation
    with temporal consistency voting to eliminate false positives.

    Fire:  High-saturation orange/red/yellow, high brightness, irregular contours.
    Smoke: Low-saturation gray, moderate brightness, large diffuse blobs with
           high solidity (no thin/linear edges).

    False-positive suppression:
      - Stricter HSV thresholds (higher min-saturation / min-value for fire).
      - Fill-ratio guard: fire regions must be compact, not thin lines.
      - Solidity guard: smoke blobs must be convex-like (solidity >= 0.45).
      - Aspect-ratio guard: extreme rectangles (walls, pipes) are rejected.
      - Temporal vote buffer: a class must be detected in >= N of the last M
        frames before it is reported, preventing single-frame flashes.
    """

    # ── Fire HSV thresholds ────────────────────────────────────────────────
    # Range 1: intense flame orange/yellow core (high saturation + peak brightness)
    #
    # ATTEMPTED FIX, REVERTED — kept here as a documented dead end so it
    # isn't re-tried blind: widening these bands (S floor 160->130, V floor
    # 210->180) plus a third near-white "hot core" band did measurably
    # improve recall on the real Fire/Fire_Images photo corpus (0.037 ->
    # 0.113 recall at the deployed 0.82 threshold — see fire_threshold_
    # analysis.csv), but validating the SAME change against a real CCTV
    # video with no fire in it at all (backend/storage/videos/
    # uploaded_cameras/cam_f00f3394.mp4, a fight scene) showed it made the
    # detector confirm "Fire" for 740/1184 frames (62% of the video, peak
    # confidence 0.90) — almost certainly skin tones and warm indoor
    # lighting newly qualifying under the loosened saturation/value floors.
    # That is a disqualifying false-positive regression for a security
    # system, so the widened bands were reverted; the local-brightness-
    # contrast gate below (a pure ADDITIONAL restriction on top of these
    # original bands, which cannot itself increase false positives) was
    # kept as a genuine, smaller improvement. Net honest result: real fire
    # recall on this class of classical HSV detector remains a disclosed,
    # UNRESOLVED limitation — see this module's audit report — that likely
    # needs per-frame flicker/temporal-variance analysis or a trained model
    # to fix properly, not another color-band adjustment.
    _FIRE_LOWER1 = np.array([0,   160, 210], dtype=np.uint8)
    _FIRE_UPPER1 = np.array([22,  255, 255], dtype=np.uint8)
    # Range 2: intense flame crimson-red wrap-around
    _FIRE_LOWER2 = np.array([170, 160, 210], dtype=np.uint8)
    _FIRE_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

    # ── Smoke HSV thresholds ──────────────────────────────────────────────
    # Low saturation (dense gray plume), high value
    # Saturation upper bound widened from 25 to 60, based on directly
    # measuring HSV values inside a real, unambiguous, visually-confirmed
    # smoke plume (a real production video where smoke went entirely
    # undetected): median saturation in the plume was 22-44 across three
    # sampled frames, with the 90th percentile up to 57-65 — i.e. this
    # smoke has a real bluish-gray tint (H~96-122) rather than being
    # perfectly achromatic, and the old S<=25 ceiling excluded most of it.
    # Not widened further than this measured range, to avoid reintroducing
    # the false positives (bright sky, pale reflections) the original
    # narrower band was tuned to exclude.
    _SMOKE_LOWER = np.array([0,   0,   90], dtype=np.uint8)
    _SMOKE_UPPER = np.array([180, 60, 200], dtype=np.uint8)

    # ── Area thresholds — RESOLUTION-RELATIVE, not a fixed pixel count ──────
    # Unified bounding-box-area gate: a candidate region smaller than 0.5%
    # of the frame's own area is rejected outright for BOTH fire and smoke,
    # before any other gate runs. This is the deliberate anti-spam filter
    # for trivial sources — a lighter flame, a stove burner flare, or a
    # 1-frame headlight reflection typically occupies well under 0.5% of a
    # normal camera frame, while a real fire/smoke incident large enough to
    # matter for a security alert does not.
    #
    # Expressed as a FRACTION of the frame's own area (not a fixed pixel
    # count) so the same 0.5% threshold means the same thing regardless of
    # camera resolution — a fixed pixel floor calibrated for a 960x540
    # frame would be far too strict on a smaller source (a real, diagnosed
    # bug on a 320x240 video: see git history) or far too loose on a 4K one.
    _FIRE_MIN_AREA_FRACTION  = 0.005
    _SMOKE_MIN_AREA_FRACTION = 0.005
    # Absolute floor regardless of resolution — guards against a
    # pathologically tiny frame (e.g. a thumbnail) confirming from a
    # handful of noise pixels just because the fraction shrank enough.
    _MIN_AREA_FLOOR_PX = 200

    # ── Temporal vote window ──────────────────────────────────────────────
    _VOTE_WINDOW = 8   # Number of recent frames to consider
    _FIRE_VOTES  = 5   # Minimum hits in window to confirm fire
    _SMOKE_VOTES = 6   # Minimum hits in window to confirm smoke

    # Minimum mean frame-to-frame grayscale change (0-255 scale) inside a
    # candidate smoke region — rejects a genuinely static bright/pale
    # surface (floor tile, wall, ceiling, glare) that happens to satisfy the
    # low-saturation/high-value color range. KEPT after direct measurement:
    # removing it (tried alongside removing the optical-flow gate below, to
    # rely solely on the 6-of-8 frame vote + the external state machine's
    # 12-window hysteresis for temporal filtering) produced a real, measured
    # false alarm — a real false-positive video (fighting, no smoke)
    # sustained a confirmed smoke event once this floor was gone. See the
    # gate itself in detect() for the exact before/after numbers.
    _SMOKE_MOTION_MIN_DIFF = 2.5

    # REMOVED: optical-flow turbulence gate (_SMOKE_FLOW_MIN_INCOHERENCE)
    # that used to run here. Measured directly against real footage, its
    # circular-variance distribution for genuine smoke (median 0.35) and for
    # a real false-positive fighting video (median 0.30) overlapped almost
    # completely — it wasn't separating anything, it was just rejecting most
    # real smoke.

    def __init__(self, fire_threshold: float = 0.82, smoke_threshold: float = 0.68):
        self.fire_threshold  = fire_threshold
        self.smoke_threshold = smoke_threshold
        # Temporal vote buffers: deque of bool (True = detected in that frame)
        from collections import deque as _deque
        self._fire_votes:  "_deque[bool]" = _deque(maxlen=self._VOTE_WINDOW)
        self._smoke_votes: "_deque[bool]" = _deque(maxlen=self._VOTE_WINDOW)
        self._prev_gray: Optional[np.ndarray] = None
        # Last raw boxes found for each class — reused when a class stays
        # "confirmed" (temporal vote still holds) but this exact frame
        # didn't produce a fresh raw candidate, so the overlay box doesn't
        # flicker on and off every other frame while genuinely ongoing
        # fire/smoke briefly dips below the color/shape gates.
        self._last_fire_boxes: List[Tuple[int, int, int, int, float]] = []
        self._last_smoke_boxes: List[Tuple[int, int, int, int, float]] = []

    def reset_temporal_state(self) -> None:
        """
        Clears the vote buffers. Call this whenever the frame stream this
        detector sees is about to have a hard discontinuity that is NOT a
        real scene cut — the one concrete case today is an uploaded video
        looping back to frame 0 for continuous "live" playback (see
        backend/app/services/video_stream_service.py's capture loop).

        Without this, the mean-frame-diff motion gate (see detect()) would
        compare the video's LAST frame against its FIRST frame as if they
        were temporally adjacent, and a color/shape candidate building up
        hits in the 6-of-8 vote window right at the end of the video would
        carry that vote history across the artificial last->first seam as
        if it were a real continuation. A real live camera feed never has
        this discontinuity; only a looping uploaded file does.
        """
        self._fire_votes.clear()
        self._smoke_votes.clear()
        self._prev_gray = None
        self._last_fire_boxes = []
        self._last_smoke_boxes = []

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_bounding_boxes(
        boxes: List[Tuple[int, int, int, int, float]], proximity: int = 40
    ) -> List[Tuple[int, int, int, int, float]]:
        """Merge overlapping or nearby bounding boxes into clean unified regions."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        merged: List[Tuple[int, int, int, int, float]] = []
        for x1, y1, x2, y2, conf in sorted_boxes:
            placed = False
            for i, (mx1, my1, mx2, my2, mconf) in enumerate(merged):
                if not (
                    x2 + proximity < mx1 or x1 - proximity > mx2
                    or y2 + proximity < my1 or y1 - proximity > my2
                ):
                    merged[i] = (
                        min(x1, mx1), min(y1, my1),
                        max(x2, mx2), max(y2, my2),
                        max(conf, mconf),
                    )
                    placed = True
                    break
            if not placed:
                merged.append((x1, y1, x2, y2, conf))
        return merged

    # ------------------------------------------------------------------
    @staticmethod
    def _solidity(contour) -> float:
        """Ratio of contour area to its convex hull area (1.0 = perfectly convex)."""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0.0
        hull  = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        return area / (hull_area + 1e-6)

    # ------------------------------------------------------------------
    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run fire & smoke detection on a BGR frame.
        Returns only temporally confirmed detections to avoid flicker/FP.
        """
        if frame is None or frame.size == 0:
            return [], frame.copy() if frame is not None else np.zeros((1, 1, 3), np.uint8)

        annotated = frame.copy()
        detections: List[Dict[str, Any]] = []

        frame_area = frame.shape[0] * frame.shape[1]
        fire_min_area = max(self._MIN_AREA_FLOOR_PX, self._FIRE_MIN_AREA_FRACTION * frame_area)
        smoke_min_area = max(self._MIN_AREA_FLOOR_PX, self._SMOKE_MIN_AREA_FRACTION * frame_area)

        # Blur slightly to reduce pixel noise before HSV conversion
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        prev_gray = self._prev_gray
        self._prev_gray = gray

        # ── 1. Fire Detection ─────────────────────────────────────────────
        fire_mask1 = cv2.inRange(hsv, self._FIRE_LOWER1, self._FIRE_UPPER1)
        fire_mask2 = cv2.inRange(hsv, self._FIRE_LOWER2, self._FIRE_UPPER2)
        fire_mask  = cv2.bitwise_or(fire_mask1, fire_mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,  kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_fire_boxes: List[Tuple[int, int, int, int, float]] = []

        # Local-SATURATION-contrast gate: a purely ADDITIONAL restriction on
        # top of the (reverted-to-original, see _FIRE_LOWER1's comment)
        # color bands above — it can only REJECT candidates the original
        # detector would have accepted, never accept new ones, so it cannot
        # introduce a new false-positive class the way the band-widening
        # attempt did.
        #
        # REVISED from an earlier local-BRIGHTNESS-contrast version of this
        # same gate, which was itself found to be a real bug, not a fix:
        # tested against a real fire-department training video (a genuine,
        # clearly visible flame venting from a window in daylight), the
        # brightness version rejected it outright — the flame's region mean
        # V was 141.8 against a frame mean V of 147.6 (+25 margin required
        # 172.6), because the bright DAYTIME SKY and pale smoke plume in
        # the same frame were themselves brighter than the fire. "Fire is
        # locally brighter than its surroundings" only holds indoors/at
        # night, not outdoors in daylight — measured directly, not assumed.
        #
        # Saturation contrast does not have that failure mode on the same
        # frame: the flame region measured S=163.4 against a frame mean of
        # only S=47.5 (sky S=39.4, similarly bright but far less
        # saturated) — a real, sustained, ~3x contrast a bright-but-pale
        # sky/smoke background cannot produce, while still only accepting
        # what the color band already flagged as candidate-orange/red.
        frame_mean_s = float(hsv[:, :, 1].mean())

        for cnt in fire_contours:
            area = cv2.contourArea(cnt)
            if area < fire_min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Aspect-ratio guard: reject extreme rectangles (thin bars/pipes)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 5.0:
                continue

            fill_ratio = area / (bw * bh + 1e-6)
            # Fill ratio guard: real fire is compact, not a thin wisp
            if fill_ratio < 0.25:
                continue

            region_mean_s = float(hsv[y:y + bh, x:x + bw, 1].mean())
            if region_mean_s < frame_mean_s + 40:
                # Not meaningfully more saturated than the rest of the
                # scene — likely an ordinary lit object of the right color
                # and brightness, not fire.
                continue

            # Confidence blends fill_ratio (shape) with saturation contrast
            # (the same signal the gate above already computed, now reused
            # rather than discarded) — a candidate that clears the gate by
            # a wide margin is genuinely more likely to be real fire than
            # one barely over it, and this is what actually let real,
            # clearly-visible flame reach a usable confidence: measured
            # directly against the fire-frame_200.jpg case above, real
            # flame has fill_ratio ~0.51 (irregular flame shape rarely
            # fills its own bounding box tightly) which the OLD fill-ratio-
            # only formula (0.58 + fill_ratio*0.38) capped at ~0.77 —
            # consistently just under FIRE_DETECTION_THRESHOLD (0.82),
            # meaning the confirmed real flame above NEVER once cleared the
            # gate across ~700 frames of clearly visible fire. Its
            # saturation contrast (~2.4x frame mean) was the strong signal
            # fill_ratio alone was missing.
            sat_contrast = (region_mean_s - frame_mean_s) / max(frame_mean_s, 1.0)
            sat_term = min(sat_contrast, 2.0) / 2.0
            conf = min(0.55 + fill_ratio * 0.25 + sat_term * 0.20, 0.98)
            if conf >= self.fire_threshold:
                raw_fire_boxes.append((x, y, x + bw, y + bh, conf))

        # Merge overlapping fire boxes (max 3)
        merged_fire_boxes = self._merge_bounding_boxes(raw_fire_boxes, proximity=45)[:3]
        frame_has_fire = len(merged_fire_boxes) > 0
        if merged_fire_boxes:
            self._last_fire_boxes = merged_fire_boxes

        # ── 2. Smoke Detection ────────────────────────────────────────────
        smoke_mask = cv2.inRange(hsv, self._SMOKE_LOWER, self._SMOKE_UPPER)
        # Subtract fire regions to avoid double-labelling
        smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(fire_mask))

        smoke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN,  smoke_kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, smoke_kernel)

        smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_smoke_boxes: List[Tuple[int, int, int, int, float]] = []

        for cnt in smoke_contours:
            area = cv2.contourArea(cnt)
            if area < smoke_min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Aspect-ratio guard: reject thin horizontal/vertical bars
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 6.0:
                continue

            fill_ratio = area / (bw * bh + 1e-6)
            # Fill ratio guard: smoke should be a diffuse blob
            if fill_ratio < 0.20:
                continue

            # Solidity guard: smoke blobs are roughly convex
            solidity = self._solidity(cnt)
            if solidity < 0.40:
                continue

            # Optical-flow incoherence gate REMOVED: measured directly
            # against real footage, its circular-variance distribution for
            # genuine smoke (median 0.35) and for a real false-positive
            # fighting video (median 0.30) overlapped almost completely — it
            # wasn't separating anything, it was just rejecting most real
            # smoke.
            #
            # Mean-frame-diff gate KEPT: tested removing this too (relying
            # only on the 6-of-8 frame vote + the external state machine's
            # 12-window hysteresis for temporal filtering), and it produced
            # a real, measured false alarm — a real false-positive video
            # (fighting, no smoke) sustained a confirmed smoke event at
            # t=42.2s once this floor was gone (mean confidence 0.660,
            # nonzero on 943/1184 frames). With this floor restored, the
            # same false-positive video confirms nothing while the real
            # smoke video still confirms correctly (mean 0.736, nonzero
            # 332/363) — this gate is doing real, necessary work on this
            # footage, unlike the flow gate above.
            if prev_gray is not None:
                roi_now = gray[y:y + bh, x:x + bw]
                roi_prev = prev_gray[y:y + bh, x:x + bw]
                mean_diff = float(cv2.absdiff(roi_now, roi_prev).mean())
                if mean_diff < self._SMOKE_MOTION_MIN_DIFF:
                    continue

            conf = min(0.50 + fill_ratio * 0.40 + solidity * 0.08, 0.92)
            if conf >= self.smoke_threshold:
                raw_smoke_boxes.append((x, y, x + bw, y + bh, conf))

        merged_smoke_boxes = self._merge_bounding_boxes(raw_smoke_boxes, proximity=60)[:3]
        frame_has_smoke = len(merged_smoke_boxes) > 0
        if merged_smoke_boxes:
            self._last_smoke_boxes = merged_smoke_boxes

        # ── 3. Temporal vote update ───────────────────────────────────────
        # Votes are based on THIS frame's raw candidate only (never the
        # cache below) — the cache is purely a display smoother, it must
        # never be able to manufacture votes and keep something "confirmed"
        # after it's genuinely gone.
        self._fire_votes.append(frame_has_fire)
        self._smoke_votes.append(frame_has_smoke)

        fire_confirmed  = sum(self._fire_votes)  >= self._FIRE_VOTES
        smoke_confirmed = sum(self._smoke_votes) >= self._SMOKE_VOTES
        if not fire_confirmed:
            self._last_fire_boxes = []
        if not smoke_confirmed:
            self._last_smoke_boxes = []

        # ── 4. Emit detections + draw only if temporally confirmed ────────
        # Uses this frame's box if it has one, otherwise the last known box
        # while still confirmed — real fire/smoke doesn't vanish and
        # reappear every other frame just because one frame's color/shape
        # gates missed it; only the flicker on the overlay should go away.
        if fire_confirmed:
            for x1, y1, x2, y2, conf in (merged_fire_boxes or self._last_fire_boxes):
                bw, bh = x2 - x1, y2 - y1
                cx, cy = x1 + bw // 2, y1 + bh // 2
                detections.append({
                    "class_name": "fire",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                })

        if smoke_confirmed:
            for x1, y1, x2, y2, conf in (merged_smoke_boxes or self._last_smoke_boxes):
                bw, bh = x2 - x1, y2 - y1
                cx, cy = x1 + bw // 2, y1 + bh // 2
                detections.append({
                    "class_name": "smoke",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                })

        return detections, annotated


# Backward-compatible alias — every existing caller (inference_engine.py,
# video_stream_service.py, video_timeline_processor.py, tests) imports this
# class as `FireSmokeDetector`. Keeping the alias means the "surgical
# upgrade" (new class name + unified 0.5% area gate) needed no changes to
# any of those call sites or their DB/alert-dispatch integration.
FireSmokeDetector = PracticalFireSmokeDetector
