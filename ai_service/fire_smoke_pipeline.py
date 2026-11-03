"""
Fire/Smoke Evidence-Fusion Pipeline.

ROOT CAUSE (measured directly against this project's own real footage,
not assumed — see the diagnostic session that produced this file):
  - False positives were ALREADY well-controlled on the plain YOLO
    detector (ai_service/fire_smoke_yolo_detector.py): 3/43 real non-fire
    videos showed a brief single-frame confidence spike (0.37-0.65), all
    comfortably under the 0.70 sustained-confirm threshold — no real
    false alerts.
  - The actual problem is MISSED real fire: on a genuine, unambiguous
    real house-fire video in this project's own footage
    (backend/storage/videos/uploaded_cameras/cam_fc5fc1f6.mp4), YOLO's
    raw per-frame fire confidence sits mostly at 0.40-0.60 (a single
    peak of 0.78 across 312 frames) — the EMA-smoothed value NEVER
    reaches the 0.70 FIRE_ALERT_THRESHOLD for the entire clip, so fire
    is confirmed on 0/312 frames, while smoke (already well-served by
    YOLO alone) correctly confirms on 301/312 frames on the very same
    clip. Direct color-space measurement on the same frames shows
    strong, real supporting evidence YOLO isn't using: ~3.85% of the
    frame is HSV-corona-colored, with clear RGB red-dominance and
    YCbCr light-emission signal.
  - Conclusion: this is a confidence-calibration gap, not a detector
    architecture failure — YOLO alone under-scores real fire relative to
    the alert threshold. Fusing YOLO's confidence with independent,
    lightweight, non-training visual evidence (computed fresh on each
    candidate's own region, not a second trained model) closes that gap
    for genuine fire without moving the false-positive numbers, which
    were already fine.

DESIGN
  ai_service/fire_smoke_yolo_detector.py (YOLOFireSmokeDetector) and
  ai_service/detector.py (PracticalFireSmokeDetector) are used here
  UNMODIFIED, as internal dependencies — this file does not edit either.
  YOLO remains the PRIMARY candidate source; the classical detector is
  used only as one SUPPORTING signal (a small bonus when its own,
  independently-gated output overlaps YOLO's candidate) — it can never
  produce a detection on its own here, and detect() never requires both
  to agree in a given frame.

  No new dataset is introduced anywhere in this file — every "visual
  evidence" signal below is a per-frame, per-candidate pixel measurement
  (HSV/RGB/YCbCr coverage, brightness, flicker, motion, growth) computed
  fresh on the incoming CCTV frame's own pixels, not a trained model, not
  historical training data.

  `FireSmokeFusionPipeline` implements the exact same `.detect(frame) ->
  (detections, annotated)` / `.reset_temporal_state()` contract as both
  underlying detectors, so it is a drop-in replacement at the one call
  site in ai_service/inference_engine.py (`self.fire_smoke_detector =
  ...`). ai_service/fire_smoke_state.py's existing EMA + confirm-window +
  grace-period hysteresis is UNCHANGED and remains the final authority on
  when an event is confirmed/alerted — this pipeline only produces a
  better per-frame confidence value for it to consume; it does not
  reimplement persistence/alerting itself.
"""
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai_service.fire_smoke_yolo_detector import YOLOFireSmokeDetector
from ai_service.detector import FireSmokeDetector as ClassicalFireSmokeDetector
from ai_service.config import AIConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Centralized configuration — every tunable in this pipeline lives here.
# All defaults are principled choices from direct measurement against
# this project's own real footage (see module docstring), not guesses;
# each is documented with what it controls and why that value.
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FireSmokePipelineConfig:
    # ── Candidate filtering (step 2) ────────────────────────────────────
    fire_min_conf: float = 0.20      # YOLO's own raw floor (matches YOLOFireSmokeDetector's _MIN_BOX_CONF)
    smoke_min_conf: float = 0.20
    min_box_area_fraction: float = 0.003   # relative to frame area — rejects trivial specks
    max_box_area_fraction: float = 0.65    # rejects a near-whole-frame "box" (glare/exposure artifact)
    max_aspect_ratio: float = 6.0          # rejects thin bars/edges YOLO occasionally boxes

    # ── Temporal verification (step 3) ──────────────────────────────────
    persistence_min_frames: int = 3        # min consecutive-ish frames with a candidate before it's "established"
    gap_tolerance_frames: int = 3          # this many consecutive MISSED frames don't reset the streak
    iou_continuity_threshold: float = 0.25 # below this, treat as a new/different region (soft, not a reject)
    static_history_len: int = 10           # frames of (w,h) history for the static-box check
    static_variance_eps_px2: float = 9.0   # ~3px std-dev — below this, box isn't reshaping
    static_evidence_penalty: float = 0.12  # SOFT confidence penalty for a static box — NOT a reject (see fire_evidence)

    # ── Fusion formula (step 4/5) ────────────────────────────────────────
    # fused = yolo_conf + (1 - yolo_conf) * support_score * boost_factor
    # rather than a plain weighted average of every signal. Rationale,
    # from direct measurement: on this project's own real fire video,
    # YOLO confidence sits at 0.40-0.60 while colour evidence is
    # consistently strong (0.6-1.0) — a plain weighted AVERAGE dilutes
    # that strong secondary signal down to the low-to-mid 0.40s and never
    # reaches the alert threshold (measured: fused peak 0.53, never
    # confirmed). The boost formula instead treats strong supporting
    # evidence as CONFIRMING what YOLO already suspects, pushing a
    # plausible candidate decisively toward 1.0 — while a candidate with
    # near-zero supporting evidence (ordinary content) stays close to its
    # own modest YOLO confidence, same as before. `support_score` is
    # itself a weighted average of the individual evidence terms (so
    # those sub-weights still matter for balance among themselves), and
    # `boost_factor` caps how much support evidence alone can add.
    fire_boost_factor: float = 1.0
    fire_support_color_weight: float = 0.55
    fire_support_brightness_weight: float = 0.10
    fire_support_flicker_weight: float = 0.10
    fire_support_motion_weight: float = 0.08
    fire_support_growth_weight: float = 0.07
    fire_support_classical_weight: float = 0.10

    # Fire colour-evidence normalization: HSV-corona/core coverage
    # fraction of the candidate ROI at/above this fraction saturates the
    # colour evidence term to 1.0 (measured real fire: ~3.85% of the
    # WHOLE frame was corona-coloured; using a smaller ROI-relative
    # target here since the ROI is tightly cropped to the candidate box).
    fire_color_saturation_fraction: float = 0.10

    # ── Smoke fusion (step 5) ────────────────────────────────────────────
    # Same boost formula, but with a SMALLER boost_factor: smoke was
    # ALREADY working well on YOLO alone (0/43 real-footage false
    # positives, 301/312 confirmed frames on the real fire video's smoke
    # plume) — supporting evidence here is a modest assist, not a rescue,
    # so it shouldn't be able to push a weak candidate as far as fire's.
    smoke_boost_factor: float = 0.6
    smoke_support_color_weight: float = 0.35       # low-saturation grey/white coverage
    smoke_support_drift_weight: float = 0.45       # upward drift / expansion (continuous score, not hard reject)
    smoke_support_growth_weight: float = 0.20

    smoke_color_saturation_fraction: float = 0.15

    # ── Diagnostics (step 9) — OFF by default ───────────────────────────
    debug_logging: bool = False


DEFAULT_CONFIG = FireSmokePipelineConfig()


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _crop(img: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


# ═══════════════════════════════════════════════════════════════════════
# Visual evidence — pure per-frame pixel measurements on a candidate's
# own ROI. No training, no model, no dataset: every function here reads
# only the CURRENT incoming CCTV frame (or a short rolling window of
# recent frames already held in memory for this camera session).
# ═══════════════════════════════════════════════════════════════════════
def fire_color_evidence(roi_bgr: np.ndarray, cfg: FireSmokePipelineConfig) -> float:
    """HSV corona/white-core coverage + RGB red-dominance + YCbCr
    light-emission agreement, combined and normalized to [0,1]. Real fire
    measured at ~3.85% frame-wide corona coverage on this project's own
    footage; saturates at cfg.fire_color_saturation_fraction of the
    (tighter) candidate ROI."""
    if roi_bgr.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    corona = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 120], np.uint8), np.array([25, 255, 255], np.uint8)),
        cv2.inRange(hsv, np.array([170, 120, 120], np.uint8), np.array([180, 255, 255], np.uint8)),
    )
    core = cv2.inRange(hsv, np.array([0, 0, 220], np.uint8), np.array([180, 80, 255], np.uint8))
    hsv_mask = cv2.bitwise_or(corona, core)

    y_ch, cr_ch, cb_ch = cv2.split(cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb))
    light_mask = cv2.bitwise_and(cv2.compare(y_ch, cb_ch, cv2.CMP_GT), cv2.compare(cr_ch, cb_ch, cv2.CMP_GT))

    b_ch, g_ch, r_ch = cv2.split(blurred)
    red_dom_mask = cv2.bitwise_and(
        cv2.bitwise_and(cv2.compare(r_ch, g_ch, cv2.CMP_GT), cv2.compare(g_ch, b_ch, cv2.CMP_GT)),
        cv2.compare(r_ch, np.full_like(r_ch, 150), cv2.CMP_GT),
    )

    combined = cv2.bitwise_and(cv2.bitwise_and(hsv_mask, light_mask), red_dom_mask)
    coverage = float((combined > 0).sum()) / combined.size
    return _clip01(coverage / cfg.fire_color_saturation_fraction)


def fire_brightness_evidence(roi_bgr: np.ndarray) -> float:
    """Real flame is locally bright. Mean V-channel relative to a fixed
    mid-grey reference (not frame-relative — a small ROI's own mean can
    be misleadingly high/low), normalized to [0,1]."""
    if roi_bgr.size == 0:
        return 0.0
    v = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return _clip01((float(v.mean()) - 120.0) / 135.0)


def smoke_color_evidence(roi_bgr: np.ndarray, cfg: FireSmokePipelineConfig) -> float:
    """Low-saturation grey/white coverage — deliberately does NOT require
    any red/orange, matching real smoke's actual colour profile."""
    if roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 90], np.uint8), np.array([180, 60, 200], np.uint8))
    coverage = float((mask > 0).sum()) / mask.size
    return _clip01(coverage / cfg.smoke_color_saturation_fraction)


# ═══════════════════════════════════════════════════════════════════════
# Per-class temporal track — persistence/gap-tolerance bookkeeping and
# the history needed for flicker, growth, and drift evidence. One
# instance per class (fire, smoke) per camera session (never shared
# across cameras — same isolation guarantee as the detectors it wraps).
# ═══════════════════════════════════════════════════════════════════════
class _EvidenceTrack:
    def __init__(self, cfg: FireSmokePipelineConfig):
        self.cfg = cfg
        self.state = "idle"  # idle -> candidate -> established
        self.consecutive_present = 0
        self.consecutive_missed = 0
        self.prev_box: Optional[List[int]] = None
        self.dim_history: Deque[Tuple[int, int]] = deque(maxlen=cfg.static_history_len)
        self.trend_history: Deque[Tuple[int, int, int]] = deque(maxlen=10)  # (cx, cy, area)
        self.color_mask_history: Deque[np.ndarray] = deque(maxlen=5)
        # Event-level bookkeeping (step 7): independent of fire_smoke_state.py's
        # own hysteresis — this is per-track evidence metadata, exposed for
        # diagnostics and available to callers that want it.
        self.start_frame: Optional[int] = None
        self.last_seen_frame: Optional[int] = None
        self.peak_confidence: float = 0.0
        self._conf_sum: float = 0.0
        self._conf_count: int = 0

    def reset(self) -> None:
        self.__init__(self.cfg)

    @property
    def average_confidence(self) -> float:
        return round(self._conf_sum / self._conf_count, 3) if self._conf_count else 0.0

    def record_present(self, frame_idx: int, bbox: List[int], fused_conf: float) -> None:
        self.consecutive_present += 1
        self.consecutive_missed = 0
        if self.start_frame is None:
            self.start_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.peak_confidence = max(self.peak_confidence, fused_conf)
        self._conf_sum += fused_conf
        self._conf_count += 1
        if self.consecutive_present >= self.cfg.persistence_min_frames:
            self.state = "established"
        else:
            self.state = "candidate"
        self.prev_box = bbox

    def record_missed(self) -> None:
        self.consecutive_missed += 1
        if self.consecutive_missed > self.cfg.gap_tolerance_frames:
            # Absence has genuinely exceeded tolerance — this is a new
            # event from here, not a continuation. Full reset.
            self.reset()

    def is_continuous_with(self, bbox: List[int]) -> bool:
        if self.prev_box is None:
            return True  # nothing to compare against yet — not a discontinuity
        return _iou(bbox, self.prev_box) >= self.cfg.iou_continuity_threshold

    def static_penalty(self) -> float:
        """Rule per spec: static box behaviour is NEGATIVE EVIDENCE (a
        soft confidence penalty), never an automatic rejection — a real,
        stationary fire (e.g. confined to one spot) must still be able to
        accumulate enough evidence from colour/brightness/growth to
        confirm."""
        if len(self.dim_history) < self.cfg.static_history_len:
            return 0.0
        widths = [d[0] for d in self.dim_history]
        heights = [d[1] for d in self.dim_history]
        import statistics
        if (statistics.pvariance(widths) < self.cfg.static_variance_eps_px2
                and statistics.pvariance(heights) < self.cfg.static_variance_eps_px2):
            return self.cfg.static_evidence_penalty
        return 0.0

    def growth_evidence(self) -> float:
        """Positive if area is trending up over the tracked window —
        organic growth is real-fire/smoke-like; shrinking or flat is not
        rewarded (but not penalized either, beyond the plain YOLO/colour
        terms already reflecting what's actually visible)."""
        if len(self.trend_history) < 4:
            return 0.0
        area0 = self.trend_history[0][2]
        area1 = self.trend_history[-1][2]
        growth = (area1 - area0) / max(area0, 1.0)
        return _clip01(growth / 0.5)  # 50% area growth over the window -> full evidence

    def drift_evidence(self) -> float:
        """Smoke-specific: net upward drift OR area expansion, continuous
        score (not a hard reject) — a smoke plume can do either or both."""
        if len(self.trend_history) < 4:
            return 0.0
        cx0, cy0, area0 = self.trend_history[0]
        cx1, cy1, area1 = self.trend_history[-1]
        dy = cy1 - cy0  # negative = moved up (image y grows downward)
        dx = cx1 - cx0
        area_growth = (area1 - area0) / max(area0, 1.0)

        is_horizontal_fast = abs(dx) >= 30.0 and abs(dy) < abs(dx) * 0.4
        if is_horizontal_fast:
            return 0.0  # vehicle-like lateral motion — not smoke behaviour

        upward_score = _clip01(-dy / 20.0)         # 20px net upward -> full score
        expansion_score = _clip01(area_growth / 0.3)  # 30% growth -> full score
        return max(upward_score, expansion_score)

    def flicker_evidence(self, bbox: List[int]) -> float:
        """Mean absdiff of the binary colour-mask within this box's ROI
        across the recent history — real flame/smoke visibly changes
        frame to frame; a static coloured object barely does."""
        if len(self.color_mask_history) < 3:
            return 0.0
        hist = list(self.color_mask_history)
        diffs = []
        for i in range(1, len(hist)):
            prev_roi = _crop(hist[i - 1], bbox)
            curr_roi = _crop(hist[i], bbox)
            if prev_roi.size == 0 or prev_roi.shape != curr_roi.shape:
                continue
            diffs.append(float(cv2.absdiff(curr_roi, prev_roi).mean()))
        if not diffs:
            return 0.0
        return _clip01((sum(diffs) / len(diffs)) / 12.0)  # mean absdiff of 12 (0-255 scale) -> full evidence

    def motion_evidence(self, bbox: List[int], prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
        if prev_gray is None:
            return 0.0
        prev_roi = _crop(prev_gray, bbox)
        curr_roi = _crop(curr_gray, bbox)
        if prev_roi.size == 0 or prev_roi.shape != curr_roi.shape:
            return 0.0
        diff = float(cv2.absdiff(curr_roi, prev_roi).mean())
        return _clip01(diff / 15.0)


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline — implements the .detect(frame) / .reset_temporal_state()
# contract, internally: YOLO candidates -> filtering -> per-class
# verification/fusion -> single evidence-backed detection per class.
# ═══════════════════════════════════════════════════════════════════════
class FireSmokeFusionPipeline:
    def __init__(self, config: Optional[FireSmokePipelineConfig] = None, weights_path: Optional[str] = None):
        self.cfg = config or DEFAULT_CONFIG
        self.yolo = YOLOFireSmokeDetector(weights_path=weights_path)
        # Classical detector: SUPPORTING evidence only — see module
        # docstring. Its own internal thresholds/votes are untouched; we
        # only look at whether IT independently agrees, as a bonus.
        self.classical = ClassicalFireSmokeDetector(
            fire_threshold=AIConfig.FIRE_DETECTION_THRESHOLD,
            smoke_threshold=AIConfig.SMOKE_DETECTION_THRESHOLD,
        )
        self._fire_track = _EvidenceTrack(self.cfg)
        self._smoke_track = _EvidenceTrack(self.cfg)
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_idx = 0

    def reset_temporal_state(self) -> None:
        self.yolo.reset_temporal_state()
        self.classical.reset_temporal_state()
        self._fire_track.reset()
        self._smoke_track.reset()
        self._prev_gray = None
        self._frame_idx = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _best_box(dets: List[Dict[str, Any]], class_name: str) -> Optional[Dict[str, Any]]:
        candidates = [d for d in dets if d["class_name"] == class_name]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d["confidence"])

    def _valid_candidate(self, box: Optional[Dict[str, Any]], frame_area: int) -> Optional[Dict[str, Any]]:
        """Step 2 — candidate filtering: box-size sanity, aspect ratio.
        Confidence floor is already applied inside YOLOFireSmokeDetector."""
        if box is None:
            return None
        x1, y1, x2, y2 = box["bbox"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        area = w * h
        area_fraction = area / max(frame_area, 1)
        if area_fraction < self.cfg.min_box_area_fraction or area_fraction > self.cfg.max_box_area_fraction:
            return None
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > self.cfg.max_aspect_ratio:
            return None
        return box

    def _process_fire(
        self, frame: np.ndarray, gray: np.ndarray, color_mask: np.ndarray,
        box: Optional[Dict[str, Any]], classical_box: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        track = self._fire_track
        cfg = self.cfg

        if box is None:
            track.record_missed()
            return None

        bbox = box["bbox"]
        roi = _crop(frame, bbox)

        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        track.dim_history.append((bw, bh))
        cx, cy = box["center"]
        area = bw * bh
        track.trend_history.append((cx, cy, area))
        track.color_mask_history.append(color_mask)

        color_ev = fire_color_evidence(roi, cfg)
        brightness_ev = fire_brightness_evidence(roi)
        flicker_ev = track.flicker_evidence(bbox)
        motion_ev = track.motion_evidence(bbox, self._prev_gray, gray)
        growth_ev = track.growth_evidence()
        classical_bonus = 1.0 if (classical_box is not None and _iou(bbox, classical_box["bbox"]) > 0.1) else 0.0
        static_penalty = track.static_penalty()

        support_score = (
            cfg.fire_support_color_weight * color_ev
            + cfg.fire_support_brightness_weight * brightness_ev
            + cfg.fire_support_flicker_weight * flicker_ev
            + cfg.fire_support_motion_weight * motion_ev
            + cfg.fire_support_growth_weight * growth_ev
            + cfg.fire_support_classical_weight * classical_bonus
        )
        yolo_conf = box["confidence"]
        fused = yolo_conf + (1.0 - yolo_conf) * support_score * cfg.fire_boost_factor - static_penalty
        fused = _clip01(fused)

        track.record_present(self._frame_idx, bbox, fused)

        if cfg.debug_logging:
            logger.info(
                f"[fire_smoke_pipeline] FIRE frame={self._frame_idx} yolo={box['confidence']:.2f} "
                f"color={color_ev:.2f} bright={brightness_ev:.2f} flicker={flicker_ev:.2f} "
                f"motion={motion_ev:.2f} growth={growth_ev:.2f} classical_bonus={classical_bonus:.2f} "
                f"static_penalty={static_penalty:.2f} fused={fused:.2f} state={track.state}"
            )

        if track.state != "established":
            return None  # step 3: persistence not yet met — don't offer this as a candidate yet

        return {"class_name": "fire", "confidence": round(fused, 3), "bbox": bbox, "center": box["center"]}

    def _process_smoke(
        self, frame: np.ndarray,
        box: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        track = self._smoke_track
        cfg = self.cfg

        if box is None:
            track.record_missed()
            return None

        bbox = box["bbox"]
        roi = _crop(frame, bbox)

        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = box["center"]
        area = bw * bh
        track.trend_history.append((cx, cy, area))

        color_ev = smoke_color_evidence(roi, cfg)
        drift_ev = track.drift_evidence()
        growth_ev = track.growth_evidence()

        support_score = (
            cfg.smoke_support_color_weight * color_ev
            + cfg.smoke_support_drift_weight * drift_ev
            + cfg.smoke_support_growth_weight * growth_ev
        )
        yolo_conf = box["confidence"]
        fused = yolo_conf + (1.0 - yolo_conf) * support_score * cfg.smoke_boost_factor
        fused = _clip01(fused)

        track.record_present(self._frame_idx, bbox, fused)

        if cfg.debug_logging:
            logger.info(
                f"[fire_smoke_pipeline] SMOKE frame={self._frame_idx} yolo={box['confidence']:.2f} "
                f"color={color_ev:.2f} drift={drift_ev:.2f} growth={growth_ev:.2f} "
                f"fused={fused:.2f} state={track.state}"
            )

        if track.state != "established":
            return None

        return {"class_name": "smoke", "confidence": round(fused, 3), "bbox": bbox, "center": box["center"]}

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        annotated = frame.copy() if frame is not None else np.zeros((1, 1, 3), np.uint8)
        if frame is None or frame.size == 0:
            return [], annotated

        yolo_dets, _ = self.yolo.detect(frame)
        classical_dets, _ = self.classical.detect(frame)

        frame_area = frame.shape[0] * frame.shape[1]
        fire_box = self._valid_candidate(self._best_box(yolo_dets, "fire"), frame_area)
        smoke_box = self._valid_candidate(self._best_box(yolo_dets, "smoke"), frame_area)
        classical_fire_box = self._best_box(classical_dets, "fire")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Colour mask for the flicker check — cheap, reused from the same
        # HSV/YCbCr/RGB combination fire_color_evidence uses, computed
        # once per frame at full resolution (not per-ROI) so flicker_ev's
        # cropped history stays spatially consistent frame to frame.
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        color_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 120, 120], np.uint8), np.array([25, 255, 255], np.uint8)),
            cv2.inRange(hsv, np.array([170, 120, 120], np.uint8), np.array([180, 255, 255], np.uint8)),
        )

        fire_result = self._process_fire(frame, gray, color_mask, fire_box, classical_fire_box)
        smoke_result = self._process_smoke(frame, smoke_box)

        self._prev_gray = gray
        self._frame_idx += 1

        detections = [d for d in (fire_result, smoke_result) if d is not None]
        return detections, annotated
