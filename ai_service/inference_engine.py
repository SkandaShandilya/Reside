"""
Inference Engine — Phase 3

Real-Time AI Pipeline Orchestrator.

Pipeline Flow:
  Raw Video Frame (BGR numpy array)
        ↓
  Person Detector (YOLOv8 / HOG)
        ↓
  Object Tracker (Centroid & Trajectory)
        ↓
  Fire/Smoke Detector (HSV + temporal voting)
        ↓
  Behaviour Engine Analysis (loitering / restricted zones / crowd — all
  deterministic, position-based rules)
        ↓
  Alert Generator (Cooldown & Deduplication check)
        ↓
  Output: Security Alert / Review Queue Item / None

NOTE ON THE NEURAL ACTIVITY CLASSIFIER (Fighting/Kicking/Snatching/Falling/
Running/Loitering/Crowd Formation/Fire/Smoke): the checkpoint loaded here
(auto-selected as whichever storage/models/*.pt reports the highest
validation accuracy among checkpoints evaluated on >=100 samples) has its
real accuracy at load time logged and exposed as `self.neural_accuracy` —
that number changes every time a new checkpoint is trained, so it is
intentionally NOT hardcoded here; a stale hardcoded percentage in a comment
is exactly the kind of claim this project explicitly must not make (see
`ai_service/training/evaluate_thresholds.py` for how to get a real,
leakage-checked number for whatever checkpoint is currently deployed).
Whatever that number is, its predictions are shown on the overlay through
the hysteresis state machine (ai_service/activity_state.py) as ANALYZING
(uncommitted) or CONFIRMED (persistence + confidence both satisfied) —
never phrased as certain on a single frame — and only ever escalate to a
persisted alert through the existing confidence-gated review-queue path
(low confidence -> review queue, never a direct "critical" alert) — they
never override or get conflated with the deterministic rule-based
detections (loitering/crowd/restricted-zone), which stay the only source
of high-confidence "confirmed" alerts backed by a non-learned signal.
"""
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from ai_service.config import AIConfig
from ai_service.detector import PersonDetector
from ai_service.fire_smoke_pipeline import FireSmokeFusionPipeline
from ai_service.tracker import PersonTracker
from ai_service.behaviour_engine import BehaviourEngine, DetectionResult
from ai_service import reid
from ai_service.alert_generator import AlertGenerator, AlertEvent
from ai_service.activity_state import ActivityStateMachine, ActivityStateResult, NORMAL, ANALYZING, CONFIRMED, RECOVERY
from ai_service.fire_smoke_state import (
    CameraFireSmokeState, FireSmokeStateResult,
    NORMAL as FS_NORMAL, POSSIBLE as FS_POSSIBLE, CONFIRMED as FS_CONFIRMED, RECOVERY as FS_RECOVERY,
    DISPLAY_NAME as FS_DISPLAY_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """
    Result of the (relatively expensive) detection/tracking/behaviour pass
    for one frame. Deliberately holds no pixels — it's cheap to cache and
    reuse across several video frames while overlay drawing (render())
    happens on every frame independently, so the video stays smooth even
    though full analysis can't keep up with the raw frame rate.
    """
    tracked_persons: List[Dict] = field(default_factory=list)
    fire_smoke_detections: List[Dict] = field(default_factory=list)
    fire_detected: bool = False
    smoke_detected: bool = False
    fire_conf: float = 0.0
    smoke_conf: float = 0.0
    # Fire/Smoke hysteresis state — SEPARATE subsystem, see
    # ai_service/fire_smoke_state.py. fire_detected/smoke_detected/fire_conf/
    # smoke_conf above stay this frame's raw detector output; these are the
    # temporally-stable POSSIBLE/CONFIRMED/RECOVERY view the overlay and
    # alerting actually use.
    fire_state: str = FS_NORMAL
    smoke_state: str = FS_NORMAL
    fire_smoothed_confidence: float = 0.0
    smoke_smoothed_confidence: float = 0.0
    fire_duration_seconds: float = 0.0
    smoke_duration_seconds: float = 0.0
    person_count: int = 0
    behaviour_label: Optional[str] = None
    behaviour_confidence: float = 0.0
    # Neural classifier's current top prediction for this camera's rolling
    # clip buffer — a suggestion, not a confirmed detection (see module
    # docstring). None until the model has seen a full clip.
    neural_label: Optional[str] = None
    neural_confidence: float = 0.0
    # Hysteresis-smoothed view of the neural classifier's output — see
    # ai_service/activity_state.py. `neural_label`/`neural_confidence` above
    # stay the raw per-window model output; these are what the overlay's
    # three-state (Normal/Analyzing/Confirmed) rendering and alert gating
    # actually key off of.
    activity_state: str = NORMAL
    activity_smoothed_confidence: float = 0.0
    is_alertable: bool = False
    alert: Optional[AlertEvent] = None
    # On the rare window where TWO DIFFERENT classes both independently hit
    # event_started (see analyze()'s neural/behaviour event lifecycle block
    # and the masking bug documented there), `alert` above carries only the
    # first/"featured" one — every other brand-new alert from the same
    # window goes here so none are ever silently dropped. Almost always
    # empty. Callers (video_stream_service.py, video_timeline_processor.py)
    # must dispatch these exactly like `alert`.
    additional_alerts: List[AlertEvent] = field(default_factory=list)
    # Refresh/finalize instructions for an already-open behaviour event's DB
    # row (peak/average/duration, and end_time on finalize) — see analyze()'s
    # neural/behaviour event lifecycle block and
    # backend/app/services/alert_dispatcher.dispatch_event_update_sync.
    # Distinct from `alert` above, which is only ever a brand-NEW row.
    behaviour_event_updates: List[Dict[str, Any]] = field(default_factory=list)
    # Same as behaviour_event_updates above, but for the SEPARATE fire/smoke
    # subsystem — kept as a distinct field (never merged with the behaviour
    # list) per the architectural mandate that the two stay independent at
    # every layer. See analyze()'s fire/smoke event lifecycle block.
    fire_smoke_event_updates: List[Dict[str, Any]] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    # Zones actually used for this analysis, already converted from the
    # normalized (0-1) coordinates they're stored in to this frame's pixel
    # coordinates — render() draws from here so overlay and detection always
    # agree on where a zone actually is, even if a caller resizes frames.
    zones_px: List[Dict] = field(default_factory=list)
    # The specific geometric evidence behind whichever gated behaviour class
    # is currently active this window ({"type": "fighting"|"snatching"|
    # "vandalism", ...gate-specific fields}) — see analyze()'s step 5f for
    # how it's populated and _draw_rich_overlays for how it's drawn. None
    # when no gated class is currently active. This is what lets the
    # overlay show a guard the SPECIFIC reason an alert fired (the two
    # people in contact, the departure vector, the asset zone) instead of
    # just a generic box — see render()'s module-level rationale.
    behaviour_proof: Optional[Dict[str, Any]] = None
    # Track IDs currently qualifying as sustained-running (see
    # BehaviourEngine.get_running_track_ids()) — the ONE source of truth
    # for "is this person running", shared between the Running alert rule
    # and the overlay's per-person box, so they can never disagree (the
    # overlay used to run its own separate, unguarded instantaneous speed
    # check).
    running_track_ids: frozenset = field(default_factory=frozenset)
    # track_id -> seconds inside a restricted zone (overlay timer / label)
    zone_dwell: Dict[int, float] = field(default_factory=dict)
    # Track IDs currently qualifying as Loitering (see BehaviourEngine.
    # person_is_loitering()) — the ONE source of truth for "is this person
    # loitering", shared between the Loitering alert rule and the
    # overlay's per-person box, so they can never disagree (same pattern
    # as running_track_ids above).
    loitering_track_ids: frozenset = field(default_factory=frozenset)
    # Track IDs currently belonging to a CONFIRMED Overcrowding cluster
    # (see BehaviourEngine.get_overcrowding_state()) — read independently
    # of whether Overcrowding happened to be this window's winning
    # DetectionResult, so the overlay's group border/banner stays visible
    # for the whole confirmed duration, not just the single window it
    # first fired.
    overcrowding_track_ids: frozenset = field(default_factory=frozenset)


class CameraInferenceSession:
    """
    Per-camera inference session maintaining tracker state and the rolling
    clip buffer the neural classifier reads from.

    fire_smoke_detector lives here (one per camera), not on the shared
    InferenceEngine — it carries per-frame temporal state (vote buffers,
    previous grayscale frame for motion checks) that must never mix
    between cameras. A single shared instance would let camera A's frames
    count toward camera B's smoke confirmation votes, and would crash
    outright the moment two cameras with different resolutions were
    processed back-to-back (their frames aren't the same shape).
    """

    def __init__(self, camera_id: int, camera_name: str, camera_location: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_location = camera_location
        self.tracker = PersonTracker()
        # Fire/Smoke: FireSmokeFusionPipeline (ai_service/
        # fire_smoke_pipeline.py) — YOLOv8 (finetune_v2, unmodified,
        # never retrained) stays the PRIMARY detector; a lightweight,
        # non-training visual-evidence layer (HSV/RGB/YCbCr colour,
        # brightness, flicker, motion, growth) plus the classical
        # detector as a supporting-only bonus signal are fused with it to
        # close a measured confidence gap — real fire in this project's
        # own footage sat at raw YOLO confidence 0.40-0.60, under the
        # 0.70 alert threshold, despite strong independent colour
        # evidence. Fixed the miss (0/312 -> 286/312 confirmed frames on
        # a real fire video) without moving false positives (0/43 real
        # non-fire videos produce a confirmed alert, before or after).
        # Same .detect(frame) contract, so fire_smoke_state.py below
        # needed no changes. See that module's docstring for the full
        # root-cause measurement and design rationale.
        self.fire_smoke_detector = FireSmokeFusionPipeline()
        self.frame_buffer: deque = deque(maxlen=AIConfig.SEQUENCE_LENGTH)
        self.cached_neural_label: Optional[str] = None
        self.cached_neural_confidence: float = 0.0
        # Full per-class softmax vector from the last successful inference
        # pass — needed so ActivityStateMachine can track every class's
        # evidence independently instead of only ever seeing the winning
        # top-1 class (see activity_state.py module docstring).
        self.cached_neural_probs: Dict[str, float] = {}
        # Scene-level hysteresis over the neural classifier's raw per-window
        # output (see ai_service/activity_state.py) — one per camera, since
        # the classifier judges the whole clip, not a per-person crop.
        self.activity_state = ActivityStateMachine()
        # Fire/Smoke hysteresis — a SEPARATE subsystem (ai_service/
        # fire_smoke_state.py), fed only from fire_smoke_detector above,
        # never from the neural activity classifier.
        self.fire_smoke_state = CameraFireSmokeState()
        # activity label -> the `now` (event_time) it was last created/
        # refreshed in the DB — see analyze()'s neural/behaviour event
        # lifecycle block. Only present while that label's event is
        # currently open (has a live DB row awaiting finalization).
        self.open_behaviour_events: Dict[str, float] = {}
        # Same bookkeeping as open_behaviour_events above, kept as a
        # SEPARATE dict (never sharing keys/state with it) so fire/smoke's
        # event lifecycle stays fully independent from behaviour's, per the
        # architectural mandate — see analyze()'s fire/smoke event
        # lifecycle block.
        self.open_fire_smoke_events: Dict[str, float] = {}
        # activity label -> the `now` its event most recently CLOSED (event_
        # ended). Used only for the RETRIGGER_COOLDOWN_SECONDS check — a
        # fresh event_started within that window of the last close reopens
        # the same row instead of creating a new one. See analyze()'s
        # fire/smoke event lifecycle block and AIConfig.RETRIGGER_COOLDOWN_
        # SECONDS's docstring.
        self.fire_smoke_closed_at: Dict[str, float] = {}
        # activity label -> the last STATE (NORMAL/POSSIBLE/CONFIRMED/
        # RECOVERY) actually logged for it — so debug logging only fires on
        # transitions, never every analysis window.
        self.fire_smoke_last_logged_state: Dict[str, str] = {}
        # Cumulative fire_detected/smoke_detected for the CURRENTLY open
        # combined "Fire/Smoke Detected" incident — OR-accumulated every
        # frame the incident stays open, reset when a new incident starts.
        # Needed because fire and smoke can each independently confirm at
        # different frames within the same open incident (their own
        # 12-window hysteresis runs independently) — capturing only the
        # single trigger-frame's snapshot would wrongly show
        # fire_detected=False for an incident that had real, sustained
        # fire a few frames after smoke happened to confirm first.
        self.fire_smoke_open_fire_ever: bool = False
        self.fire_smoke_open_smoke_ever: bool = False

        # Fighting Gate 3/4 — per-pair rolling crop buffers (keyed by
        # (low_track_id, high_track_id)) feeding the SAME deployed
        # checkpoint on just that pair's cropped, padded region, and which
        # specific pair (if any) currently owns this camera's "active
        # fight" slot — see analyze()'s Fighting block and behaviour_
        # engine.py's FightingPairState. One active incident at a time per
        # camera, same one-result-per-window convention the rest of this
        # module already uses. The crop buffer is keyed by camera (not by
        # track_id pair) — see _fighting_gate3_prob's docstring for why.
        self.fighting_incident_buffer: deque = deque(maxlen=AIConfig.SEQUENCE_LENGTH)
        self.active_fighting_pair_key: Optional[Tuple[int, int]] = None
        self.fighting_last_pair_seen_at: Optional[float] = None

        # Real, measured per-stage timings (ms) from this camera's most
        # recent analyze() call — see analyze()'s timing instrumentation.
        # Populated every call so callers/tests can inspect actual
        # hardware performance without re-deriving it from logs.
        self.last_stage_timings_ms: Dict[str, float] = {}

        # Loitering Re-ID — this camera's own rolling cache of each active
        # track's most recent visual descriptor (ai_service/reid.py) and
        # effective dwell time (behaviour_engine.py's get_effective_dwell),
        # refreshed every analyze() call. Needed because by the time a
        # track is EVICTED (gone from tracked_persons), there's no current
        # frame crop of them to build a descriptor from — this cache holds
        # their last-known state from while they were still visible, so
        # registering their exit uses real, recent evidence, not a stale
        # or missing one.
        self.track_signatures: Dict[int, Dict[str, Any]] = {}
        self.track_last_dwell: Dict[int, float] = {}


class InferenceEngine:
    """
    Singleton AI Orchestrator that handles multi-camera real-time video inference.
    """

    def __init__(self, config: AIConfig = AIConfig()):
        self.config = config
        self.detector = PersonDetector(
            model_name=config.YOLO_MODEL,
            confidence_threshold=config.YOLO_CONFIDENCE,
            iou_threshold=config.DETECTION_IOU_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
        )
        self.behaviour_engine = BehaviourEngine({
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "loitering_demo_dwell_seconds": config.LOITERING_DWELL_TIME_SECONDS,
            "loitering_demo_radius_px": config.LOITERING_MAX_RADIUS_PX,
            "loitering_dwell_seconds": config.LOITERING_DWELL_SECONDS,
            "loiter_radius_px": config.LOITER_RADIUS_PX,
            "loitering_pacing_window_seconds": config.LOITERING_PACING_WINDOW_SECONDS,
            "loitering_pacing_radius_px": config.LOITERING_PACING_RADIUS_PX,
            "reid_similarity_threshold": config.RE_ID_SIMILARITY_THRESHOLD,
            "reid_time_window_seconds": config.RE_ID_TIME_WINDOW_MINUTES * 60,
            "crowd_threshold": config.CROWD_THRESHOLD,
            "crowd_persist_seconds": config.CROWD_PERSIST_SECONDS,
            "crowd_max_avg_distance_px": config.CROWD_MAX_AVG_DISTANCE_PX,
            "overcrowding_min_count": config.OVERCROWDING_MIN_COUNT,
            "overcrowding_max_radius_px": config.OVERCROWDING_MAX_RADIUS_PX,
            "overcrowding_dwell_seconds": config.OVERCROWDING_DWELL_SECONDS,
            "overcrowding_pretrigger_gap_seconds": config.OVERCROWDING_PRETRIGGER_GAP_SECONDS,
            "running_speed_px_s": config.RUNNING_SPEED_PX_S,
            "running_min_seconds": config.RUNNING_MIN_SECONDS,
            "vandalism_min_seconds": config.VANDALISM_MIN_SECONDS,
            "vandalism_aggression_threshold": config.VANDALISM_AGGRESSION_THRESHOLD,
            "falling_aspect_ratio_max": config.FALLING_ASPECT_RATIO_MAX,
            "falling_motionless_px": config.FALLING_MOTIONLESS_PX,
            "falling_motionless_seconds": config.FALLING_MOTIONLESS_SECONDS,
            "snatching_contact_px": config.SNATCHING_CONTACT_PX,
            "snatching_departure_px_s": config.SNATCHING_DEPARTURE_PX_S,
            "snatching_departure_window_seconds": config.SNATCHING_DEPARTURE_WINDOW_SECONDS,
        })
        self.alert_generator = AlertGenerator(
            cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
        )
        # A SEPARATE alert generator for fire/smoke — deliberately not
        # shared with the behaviour alert_generator above, so the two
        # systems stay logically independent. cooldown_seconds=0: dedup is
        # now fully structural (edge-triggered on event_started, one row
        # per event) plus an explicit, purpose-built
        # AIConfig.RETRIGGER_COOLDOWN_SECONDS post-close reopen check in
        # analyze() below — stacking AlertGenerator's OWN generic cooldown
        # on top (keyed off alert CREATION time, not the event's actual
        # CLOSE time) would be a second, competing dedup mechanism that
        # could incorrectly block a genuinely new event or fail to block a
        # too-fast reflicker, depending on how long the prior event had
        # been open. See ai_service/fire_smoke_state.py's module docstring.
        self.fire_smoke_alert_generator = AlertGenerator(
            cooldown_seconds=0,
            confidence_threshold=config.FIRE_ALERT_THRESHOLD,  # only used for send_to_review routing; is_alertable already gates confirmation
        )

        # Fire & Smoke Detector: intentionally NOT instantiated here — it
        # carries per-frame temporal state (vote buffers, previous frame for
        # motion checks) that must stay isolated per camera, so each
        # CameraInferenceSession creates and owns its own instance instead.

        # Neural activity classifier — see module docstring for how its
        # output is gated (display-only "AI:" suggestion, never a direct
        # confirmed alert).
        self.neural_model = None
        self.neural_device = "cpu"
        self.neural_classes: List[str] = config.DEFAULT_ACTIVITY_CLASSES
        self.neural_accuracy: float = 0.0
        self._load_best_neural_model()

        # Camera sessions
        self._sessions: Dict[int, CameraInferenceSession] = {}

    def _get_deployed_checkpoint(self) -> Optional[Tuple[str, Path]]:
        """
        The DB's `model_versions.is_deployed` flag (set only by an admin via
        POST /api/models/{id}/deploy or /rollback — see backend/app/api/
        models.py) is the single authoritative answer to "which checkpoint
        should inference use". It must never be second-guessed by
        accuracy-based auto-selection: an explicit deployment always wins.
        Returns (version, path) for the deployed checkpoint's file if the DB
        has a deployed row AND its file can be found on disk, else None (no
        deployed row, unreadable DB, or the file is missing) so the caller
        can fall back to auto-selection.
        """
        import sqlite3

        db_candidates = [
            Path("./backend/sentinel.db"),
            Path("./sentinel.db"),
            Path(__file__).resolve().parent.parent / "backend" / "sentinel.db",
        ]
        db_path = next((d for d in db_candidates if d.exists()), None)
        if db_path is None:
            return None

        try:
            con = sqlite3.connect(str(db_path))
            # `model_versions` now holds rows for more than one independent
            # model type (this behaviour CNN's own "Time-Distributed CNN"
            # rows, plus the separate Fire/Smoke neural classifier's
            # "FireSmoke-CNN" rows — see ai_service/fire_smoke_model.py) —
            # each with its OWN independent "deployed" flag. Without this
            # exclusion, `is_deployed = 1 LIMIT 1` with no ORDER BY could
            # return EITHER model's row depending on unspecified SQLite
            # row ordering the moment a second architecture's row also had
            # is_deployed=1 — confirmed directly: after registering the
            # first FireSmoke-CNN checkpoint, two rows had is_deployed=1
            # simultaneously (each architecture's own deploy action, quite
            # correctly, never touched the other's flag). Scoping this
            # query to this model's own architecture is what actually makes
            # the two independent "deployed" concepts safe to coexist.
            row = con.execute(
                "SELECT version, weights_path FROM model_versions "
                "WHERE is_deployed = 1 AND architecture != 'FireSmoke-CNN' LIMIT 1"
            ).fetchone()
            con.close()
        except Exception as e:
            logger.warning(f"Could not read deployed checkpoint from {db_path}: {e}")
            return None

        if not row or not row[1]:
            return None

        version, weights_path = row
        filename = Path(weights_path).name
        for d in self._checkpoint_search_dirs():
            for candidate in (d / filename, d / "archived" / filename):
                if candidate.exists():
                    return version, candidate

        logger.warning(
            f"DB marks version {version} as deployed but its checkpoint file "
            f"'{filename}' was not found on disk — falling back to auto-selection."
        )
        return None

    @staticmethod
    def _checkpoint_search_dirs() -> List[Path]:
        return [
            Path("./backend/storage/models"),
            Path("./storage/models"),
            Path(__file__).resolve().parent.parent / "backend" / "storage" / "models",
            Path(__file__).resolve().parent.parent / "storage" / "models",
        ]

    def _apply_checkpoint(self, path: Path, ckpt: dict, accuracy: float, torch, TimeDistributedCNN, source: str) -> bool:
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            classes = ckpt.get("activity_classes") or self.neural_classes
            model = TimeDistributedCNN(num_classes=len(classes), sequence_length=AIConfig.SEQUENCE_LENGTH)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            self.neural_model = model
            self.neural_device = device
            self.neural_classes = classes
            self.neural_accuracy = float(accuracy)
            logger.info(f"Loaded neural activity classifier {path.name} ({source}) on {device}.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load neural checkpoint {path}: {e}")
            return False

    def _load_best_neural_model(self) -> None:
        """
        Deployment is deterministic: if the DB has an explicitly deployed
        checkpoint (see _get_deployed_checkpoint), that one is loaded and
        auto-selection never runs. Only when nothing is explicitly deployed
        (or its file is missing) does this fall back to scanning
        storage/models/*.pt for whichever checkpoint reports the highest
        real accuracy (stored in the checkpoint itself by fine_tune.py).
        Falls back to no neural model at all — not a random-weight one — if
        nothing usable is found either way, since untrained-random output is
        worse than having no suggestion at all.
        """
        try:
            import torch
            from ai_service.activity_model import TimeDistributedCNN
        except ImportError:
            logger.info("PyTorch not available — neural activity suggestions disabled.")
            return

        deployed = self._get_deployed_checkpoint()
        if deployed is not None:
            version, path = deployed
            try:
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(f"Could not read explicitly deployed checkpoint {path}: {e}")
                ckpt = None
            if isinstance(ckpt, dict):
                acc = ckpt.get("accuracy", 0.0) or 0.0
                if self._apply_checkpoint(path, ckpt, acc, torch, TimeDistributedCNN, f"explicitly deployed v{version}"):
                    return
            logger.warning(f"Falling back to auto-selection after failing to load deployed v{version}.")

        checkpoint_dir = next((d for d in self._checkpoint_search_dirs() if d.exists()), None)
        if checkpoint_dir is None:
            logger.info("No storage/models directory found — neural activity suggestions disabled.")
            return

        # A checkpoint evaluated on only a handful of samples can report an
        # inflated accuracy that isn't comparable to one properly evaluated
        # on hundreds of held-out samples — require a minimum validation
        # size before a checkpoint is even eligible, then pick the highest
        # accuracy among those that qualify. Older checkpoints predating
        # this field (validation_samples_count missing) are skipped rather
        # than assumed trustworthy.
        min_validation_samples = 100
        best_path, best_ckpt, best_acc = None, None, -1.0
        skipped_small = []
        for path in checkpoint_dir.glob("suspicious_activity_v*.pt"):
            try:
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(f"Could not read checkpoint {path.name}: {e}")
                continue
            if not isinstance(ckpt, dict):
                continue
            val_count = ckpt.get("validation_samples_count", 0)
            acc = ckpt.get("accuracy", -1.0)
            if val_count < min_validation_samples:
                skipped_small.append((path.name, val_count))
                continue
            if acc is not None and acc > best_acc:
                best_path, best_ckpt, best_acc = path, ckpt, acc

        if skipped_small:
            logger.info(
                f"Skipped {len(skipped_small)} checkpoint(s) evaluated on fewer than "
                f"{min_validation_samples} validation samples (not comparable): {skipped_small}"
            )

        if best_ckpt is None or best_acc <= 0:
            logger.info("No usable trained checkpoint found — neural activity suggestions disabled.")
            return

        self._apply_checkpoint(
            best_path, best_ckpt, best_acc, torch, TimeDistributedCNN,
            f"auto-selected, no explicit deployment — highest eligible accuracy {best_acc:.1%}",
        )

    def get_or_create_session(
        self, camera_id: int, camera_name: str = "", camera_location: str = ""
    ) -> CameraInferenceSession:
        if camera_id not in self._sessions:
            self._sessions[camera_id] = CameraInferenceSession(
                camera_id=camera_id,
                camera_name=camera_name or f"Camera {camera_id}",
                camera_location=camera_location or "General Area",
            )
        return self._sessions[camera_id]

    def analyze(
        self,
        camera_id: int,
        frame: np.ndarray,
        camera_name: str = "",
        camera_location: str = "",
        zones: List[Dict] = None,
        fire_enabled: bool = True,
        smoke_enabled: bool = True,
        event_time: Optional[float] = None,
        session: Optional["CameraInferenceSession"] = None,
    ) -> FrameAnalysis:
        """
        Run detection/tracking/behaviour analysis on one frame. This is the
        expensive part of the pipeline (YOLO inference etc.) — callers that
        need to keep a video stream smooth should run this on a background
        thread/cadence and reuse the latest result via render() on every
        displayed frame instead of calling this per frame.

        `fire_enabled`/`smoke_enabled` are per-camera toggles, independent
        of the behaviour pipeline — see Camera.detection_config.

        `event_time`: the "clock" fed to the fire/smoke and activity
        hysteresis state machines for duration/timestamp bookkeeping.
        Defaults to real wall-clock time (`time.time()`), which is correct
        for live streaming (frames really do arrive at that rate). Offline
        full-video-timeline analysis (ai_service/video_timeline_processor.py)
        can run faster or slower than real-time, so it passes a synthesized
        epoch (`job_start_epoch + video_offset_seconds`) instead — this
        keeps every downstream duration/absolute-timestamp calculation
        correct without any live/offline branching in this method or in
        alert_dispatcher.py.

        `session`: inject a specific `CameraInferenceSession` instead of
        resolving/creating one from the shared per-camera registry. Used by
        the offline timeline processor so a full-file batch pass never
        shares tracker/fire-smoke-detector/activity-state with (and never
        corrupts) a concurrently live-viewed session for the same
        camera_id. Live streaming never passes this — defaults to the
        existing shared-session behavior, unchanged.
        """
        start_time = time.time()  # always real wall-clock — used only for processing-latency telemetry below
        now = event_time if event_time is not None else start_time
        session = session or self.get_or_create_session(camera_id, camera_name, camera_location)

        # Real, measured per-stage timing (ms) — time.perf_counter() (not
        # time.time(), which is a wall-clock and can jump/skew) around
        # each of the 4 requested boundaries: frame scaling/pre-process,
        # YOLO GPU inference, tracker update, and the gate/neural-model
        # stage. detect_preprocess_ms/detect_inference_ms come from
        # Ultralytics' OWN internal timing (results[0].speed, set on
        # self.detector.last_yolo_speed_ms) rather than being hand-rolled
        # around the single opaque detector.detect() call, which can't
        # see the preprocess/inference boundary inside Ultralytics'
        # predict(). Populated onto session.last_stage_timings_ms every
        # call (not just logged) so a caller/test can inspect the real
        # numbers directly — see profile_stages.py-style validation.
        timings: Dict[str, float] = {}
        _t0 = time.perf_counter()

        # ── 1. Person Detection (YOLOv8 / HOG) ─────────────────────────────
        raw_detections, _ = self.detector.detect(frame)
        timings["detect_total_ms"] = round((time.perf_counter() - _t0) * 1000, 3)
        yolo_speed = getattr(self.detector, "last_yolo_speed_ms", {}) or {}
        timings["detect_preprocess_ms"] = round(yolo_speed.get("preprocess", 0.0), 3)
        timings["detect_inference_ms"] = round(yolo_speed.get("inference", 0.0), 3)
        timings["detect_postprocess_ms"] = round(yolo_speed.get("postprocess", 0.0), 3)

        # ── 2. Fire & Smoke Detection — SEPARATE subsystem from behaviour ───
        # (HSV segmentation + optical-flow turbulence gate; see detector.py).
        # FireSmokeDetector.detect() already applies its own frame-to-frame
        # vote (suppresses single-frame flashes) before returning anything —
        # what comes out of it is this frame's "raw candidate", which then
        # feeds the fire_smoke_state hysteresis machine below for the
        # higher-level POSSIBLE/CONFIRMED/RECOVERY decision that actually
        # gates the overlay wording and alerting.
        if fire_enabled or smoke_enabled:
            fire_smoke_detections, _ = session.fire_smoke_detector.detect(frame)
        else:
            fire_smoke_detections = []
        if not fire_enabled:
            fire_smoke_detections = [d for d in fire_smoke_detections if d["class_name"] != "fire"]
        if not smoke_enabled:
            fire_smoke_detections = [d for d in fire_smoke_detections if d["class_name"] != "smoke"]

        fire_detected  = any(d["class_name"] == "fire"  for d in fire_smoke_detections)
        smoke_detected = any(d["class_name"] == "smoke" for d in fire_smoke_detections)
        fire_conf  = max((d["confidence"] for d in fire_smoke_detections if d["class_name"] == "fire"),  default=0.0)
        smoke_conf = max((d["confidence"] for d in fire_smoke_detections if d["class_name"] == "smoke"), default=0.0)

        fire_state_result: FireSmokeStateResult = session.fire_smoke_state.fire.update(fire_conf, now)
        smoke_state_result: FireSmokeStateResult = session.fire_smoke_state.smoke.update(smoke_conf, now)

        # ── 3. Object Tracking ─────────────────────────────────────────────
        _t_tracker0 = time.perf_counter()
        tracked_persons = session.tracker.update(raw_detections)
        timings["tracker_ms"] = round((time.perf_counter() - _t_tracker0) * 1000, 3)
        person_count = len(tracked_persons)

        # ── 3b. Loitering Re-ID — cache signatures, link returns, register exits ──
        # Deliberately placed BEFORE step 4 (behaviour_engine.analyze_activity,
        # which runs the actual Loitering check) so a re-entry match made
        # THIS window is already visible to that same window's loitering
        # evaluation — a brand-new track linked to a returning identity
        # should be able to trigger Repeat Loitering immediately if the
        # combined dwell already crosses the threshold, not one window late.
        _t_reid0 = time.perf_counter()
        for p in tracked_persons:
            tid = p.get("track_id")
            if tid is None:
                continue
            signature = reid.extract_descriptor(frame, p.get("bbox"))
            if signature is not None:
                session.track_signatures[tid] = signature
            if p.get("is_new_track") and signature is not None:
                self.behaviour_engine.reid_try_link(camera_id, tid, signature, now)
            session.track_last_dwell[tid] = self.behaviour_engine.get_effective_dwell(camera_id, p)

        for evicted_tid in session.tracker.last_evicted_track_ids:
            sig = session.track_signatures.pop(evicted_tid, None)
            last_dwell = session.track_last_dwell.pop(evicted_tid, 0.0)
            if sig is not None:
                self.behaviour_engine.reid_register_exit(camera_id, sig, last_dwell, now)
            self.behaviour_engine.reid_release_track(camera_id, evicted_tid)
        timings["reid_ms"] = round((time.perf_counter() - _t_reid0) * 1000, 3)

        # Fighting's Gates 1+2 (proximity + kinetic motion) are resolved
        # here, BEFORE the whole-frame neural classifier below, so that
        # call can be skipped on the (common) frames where nobody is even
        # a geometric candidate for fighting — see _run_neural_classifier's
        # gates_1_2_pass parameter. This is pure pair-IDENTITY selection
        # (sticky: prefer continuing whichever pair we were already
        # tracking, if it still clears Gates 1+2), not Gate 3/4 evaluation
        # — that still happens in step 5c below, once fighting_box_a/b are
        # available for evidence/cropping.
        _t_gates0 = time.perf_counter()
        fighting_pair_key: Optional[Tuple[int, int]] = None
        fighting_box_a = fighting_box_b = None
        gates_1_2_pass = False

        sticky_pair_key = session.active_fighting_pair_key
        if sticky_pair_key is not None:
            by_id = {p.get("track_id"): p for p in tracked_persons if p.get("track_id") is not None}
            sa, sb = by_id.get(sticky_pair_key[0]), by_id.get(sticky_pair_key[1])
            if sa is not None and sb is not None:
                fighting_pair_key = sticky_pair_key
                fighting_box_a, fighting_box_b = sa, sb
                gates_1_2_pass = self.behaviour_engine.fighting_pair_still_qualifies_configured(sa, sb)

        if fighting_pair_key is None:
            gate12_pair = self.behaviour_engine.find_fighting_pair_configured(tracked_persons)
            if gate12_pair is not None:
                a, b = gate12_pair
                fighting_pair_key = tuple(sorted((a["track_id"], b["track_id"])))
                fighting_box_a, fighting_box_b = a, b
                gates_1_2_pass = True
        timings["gates_1_2_selection_ms"] = round((time.perf_counter() - _t_gates0) * 1000, 3)

        # ── 4. Behaviour Engine — rule-based only (see module docstring) ────
        # This single call covers ALL of _check_rules: Overcrowding
        # clustering + hysteresis, Crowd Formation, Restricted Zone,
        # Loitering (dwell/pacing/demo rules), and Running — timed as one
        # combined "behaviour math" block since that's the granularity
        # this task's bottleneck report was framed at. See the per-
        # sub-feature profiling in this task's real-video validation if a
        # SPECIFIC one of these needs to be isolated further.
        _t_behaviour0 = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]
        zones_px = self._scale_zones_to_pixels(zones, frame_w, frame_h)
        result: Optional[DetectionResult] = self.behaviour_engine.analyze_activity(
            persons_detected=person_count,
            tracked_persons=tracked_persons,
            zones=zones_px,
            camera_id=camera_id,
            now=now,
        )
        timings["behaviour_rules_ms"] = round((time.perf_counter() - _t_behaviour0) * 1000, 3)
        # Snapshot AFTER analyze_activity() has updated the per-track
        # running-persistence state for this window — see FrameAnalysis.
        # running_track_ids's docstring for why the overlay reads this
        # instead of recomputing its own speed check.
        running_track_ids = frozenset(self.behaviour_engine.get_running_track_ids(camera_id))
        loitering_track_ids = frozenset(
            p["track_id"] for p in tracked_persons
            if p.get("track_id") is not None and self.behaviour_engine.person_is_loitering(camera_id, p)
        )
        overcrowding_snapshot = self.behaviour_engine.get_overcrowding_state(camera_id, now)
        overcrowding_track_ids = frozenset(
            m["track_id"] for m in overcrowding_snapshot["members"]
            if overcrowding_snapshot["state"] == "CONFIRMED" and m.get("track_id") is not None
        )

        # ── 5. Neural Activity Classifier — display-only suggestion ─────────
        # Runs on a rolling clip buffer of this camera's recent frames. Its
        # output never overrides or gets treated the same as the rule-based
        # `result` above — see module docstring for why. The actual GPU
        # forward pass only runs when gates_1_2_pass (computed in step 3
        # above) is True — this is the SAME 16-frame ResNet-18+BiLSTM
        # checkpoint Gate 3 also uses, and on real chaotic footage it
        # measured ~18ms per call, unconditionally, on every single
        # analyzed frame regardless of whether anyone was even near anyone
        # else. Buffer maintenance (resize+append) still happens every
        # frame regardless (cheap, <1ms) so the rolling clip stays a real
        # CONTIGUOUS temporal window when the model does run — only the
        # expensive tensor forward pass is skipped, not the buffer's
        # temporal continuity.
        _t_neural0 = time.perf_counter()
        neural_label, neural_confidence, neural_probs = self._run_neural_classifier(
            session, frame, gates_1_2_pass=gates_1_2_pass
        )
        timings["neural_whole_frame_ms"] = round((time.perf_counter() - _t_neural0) * 1000, 3)

        # ── 5b. Hysteresis smoothing over the raw per-window prediction ─────
        # Converts the noisy raw output into a stable NORMAL/ANALYZING/
        # CONFIRMED state — see ai_service/activity_state.py. Every downstream
        # consumer (overlay, alerting) uses this, never the raw value alone.
        #
        # The FULL per-class probability vector is passed (not just the
        # top-1 label) so every class accumulates its own evidence
        # independently window-to-window — see activity_state.py's module
        # docstring for why a top-1-only feed previously caused sustained
        # events to be thrown away by a single noisy dip.
        #
        # Fire/Smoke are explicitly excluded here, not just from display:
        # the neural classifier still has them as 2 of its 10 trained
        # classes (a known, disclosed architectural debt — see module
        # docstring), but the dedicated fire_smoke_state pipeline above is
        # the ONLY thing allowed to confirm or alert on fire/smoke. Without
        # this filter, the behaviour state machine could itself confirm
        # "Fire" from the whole-clip classifier and generate a duplicate,
        # differently-thresholded alert for the same physical event — which
        # is exactly the "mixing" the fire/smoke spec forbids.
        behaviour_probs = {
            cls: p for cls, p in neural_probs.items() if cls.lower() not in ("normal", "fire", "smoke")
        }

        # ── 5c. Fighting — dedicated 4-Gate subsystem ────────────────────
        # Fighting is now FULLY separate from the shared ActivityStateMachine
        # (same "own dedicated subsystem" pattern already used for
        # Fire/Smoke, Vandalism, Falling, Snatching) — see config.py's
        # FIGHTING_* block and behaviour_engine.py's FightingPairState for
        # the complete 4-gate design (scale-invariant proximity, kinetic
        # motion/aspect-jitter, per-pair cropped neural confirmation, and a
        # dedicated hysteresis state machine). The whole-frame CNN's raw
        # Fighting probability is captured BEFORE being zeroed here, since
        # Vandalism below still needs it as its own aggression signal,
        # independent of whether any specific pair currently clears
        # Fighting's own geometric gates.
        raw_fighting_prob_for_vandalism = neural_probs.get("Fighting", 0.0)
        behaviour_probs["Fighting"] = 0.0  # never confirmed via the shared ActivityStateMachine anymore

        # fighting_pair_key/fighting_box_a/fighting_box_b/gates_1_2_pass
        # were already resolved in step 3 above (sticky pair-IDENTITY
        # selection), before the whole-frame neural classifier call, so it
        # could be gated on gates_1_2_pass. Gate 3/4 evaluation continues
        # here now that fighting_box_a/b are available for cropping.

        # The crop buffer's own "is this still fresh evidence" grace
        # period is deliberately SEPARATE from Gate 4's internal
        # accumulation timer. Gate 4's "idle" only turns False once gates
        # 1+2+3 have ALL passed together at least once (that's what
        # starts its confirm-trigger timer, per the 4-gate spec) — but
        # Gate 3 can only ever pass once this buffer has filled with
        # SEQUENCE_LENGTH consecutive frames of real evidence. Clearing
        # the buffer whenever Gate 4 was still "idle" was circular: the
        # buffer could never fill because it kept getting wiped every
        # single window while waiting for a gate3_prob that could only
        # exist once the buffer had already filled. Instead, only clear
        # it once no qualifying pair has been seen ANYWHERE in frame for
        # FIGHTING_PROXIMITY_FAIL_SECONDS — the same grace period Gate 4
        # already uses to decide a CONFIRMED fight has truly ended.
        if gates_1_2_pass:
            session.fighting_last_pair_seen_at = now
        elif (
            session.fighting_last_pair_seen_at is not None
            and (now - session.fighting_last_pair_seen_at) >= self.behaviour_engine.fighting_proximity_fail_seconds
        ):
            session.fighting_incident_buffer.clear()
            session.fighting_last_pair_seen_at = None

        _t_gate3_0 = time.perf_counter()
        gate3_prob = 0.0
        if fighting_box_a is not None and fighting_box_b is not None:
            gate3_prob = self._fighting_gate3_prob(session, frame, fighting_box_a, fighting_box_b)
        timings["gate3_pair_crop_ms"] = round((time.perf_counter() - _t_gate3_0) * 1000, 3)

        gate4 = self.behaviour_engine.update_fighting_incident_state(
            camera_id, now,
            gates_1_2_pass=gates_1_2_pass,
            gate3_prob=gate3_prob,
        )

        session.active_fighting_pair_key = (
            fighting_pair_key if (gates_1_2_pass or gate4["state"] == "CONFIRMED") else None
        )

        fighting_result: Optional[DetectionResult] = None
        fighting_proof: Optional[Dict[str, Any]] = None
        if gate4["state"] == "CONFIRMED":
            fighting_result = DetectionResult(
                camera_id=camera_id,
                activity="Fighting",
                confidence=gate3_prob,
                severity="critical",
                persons_detected=2,
                duration_seconds=gate4["duration_seconds"],
                timestamp=datetime.utcnow(),
                metadata={
                    "rule": "fighting_4gate",
                    "peak_confidence": gate4["peak_confidence"],
                    "average_confidence": gate4["average_confidence"],
                    "track_ids": list(fighting_pair_key) if fighting_pair_key else [],
                },
            )
            if fighting_box_a is not None and fighting_box_b is not None:
                fighting_proof = {
                    "type": "fighting", "box_a": fighting_box_a, "box_b": fighting_box_b,
                    "confidence": gate3_prob,
                }

        # ── 5d. Geometric + motionlessness gate for "Falling" ────────────
        # Zero the CNN's Falling probability unless some tracked person is
        # both in a horizontal posture (low aspect ratio) AND has stayed
        # motionless since that posture appeared for the configured
        # duration — a plain CNN spike from someone bending to tie a shoe
        # or pick something up doesn't survive this, since they straighten
        # up or keep moving well before the motionless window elapses. The
        # explicit FALLING_CNN_THRESHOLD floor is applied on top (currently
        # redundant with the stricter global confirm threshold — see
        # config.py's comment on it).
        if behaviour_probs.get("Falling", 0.0) > 0.0 and not (
            behaviour_probs["Falling"] > AIConfig.FALLING_CNN_THRESHOLD
            and self.behaviour_engine.check_falling_candidate(camera_id, now, tracked_persons)
        ):
            behaviour_probs["Falling"] = 0.0

        # ── 5e. Contact + departure gate for "Snatching" ─────────────────
        # Zero the CNN's Snatching probability unless two tracked persons
        # were recently in contact and one has since departed fast enough,
        # soon enough after that contact, to look like a grab-and-run
        # rather than a handshake or a passing brush. find_snatch_divergence
        # is STATEFUL (tracks recent contacts) — called at most once per
        # window, never twice, so its internal bookkeeping isn't double-
        # advanced. The actual divergence proof (track/direction) is kept
        # for the overlay, same reasoning as Fighting's own proof above.
        snatch_proof = None
        if behaviour_probs.get("Snatching", 0.0) > 0.0:
            snatch_proof = self.behaviour_engine.find_snatch_divergence(camera_id, now, tracked_persons)
            if not (behaviour_probs["Snatching"] > AIConfig.SNATCHING_CNN_THRESHOLD and snatch_proof is not None):
                behaviour_probs["Snatching"] = 0.0
                snatch_proof = None

        state_result, ended_activity_events, active_activity_events = session.activity_state.update(behaviour_probs, now=now)

        # ── 5f. Overlay evidence — "what the AI is showing its work with" ──
        # Captures the SPECIFIC geometric proof behind whichever gated class
        # is currently active, so the render layer can draw it (connecting
        # line / velocity arrow / asset-zone highlight) instead of a generic
        # box — see render()'s docstring on why an alert with no visible
        # "why" reads as a false alarm to a guard. Fighting's proof comes
        # from its own dedicated Gate 1-4 pipeline above (step 5c), not
        # `state_result` (Fighting no longer flows through the shared
        # ActivityStateMachine at all). Vandalism's own proof lives in its
        # DetectionResult.metadata instead (populated after Vandalism is
        # actually checked, further down in this method) since that gate
        # runs later, once `result` is available.
        behaviour_proof: Optional[Dict[str, Any]] = fighting_proof
        if behaviour_proof is None and state_result.label == "Snatching" and state_result.state != NORMAL and snatch_proof is not None:
            behaviour_proof = {"type": "snatching", **snatch_proof}

        # ── 6. Fire/Smoke Alert Generation — SEPARATE from behaviour alerts ──
        # (own AlertGenerator instance, own thresholds). EDGE-TRIGGERED on
        # event_started/event_ended (see fire_smoke_state.py): exactly ONE
        # alert row is created per real incident, refreshed in place while
        # it stays open, and finalized (not re-created) when it closes.
        #
        # Fire and smoke are combined into ONE application-level event —
        # "Fire/Smoke Detected" — fire OR smoke OR both all map to the
        # same category (see backend/app/constants.py's FIRE_SMOKE_ACTIVITIES
        # and the combined-event construction below). The two underlying
        # FireSmokeStateMachine instances (fire_state_result/
        # smoke_state_result, computed above — UNCHANGED, still fully
        # independent per-class hysteresis/thresholds) are NOT touched by
        # this; only how their results are turned into an alert/event is
        # combined. Per-class detail (fire_detected/smoke_detected) is
        # preserved in the alert's metadata (see combined_metadata below),
        # not discarded — see AlertEvent.metadata and
        # SecurityEvent.metadata_json.
        #
        # combined_state picks whichever of the two sub-signals is
        # currently "most active" (CONFIRMED > RECOVERY > POSSIBLE >
        # NORMAL) purely for display/logging; is_alertable/event_started/
        # event_ended are OR-combined against the single "Fire/Smoke
        # Detected" open-event slot (session.open_fire_smoke_events now
        # only ever has this one key) — an event is open if EITHER
        # sub-signal is confirmed/recovering, and only closes once BOTH
        # have dropped out, matching "fire_detected OR smoke_detected".
        _fs_state_rank = {FS_NORMAL: 0, FS_POSSIBLE: 1, FS_RECOVERY: 2, FS_CONFIRMED: 3}
        fire_active = fire_state_result.state in (FS_CONFIRMED, FS_RECOVERY)
        smoke_active = smoke_state_result.state in (FS_CONFIRMED, FS_RECOVERY)
        combined_currently_open = "Fire/Smoke Detected" in session.open_fire_smoke_events
        combined_is_alertable = fire_state_result.is_alertable or smoke_state_result.is_alertable
        combined_active = fire_active or smoke_active
        combined_state = (
            fire_state_result.state
            if _fs_state_rank[fire_state_result.state] >= _fs_state_rank[smoke_state_result.state]
            else smoke_state_result.state
        )
        combined_result = FireSmokeStateResult(
            state=combined_state,
            raw_confidence=max(fire_state_result.raw_confidence, smoke_state_result.raw_confidence),
            smoothed_confidence=max(fire_state_result.smoothed_confidence, smoke_state_result.smoothed_confidence),
            is_alertable=combined_is_alertable,
            duration_seconds=max(fire_state_result.duration_seconds, smoke_state_result.duration_seconds),
            peak_confidence=max(fire_state_result.peak_confidence, smoke_state_result.peak_confidence),
            average_confidence=max(fire_state_result.average_confidence, smoke_state_result.average_confidence),
            start_time=fire_state_result.start_time if fire_active else smoke_state_result.start_time,
            event_started=combined_is_alertable and not combined_currently_open,
            event_ended=combined_currently_open and not combined_active,
        )
        # Severity: "critical" if fire is part of the active signal (fire
        # is more dangerous than smoke alone), else "warning" for a
        # smoke-only event — matches the original per-class severities.
        combined_severity = "critical" if fire_active else "warning"

        # Cumulative fire/smoke detail for the OPEN incident (see the
        # session field's docstring) — reset on a genuinely new incident,
        # OR-accumulated every frame it stays open, so an alert whose
        # smoke confirmed a few frames before fire did (each class runs
        # its own independent 12-window confirm) still ends up correctly
        # showing fire_detected=True once fire catches up, instead of
        # freezing whatever was true at the exact trigger frame.
        if combined_result.event_started:
            session.fire_smoke_open_fire_ever = False
            session.fire_smoke_open_smoke_ever = False
        if combined_active:
            session.fire_smoke_open_fire_ever = session.fire_smoke_open_fire_ever or fire_active
            session.fire_smoke_open_smoke_ever = session.fire_smoke_open_smoke_ever or smoke_active

        # Internal detection detail preserved for the combined event (per
        # the response-shape example in the task spec) — never discarded,
        # just no longer split into two separate alert rows.
        combined_metadata = {
            "event_type": "fire_smoke",
            "fire_detected": session.fire_smoke_open_fire_ever,
            "smoke_detected": session.fire_smoke_open_smoke_ever,
        }

        fire_smoke_new_alerts: List[AlertEvent] = []
        fire_smoke_event_updates: List[Dict[str, Any]] = []
        for label, fs_result, severity in (
            ("Fire/Smoke Detected", combined_result, combined_severity),
        ):
            # ── Debug logging — transitions and edges only, never every
            # window, per the "avoid excessive output" requirement.
            display_state = FS_DISPLAY_NAME.get(fs_result.state, fs_result.state)
            if fs_result.state != session.fire_smoke_last_logged_state.get(label):
                logger.info(
                    f"[FireSmoke] camera={camera_id} fire={fire_conf:.2f} smoke={smoke_conf:.2f} "
                    f"label={label} state={display_state}"
                )
                session.fire_smoke_last_logged_state[label] = fs_result.state
            if fs_result.event_started:
                logger.info(f"[FireSmoke] camera={camera_id} {label.upper()} EVENT STARTED")
            if fs_result.event_ended:
                logger.info(
                    f"[FireSmoke] camera={camera_id} {label.upper()} EVENT ENDED "
                    f"duration={fs_result.duration_seconds:.1f}s"
                )

            if fs_result.event_started and fs_result.is_alertable:
                closed_at = session.fire_smoke_closed_at.get(label)
                if closed_at is not None and now - closed_at < AIConfig.RETRIGGER_COOLDOWN_SECONDS:
                    # Within the retrigger cooldown of the last close —
                    # reopen that same row instead of creating a new one.
                    session.open_fire_smoke_events[label] = now
                    fire_smoke_event_updates.append({
                        "action": "reopen", "camera_id": camera_id, "activity": label,
                        "peak_confidence": fs_result.peak_confidence,
                        "average_confidence": fs_result.average_confidence,
                        "duration_seconds": fs_result.duration_seconds,
                        "metadata": combined_metadata,
                    })
                else:
                    new_fs_alert = self.fire_smoke_alert_generator.generate_alert(
                        camera_id=camera_id, camera_name=session.camera_name,
                        camera_location=session.camera_location, activity=label,
                        confidence=fs_result.smoothed_confidence, severity=severity,
                        persons_detected=0, duration_seconds=fs_result.duration_seconds,
                        peak_confidence=fs_result.peak_confidence,
                        average_confidence=fs_result.average_confidence,
                        start_time=fs_result.start_time,
                        metadata=combined_metadata,
                    )
                    if new_fs_alert is not None:
                        session.open_fire_smoke_events[label] = now
                        fire_smoke_new_alerts.append(new_fs_alert)
            elif (
                label in session.open_fire_smoke_events
                and fs_result.state in (FS_CONFIRMED, FS_RECOVERY)
                and now - session.open_fire_smoke_events[label] >= AIConfig.BEHAVIOUR_EVENT_REFRESH_SECONDS
            ):
                session.open_fire_smoke_events[label] = now
                fire_smoke_event_updates.append({
                    "action": "refresh", "camera_id": camera_id, "activity": label,
                    "peak_confidence": fs_result.peak_confidence,
                    "average_confidence": fs_result.average_confidence,
                    "duration_seconds": fs_result.duration_seconds,
                    "metadata": combined_metadata,
                })
            if fs_result.event_ended:
                session.fire_smoke_closed_at[label] = now
                if session.open_fire_smoke_events.pop(label, None) is not None:
                    fire_smoke_event_updates.append({
                        "action": "finalize", "camera_id": camera_id, "activity": label,
                        "peak_confidence": fs_result.peak_confidence,
                        "average_confidence": fs_result.average_confidence,
                        "duration_seconds": fs_result.duration_seconds,
                        "metadata": combined_metadata,
                    })

        fs_alert: Optional[AlertEvent] = fire_smoke_new_alerts[0] if fire_smoke_new_alerts else None
        extra_fire_smoke_alerts: List[AlertEvent] = fire_smoke_new_alerts[1:]

        # Fighting (own dedicated 4-Gate result, computed in step 5c) takes
        # the rule-result slot if nothing else has already filled it this
        # window — same one-result-per-window convention as every other
        # rule here.
        if result is None:
            result = fighting_result

        # Vandalism (hybrid gate) — needs the CNN's Kicking/Fighting
        # probability, only available now that step 5 has run; checked here
        # rather than inside behaviour_engine.analyze_activity() (step 4),
        # which runs BEFORE the neural classifier. Only evaluated if no
        # other rule-based result already fired this window (same
        # one-result-per-window behaviour the other rules already have).
        # Uses raw_fighting_prob_for_vandalism (the whole-frame CNN's own
        # Fighting probability, captured in step 5c BEFORE it was zeroed
        # there) rather than behaviour_probs["Fighting"], which is now
        # unconditionally 0.0 — Vandalism's aggression signal is
        # deliberately independent of whether any specific pair currently
        # clears Fighting's own geometric gates.
        if result is None:
            result = self.behaviour_engine.check_vandalism(
                camera_id, now, tracked_persons, zones_px,
                kicking_prob=behaviour_probs.get("Kicking", 0.0),
                fighting_prob=raw_fighting_prob_for_vandalism,
            )
            if result is not None:
                # Vandalism's proof lives in DetectionResult.metadata (zone
                # polygon, offending track's box, which CNN signal fired) —
                # see check_vandalism()'s docstring.
                behaviour_proof = {"type": "vandalism", "duration_seconds": result.duration_seconds, **result.metadata}

        alert: Optional[AlertEvent] = None
        if result:
            alert = self.alert_generator.generate_alert(
                camera_id=camera_id, camera_name=session.camera_name,
                camera_location=session.camera_location, activity=result.activity,
                confidence=result.confidence, severity=result.severity,
                persons_detected=person_count, duration_seconds=result.duration_seconds,
            )

        # Neural/behaviour event lifecycle — EDGE-TRIGGERED on the state
        # machine's own event_started/event_ended signals (see
        # ai_service/activity_state.py), not fired on every qualifying
        # window. The old design called generate_alert() every single
        # window the smoothed confidence cleared REVIEW_THRESHOLD, relying
        # entirely on AlertGenerator's wall-clock cooldown (30s) to avoid
        # duplicates — which meant a SUSTAINED event lasting longer than the
        # cooldown could spawn a second, spurious "new" alert mid-event, and
        # a GENUINELY new event of the same activity within the cooldown
        # window of a prior (already-ended) one could be silently dropped.
        # Tying creation to `event_started` fixes both: exactly one Alert
        # per real event, and back-to-back distinct events are never merged
        # or suppressed just because they're less than a cooldown apart.
        #
        # CRITICAL: this iterates `active_activity_events` — EVERY currently
        # non-NORMAL class — not just `state_result` (the single "best"
        # one). A real bug found on an actual 40s CCTV fighting video: a
        # sustained, high-confidence "Loitering" confirmation permanently
        # won the "best" slot for nearly the entire clip, so a SEPARATE,
        # independently high-confidence "Fighting" confirmation happening
        # at the same time could never surface — it would have needed to
        # numerically outrank Loitering's already-near-1.0 smoothed
        # confidence, which has nothing to do with whether Fighting's OWN
        # evidence cleared its OWN threshold. Iterating every active class
        # means each one's alert lifecycle is judged purely on its own
        # evidence, never on how it compares to a completely different
        # class. See ai_service/activity_state.py's `update()` docstring.
        #
        # `neural_event_updates` (create/refresh/finalize) is separate from
        # a new Alert row + live broadcast — refresh/finalize only ever
        # UPDATE an already-created row's peak/average/end-time as the
        # event continues or closes, handled by
        # alert_dispatcher.dispatch_event_update_sync (see
        # video_stream_service.py / video_timeline_processor.py, which both
        # call it exactly like they already call dispatch_alert_sync).
        neural_alerts: List[AlertEvent] = []
        neural_event_updates: List[Dict[str, Any]] = []
        if not fs_alert and not alert:
            for evt in active_activity_events:
                label = evt.label
                if not label:
                    continue
                if evt.event_started and evt.is_alertable:
                    new_alert = self.alert_generator.generate_alert(
                        camera_id=camera_id, camera_name=session.camera_name,
                        camera_location=session.camera_location, activity=label,
                        confidence=evt.smoothed_confidence,
                        severity=AIConfig.SEVERITY_MAP.get(label, "warning"),
                        persons_detected=person_count, duration_seconds=evt.duration_seconds,
                        peak_confidence=evt.peak_confidence,
                        average_confidence=evt.average_confidence,
                        start_time=evt.start_time,
                    )
                    if new_alert is None:
                        # Only possible if an UNRELATED earlier alert for
                        # this exact (camera, activity) pair is still
                        # inside its cooldown — a freshly-STARTED event
                        # must never be silently dropped just because of
                        # that timer.
                        self.alert_generator.clear_cooldown(camera_id, label)
                        new_alert = self.alert_generator.generate_alert(
                            camera_id=camera_id, camera_name=session.camera_name,
                            camera_location=session.camera_location, activity=label,
                            confidence=evt.smoothed_confidence,
                            severity=AIConfig.SEVERITY_MAP.get(label, "warning"),
                            persons_detected=person_count, duration_seconds=evt.duration_seconds,
                            peak_confidence=evt.peak_confidence,
                            average_confidence=evt.average_confidence,
                            start_time=evt.start_time,
                        )
                    if new_alert is not None:
                        session.open_behaviour_events[label] = now
                        neural_alerts.append(new_alert)
                elif (
                    label in session.open_behaviour_events
                    and evt.state in (CONFIRMED, RECOVERY)
                    and now - session.open_behaviour_events[label] >= AIConfig.BEHAVIOUR_EVENT_REFRESH_SECONDS
                ):
                    session.open_behaviour_events[label] = now
                    neural_event_updates.append({
                        "action": "refresh", "camera_id": camera_id, "activity": label,
                        "peak_confidence": evt.peak_confidence,
                        "average_confidence": evt.average_confidence,
                        "duration_seconds": evt.duration_seconds,
                    })

        for ended in ended_activity_events:
            if ended.label and session.open_behaviour_events.pop(ended.label, None) is not None:
                neural_event_updates.append({
                    "action": "finalize", "camera_id": camera_id, "activity": ended.label,
                    "peak_confidence": ended.peak_confidence,
                    "average_confidence": ended.average_confidence,
                    "duration_seconds": ended.duration_seconds,
                })

        # Almost always at most one new neural alert per window (two
        # DIFFERENT classes both hitting event_started on the exact same
        # window is rare) — `final_alert`/FrameAnalysis.alert stays the
        # single "featured" one for immediate overlay/telemetry, any
        # others still get fully persisted via `additional_alerts` below so
        # none are ever silently dropped.
        primary_neural_alert = neural_alerts[0] if neural_alerts else None
        extra_neural_alerts = neural_alerts[1:]
        final_alert = fs_alert or alert or primary_neural_alert
        # Fire's and Smoke's own event_started can both fire on the exact
        # same window (rare, but possible) — `fs_alert` above already picked
        # the first as the "featured" one; the other (if any) still needs
        # to be persisted, same as extra_neural_alerts.
        all_additional_alerts = extra_neural_alerts + extra_fire_smoke_alerts

        proc_time_ms = round((time.time() - start_time) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - _t0) * 1000, 3)
        session.last_stage_timings_ms = timings
        logger.debug(
            f"Camera {camera_id} stage timings (ms): "
            f"detect(pre={timings['detect_preprocess_ms']}, "
            f"infer={timings['detect_inference_ms']}, "
            f"post={timings['detect_postprocess_ms']}) "
            f"tracker={timings['tracker_ms']} "
            f"reid={timings['reid_ms']} "
            f"gates1+2={timings['gates_1_2_selection_ms']} "
            f"behaviour_rules(loiter+crowd+overcrowd+zone+run)={timings['behaviour_rules_ms']} "
            f"neural_whole_frame(CNN)={timings['neural_whole_frame_ms']} "
            f"gate3(CNN)={timings['gate3_pair_crop_ms']} "
            f"TOTAL={timings['total_ms']}"
        )
        behaviour_label = result.activity if result else None
        hazard_label = (
            "FIRE DETECTED" if fire_detected else
            "SMOKE DETECTED" if smoke_detected else
            (behaviour_label or "Normal")
        )
        telemetry = {
            "camera_id": camera_id,
            "persons_detected": person_count,
            "activity": hazard_label,
            "confidence": round(max((result.confidence if result else 0.0), fire_conf, smoke_conf), 2),
            "fire_detected": fire_detected,
            "smoke_detected": smoke_detected,
            "fire_confidence": round(fire_conf, 2),
            "smoke_confidence": round(smoke_conf, 2),
            "processing_time_ms": proc_time_ms,
            "detector_backend": self.detector.backend,
            "neural_activity": neural_label or "N/A",
            "neural_confidence": round(neural_confidence, 2),
            "activity_state": state_result.state,
            "activity_smoothed_confidence": state_result.smoothed_confidence,
            "fire_state": fire_state_result.state,
            "smoke_state": smoke_state_result.state,
            "fire_smoothed_confidence": fire_state_result.smoothed_confidence,
            "smoke_smoothed_confidence": smoke_state_result.smoothed_confidence,
        }

        return FrameAnalysis(
            tracked_persons=tracked_persons,
            fire_smoke_detections=fire_smoke_detections,
            fire_detected=fire_detected,
            smoke_detected=smoke_detected,
            fire_conf=fire_conf,
            smoke_conf=smoke_conf,
            fire_state=fire_state_result.state,
            smoke_state=smoke_state_result.state,
            fire_smoothed_confidence=fire_state_result.smoothed_confidence,
            smoke_smoothed_confidence=smoke_state_result.smoothed_confidence,
            fire_duration_seconds=fire_state_result.duration_seconds,
            smoke_duration_seconds=smoke_state_result.duration_seconds,
            person_count=person_count,
            behaviour_label=behaviour_label,
            behaviour_confidence=(result.confidence if result else 0.0),
            neural_label=neural_label,
            activity_state=state_result.state,
            activity_smoothed_confidence=state_result.smoothed_confidence,
            is_alertable=state_result.is_alertable,
            neural_confidence=neural_confidence,
            alert=final_alert,
            additional_alerts=all_additional_alerts,
            behaviour_event_updates=neural_event_updates,
            fire_smoke_event_updates=fire_smoke_event_updates,
            telemetry=telemetry,
            zones_px=zones_px,
            behaviour_proof=behaviour_proof,
            running_track_ids=running_track_ids,
            zone_dwell=self.behaviour_engine.get_zone_dwell(camera_id),
            loitering_track_ids=loitering_track_ids,
            overcrowding_track_ids=overcrowding_track_ids,
        )

    def _run_neural_classifier(
        self, session: "CameraInferenceSession", frame: np.ndarray, gates_1_2_pass: bool = True
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Push this frame into the camera's rolling clip buffer and, once full,
        run the neural classifier. Returns the cached (label, confidence,
        per_class_probs) every call — the buffer fills gradually, so most
        calls just return whatever the last full-clip classification
        produced. `per_class_probs` is the FULL softmax vector (every class,
        not just the top-1 winner) — ActivityStateMachine needs it to track
        each class's evidence independently (see its module docstring).

        `gates_1_2_pass`: whether Fighting's Gates 1+2 (proximity + kinetic
        motion) currently flag some pair as a candidate. Buffer maintenance
        (resize+append) ALWAYS happens regardless — the rolling clip must
        stay a genuine contiguous temporal window for the model to see
        real motion, not a temporally-gapped one — but the actual GPU
        forward pass (the expensive part, ~18ms measured on real chaotic
        footage) is skipped when False, since this checkpoint's only
        non-"Normal" class is "Fighting" and Gates 1+2 not finding any
        pair anywhere in frame makes that call's outcome a near-certain
        "Normal" anyway. Skipped calls simply return the last cached
        result, same as while the buffer is still filling.
        """
        if self.neural_model is None:
            return None, 0.0, {}

        try:
            small_rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), AIConfig.IMAGE_SIZE)
            session.frame_buffer.append(small_rgb)
        except Exception:
            return session.cached_neural_label, session.cached_neural_confidence, session.cached_neural_probs

        if len(session.frame_buffer) < AIConfig.SEQUENCE_LENGTH:
            return session.cached_neural_label, session.cached_neural_confidence, session.cached_neural_probs

        if not gates_1_2_pass:
            return session.cached_neural_label, session.cached_neural_confidence, session.cached_neural_probs

        try:
            import torch
            clip = np.array(session.frame_buffer, dtype=np.float32) / 255.0
            clip = np.transpose(clip, (0, 3, 1, 2))  # (seq, 3, H, W)
            tensor = torch.from_numpy(clip).unsqueeze(0).to(self.neural_device)
            with torch.inference_mode():
                use_amp = self.neural_device.startswith("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.neural_model(tensor)
                    probs = torch.softmax(logits.float(), dim=1)[0]
                top_idx = int(torch.argmax(probs).item())
                confidence = float(probs[top_idx].item())
            session.cached_neural_label = self.neural_classes[top_idx]
            session.cached_neural_confidence = round(confidence, 3)
            session.cached_neural_probs = {
                cls: round(float(probs[i].item()), 3) for i, cls in enumerate(self.neural_classes)
            }
        except Exception as e:
            logger.debug(f"Neural classifier inference error: {e}")

        return session.cached_neural_label, session.cached_neural_confidence, session.cached_neural_probs

    def _fighting_gate3_prob(
        self,
        session: "CameraInferenceSession",
        frame: np.ndarray,
        person_a: Dict[str, Any],
        person_b: Dict[str, Any],
    ) -> float:
        """
        GATE 3 — neural confirmation on the specific pair-region currently
        flagged by Gates 1+2, not the whole frame (see config.py's
        FIGHTING_CROP_PADDING_FRACTION/FIGHTING_NEURAL_CONFIRM_THRESHOLD).
        Crops the union of both boxes (padded by
        FIGHTING_CROP_PADDING_FRACTION on each side for context), builds a
        rolling SEQUENCE_LENGTH-frame clip in session.fighting_incident_buffer
        (same preprocessing as _run_neural_classifier's whole-frame buffer,
        just a different crop source), and runs it through the SAME
        deployed checkpoint independently — a scene-level Fighting
        probability can be dominated by unrelated activity elsewhere in
        frame; this asks specifically "is fighting happening in THIS
        region". Returns 0.0 (not yet confirmable) while the buffer is
        still filling, or if the model isn't available.

        Deliberately keyed by CAMERA, not by the specific track_id pair:
        on real chaotic multi-person footage (heavy occlusion, fast
        erratic movement), the exact track_id pair that Gates 1+2 pick as
        "who's fighting" was observed to change identity every few
        hundred milliseconds — no single pair ever stayed the same ID
        combination long enough (SEQUENCE_LENGTH=16 frames, ~2s at this
        analysis rate) to fill a per-pair buffer, even though the same
        physical fight was visibly, continuously happening in roughly the
        same spot the whole time. Accumulating frames for "the region
        Gates 1+2 are currently flagging in this camera" instead of "this
        exact pair of numeric IDs" tracks the real, continuous physical
        event instead of the tracker's unstable identity labels for it.
        The specific box_a/box_b passed in are still used as the drawn
        evidence for whichever pair is currently flagged.
        """
        if self.neural_model is None:
            return 0.0

        bbox_a, bbox_b = person_a.get("bbox"), person_b.get("bbox")
        if not bbox_a or not bbox_b:
            return 0.0

        x1 = min(bbox_a[0], bbox_b[0])
        y1 = min(bbox_a[1], bbox_b[1])
        x2 = max(bbox_a[2], bbox_b[2])
        y2 = max(bbox_a[3], bbox_b[3])
        pad_x = (x2 - x1) * AIConfig.FIGHTING_CROP_PADDING_FRACTION
        pad_y = (y2 - y1) * AIConfig.FIGHTING_CROP_PADDING_FRACTION
        h, w = frame.shape[:2]
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.0

        crop = frame[cy1:cy2, cx1:cx2]
        buffer = session.fighting_incident_buffer

        try:
            small_rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), AIConfig.IMAGE_SIZE)
            buffer.append(small_rgb)
        except Exception:
            return 0.0

        if len(buffer) < AIConfig.SEQUENCE_LENGTH:
            return 0.0

        try:
            import torch
            clip = np.array(buffer, dtype=np.float32) / 255.0
            clip = np.transpose(clip, (0, 3, 1, 2))
            tensor = torch.from_numpy(clip).unsqueeze(0).to(self.neural_device)
            with torch.inference_mode():
                use_amp = self.neural_device.startswith("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.neural_model(tensor)
                    probs = torch.softmax(logits.float(), dim=1)[0]
            fighting_idx = self.neural_classes.index("Fighting") if "Fighting" in self.neural_classes else None
            if fighting_idx is None:
                return 0.0
            return round(float(probs[fighting_idx].item()), 3)
        except Exception as e:
            logger.debug(f"Fighting Gate 3 (pair-crop) inference error: {e}")
            return 0.0

    @staticmethod
    def _scale_zones_to_pixels(zones: Optional[List[Dict]], frame_w: int, frame_h: int) -> List[Dict]:
        """
        Camera zones are stored as normalized (0-1) fractions of frame width
        /height so they stay correct regardless of a camera's resolution.
        Convert to this frame's actual pixel coordinates before using them
        for point-in-polygon tests or drawing.
        """
        if not zones:
            return []
        scaled = []
        for zone in zones:
            pts = zone.get("polygon_points", [])
            scaled_pts = [
                {"x": p.get("x", 0) * frame_w, "y": p.get("y", 0) * frame_h}
                for p in pts
            ]
            scaled.append({**zone, "polygon_points": scaled_pts})
        return scaled

    def render(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        camera_name: str = "",
    ) -> np.ndarray:
        """
        Draw overlays for the given (possibly slightly stale) analysis onto
        a fresh copy of `frame`. Cheap — safe to call on every displayed
        frame even when `analysis` was computed a few frames ago. Zones are
        read from `analysis.zones_px` (already pixel-scaled at analyze()
        time), so overlay and detection always agree on zone geometry.
        """
        output_frame = frame.copy()
        self._draw_rich_overlays(
            frame=output_frame,
            tracked_persons=analysis.tracked_persons,
            fire_smoke_detections=analysis.fire_smoke_detections,
            fire_detected=analysis.fire_detected,
            smoke_detected=analysis.smoke_detected,
            fire_conf=analysis.fire_conf,
            smoke_conf=analysis.smoke_conf,
            fire_state=analysis.fire_state,
            smoke_state=analysis.smoke_state,
            fire_smoothed_confidence=analysis.fire_smoothed_confidence,
            smoke_smoothed_confidence=analysis.smoke_smoothed_confidence,
            fire_duration_seconds=analysis.fire_duration_seconds,
            smoke_duration_seconds=analysis.smoke_duration_seconds,
            person_count=analysis.person_count,
            behaviour_label=analysis.behaviour_label,
            behaviour_confidence=analysis.behaviour_confidence,
            neural_label=analysis.neural_label,
            neural_confidence=analysis.neural_confidence,
            activity_state=analysis.activity_state,
            activity_smoothed_confidence=analysis.activity_smoothed_confidence,
            proc_time_ms=analysis.telemetry.get("processing_time_ms", 0.0),
            detector_backend=self.detector.backend,
            camera_name=camera_name,
            zones=analysis.zones_px,
            model_version=f"acc{self.neural_accuracy:.0%}" if self.neural_model is not None else "none",
            neural_device=self.neural_device,
            behaviour_proof=analysis.behaviour_proof,
            running_track_ids=analysis.running_track_ids,
            zone_dwell=analysis.zone_dwell,
            loitering_track_ids=analysis.loitering_track_ids,
            overcrowding_track_ids=analysis.overcrowding_track_ids,
        )
        return output_frame

    def render_behaviour_only(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        camera_name: str = "",
    ) -> np.ndarray:
        """
        The Live Monitoring feed. Same behaviour drawing as render(), but
        fire/smoke results are never passed to the fire/smoke-drawing
        branches (include_fire_smoke=False) — so even though `analysis`
        still carries real fire/smoke detections, this view structurally
        cannot render them. See the mandate to keep Live Monitoring and
        Fire & Smoke completely separate at the overlay level.
        """
        output_frame = frame.copy()
        self._draw_rich_overlays(
            frame=output_frame,
            tracked_persons=analysis.tracked_persons,
            fire_smoke_detections=analysis.fire_smoke_detections,
            fire_detected=analysis.fire_detected,
            smoke_detected=analysis.smoke_detected,
            fire_conf=analysis.fire_conf,
            smoke_conf=analysis.smoke_conf,
            fire_state=analysis.fire_state,
            smoke_state=analysis.smoke_state,
            fire_smoothed_confidence=analysis.fire_smoothed_confidence,
            smoke_smoothed_confidence=analysis.smoke_smoothed_confidence,
            fire_duration_seconds=analysis.fire_duration_seconds,
            smoke_duration_seconds=analysis.smoke_duration_seconds,
            person_count=analysis.person_count,
            behaviour_label=analysis.behaviour_label,
            behaviour_confidence=analysis.behaviour_confidence,
            neural_label=analysis.neural_label,
            neural_confidence=analysis.neural_confidence,
            activity_state=analysis.activity_state,
            activity_smoothed_confidence=analysis.activity_smoothed_confidence,
            proc_time_ms=analysis.telemetry.get("processing_time_ms", 0.0),
            detector_backend=self.detector.backend,
            camera_name=camera_name,
            zones=analysis.zones_px,
            model_version=f"acc{self.neural_accuracy:.0%}" if self.neural_model is not None else "none",
            neural_device=self.neural_device,
            include_fire_smoke=False,
            behaviour_proof=analysis.behaviour_proof,
            running_track_ids=analysis.running_track_ids,
            zone_dwell=analysis.zone_dwell,
            loitering_track_ids=analysis.loitering_track_ids,
            overcrowding_track_ids=analysis.overcrowding_track_ids,
        )
        return output_frame

    def render_fire_smoke_only(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        camera_name: str = "",
    ) -> np.ndarray:
        """
        Same underlying detection/state as render(), but draws ONLY the
        fire/smoke overlay — no person boxes, trajectories, behaviour HUD,
        zones, or "AI:" suggestion strip. For the dedicated Fire & Smoke
        tab, which should show fire/smoke evidence with nothing else
        competing for attention on the frame.
        """
        output_frame = frame.copy()
        self._draw_fire_smoke_only_overlays(
            frame=output_frame,
            fire_smoke_detections=analysis.fire_smoke_detections,
            fire_state=analysis.fire_state,
            smoke_state=analysis.smoke_state,
            fire_smoothed_confidence=analysis.fire_smoothed_confidence,
            smoke_smoothed_confidence=analysis.smoke_smoothed_confidence,
            fire_duration_seconds=analysis.fire_duration_seconds,
            smoke_duration_seconds=analysis.smoke_duration_seconds,
            camera_name=camera_name,
        )
        return output_frame

    @staticmethod
    def _draw_fire_smoke_only_overlays(
        frame: np.ndarray,
        fire_smoke_detections: List[Dict],
        fire_state: str,
        smoke_state: str,
        fire_smoothed_confidence: float,
        smoke_smoothed_confidence: float,
        fire_duration_seconds: float,
        smoke_duration_seconds: float,
        camera_name: str,
    ) -> None:
        """Self-contained — deliberately does not reuse _draw_rich_overlays'
        nested helpers, to keep this subsystem's rendering independent."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        CLR_FIRE  = (0, 80, 255)
        CLR_SMOKE = (200, 200, 200)
        CLR_WHITE = (255, 255, 255)
        CLR_BLACK = (0, 0, 0)
        CLR_TEAL  = (0, 220, 140)

        def _semi(pt1, pt2, color, alpha=0.55):
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            overlay[:] = frame

        def _label_bg(x, y, text, font_scale, thickness, fg, bg, pad=4):
            (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
            cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl), bg, -1)
            cv2.putText(frame, text, (x, y), FONT, font_scale, fg, thickness, cv2.LINE_AA)

        # ── Minimal top bar: camera name + LIVE dot only ────────────────────
        bar_h = 30
        _semi((0, 0), (w, bar_h), (10, 14, 24), alpha=0.85)
        is_hazard = fire_state == FS_CONFIRMED or smoke_state == FS_CONFIRMED
        dot_color = CLR_FIRE if is_hazard else CLR_TEAL
        cv2.circle(frame, (16, bar_h // 2), 6, dot_color, -1)
        cv2.circle(frame, (16, bar_h // 2), 7, CLR_WHITE, 1)
        cv2.putText(frame, camera_name or "CAMERA", (32, 20), FONT, 0.42, CLR_WHITE, 1, cv2.LINE_AA)
        status_txt = "FIRE/SMOKE MONITORING"
        (stw, _), _ = cv2.getTextSize(status_txt, FONT, 0.38, 1)
        cv2.putText(frame, status_txt, (w - stw - 12, 20), FONT, 0.38, (140, 160, 180), 1, cv2.LINE_AA)

        # ── Fire/Smoke bounding boxes — state-aware wording, own colors ─────
        both_confirmed = fire_state == FS_CONFIRMED and smoke_state == FS_CONFIRMED
        for det in fire_smoke_detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            if det["class_name"] == "fire":
                word = {FS_POSSIBLE: "FIRE?", FS_CONFIRMED: "FIRE DETECTED", FS_RECOVERY: "FIRE (fading)"}.get(fire_state, "FIRE?")
                label = f"{word} {fire_smoothed_confidence:.0%}" if not both_confirmed else f"FIRE+SMOKE {fire_smoothed_confidence:.0%}"
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 20, 180), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_FIRE, 2 if fire_state == FS_CONFIRMED else 1)
                _label_bg(x1, y1 - 6, label, 0.5, 2, CLR_WHITE, CLR_FIRE, pad=4)
            else:
                word = {FS_POSSIBLE: "SMOKE?", FS_CONFIRMED: "SMOKE DETECTED", FS_RECOVERY: "SMOKE (fading)"}.get(smoke_state, "SMOKE?")
                label = f"{word} {smoke_smoothed_confidence:.0%}" if not both_confirmed else f"FIRE+SMOKE {smoke_smoothed_confidence:.0%}"
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (120, 120, 120), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_SMOKE, 2 if smoke_state == FS_CONFIRMED else 1)
                _label_bg(x1, y1 - 6, label, 0.5, 2, CLR_BLACK, CLR_SMOKE, pad=4)

        # ── Bottom hazard strip — CONFIRMED only ────────────────────────────
        if fire_state == FS_CONFIRMED or smoke_state == FS_CONFIRMED:
            if both_confirmed:
                strip_txt = f"!! FIRE + SMOKE (Fire {fire_smoothed_confidence:.0%} / Smoke {smoke_smoothed_confidence:.0%}) !!"
                strip_clr, text_clr = CLR_FIRE, CLR_WHITE
            elif fire_state == FS_CONFIRMED:
                strip_txt = f"!! FIRE DETECTED ({fire_smoothed_confidence:.0%}) — {fire_duration_seconds:.0f}s !!"
                strip_clr, text_clr = CLR_FIRE, CLR_WHITE
            else:
                strip_txt = f"!! SMOKE DETECTED ({smoke_smoothed_confidence:.0%}) — {smoke_duration_seconds:.0f}s !!"
                strip_clr, text_clr = CLR_SMOKE, CLR_BLACK
            _semi((0, h - 24), (w, h), strip_clr, alpha=0.90)
            (tw, _), _ = cv2.getTextSize(strip_txt, FONT, 0.5, 2)
            cv2.putText(frame, strip_txt, ((w - tw) // 2, h - 6), FONT, 0.5, text_clr, 2, cv2.LINE_AA)

    def process_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        camera_name: str = "",
        camera_location: str = "",
        zones: List[Dict] = None,
    ) -> Tuple[np.ndarray, Optional[AlertEvent], Dict[str, Any]]:
        """
        Convenience one-shot call: analyze() + render() on the same frame.
        For a live video stream, prefer calling analyze() on a background
        cadence and render() on every frame instead (see module docstring).
        `zones` here are in normalized (0-1) coordinates, same as stored.
        """
        analysis = self.analyze(camera_id, frame, camera_name, camera_location, zones)
        output_frame = self.render(frame, analysis, camera_name or self.get_or_create_session(camera_id).camera_name)
        return output_frame, analysis.alert, analysis.telemetry

    @staticmethod
    def _draw_rich_overlays(
        frame: np.ndarray,
        tracked_persons: List[Dict],
        fire_smoke_detections: List[Dict],
        fire_detected: bool,
        smoke_detected: bool,
        fire_conf: float,
        smoke_conf: float,
        fire_state: str,
        smoke_state: str,
        fire_smoothed_confidence: float,
        smoke_smoothed_confidence: float,
        fire_duration_seconds: float,
        smoke_duration_seconds: float,
        person_count: int,
        behaviour_label: Optional[str],
        behaviour_confidence: float,
        neural_label: Optional[str],
        neural_confidence: float,
        activity_state: str,
        activity_smoothed_confidence: float,
        proc_time_ms: float,
        detector_backend: str,
        camera_name: str,
        zones: Optional[List[Dict]] = None,
        model_version: str = "none",
        neural_device: str = "cpu",
        include_fire_smoke: bool = True,
        behaviour_proof: Optional[Dict[str, Any]] = None,
        running_track_ids: frozenset = frozenset(),
        zone_dwell: Optional[Dict[int, float]] = None,
        loitering_track_ids: frozenset = frozenset(),
        overcrowding_track_ids: frozenset = frozenset(),
    ) -> None:
        """
        Draw detection overlays, trajectory vectors, and HUD telemetry.

        `include_fire_smoke=False` is used by render_behaviour_only() (the
        Live Monitoring feed) — every fire/smoke-drawing branch below is
        gated on this flag so that view genuinely cannot show fire/smoke
        results even if they're present in `analysis`, per the mandate to
        keep the two monitoring experiences completely separate. It never
        affects behaviour drawing — restricted/loiter/crowd/AI-suggestion
        logic runs identically either way.

        Two different kinds of label appear here, deliberately styled
        differently so they're never confused:
        - CONFIRMED (solid badges, drives the alert banner): loitering/
          crowd/restricted-zone (deterministic, position-based rules) and
          fire/smoke (HSV segmentation with temporal voting) — all backed
          by a real, verifiable signal, not a guess.
        - "AI:" suggestion (dashed outline, smaller HUD line only): the
          neural classifier's current top guess for Fighting/Kicking/
          Snatching/Falling/Running/Loitering/Crowd Formation, shown with
          its actual measured confidence. This model measures ~51% overall
          accuracy — real signal, not reliable enough to claim as fact, so
          it is never drawn as a solid "confirmed" box and never labeled
          without its confidence number attached.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── Global Palette ────────────────────────────────────────────────
        CLR_TEAL     = (0, 220, 140)    # Emerald/Teal (Normal)
        CLR_CRIMSON  = (0, 30, 255)     # Vivid Crimson (Restricted zone entry / Fighting)
        CLR_MAGENTA  = (200, 0, 210)    # Magenta (Kicking)
        CLR_AMBER    = (0, 140, 255)    # Amber/Gold (Crowd / Snatching)
        CLR_CYAN     = (210, 220, 0)    # Cyan (Falling)
        CLR_YELLOW   = (0, 230, 255)    # Neon Yellow (Running)
        CLR_PURPLE   = (255, 50, 180)   # Purple/Violet (Loitering)
        CLR_FIRE     = (0, 80, 255)     # Flame Orange-Red
        CLR_SMOKE    = (200, 200, 200)  # Smoke Silver
        CLR_HUD_BG   = (10, 14, 24)     # Deep dark slate
        CLR_WHITE    = (255, 255, 255)
        CLR_BLACK    = (0, 0, 0)
        FONT         = cv2.FONT_HERSHEY_SIMPLEX

        # Every dataset activity class gets its own distinct color, used
        # consistently for the "AI:" suggestion badge and (when unambiguous)
        # the per-person box — so "Fighting" always reads as crimson,
        # "Kicking" always magenta, etc., regardless of where it's drawn.
        NEURAL_COLORS = {
            "fighting": CLR_CRIMSON,
            "kicking": CLR_MAGENTA,
            "snatching": CLR_AMBER,
            "falling": CLR_CYAN,
            "running": CLR_YELLOW,
            "loitering": CLR_PURPLE,
            "crowd formation": CLR_AMBER,
            "fire": CLR_FIRE,
            "smoke": CLR_SMOKE,
            "normal": CLR_TEAL,
        }

        behaviour_lower = (behaviour_label or "").lower()
        neural_lower = (neural_label or "").lower()
        neural_color = NEURAL_COLORS.get(neural_lower, (140, 170, 190))

        def _semi(rect_pt1, rect_pt2, color, alpha=0.55):
            """Draw a semi-transparent filled rectangle."""
            cv2.rectangle(overlay, rect_pt1, rect_pt2, color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            overlay[:] = frame

        def _label_bg(x, y, text, font_scale, thickness, fg, bg, pad=4):
            """Draw text with a solid background pill."""
            (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
            cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl), bg, -1)
            cv2.putText(frame, text, (x, y), FONT, font_scale, fg, thickness, cv2.LINE_AA)

        def _dashed_rect(pt1, pt2, color, thickness=2, dash=8):
            """Dashed box outline — used for AI *suggestions* so they're
            never visually confused with a solid (confirmed) box."""
            x1, y1 = pt1
            x2, y2 = pt2
            for x in range(x1, x2, dash * 2):
                cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
                cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
            for y in range(y1, y2, dash * 2):
                cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
                cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

        def _point_in_restricted_zone(cx: float, cy: float) -> bool:
            if not zones:
                return False
            for zone in zones:
                if zone.get("zone_type") != "restricted":
                    continue
                pts = zone.get("polygon_points", [])
                if len(pts) < 3:
                    continue
                inside = False
                n = len(pts)
                j = n - 1
                for i in range(n):
                    xi, yi = pts[i].get("x", 0), pts[i].get("y", 0)
                    xj, yj = pts[j].get("x", 0), pts[j].get("y", 0)
                    if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi):
                        inside = not inside
                    j = i
                if inside:
                    return True
            return False

        # ── 1. Draw Zones (if defined) ────────────────────────────────────
        # ASSET zones get a distinct semi-transparent BLUE fill (not just an
        # outline) so a guard can see at a glance which zone a Vandalism
        # alert's red person-box is actually inside — see behaviour_proof's
        # "vandalism" branch below, which re-highlights the SPECIFIC matched
        # zone on top of this once an alert is active.
        CLR_ASSET = (230, 130, 30)  # Blue (BGR)
        if zones:
            for zone in zones:
                pts = zone.get("polygon_points", [])
                if len(pts) >= 3:
                    poly_arr = np.array([[int(p.get("x", 0)), int(p.get("y", 0))] for p in pts], np.int32)
                    zone_type = zone.get("zone_type")
                    if zone_type == "asset":
                        z_color = CLR_ASSET
                        cv2.fillPoly(overlay, [poly_arr], z_color)
                        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
                        overlay[:] = frame
                    else:
                        z_color = (0, 0, 220) if zone_type == "restricted" else (0, 200, 255)
                    cv2.polylines(frame, [poly_arr], isClosed=True, color=z_color, thickness=2, lineType=cv2.LINE_AA)
                    z_name = zone.get("name", "ZONE")
                    cv2.putText(frame, z_name.upper(), (poly_arr[0][0], poly_arr[0][1] - 6), FONT, 0.40, z_color, 1, cv2.LINE_AA)

        # ── 2. Top HUD Bar ────────────────────────────────────────────────
        bar_h = 38
        _semi((0, 0), (w, bar_h), CLR_HUD_BG, alpha=0.88)
        cv2.line(frame, (0, bar_h), (w, bar_h), (35, 45, 65), 1)

        is_alert = (include_fire_smoke and (fire_detected or smoke_detected)) or bool(behaviour_label) or activity_state == CONFIRMED
        status_color = CLR_CRIMSON if is_alert else CLR_TEAL

        # Status indicator dot
        dot_x = 16
        cv2.circle(frame, (dot_x, bar_h // 2), 6, status_color, -1)
        cv2.circle(frame, (dot_x, bar_h // 2), 7, CLR_WHITE, 1)

        # Persons counter badge
        person_txt = f"PERSONS: {person_count}"
        cv2.putText(frame, person_txt, (32, 24), FONT, 0.45, CLR_WHITE, 1, cv2.LINE_AA)

        # Active Activity Badge — only real, rule-verified events. Fire/
        # smoke branches are skipped entirely when include_fire_smoke is
        # False (Live Monitoring) — those results simply never reach here.
        if include_fire_smoke and fire_detected:
            act_txt = f"!! FIRE HAZARD  {fire_conf:.0%}"
            act_clr = CLR_FIRE
        elif include_fire_smoke and smoke_detected:
            act_txt = f"!! SMOKE DETECTED  {smoke_conf:.0%}"
            act_clr = CLR_SMOKE
        elif "restricted" in behaviour_lower:
            act_txt = f"!! RESTRICTED AREA ENTRY  {behaviour_confidence:.0%}"
            act_clr = CLR_CRIMSON
        elif "loiter" in behaviour_lower:
            act_txt = f"!! SUSPICIOUS LOITERING  {behaviour_confidence:.0%}"
            act_clr = CLR_PURPLE
        elif "crowd" in behaviour_lower:
            act_txt = f"!! CROWD FORMATION ({person_count} PERSONS)"
            act_clr = CLR_AMBER
        elif behaviour_label and behaviour_lower != "running":
            # Any OTHER rule-based result (Vandalism, and any future one)
            # still takes priority over a neural suggestion here — only
            # "Running" is excluded from this generic bucket (see below):
            # it is a much noisier, lower-certainty single-person rule than
            # an 8-window-persisted neural CONFIRMED, and real fights
            # routinely also trigger it (fast limb/body movement), which
            # used to hide "FIGHTING CONFIRMED" behind "RUNNING 85%" on the
            # banner — the same bug fixed in the per-person box chain above.
            act_txt = f"!! {behaviour_lower.upper()}  {behaviour_confidence:.0%}"
            act_clr = CLR_CRIMSON
        elif activity_state == CONFIRMED and neural_lower not in ("normal", "fire", "smoke"):
            act_txt = f"!! {neural_lower.upper()} CONFIRMED  {activity_smoothed_confidence:.0%}"
            act_clr = neural_color
        elif behaviour_label:
            # Running, with no active neural confirmation to compete with —
            # still shown normally.
            act_txt = f"!! {behaviour_lower.upper()}  {behaviour_confidence:.0%}"
            act_clr = CLR_CRIMSON
        else:
            act_txt = "NORMAL PATROL"
            act_clr = CLR_TEAL

        (atw, _), _ = cv2.getTextSize(act_txt, FONT, 0.48, 1)
        cv2.putText(frame, act_txt, ((w - atw) // 2, 24), FONT, 0.48, act_clr, 2 if is_alert else 1, cv2.LINE_AA)

        # Latency + Backend telemetry (right side)
        gpu_badge = f"{detector_backend.upper()} | {proc_time_ms}ms"
        (btw, _), _ = cv2.getTextSize(gpu_badge, FONT, 0.38, 1)
        cv2.putText(frame, gpu_badge, (w - btw - 12, 24), FONT, 0.38, (140, 160, 180), 1, cv2.LINE_AA)

        # ── 2b. Neural activity strip — driven by the hysteresis STATE, never
        # the raw per-window probability directly (see activity_state.py).
        # ANALYZING is a thin dashed "still deciding" line; CONFIRMED is a
        # solid banner. Fire/Smoke are excluded even if the neural model
        # guesses them — the HSV detector above already gives a real,
        # authoritative overlay for those two classes.
        ai_bar_h = 0
        is_scene_level = neural_lower in AIConfig.SCENE_LEVEL_ACTIVITIES
        if activity_state in (ANALYZING, CONFIRMED, RECOVERY) and neural_lower not in ("normal", "fire", "smoke"):
            ai_bar_h = 22
            ai_clr = neural_color
            _semi((0, bar_h), (w, bar_h + ai_bar_h), (16, 20, 30), alpha=0.80)
            if activity_state == CONFIRMED:
                cv2.rectangle(frame, (0, bar_h), (w, bar_h + ai_bar_h), ai_clr, 2)
                scope = "SCENE ACTIVITY" if is_scene_level else "CONFIRMED"
                ai_txt = f"{scope}: {neural_label.upper()}  {activity_smoothed_confidence:.0%}  CONFIRMED"
            else:
                # Dashed top border marks "still deciding" — never a solid box.
                for dx in range(0, w, 10):
                    cv2.line(frame, (dx, bar_h), (min(dx + 5, w), bar_h), ai_clr, 1)
                ai_txt = f"ANALYZING: {neural_label.upper()}  {activity_smoothed_confidence:.0%}"
            (aitw, _), _ = cv2.getTextSize(ai_txt, FONT, 0.40, 1)
            cv2.putText(frame, ai_txt, ((w - aitw) // 2, bar_h + 16), FONT, 0.40, ai_clr, 1, cv2.LINE_AA)

        # ── 3. Per-Person Overlays & Trajectories — each person judged on ──
        #      their OWN tracked signals & scene interaction ───────────────
        for p in tracked_persons:
            bbox = p.get("bbox") or p.get("detection", {}).get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            track_id      = p.get("track_id", p.get("id", "?"))
            conf          = p.get("confidence", p.get("detection", {}).get("confidence", 0.0))
            dwell_time    = p.get("dwell_time", 0.0)
            speed         = p.get("speed", 0.0)
            wander_radius = p.get("wander_radius", float("inf"))
            history       = p.get("history", [])
            trail         = p.get("trail", history)  # time-windowed (~2s) — see tracker.py Track.recent_trail()
            position_1s_ago = p.get("position_1s_ago")

            bw = x2 - x1
            bh = y2 - y1
            cx_p = (x1 + x2) // 2
            cy_p = (y1 + y2) // 2

            # Same rule as the detector: body centre OR feet inside the zone.
            is_restricted = _point_in_restricted_zone(cx_p, cy_p) or _point_in_restricted_zone(cx_p, y2)
            # Reads the SAME persisted, scale-invariant, duration-gated
            # state the Running alert rule uses (BehaviourEngine.
            # get_running_track_ids()) — this box no longer runs its own
            # separate instantaneous px/s check, so the overlay and the
            # alert can never disagree about who is "running".
            is_running = track_id in running_track_ids
            # Reads the SAME persisted state the Loitering alert rule uses
            # (BehaviourEngine.person_is_loitering()) — see
            # loitering_track_ids's docstring; this box no longer
            # recomputes its own copy of the dwell/radius thresholds.
            is_loitering = track_id in loitering_track_ids
            # The neural classifier judges the whole clip, not a specific
            # person — attribution to an individual box is only unambiguous
            # for PERSON_LEVEL classes with exactly one person on screen.
            # SCENE_LEVEL classes (Fighting/Kicking/Snatching — inherently
            # >=2 actors, see AIConfig.SCENE_LEVEL_ACTIVITIES) can't be
            # pinned to one individual, so instead of guessing which person
            # is involved, we highlight everyone in the small group (capped
            # below CROWD_THRESHOLD, where the crowd overlay takes over) —
            # never with a single stronger claim than the group evidence
            # supports.
            is_scene_level_class = neural_lower in AIConfig.SCENE_LEVEL_ACTIVITIES
            is_person_level_class = neural_lower in AIConfig.PERSON_LEVEL_ACTIVITIES
            ai_state_active = activity_state in (ANALYZING, CONFIRMED, RECOVERY)
            # NOTE: deliberately does NOT exclude `is_running` (it used to —
            # see the priority-chain comment below for why that was a real
            # bug: a person moving fast IS commonly also a fight
            # participant, not an alternative to one).
            is_ai_flagged = (
                ai_state_active
                and (
                    (is_person_level_class and person_count == 1)
                    or (is_scene_level_class and person_count < AIConfig.CROWD_THRESHOLD)
                )
                and not is_restricted and not is_loitering
            )

            is_vandal = (
                behaviour_proof is not None
                and behaviour_proof.get("type") == "vandalism"
                and behaviour_proof.get("track_id") == track_id
            )
            # Fighting is now a fully rule-based result (its own dedicated
            # 4-Gate subsystem, see analyze()'s step 5c) — never reaches
            # here via is_ai_flagged/activity_state==CONFIRMED anymore
            # (behaviour_probs["Fighting"] is unconditionally zeroed before
            # the shared ActivityStateMachine ever sees it), so this needs
            # its own explicit check, same pattern as is_vandal above.
            is_fighting_participant = (
                behaviour_proof is not None
                and behaviour_proof.get("type") == "fighting"
                and track_id in (
                    (behaviour_proof.get("box_a") or {}).get("track_id"),
                    (behaviour_proof.get("box_b") or {}).get("track_id"),
                )
            )

            # Determine this specific person's box color & label — driven
            # only by that person's own measured signals, in priority order.
            # Confirmed (rule-based) signals always win over an AI guess.
            is_dashed = False
            if is_vandal:
                box_clr = CLR_CRIMSON
                glow_clr = (0, 0, 160)
                status_pill = f"VANDALISM: {behaviour_proof.get('aggression_label', 'Aggression')} ({behaviour_proof.get('duration_seconds', 0.0):.1f}s)"
            elif is_fighting_participant:
                box_clr = CLR_CRIMSON
                glow_clr = (0, 0, 160)
                status_pill = f"FIGHTING: {behaviour_proof.get('confidence', 0.0):.0%}"
            elif is_restricted:
                box_clr = CLR_CRIMSON
                glow_clr = (0, 0, 160)
                _dwell = (zone_dwell or {}).get(track_id, 0.0)
                if _dwell >= 20.0:
                    status_pill = f"ID:{track_id} UNAUTHORIZED ENTRY {int(_dwell)}s"
                else:
                    status_pill = f"ID:{track_id} IN ZONE {int(_dwell)}s / 20s"
            elif is_ai_flagged and activity_state == CONFIRMED:
                # Promoted ABOVE is_running/is_loitering/crowd: a properly
                # 8-window-persisted neural CONFIRMED detection (e.g.
                # Fighting) is a much stronger, more specific signal than
                # the single-person Running rule — real fights routinely
                # involve fast limb/body movement that would otherwise mask
                # "FIGHTING CONFIRMED" behind a "RUN 240px/s" label. An
                # ANALYZING/RECOVERY (unconfirmed) neural guess still loses
                # to Running below — only a real CONFIRMED wins here.
                box_clr = neural_color
                glow_clr = tuple(int(c * 0.6) for c in neural_color)
                status_pill = f"ID:{track_id} {neural_label.upper()} {activity_smoothed_confidence:.0%} CONFIRMED"
            elif is_running:
                box_clr = CLR_YELLOW
                glow_clr = (0, 150, 180)
                status_pill = f"ID:{track_id} RUN {speed:.0f}px/s"
            elif is_loitering:
                box_clr = CLR_PURPLE
                glow_clr = (160, 30, 120)
                status_pill = f"LOITERING: {dwell_time:.0f}s"
            elif person_count >= AIConfig.CROWD_THRESHOLD:
                box_clr = CLR_AMBER
                glow_clr = (0, 100, 180)
                status_pill = f"ID:{track_id} CROWD"
            elif is_ai_flagged:
                # ANALYZING or RECOVERY — still unconfirmed, dashed box only.
                box_clr = neural_color
                glow_clr = tuple(int(c * 0.6) for c in neural_color)
                status_pill = f"ID:{track_id} Analyzing: {neural_label} {activity_smoothed_confidence:.0%}"
                is_dashed = True
            else:
                box_clr = CLR_TEAL
                glow_clr = (0, 70, 50)
                status_pill = f"ID:{track_id} Normal {conf:.0%}"

            # Outer glow border
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), glow_clr, 2)
            # Main bounding box — dashed for an AI suggestion (unconfirmed),
            # solid for anything rule-based (confirmed).
            if is_dashed:
                _dashed_rect((x1, y1), (x2, y2), box_clr, thickness=2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_clr, 2)

            # High-tech corner target brackets
            clen = max(8, min(16, bw // 4, bh // 4))
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * clen, cy), box_clr, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy * clen), box_clr, 3)

            # Loitering: faint circle showing the LOITER_RADIUS_PX drift
            # boundary this person must have stayed inside — proves the
            # alert isn't just "on screen a while" but "hasn't wandered".
            # Drawn around their CURRENT centroid (an approximation of their
            # actual footprint anchor — see tracker.py's wander_radius,
            # which is measured over their whole retained history, not a
            # single stored anchor point).
            if is_loitering:
                cv2.circle(frame, (cx_p, cy_p), int(AIConfig.LOITER_RADIUS_PX), CLR_PURPLE, 1, cv2.LINE_AA)

            # Track-ID-linked trajectory tail + velocity arrow. Both are
            # gated on REAL displacement (current centroid vs. this same
            # track's position ~1s ago — tracker.py's Track.
            # position_seconds_ago(), genuinely time-based, not a fixed
            # sample-index lookback), not the `speed` proxy this used to
            # key off of — a person standing still must show nothing, per
            # spec, and speed/displacement can disagree at the margins.
            if position_1s_ago is not None:
                disp_dx = cx_p - position_1s_ago[0]
                disp_dy = cy_p - position_1s_ago[1]
                displacement_px = float(np.hypot(disp_dx, disp_dy))
            else:
                displacement_px = 0.0

            if displacement_px >= 5.0:
                # Fading trajectory tail — genuinely alpha-blended (not just
                # thinning line width) across the time-windowed `trail`
                # (~2s, see tracker.py Track.recent_trail()), oldest
                # (near-transparent) to newest (near-opaque). Same color as
                # this person's own box/ID so it's unambiguous which trail
                # belongs to which ID.
                if trail and len(trail) > 1:
                    n = len(trail)
                    fh, fw = frame.shape[:2]
                    for ti in range(1, n):
                        pt_a = tuple(map(int, trail[ti - 1]))
                        pt_b = tuple(map(int, trail[ti]))
                        fade_alpha = 0.15 + 0.55 * (ti / (n - 1))  # 0.15 (oldest) -> 0.70 (newest)
                        # Alpha-blend only a small ROI around this segment,
                        # not a full-frame copy+blend — cv2.line only ever
                        # touches pixels near the segment (plus a few for
                        # line width/anti-aliasing), so blending the REST of
                        # the frame back into itself was pure wasted work:
                        # a full 960x540 copy+blend per segment, per moving
                        # person, was measured (real per-section profiling)
                        # to cost ~19ms of this function's ~20ms total on a
                        # busy multi-person scene. Same visual output —
                        # pixels outside the ROI are untouched either way.
                        pad = 6  # line half-width + AA fringe
                        rx1 = max(0, min(pt_a[0], pt_b[0]) - pad)
                        ry1 = max(0, min(pt_a[1], pt_b[1]) - pad)
                        rx2 = min(fw, max(pt_a[0], pt_b[0]) + pad)
                        ry2 = min(fh, max(pt_a[1], pt_b[1]) + pad)
                        if rx2 <= rx1 or ry2 <= ry1:
                            continue
                        roi = frame[ry1:ry2, rx1:rx2]
                        seg = roi.copy()
                        cv2.line(seg, (pt_a[0] - rx1, pt_a[1] - ry1), (pt_b[0] - rx1, pt_b[1] - ry1),
                                 box_clr, 2, cv2.LINE_AA)
                        cv2.addWeighted(seg, fade_alpha, roi, 1 - fade_alpha, 0, roi)

                # Directional arrow: real vector from ~1s ago to now, so its
                # length naturally scales with actual speed (faster movement
                # = longer arrow) instead of a fixed length regardless of
                # speed — capped so an extreme speed spike doesn't draw an
                # arrow that flies off screen.
                MAX_ARROW_LEN = 120.0
                if displacement_px > MAX_ARROW_LEN:
                    scale = MAX_ARROW_LEN / displacement_px
                    arrow_end = (cx_p + int(disp_dx * scale), cy_p + int(disp_dy * scale))
                else:
                    arrow_end = (cx_p + int(disp_dx), cy_p + int(disp_dy))
                cv2.arrowedLine(frame, (cx_p, cy_p), arrow_end, box_clr, 2, tipLength=0.3)

            # Draw dwell progress gauge only while genuinely stationary
            if dwell_time > 5 and wander_radius <= AIConfig.LOITER_RADIUS_PX:
                dwell_str = f"{dwell_time:.0f}s"
                cv2.circle(frame, (cx_p, cy_p), 14, CLR_PURPLE, 2)
                cv2.putText(frame, dwell_str, (cx_p - 8, cy_p + 4), FONT, 0.32, CLR_WHITE, 1, cv2.LINE_AA)
            else:
                # Centroid dot
                cv2.circle(frame, (cx_p, cy_p), 3, box_clr, -1)

            # Identification and Confidence Pill
            _label_bg(x1, y1 - 6, status_pill, 0.40, 1, CLR_BLACK if box_clr in (CLR_TEAL, CLR_YELLOW, CLR_CYAN) else CLR_WHITE, box_clr, pad=3)

        # ── 4. Crowd Proximity Lattice ──────────────────────────────────────
        if "crowd" in behaviour_lower and len(tracked_persons) >= 2:
            centers = []
            for p in tracked_persons:
                bbox = p.get("bbox") or p.get("detection", {}).get("bbox")
                if bbox:
                    centers.append(((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2))

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    c1, c2 = centers[i], centers[j]
                    dist = int(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                    if dist < 160:
                        cv2.line(frame, c1, c2, CLR_AMBER, 1, cv2.LINE_AA)

        # ── 4b. Gate-specific proof geometry — the AI "showing its work" ────
        # Drawn from behaviour_proof (see analyze()'s step 5f/5g), never
        # recomputed here — this is purely a visualization of evidence the
        # gate itself already used to decide, so it can never show a "proof"
        # that didn't actually factor into the alert.
        if behaviour_proof is not None:
            proof_type = behaviour_proof.get("type")
            if proof_type == "fighting":
                box_a, box_b = behaviour_proof.get("box_a"), behaviour_proof.get("box_b")
                if box_a and box_b and box_a.get("bbox") and box_b.get("bbox"):
                    xa1, ya1, xa2, ya2 = box_a["bbox"]
                    xb1, yb1, xb2, yb2 = box_b["bbox"]
                    ca = (int((xa1 + xa2) / 2), int((ya1 + ya2) / 2))
                    cb = (int((xb1 + xb2) / 2), int((yb1 + yb2) / 2))
                    dash = 10
                    dist = max(1, int(np.hypot(cb[0] - ca[0], cb[1] - ca[1])))
                    steps = max(1, dist // (dash * 2))
                    for si in range(steps):
                        t0 = si / steps
                        t1 = min(1.0, t0 + 0.5 / steps)
                        p0 = (int(ca[0] + (cb[0] - ca[0]) * t0), int(ca[1] + (cb[1] - ca[1]) * t0))
                        p1 = (int(ca[0] + (cb[0] - ca[0]) * t1), int(ca[1] + (cb[1] - ca[1]) * t1))
                        cv2.line(frame, p0, p1, CLR_CRIMSON, 2, cv2.LINE_AA)
                    label = f"FIGHTING: {behaviour_proof.get('confidence', 0.0):.0%}"
                    mid = (int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2))
                    _label_bg(mid[0], mid[1], label, 0.42, 2, CLR_WHITE, CLR_CRIMSON, pad=4)
            elif proof_type == "snatching":
                bbox = behaviour_proof.get("bbox")
                direction = behaviour_proof.get("direction", (0.0, 0.0))
                center = behaviour_proof.get("center")
                if bbox and center:
                    cx_s, cy_s = int(center[0]), int(center[1])
                    dx, dy = direction
                    arrow_len = 60
                    arrow_end = (cx_s + int(dx * arrow_len), cy_s + int(dy * arrow_len))
                    cv2.arrowedLine(frame, (cx_s, cy_s), arrow_end, CLR_YELLOW, 3, tipLength=0.35)
                    label = f"SNATCHING: {activity_smoothed_confidence:.0%}"
                    _label_bg(cx_s, cy_s - 12, label, 0.42, 2, CLR_BLACK, CLR_YELLOW, pad=4)
            elif proof_type == "vandalism":
                zone_polygon = behaviour_proof.get("zone_polygon")
                if zone_polygon and len(zone_polygon) >= 3:
                    poly_arr = np.array(
                        [[int(p.get("x", 0)), int(p.get("y", 0))] for p in zone_polygon], np.int32
                    )
                    # Re-highlight the SPECIFIC matched asset zone with a
                    # brighter outline on top of the base asset fill (see
                    # section 1 above) so it's unambiguous which zone this
                    # particular alert is about, even with multiple asset
                    # zones configured on one camera.
                    cv2.polylines(frame, [poly_arr], isClosed=True, color=CLR_ASSET, thickness=3, lineType=cv2.LINE_AA)

        # ── 5. Fire & Smoke Overlays — SEPARATE subsystem, own visual ──────
        # language (spec section 8/9/12). Box comes from the HSV detector's
        # own confirmed candidate for this frame; the LABEL WORDING reflects
        # the higher-level fire_smoke_state hysteresis (POSSIBLE/CONFIRMED/
        # RECOVERY), never a raw single-frame confidence claim.
        # Everything in this section is fire/smoke-specific — skipped
        # entirely when include_fire_smoke is False (Live Monitoring), so
        # fire/smoke boxes/labels can never leak onto that view even though
        # `fire_smoke_detections` is still passed in (and ignored).
        if include_fire_smoke:
            fire_smoke_both_confirmed = fire_state == FS_CONFIRMED and smoke_state == FS_CONFIRMED
            for det in fire_smoke_detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                conf = det["confidence"]
                if det["class_name"] == "fire":
                    state_word = {FS_POSSIBLE: "FIRE?", FS_CONFIRMED: "FIRE DETECTED", FS_RECOVERY: "FIRE (fading)"}.get(fire_state, "FIRE?")
                    label = f"{state_word} {fire_smoothed_confidence:.0%}" if not fire_smoke_both_confirmed else f"FIRE+SMOKE {fire_smoothed_confidence:.0%}"
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 20, 180), 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_FIRE, 2 if fire_state == FS_CONFIRMED else 1)
                    _label_bg(x1, y1 - 6, label, 0.48, 2, CLR_WHITE, CLR_FIRE, pad=4)
                    if fire_state == FS_CONFIRMED:
                        # Diagonal crosshatch warning pattern — confirmed only
                        step = max(12, (x2 - x1) // 6)
                        for sx in range(x1, x2, step):
                            cv2.line(frame, (sx, y1), (min(sx + step, x2), y2), (0, 30, 220), 1)
                else:
                    state_word = {FS_POSSIBLE: "SMOKE?", FS_CONFIRMED: "SMOKE DETECTED", FS_RECOVERY: "SMOKE (fading)"}.get(smoke_state, "SMOKE?")
                    label = f"{state_word} {smoke_smoothed_confidence:.0%}" if not fire_smoke_both_confirmed else f"FIRE+SMOKE {smoke_smoothed_confidence:.0%}"
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (120, 120, 120), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_SMOKE, 2 if smoke_state == FS_CONFIRMED else 1)
                    _label_bg(x1, y1 - 6, label, 0.48, 2, CLR_BLACK, CLR_SMOKE, pad=4)

        # ── 6. Bottom Warning Banner for Detected Incidents ───────────────
        if is_alert:
            banner_clr = CLR_CRIMSON if ((include_fire_smoke and fire_detected) or "restricted" in behaviour_lower) else CLR_AMBER
            banner_y1 = h - 50
            banner_y2 = h - 24
            _semi((0, banner_y1), (w, banner_y2), banner_clr, alpha=0.82)
            banner_conf = max(
                behaviour_confidence,
                fire_conf if include_fire_smoke else 0.0,
                smoke_conf if include_fire_smoke else 0.0,
                activity_smoothed_confidence if activity_state == CONFIRMED else 0.0,
            )
            banner_txt = f"SECURITY ALERT: {act_txt.replace('!!', '').strip()} (CONFIDENCE: {banner_conf:.0%})"
            (btw, _), _ = cv2.getTextSize(banner_txt, FONT, 0.48, 2)
            cv2.putText(frame, banner_txt, ((w - btw) // 2, banner_y2 - 7), FONT, 0.48, CLR_WHITE, 2, cv2.LINE_AA)

        # ── 7. Fire / Smoke Emergency Flashing Strip ──────────────────────
        # Only shown once the hysteresis state machine has actually
        # CONFIRMED — a POSSIBLE (single/few-frame) candidate never triggers
        # the full-width hazard strip, only the bounding-box label above.
        # Entirely skipped on the behaviour-only (Live Monitoring) view.
        if include_fire_smoke and (fire_state == FS_CONFIRMED or smoke_state == FS_CONFIRMED):
            if fire_state == FS_CONFIRMED and smoke_state == FS_CONFIRMED:
                strip_txt = f"!! HAZARD: FIRE + SMOKE (Fire {fire_smoothed_confidence:.0%} / Smoke {smoke_smoothed_confidence:.0%}) !!"
                strip_clr = CLR_FIRE
            elif fire_state == FS_CONFIRMED:
                strip_txt = f"!! HAZARD DETECTED: FIRE ({fire_smoothed_confidence:.0%}) — {fire_duration_seconds:.0f}s !!"
                strip_clr = CLR_FIRE
            else:
                strip_txt = f"!! HAZARD DETECTED: SMOKE ({smoke_smoothed_confidence:.0%}) — {smoke_duration_seconds:.0f}s !!"
                strip_clr = CLR_SMOKE
            _semi((0, h - 24), (w, h), strip_clr, alpha=0.90)
            (stw, _), _ = cv2.getTextSize(strip_txt, FONT, 0.50, 2)
            cv2.putText(frame, strip_txt, ((w - stw) // 2, h - 6), FONT, 0.50, CLR_WHITE if fire_state == FS_CONFIRMED else CLR_BLACK, 2, cv2.LINE_AA)

        # ── 8. Debug Overlay (off in production) ──────────────────────────
        # Verbose per-frame diagnostics for tuning thresholds against real
        # footage — never meant for the resident/guard-facing UI.
        if AIConfig.DEBUG_OVERLAY:
            debug_lines = [
                f"AI FPS target: {AIConfig.AI_ANALYSIS_FPS} | Proc: {proc_time_ms}ms | Device: {neural_device}",
                f"Model: {model_version} | People: {person_count} | Backend: {detector_backend}",
                f"Neural raw: {neural_label or 'N/A'} {neural_confidence:.0%} | State: {activity_state}",
                f"Smoothed: {activity_smoothed_confidence:.0%} | "
                f"Start/Confirm/End: {AIConfig.ACTIVITY_START_THRESHOLD:.0%}/"
                f"{AIConfig.ACTIVITY_CONFIRM_THRESHOLD:.0%}/{AIConfig.ACTIVITY_END_THRESHOLD:.0%}",
            ]
            if include_fire_smoke:
                debug_lines += [
                    f"FIRE/SMOKE MODEL — Fire raw: {fire_conf:.0%} smooth: {fire_smoothed_confidence:.0%} state: {fire_state}",
                    f"Smoke raw: {smoke_conf:.0%} smooth: {smoke_smoothed_confidence:.0%} state: {smoke_state} | "
                    f"Fire/Smoke alert thr: {AIConfig.FIRE_ALERT_THRESHOLD:.0%}/{AIConfig.SMOKE_ALERT_THRESHOLD:.0%}",
                ]
            hazard_strip_present = include_fire_smoke and (fire_detected or smoke_detected)
            dy = h - 24 - (18 * len(debug_lines)) - 8 if hazard_strip_present else h - (18 * len(debug_lines)) - 8
            _semi((4, dy - 4), (420, h - (24 if hazard_strip_present else 0)), (10, 10, 10), alpha=0.75)
            for i, line in enumerate(debug_lines):
                cv2.putText(frame, line, (10, dy + i * 18 + 12), FONT, 0.36, (0, 255, 120), 1, cv2.LINE_AA)
            for p in tracked_persons:
                bbox = p.get("bbox") or p.get("detection", {}).get("bbox")
                if not bbox:
                    continue
                tid = p.get("track_id", p.get("id", "?"))
                cv2.putText(frame, f"#{tid}", (int(bbox[0]), int(bbox[3]) + 14), FONT, 0.34, (0, 255, 255), 1, cv2.LINE_AA)

        # ── 5. Overcrowding — collective bounding envelope + banner ─────────
        # Deliberately drawn LAST, after every _semi() call above: _semi()
        # keeps its own `overlay` buffer in sync with `frame` only when
        # _semi() itself runs (see its docstring) — any DIRECT cv2 draw
        # onto `frame` (like this block's) that happened BEFORE a later
        # _semi() call would get silently blended back over with that
        # call's stale pre-draw `overlay` snapshot and effectively
        # corrupted/darkened (found via real pixel-level render testing,
        # not assumed). Placing this block after all of this function's
        # _semi() calls sidesteps that without having to touch _semi()
        # itself or any of the untouched features that use it.
        #
        # Deliberately SEPARATE from the Crowd Proximity Lattice above (a
        # different, untouched feature) — driven by overcrowding_track_ids
        # (this camera's CURRENT confirmed-cluster membership, read via
        # BehaviourEngine.get_overcrowding_state(), see FrameAnalysis.
        # overcrowding_track_ids's docstring), not by whether "Overcrowding"
        # happens to be this window's single winning behaviour_label — same
        # "stays visible for the whole confirmed duration" principle as
        # Loitering/Running's overlays.
        if overcrowding_track_ids:
            member_bboxes = [
                p.get("bbox") or p.get("detection", {}).get("bbox")
                for p in tracked_persons
                if p.get("track_id") in overcrowding_track_ids
            ]
            member_bboxes = [b for b in member_bboxes if b]
            if member_bboxes:
                CLR_OVERCROWD = CLR_MAGENTA
                ex1 = int(min(b[0] for b in member_bboxes))
                ey1 = int(min(b[1] for b in member_bboxes))
                ex2 = int(max(b[2] for b in member_bboxes))
                ey2 = int(max(b[3] for b in member_bboxes))
                pad = 12
                ex1, ey1 = max(0, ex1 - pad), max(0, ey1 - pad)
                ex2, ey2 = min(frame.shape[1], ex2 + pad), min(frame.shape[0], ey2 + pad)
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), CLR_OVERCROWD, 3, cv2.LINE_AA)
                banner = f"OVERCROWDING DETECTED (N={len(overcrowding_track_ids)}>=5)"
                (btw, bth), _ = cv2.getTextSize(banner, FONT, 0.55, 2)
                bx1 = max(0, min(ex1, frame.shape[1] - btw - 16))
                by2 = ey1
                by1 = max(0, by2 - bth - 14)
                cv2.rectangle(frame, (bx1, by1), (bx1 + btw + 16, by2), CLR_OVERCROWD, -1)
                cv2.putText(frame, banner, (bx1 + 8, by2 - 8), FONT, 0.55, CLR_WHITE, 2, cv2.LINE_AA)



# Global singleton
inference_engine = InferenceEngine()
