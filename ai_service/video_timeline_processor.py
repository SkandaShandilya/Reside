"""
Uploaded Video Full-Timeline Analyzer — Phase 2.

Unlike live-camera streaming (backend/app/services/video_stream_service.py
— continuous, latest-frame-wins, intentionally drops stale work when the
CPU falls behind so playback stays smooth), an uploaded video's ENTIRE
duration must be analyzed: a short suspicious event anywhere in a
multi-minute file must not be skippable just because inference is slower
than the file's native frame rate, and there may be nobody watching a live
MJPEG stream of it at all.

Flow:
  decode the file start-to-finish (OpenCV), sampling frames at a fixed,
  bounded rate (AIConfig.UPLOAD_ANALYSIS_FPS) — never paced to real-time,
  never looping — so the whole file gets covered in bounded time regardless
  of CPU speed
        |
  ai_service.inference_engine.InferenceEngine.analyze() — the EXACT SAME
  pipeline live cameras use (person detection/tracking, rule-based
  behaviour_engine, fire/smoke detector + hysteresis, neural classifier +
  ActivityStateMachine) — called once per sampled frame, in chronological
  order, using a dedicated CameraInferenceSession isolated from any
  concurrently-live-viewed session for the same camera_id (never sharing
  tracker/fire-smoke/activity-state), and an `event_time` synthesized as
  `job_start_epoch + video_offset_seconds` so every downstream duration and
  absolute timestamp is correct regardless of how much faster or slower
  than real-time this batch actually runs
        |
  same alert_dispatcher.dispatch_alert_sync / dispatch_event_update_sync
  path live cameras use, so timeline results, alerts, and review-queue
  routing behave identically regardless of source

Because this reuses analyze()'s existing rolling neural-classifier buffer
(filled one sampled frame at a time, continuously advancing across the
WHOLE file) together with Phase 1's per-class ActivityStateMachine, this is
already a maximally-overlapping sliding window over the video's full
duration (stride = 1 analysis tick) — a finer-grained superset of a
discrete fixed window/stride scheme, without a second, separate windowing
implementation to keep in sync with ai_service/training/dataset.py's.

Known limitation (disclosed, not silently skipped): position-based
rule-based behaviour_engine detections that depend on continuous frame-to-
frame tracking (loitering dwell time, restricted-zone entry) receive
per-camera zone data as an empty list here — restricted-zone/loitering
checks are effectively inert for this batch pass. They still work normally
for a live-viewed stream of the same uploaded file. Wiring zones through
would require an additional DB read per job; left as a follow-up since
Phase 2's explicit focus is the neural/behaviour timeline + fire/smoke, not
rule-based zone geometry.

Runs entirely on a background daemon thread — never blocks the FastAPI
event loop. Progress is exposed via an in-memory registry
(`get_job_status`) the frontend can poll.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

from ai_service.config import AIConfig
from ai_service.inference_engine import CameraInferenceSession

logger = logging.getLogger(__name__)

# See backend/app/services/video_stream_service.py's FIRE_SMOKE_LABELS —
# same constant, duplicated here rather than imported since this module is
# deliberately backend-agnostic (see module docstring) and that one lives
# in a backend service module. Fire and smoke are combined into one
# activity now (see inference_engine.py's combined-event construction).
FIRE_SMOKE_LABELS = ("Fire/Smoke Detected",)
MAX_FIRE_SMOKE_INCIDENT_FRAMES = 1800


@dataclass
class TimelineJobStatus:
    status: str = "idle"  # idle | running | completed | failed
    progress_percent: int = 0
    current_offset_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    frames_sampled: int = 0
    events_detected: int = 0
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


_jobs: Dict[int, TimelineJobStatus] = {}
_jobs_lock = threading.Lock()


def get_job_status(camera_id: int) -> Optional[TimelineJobStatus]:
    with _jobs_lock:
        job = _jobs.get(camera_id)
        return field_copy(job) if job else None


def field_copy(job: TimelineJobStatus) -> TimelineJobStatus:
    """Shallow copy so a caller reading status can't race the worker thread
    mutating it in place."""
    return TimelineJobStatus(**job.__dict__)


def is_running(camera_id: int) -> bool:
    with _jobs_lock:
        job = _jobs.get(camera_id)
        return bool(job and job.status == "running")


def start_full_timeline_analysis(
    camera_id: int,
    camera_name: str,
    camera_location: str,
    video_path: str,
) -> bool:
    """
    Kicks off a background thread that decodes `video_path` start-to-finish
    and runs it through the full analysis pipeline. Returns False (no-op)
    if a job for this camera is already running or the file doesn't exist;
    True once the background thread has been started.
    """
    if is_running(camera_id):
        logger.info(f"Camera {camera_id}: full-timeline analysis already running — ignoring duplicate request.")
        return False
    if not Path(video_path).exists():
        logger.warning(f"Camera {camera_id}: cannot start full-timeline analysis — file not found: {video_path}")
        return False

    with _jobs_lock:
        _jobs[camera_id] = TimelineJobStatus(status="running", started_at=time.time())

    thread = threading.Thread(
        target=_run_job,
        args=(camera_id, camera_name, camera_location, video_path),
        daemon=True,
        name=f"timeline-{camera_id}",
    )
    thread.start()
    return True


def _run_job(camera_id: int, camera_name: str, camera_location: str, video_path: str) -> None:
    from ai_service.inference_engine import inference_engine  # shared singleton — model already loaded

    job = _jobs[camera_id]
    job_start_epoch = time.time()

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 240:
            source_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = (total_frames / source_fps) if total_frames > 0 else 0.0

        with _jobs_lock:
            job.total_duration_seconds = round(total_duration, 1)

        # Sample every Kth frame so the WHOLE file is covered at a bounded,
        # fixed temporal density (AIConfig.UPLOAD_ANALYSIS_FPS) regardless
        # of the source's native fps or the video's length — never paced to
        # real-time (this can and should run faster than real-time on a
        # capable CPU), and never skipping large sections.
        sample_every_n = max(1, round(source_fps / AIConfig.UPLOAD_ANALYSIS_FPS))

        # Isolated session — NOT registered in inference_engine._sessions,
        # so this batch pass never shares tracker/fire-smoke-detector/
        # activity-state with (and can never corrupt) a concurrently
        # live-viewed stream of the same camera_id.
        session = CameraInferenceSession(camera_id, camera_name, camera_location)

        # Fire/Smoke-only pre-event ring buffer (raw BGR frames) — a batch
        # job has no "must reach the frontend immediately" pressure the
        # live path has, so instead of dispatching a same-window single
        # frame as the whole "clip" (the previous behaviour, and a
        # confirmed root cause of blank/near-empty uploaded-video Fire/
        # Smoke clips), the alert is held until the incident has fully
        # closed (+ a short post-event grace) and dispatched exactly ONCE
        # with a real pre+active+post clip already attached. Behaviour
        # alerts are untouched — see _dispatch_results below, which still
        # uses the original single-frame path for every non-fire/smoke
        # activity.
        pre_buffer_len = max(1, round(AIConfig.FIRE_SMOKE_PRE_EVENT_SECONDS * AIConfig.UPLOAD_ANALYSIS_FPS))
        pre_buffer: "deque" = deque(maxlen=pre_buffer_len)
        fire_smoke_incidents: Dict[str, dict] = {}
        # Same shared, correctly-rooted VideoProcessor the live path uses
        # (see backend/app/services/alert_dispatcher.py._get_video_processor)
        # — narrow, lazy backend import, same pattern _dispatch_results
        # below already uses for alert_dispatcher itself.
        from app.services.alert_dispatcher import _get_video_processor
        video_processor = _get_video_processor()

        frame_index = 0
        frames_sampled = 0
        events_detected = 0
        event_time = job_start_epoch
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_every_n == 0:
                video_offset_seconds = frame_index / source_fps
                event_time = job_start_epoch + video_offset_seconds

                h, w = frame.shape[:2]
                if w > 960:
                    target_w = 960
                    target_h = int(h * target_w / w)
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                try:
                    result = inference_engine.analyze(
                        camera_id=camera_id,
                        frame=frame,
                        camera_name=camera_name,
                        camera_location=camera_location,
                        zones=[],  # see module docstring's disclosed limitation
                        fire_enabled=True,
                        smoke_enabled=True,
                        event_time=event_time,
                        session=session,
                    )
                    pre_buffer.append(frame.copy())
                    events_detected += _dispatch_results(
                        result, frame, event_time,
                        camera_id, pre_buffer, fire_smoke_incidents, video_processor,
                    )
                except Exception as e:
                    logger.warning(f"Camera {camera_id}: analysis error at offset {video_offset_seconds:.1f}s: {e}")

                frames_sampled += 1
                with _jobs_lock:
                    job.current_offset_seconds = round(video_offset_seconds, 1)
                    job.frames_sampled = frames_sampled
                    job.events_detected = events_detected
                    job.progress_percent = (
                        min(99, int(100 * video_offset_seconds / total_duration)) if total_duration > 0 else 0
                    )

            frame_index += 1

        # File exhausted while an incident was still open (e.g. the fire
        # never "ends" before the recording does) — flush whatever was
        # collected rather than silently dropping it.
        for label, incident in fire_smoke_incidents.items():
            _finalize_uploaded_incident(camera_id, label, incident, video_processor, event_time)
            events_detected += 1

        with _jobs_lock:
            job.events_detected = events_detected
            job.progress_percent = 100
            job.status = "completed"
            job.finished_at = time.time()
        logger.info(
            f"Camera {camera_id}: full-timeline analysis complete — "
            f"{frames_sampled} windows sampled over {total_duration:.1f}s, "
            f"{events_detected} event(s) detected."
        )
    except Exception as e:
        with _jobs_lock:
            job.status = "failed"
            job.error = str(e)
            job.finished_at = time.time()
        logger.error(f"Camera {camera_id}: full-timeline analysis failed: {e}")
    finally:
        cap.release()


def _dispatch_results(
    result, frame, event_time: float,
    camera_id: int, pre_buffer: "deque", fire_smoke_incidents: Dict[str, dict],
    video_processor,
) -> int:
    """
    Reuses the exact same alert_dispatcher path live streaming uses (see
    backend/app/services/video_stream_service.py's `_inference_worker`) so
    timeline-detected events, DB rows, and WebSocket broadcasts behave
    identically regardless of source. This is the one place this module
    reaches into the backend package (ai_service is otherwise
    backend-agnostic) — kept as a deliberately narrow, lazy import rather
    than restructuring the whole call chain around a callback, since this
    module only ever runs inside the backend process that already owns
    alert_dispatcher's event loop.

    Fire/Smoke alerts are routed through the incident buffer (pre_buffer +
    fire_smoke_incidents) instead of being dispatched immediately with a
    single-frame clip — see _run_job's docstring comment above its
    construction. Every other activity (behaviour) is dispatched exactly as
    before, untouched. Finalizes and dispatches the ONE alert as soon as
    the state machine's own "finalize" fires — no additional grace period
    is layered on top of fire_smoke_state.py's own END_GRACE_SECONDS, which
    already delays "finalize" until confidence has genuinely stayed low for
    a while; frames recorded during that grace are already part of the
    incident's "active" recording by the time finalize fires, so they
    already serve as the clip's post-event tail. (An earlier version added
    a second, independent post-event grace here — redundant with the state
    machine's own, and it could push finalization past the end of a short
    video entirely; removed.)

    A fast reflicker within RETRIGGER_COOLDOWN_SECONDS of a close ("reopen"
    — see AIConfig.RETRIGGER_COOLDOWN_SECONDS) arrives AFTER this function
    has already dispatched-and-forgotten the first incident, so it (and any
    later "finalize"/"refresh" for that same continuation) is forwarded as
    an ordinary event-update against the already-created row — exactly the
    mechanism the live path uses — rather than starting a second buffered
    incident.

    Returns how many NEW alerts were created this call (for the caller's
    events_detected counter — fire/smoke alerts count once their incident
    is actually dispatched, which may be on a LATER call than the one that
    first saw them).
    """
    from app.services import alert_dispatcher

    new_alert_count = 0
    all_new_alerts = ([result.alert] if result.alert is not None else []) + list(result.additional_alerts)
    for a in all_new_alerts:
        if a.activity in FIRE_SMOKE_LABELS:
            if a.activity not in fire_smoke_incidents:
                fire_smoke_incidents[a.activity] = {
                    "alert": a,
                    "frames": deque(pre_buffer, maxlen=MAX_FIRE_SMOKE_INCIDENT_FRAMES),
                }
        else:
            ok, jpeg = cv2.imencode(".jpg", frame)
            alert_dispatcher.dispatch_alert_sync(a, [jpeg.tobytes()] if ok else [])
            new_alert_count += 1

    for label, incident in fire_smoke_incidents.items():
        incident["frames"].append(frame.copy())

    for update in result.behaviour_event_updates:
        alert_dispatcher.dispatch_event_update_sync(update, event_time)

    for update in result.fire_smoke_event_updates:
        label = update["activity"]
        incident = fire_smoke_incidents.get(label)
        if incident is not None and update["action"] == "finalize":
            incident["finalize_update"] = update
            del fire_smoke_incidents[label]
            _finalize_uploaded_incident(camera_id, label, incident, video_processor, event_time)
            new_alert_count += 1
        elif incident is None:
            # No buffered incident for this label — either its own
            # finalize already dispatched it (this is a later reopen/
            # refresh/finalize for the same continuation), or the
            # "start" window genuinely wasn't seen. Either way, the
            # correct row (if any) is found by camera+activity, same as
            # the live path.
            alert_dispatcher.dispatch_event_update_sync(update, event_time)

    return new_alert_count


def _finalize_uploaded_incident(
    camera_id: int, activity: str, incident: dict, video_processor, event_time: float,
) -> None:
    """
    Encodes the full pre+active+post frame list collected for one closed
    Fire/Smoke incident and dispatches it as ONE fully-formed alert
    (already-known end_time/peak/average/duration from the state machine's
    own "finalize" update — see fire_smoke_event_updates in
    ai_service/inference_engine.py) — no separate "update_clip" round trip
    needed here the way the live path requires, since a batch job has no
    live audience waiting on immediate delivery.
    """
    from app.services import alert_dispatcher

    frames = list(incident["frames"])
    alert = incident["alert"]
    finalize_update = incident.get("finalize_update")
    end_time = None
    if finalize_update:
        alert.peak_confidence = finalize_update.get("peak_confidence")
        alert.average_confidence = finalize_update.get("average_confidence")
        alert.duration_seconds = finalize_update.get("duration_seconds")
        # Cumulative fire_detected/smoke_detected (see
        # ai_service/inference_engine.py's combined_metadata) — the
        # ORIGINAL alert object's metadata was only a snapshot from the
        # moment the incident started (e.g. smoke-only, if fire hadn't
        # confirmed yet); the finalize update carries the full picture
        # accumulated over the incident's whole lifetime.
        if "metadata" in finalize_update:
            alert.metadata = finalize_update["metadata"]
        # `event_time` is the video-offset-anchored epoch timestamp for the
        # window that produced this "finalize" (job_start_epoch +
        # video_offset_seconds — see _run_job), matching how every other
        # timestamp in this module is computed. Using real wall-clock time()
        # here instead would make a batch job that runs faster or slower
        # than real-time report a wrong end_time relative to start_time.
        end_time = event_time

    clip_path = video_processor.save_clip(frames, camera_id, fps=AIConfig.UPLOAD_ANALYSIS_FPS, prefix="firesmoke")
    thumbnail_path = video_processor.save_snapshot(frames[0], camera_id, prefix="firesmoke") if frames else None

    alert_dispatcher.dispatch_alert_sync(
        alert, recent_jpeg_frames=None, clip_path=clip_path, thumbnail_path=thumbnail_path, end_time=end_time,
    )
    logger.info(
        f"Camera {camera_id}: uploaded-video {activity} incident finalized — "
        f"{len(frames)} frames, clip={clip_path}, thumbnail={thumbnail_path}."
    )
