"""
Alert Dispatcher — bridges the AI pipeline's synchronous capture threads to
the async DB session + WebSocket broadcast the rest of the app uses.

Why this exists: ai_service.inference_engine.analyze() already computes an
AlertEvent when a rule-based detection clears its threshold, but until now
nothing ever did anything with it — it wasn't written to the database and
never reached the WebSocket. The detection logic worked; the delivery path
from "detected" to "visible in the app" didn't exist. This module is that
path.

Flow implemented here (see PROJECT spec section 33):
  AlertEvent (from inference_engine)
        |
  dispatch_alert_sync()  <-- called from the capture thread
        |  asyncio.run_coroutine_threadsafe -> main event loop
        v
  _dispatch_alert_async()
        |
        +-- SecurityEvent row (always)
        +-- confidence >= CONFIDENCE_THRESHOLD -> Alert row (confirmed,
        |     UNRESOLVED) + live WebSocket broadcast to guards/admins
        +-- otherwise (send_to_review) -> IncidentReview row only, no
              live "confirmed" alert broadcast (human-in-the-loop: an
              uncertain detection needs a human to look at it before it's
              treated as a real incident)

Thumbnail + short video clip (the frames leading up to and including the
triggering frame, pulled from the stream's own rolling JPEG buffer — see
FrameBuffer.recent() in video_stream_service.py) are saved best-effort;
a failure there never blocks the alert itself from being created.
"""
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_main_loop: Optional[asyncio.AbstractEventLoop] = None
_dedupe_lock = threading.Lock()

_video_processor = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once from app startup so capture threads can schedule async work."""
    global _main_loop
    _main_loop = loop


def _get_video_processor():
    """
    Lazily-constructed, module-level VideoProcessor rooted at the backend's
    real storage path (settings.STORAGE_PATH) — NOT ai_service.video_
    processor's own default singleton, which defaults to a cwd-relative
    "./storage" and would silently write files in the wrong place depending
    on the process's working directory. Reused (not duplicated) by both the
    live-camera and uploaded-video paths so there is exactly one place that
    writes+verifies alert clips/snapshots (see ai_service/video_processor.py).
    """
    global _video_processor
    if _video_processor is None:
        from app.config import settings
        from ai_service.video_processor import VideoProcessor
        _video_processor = VideoProcessor(settings.STORAGE_PATH)
    return _video_processor


def _save_thumbnail(camera_id: int, jpeg_bytes: Optional[bytes]) -> Optional[str]:
    if not jpeg_bytes:
        return None
    import cv2
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        logger.warning(f"Could not decode alert thumbnail frame for camera {camera_id}.")
        return None
    return _get_video_processor().save_snapshot(frame, camera_id, prefix="alert")


def _save_clip(camera_id: int, jpeg_frames: list) -> Optional[str]:
    """
    Write the frames already sitting in the stream's rolling buffer (the
    seconds leading up to and including the trigger) out as an mp4. For a
    behaviour alert this is "before + event" only — building the after-
    portion would mean holding the alert open and delaying delivery until
    those future frames exist, which trades off real-time notification for
    a slightly longer clip, and section 33 wants the alert to reach the
    frontend immediately. Fire/Smoke alerts get a fuller pre+active+post
    clip instead, attached via a later "update_clip" event-update once the
    incident actually closes — see video_stream_service.py's fire/smoke
    incident recording and _dispatch_event_update_async's "update_clip"
    handling below.
    """
    if not jpeg_frames:
        return None
    return _get_video_processor().save_clip_from_jpegs(jpeg_frames, camera_id, fps=15.0, prefix="alert")


def dispatch_alert_sync(
    alert_event,
    recent_jpeg_frames: Optional[list] = None,
    clip_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    end_time: Optional[float] = None,
) -> None:
    """
    Entry point called from the (synchronous, background-thread) capture
    loop whenever inference_engine.analyze() returns an alert. Schedules
    the actual DB write + WebSocket broadcast onto the main asyncio event
    loop and returns immediately — never blocks the capture thread.

    `clip_path`/`thumbnail_path`, if given, are already-saved relative
    storage paths (see ai_service/video_processor.py) that should be used
    AS-IS instead of deriving them from `recent_jpeg_frames` — used by the
    uploaded-video Fire/Smoke path (ai_service/video_timeline_processor.py),
    which has already built and verified a full pre+active+post clip before
    dispatching, since a batch job has no "must deliver immediately" reason
    to send a same-window single-frame clip the way the live path does.
    `end_time` (epoch seconds), if given, closes the row at creation instead
    of leaving it open for a later "finalize" update — also uploaded-video-
    only, since by the time it dispatches the incident has already ended.
    """
    if _main_loop is None:
        logger.warning("Alert dispatcher has no event loop registered yet — dropping alert.")
        return
    try:
        asyncio.run_coroutine_threadsafe(
            _dispatch_alert_async(alert_event, recent_jpeg_frames or [], clip_path, thumbnail_path, end_time),
            _main_loop,
        )
    except Exception as e:
        logger.error(f"Failed to schedule alert dispatch: {e}")


async def _dispatch_alert_async(
    alert_event,
    recent_jpeg_frames: list,
    precomputed_clip_path: Optional[str] = None,
    precomputed_thumbnail_path: Optional[str] = None,
    precomputed_end_time: Optional[float] = None,
) -> None:
    from app.database import async_session_factory
    from app.models.security_event import SecurityEvent, SeverityLevel
    from app.models.alert import Alert, AlertStatus
    from app.models.incident_review import IncidentReview
    from app.websocket.manager import ws_manager
    from ai_service.config import AIConfig

    loop = asyncio.get_running_loop()
    if precomputed_thumbnail_path is not None or precomputed_clip_path is not None:
        thumbnail_path = precomputed_thumbnail_path
        clip_path = precomputed_clip_path
    else:
        thumbnail_path = await loop.run_in_executor(
            None, _save_thumbnail, alert_event.camera_id, recent_jpeg_frames[-1] if recent_jpeg_frames else None
        )
        clip_path = await loop.run_in_executor(None, _save_clip, alert_event.camera_id, recent_jpeg_frames)

    severity_map = {"critical": SeverityLevel.CRITICAL, "warning": SeverityLevel.WARNING, "info": SeverityLevel.INFO}
    event_severity = severity_map.get(alert_event.severity, SeverityLevel.WARNING)

    event_start_dt = (
        datetime.fromtimestamp(alert_event.start_time, tz=timezone.utc)
        if alert_event.start_time else None
    )
    event_end_dt = (
        datetime.fromtimestamp(precomputed_end_time, tz=timezone.utc)
        if precomputed_end_time else None
    )

    async with async_session_factory() as db:
        try:
            event = SecurityEvent(
                camera_id=alert_event.camera_id,
                activity_class=alert_event.activity,
                confidence=alert_event.confidence,
                severity=event_severity,
                persons_detected=alert_event.persons_detected,
                duration_seconds=alert_event.duration_seconds,
                video_clip_path=clip_path,
                thumbnail_path=thumbnail_path,
                model_version=None,
                timestamp=alert_event.timestamp or datetime.now(timezone.utc),
                # Timeline-aware fields (Phase 2) — start_time/peak/average
                # only ever populated by the ActivityStateMachine-driven
                # neural/behaviour event path (see ai_service/
                # inference_engine.py); rule-based and fire/smoke alerts
                # leave these None rather than fabricating a value.
                # end_time stays NULL (until dispatch_event_update_sync's
                # "finalize" action closes this same row out) UNLESS the
                # caller already knows the incident is over — see
                # precomputed_end_time above.
                start_time=event_start_dt,
                end_time=event_end_dt,
                peak_confidence=alert_event.peak_confidence,
                average_confidence=alert_event.average_confidence,
                # Internal detail preserved for combined events (e.g.
                # Fire/Smoke Detected's fire_detected/smoke_detected) —
                # None for every other alert type, which has nothing to
                # store here. Existing column ("metadata" in the DB,
                # metadata_json on the Python side — "metadata" is a
                # reserved attribute name on SQLAlchemy's declarative
                # base), no migration needed.
                metadata_json=getattr(alert_event, "metadata", None),
            )
            db.add(event)
            await db.flush()
            await db.refresh(event)

            if alert_event.send_to_review:
                # Confidence didn't clear CONFIDENCE_THRESHOLD — route to
                # the human review queue instead of a confirmed live alert.
                review = IncidentReview(
                    event_id=event.id,
                    video_path=clip_path,
                    thumbnail_path=thumbnail_path,
                    camera_id=alert_event.camera_id,
                    camera_name=alert_event.camera_name,
                    original_prediction=alert_event.activity,
                    original_confidence=alert_event.confidence,
                    timestamp=event.timestamp,
                )
                db.add(review)
                await db.commit()
                logger.info(
                    f"Camera {alert_event.camera_id}: '{alert_event.activity}' "
                    f"({alert_event.confidence:.0%}) routed to review queue (below "
                    f"{AIConfig.CONFIDENCE_THRESHOLD:.0%} confirmation threshold)."
                )
                return

            alert = Alert(
                event_id=event.id,
                camera_id=alert_event.camera_id,
                camera_name=alert_event.camera_name,
                camera_location=alert_event.camera_location,
                activity=alert_event.activity,
                confidence=alert_event.confidence,
                severity=alert_event.severity,
                status=AlertStatus.UNRESOLVED,
                persons_detected=alert_event.persons_detected,
                duration_seconds=alert_event.duration_seconds,
                video_clip_path=clip_path,
                thumbnail_path=thumbnail_path,
                source="ai",
                start_time=event_start_dt,
                end_time=event_end_dt,
                peak_confidence=alert_event.peak_confidence,
                average_confidence=alert_event.average_confidence,
            )
            db.add(alert)
            await db.commit()
            await db.refresh(alert)

            await ws_manager.broadcast_alert({
                "type": "security_alert",
                "alert_id": alert.id,
                "event_id": event.id,
                "camera_id": alert.camera_id,
                "camera_name": alert.camera_name,
                "camera_location": alert.camera_location,
                "activity": alert.activity,
                "confidence": alert.confidence,
                "severity": alert.severity,
                "persons_detected": alert.persons_detected,
                "duration_seconds": alert.duration_seconds,
                "thumbnail_path": alert.thumbnail_path,
                "video_clip_path": alert.video_clip_path,
                "status": alert.status.value,
                "timestamp": alert.created_at.isoformat() if alert.created_at else datetime.now(timezone.utc).isoformat(),
            })
            logger.info(
                f"Camera {alert.camera_id}: ALERT #{alert.id} '{alert.activity}' "
                f"({alert.confidence:.0%}, {alert.severity}) created and broadcast."
            )
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to persist/broadcast alert: {e}")


def dispatch_event_update_sync(update: dict, event_time: Optional[float] = None) -> None:
    """
    Entry point called from a (synchronous, background-thread) capture loop
    or the offline video-timeline processor whenever
    ai_service.inference_engine.FrameAnalysis.behaviour_event_updates
    contains a "refresh" or "finalize" action for an already-open event
    (created earlier via dispatch_alert_sync -> _dispatch_alert_async).
    Schedules the DB write onto the main asyncio event loop and returns
    immediately — never blocks the calling thread. `update` is one dict
    from that list (see ai_service/inference_engine.py's neural/behaviour
    event lifecycle block for its exact shape).
    """
    if _main_loop is None:
        logger.warning("Alert dispatcher has no event loop registered yet — dropping event update.")
        return
    try:
        asyncio.run_coroutine_threadsafe(
            _dispatch_event_update_async(update, event_time),
            _main_loop,
        )
    except Exception as e:
        logger.error(f"Failed to schedule event update dispatch: {e}")


async def _dispatch_event_update_async(update: dict, event_time: Optional[float]) -> None:
    """
    "refresh"/"finalize": finds the currently-OPEN SecurityEvent/Alert row
    for (camera_id, activity) — i.e. the most recent one with end_time IS
    NULL — and updates its peak/average/duration ("refresh") or
    additionally sets end_time ("finalize"). Looked up by (camera_id,
    activity, end_time IS NULL) rather than by a row id threaded back
    across the sync/async boundary, since at most one such row can be open
    at a time per (camera, activity) — dispatch_alert_sync's "create" path
    is the only thing that ever opens one, and inference_engine.py's
    `open_behaviour_events`/`open_fire_smoke_events` bookkeeping guarantees
    only one create happens before the matching finalize.

    "reopen" (fire/smoke's RETRIGGER_COOLDOWN_SECONDS mechanism — see
    AIConfig.RETRIGGER_COOLDOWN_SECONDS and ai_service/inference_engine.py's
    fire/smoke event lifecycle block): finds the MOST RECENTLY CLOSED row
    instead (end_time IS NOT NULL) and clears its end_time back to None,
    treating a fast reflicker right after a close as a continuation of the
    same incident rather than creating a new alert row. Deliberately does
    NOT touch `status` — if a Guard/Admin already acknowledged/resolved
    the alert before it reopened, that human decision is left alone; only
    the timeline fields are updated.

    An update arriving for a row that doesn't exist in the expected state
    (e.g. the initial `generate_alert()` call returned None) simply finds
    nothing to update — logged at debug level, not an error, since this is
    expected to happen occasionally.
    """
    from sqlalchemy import select
    from app.database import async_session_factory
    from app.models.security_event import SecurityEvent
    from app.models.alert import Alert

    camera_id = update["camera_id"]
    activity = update["activity"]
    action = update["action"]
    now_dt = datetime.fromtimestamp(event_time, tz=timezone.utc) if event_time else datetime.now(timezone.utc)

    async with async_session_factory() as db:
        try:
            if action == "update_clip":
                event_row = (await db.execute(
                    select(SecurityEvent)
                    .where(SecurityEvent.camera_id == camera_id, SecurityEvent.activity_class == activity)
                    .order_by(SecurityEvent.id.desc()).limit(1)
                )).scalar_one_or_none()
                alert_row = (await db.execute(
                    select(Alert)
                    .where(Alert.camera_id == camera_id, Alert.activity == activity)
                    .order_by(Alert.id.desc()).limit(1)
                )).scalar_one_or_none()
                if event_row is None and alert_row is None:
                    logger.debug(f"update_clip for camera {camera_id}/{activity} found no matching row.")
                    return
                for row in (event_row, alert_row):
                    if row is None:
                        continue
                    if update.get("video_clip_path"):
                        row.video_clip_path = update["video_clip_path"]
                    if update.get("thumbnail_path"):
                        row.thumbnail_path = update["thumbnail_path"]
                await db.commit()
                logger.info(f"Camera {camera_id}: event '{activity}' clip updated to {update.get('video_clip_path')}.")
                return

            end_time_filter = SecurityEvent.end_time.is_(None) if action != "reopen" else SecurityEvent.end_time.is_not(None)
            alert_end_time_filter = Alert.end_time.is_(None) if action != "reopen" else Alert.end_time.is_not(None)

            event_row = (await db.execute(
                select(SecurityEvent)
                .where(
                    SecurityEvent.camera_id == camera_id,
                    SecurityEvent.activity_class == activity,
                    end_time_filter,
                )
                .order_by(SecurityEvent.id.desc())
                .limit(1)
            )).scalar_one_or_none()
            alert_row = (await db.execute(
                select(Alert)
                .where(
                    Alert.camera_id == camera_id,
                    Alert.activity == activity,
                    alert_end_time_filter,
                )
                .order_by(Alert.id.desc())
                .limit(1)
            )).scalar_one_or_none()

            if event_row is None and alert_row is None:
                logger.debug(
                    f"Event update '{action}' for camera {camera_id}/{activity} "
                    "found no matching row — nothing to update."
                )
                return

            for row in (event_row, alert_row):
                if row is None:
                    continue
                row.peak_confidence = update.get("peak_confidence")
                row.average_confidence = update.get("average_confidence")
                row.duration_seconds = update.get("duration_seconds")
                if action == "finalize":
                    row.end_time = now_dt
                elif action == "reopen":
                    row.end_time = None
            # Cumulative fire_detected/smoke_detected (see
            # ai_service/inference_engine.py's combined_metadata) — only
            # SecurityEvent carries a metadata column; Alert doesn't.
            if event_row is not None and "metadata" in update:
                event_row.metadata_json = update["metadata"]

            await db.commit()
            logger.debug(
                f"Camera {camera_id}: event '{activity}' {action}d "
                f"(peak={update.get('peak_confidence')}, avg={update.get('average_confidence')})."
            )
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to persist event update ({action}) for camera {camera_id}/{activity}: {e}")
