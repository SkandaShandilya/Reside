"""
Stream API Router — Phase 2

MJPEG streaming endpoints for live camera feeds.

Endpoints:
  GET /api/stream/{camera_id}         - MJPEG live stream for a single camera
  POST /api/stream/{camera_id}/start  - Start background capture for a camera
  POST /api/stream/{camera_id}/stop   - Stop background capture for a camera
  GET /api/stream/status              - Status of all running streams
  GET /api/stream/{camera_id}/status  - Status of one camera stream
  POST /api/stream/{camera_id}/snapshot - Return single JPEG snapshot
"""
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.database import get_async_session
from app.models.camera import Camera, CameraSourceType
from app.models.camera_zone import CameraZone
from app.models.user import User, UserRole
from app.dependencies import require_role, require_guard
from app.utils.security import decode_access_token, decrypt_secret
from app.services.video_stream_service import video_stream_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stream", tags=["Stream"])

STREAM_FPS = 30  # Match source FPS for smooth playback



_bearer = HTTPBearer(auto_error=False)


async def get_stream_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(None, description="JWT token for browser img src auth"),
    db: AsyncSession = Depends(get_async_session),
) -> User:
    """
    Auth dependency for MJPEG stream endpoints.
    Accepts token from Authorization header OR ?token= query param
    (browser <img> tags cannot send custom headers).
    """
    raw_token = None
    if credentials:
        raw_token = credentials.credentials
    elif token and token not in ("null", "undefined", ""):
        raw_token = token

    if raw_token:
        payload = decode_access_token(raw_token)
        if payload and payload.get("sub"):
            user_id = payload.get("sub")
            try:
                from sqlalchemy import select as _select
                result = await db.execute(_select(User).where(User.id == int(user_id)))
                user = result.scalar_one_or_none()
                if user and user.is_active:
                    # Live camera feeds are a security-monitoring surface —
                    # a Resident must not be able to view them, whether via
                    # the app UI or a hand-crafted request against this
                    # endpoint directly (hence the check here, not just in
                    # the frontend route).
                    if user.role == UserRole.RESIDENT:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Residents do not have access to live camera feeds.",
                        )
                    return user
            except HTTPException:
                raise
            except Exception:
                pass

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


async def _get_camera_or_404(camera_id: int, session: AsyncSession) -> Camera:
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return cam


async def _fetch_zones_as_dicts(camera_id: int, session: AsyncSession) -> list:
    """Load a camera's active zones as plain dicts for the AI pipeline."""
    result = await session.execute(
        select(CameraZone).where(CameraZone.camera_id == camera_id, CameraZone.is_active == True)  # noqa: E712
    )
    zones = result.scalars().all()
    return [
        {
            "name": z.name,
            "zone_type": z.zone_type.value if hasattr(z.zone_type, "value") else str(z.zone_type),
            "polygon_points": z.polygon_points,
        }
        for z in zones
    ]


# ---------------------------------------------------------------------------
# MJPEG Live Stream
# ---------------------------------------------------------------------------

@router.get("/{camera_id}", summary="MJPEG live stream for a camera")
async def live_stream(
    camera_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_stream_user),
):

    """
    Returns a multipart/x-mixed-replace MJPEG stream.
    Use as the `src` of an <img> tag in the browser.

    The stream manager auto-starts the camera capture if not already running.
    """
    cam = await _get_camera_or_404(camera_id, session)

    # Auto-start capture if not already running
    status_info = video_stream_service.get_status(camera_id)
    if not status_info["running"]:
        source_url = _resolve_source_url(cam)
        zones = await _fetch_zones_as_dicts(camera_id, session)
        started = video_stream_service.start_stream(
            camera_id=camera_id,
            source_url=source_url,
            source_type=cam.source_type.value,
            username=cam.username or "",
            password=decrypt_secret(cam.encrypted_password) or "",
            ai_enabled=cam.ai_enabled,
            zones=zones,
            detection_config=cam.detection_config,
        )
        logger.info(f"Auto-started stream for camera {camera_id}: {started}")

    return StreamingResponse(
        video_stream_service.frame_generator(camera_id=camera_id, fps=STREAM_FPS),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/fire-smoke", summary="MJPEG stream with ONLY the fire/smoke overlay")
async def live_stream_fire_smoke(
    camera_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_stream_user),
):
    """
    Same camera feed as the main stream, but rendered with
    inference_engine.render_fire_smoke_only() — no person boxes, behaviour
    HUD, zones, or "AI:" suggestion strip. For the dedicated Fire & Smoke
    tab, which should show fire/smoke evidence with nothing else on frame.
    """
    cam = await _get_camera_or_404(camera_id, session)

    status_info = video_stream_service.get_status(camera_id)
    if not status_info["running"]:
        source_url = _resolve_source_url(cam)
        zones = await _fetch_zones_as_dicts(camera_id, session)
        started = video_stream_service.start_stream(
            camera_id=camera_id,
            source_url=source_url,
            source_type=cam.source_type.value,
            username=cam.username or "",
            password=decrypt_secret(cam.encrypted_password) or "",
            ai_enabled=cam.ai_enabled,
            zones=zones,
            detection_config=cam.detection_config,
        )
        logger.info(f"Auto-started stream for camera {camera_id}: {started}")

    return StreamingResponse(
        video_stream_service.frame_generator(camera_id=camera_id, fps=STREAM_FPS, variant="fire_smoke"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket frame push — smooth AND scalable
# ---------------------------------------------------------------------------
# The MJPEG endpoints above give smooth playback but each holds one HTTP/1.1
# connection open forever, and browsers cap concurrent connections to ~6 per
# host SHARED ACROSS EVERY OPEN TAB — with Live Monitoring's grid, the Fire &
# Smoke grid, and any detail views open together, feeds beyond the first ~6
# never connected at all (see hooks/useSnapshotStream.ts, added as the fix).
# Polling a single-JPEG snapshot on an interval fixed that (each request
# completes immediately, no held-open slot) but trades away smoothness —
# a human eye reads anything slower than ~10fps as "choppy", and a poll
# interval loose enough to avoid hammering the server (300-500ms, 2-3fps)
# reads as choppy AND visibly delayed.
#
# A WebSocket sidesteps the tradeoff entirely: it is not counted against the
# browser's HTTP/1.1 per-host connection limit (browsers allow far more
# concurrent WebSocket connections per origin), so it scales the same way
# polling does, while pushing frames as fast as the capture loop actually
# produces them (capped below at 20fps purely to bound bandwidth/CPU, not to
# work around a connection limit) — no polling delay, no choppiness.
@router.websocket("/{camera_id}/ws")
async def stream_ws(
    websocket: WebSocket,
    camera_id: int,
    variant: str = "full",
    token: Optional[str] = None,
):
    """
    Pushes JPEG frames as binary WebSocket messages, one per available new
    frame, throttled to at most ~60fps. `variant=fire_smoke` selects the
    fire/smoke-only overlay buffer, matching the MJPEG/snapshot endpoints'
    same-named param. Auth via `?token=` (a WebSocket can't send a custom
    Authorization header) — same reasoning as get_stream_user for the
    plain HTTP endpoints above, just applied manually since FastAPI's
    Depends-based auth doesn't attach to websocket routes the same way.
    """
    if not token:
        await websocket.close(code=4401)
        return
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        await websocket.close(code=4401)
        return
    # Same restriction as get_stream_user above, applied manually since
    # this is a raw websocket route with no Depends() auth — a Resident's
    # otherwise-valid token must still not open a live frame feed.
    if payload.get("role") == "resident":
        await websocket.close(code=4403)
        return

    await websocket.accept()

    from app.database import async_session_factory
    async with async_session_factory() as session:
        cam_result = await session.execute(select(Camera).where(Camera.id == camera_id))
        cam = cam_result.scalar_one_or_none()
        if not cam:
            await websocket.close(code=4404)
            return

        status_info = video_stream_service.get_status(camera_id)
        if not status_info["running"]:
            source_url = _resolve_source_url(cam)
            zones = await _fetch_zones_as_dicts(camera_id, session)
            video_stream_service.start_stream(
                camera_id=camera_id,
                source_url=source_url,
                source_type=cam.source_type.value,
                username=cam.username or "",
                password=decrypt_secret(cam.encrypted_password) or "",
                ai_enabled=cam.ai_enabled,
                zones=zones,
                detection_config=cam.detection_config,
            )

    max_fps = 60
    min_interval = 1.0 / max_fps
    last_sent_frame = None
    try:
        while True:
            loop_start = asyncio.get_event_loop().time()
            info = video_stream_service._streams.get(camera_id)
            source_buffer = (
                info.fire_smoke_buffer if (variant == "fire_smoke" and info and info.fire_smoke_buffer)
                else info.buffer if info else None
            )
            frame = source_buffer.latest() if source_buffer else None
            if frame is not None and frame is not last_sent_frame:
                await websocket.send_bytes(frame)
                last_sent_frame = frame
            elapsed = asyncio.get_event_loop().time() - loop_start
            await asyncio.sleep(max(0.0, min_interval - elapsed))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Camera {camera_id} stream_ws closed: {e}")


# ---------------------------------------------------------------------------
# Manual Stream Control
# ---------------------------------------------------------------------------

@router.post("/{camera_id}/start", summary="Start camera stream capture")
async def start_stream(
    camera_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user=Depends(require_role(UserRole.GUARD, UserRole.ADMIN, UserRole.SUPER_ADMIN)),
):
    cam = await _get_camera_or_404(camera_id, session)
    source_url = _resolve_source_url(cam)
    zones = await _fetch_zones_as_dicts(camera_id, session)

    started = video_stream_service.start_stream(
        camera_id=camera_id,
        source_url=source_url,
        source_type=cam.source_type.value,
        username=cam.username or "",
        password=decrypt_secret(cam.encrypted_password) or "",
        ai_enabled=cam.ai_enabled,
        zones=zones,
        detection_config=cam.detection_config,
    )
    return {"camera_id": camera_id, "started": started, "message": "Stream started" if started else "Already running"}


@router.post("/{camera_id}/stop", summary="Stop camera stream capture")
async def stop_stream(
    camera_id: int,
    current_user=Depends(require_role(UserRole.GUARD, UserRole.ADMIN, UserRole.SUPER_ADMIN)),
):
    video_stream_service.stop_stream(camera_id)
    return {"camera_id": camera_id, "stopped": True}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status/all", summary="Status of all running streams")
async def all_stream_status(current_user=Depends(require_guard)):
    return {"streams": video_stream_service.get_all_status()}


@router.get("/{camera_id}/status", summary="Status of a single camera stream")
async def camera_stream_status(
    camera_id: int,
    current_user=Depends(require_guard),
):
    return video_stream_service.get_status(camera_id)


@router.get("/{camera_id}/telemetry", summary="Latest AI telemetry for a camera (activity + fire/smoke state)")
async def camera_telemetry(
    camera_id: int,
    current_user=Depends(require_guard),
):
    """
    Snapshot of the most recent analysis pass — includes fire_state/
    smoke_state/fire_smoothed_confidence/smoke_smoothed_confidence (see
    ai_service/fire_smoke_state.py) alongside the existing behaviour
    telemetry. Polled by the frontend (e.g. the Fire & Smoke tab) rather
    than decoding the MJPEG stream to read status.
    """
    return video_stream_service.get_telemetry(camera_id)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@router.get("/{camera_id}/snapshot", summary="Get a single JPEG snapshot")
async def snapshot(
    camera_id: int,
    variant: Optional[str] = Query(None, description="'fire_smoke' for the fire/smoke-only overlay, default the full behaviour overlay"),
    session: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_stream_user),
):
    """
    Returns the latest JPEG frame as a single image response — the
    poll-based alternative to the MJPEG endpoints above (see frontend
    hooks/useSnapshotStream.ts and pages/Cameras.tsx's CameraThumbnail for
    why: a persistent multipart MJPEG response never completes, so each one
    permanently occupies one of the browser's shared per-host HTTP/1.1
    connection slots — a plain single-JPEG request completes immediately
    and releases its slot, so effectively unlimited feeds can poll this
    concurrently across as many open tabs as the app has). `variant`
    mirrors the MJPEG endpoints' same-named param, reading from the same
    fire_smoke_buffer used by /fire-smoke instead of the behaviour buffer.
    """
    import asyncio
    cam = await _get_camera_or_404(camera_id, session)

    # Auto-start if needed
    status_info = video_stream_service.get_status(camera_id)
    if not status_info["running"]:
        source_url = _resolve_source_url(cam)
        video_stream_service.start_stream(
            camera_id=camera_id,
            source_url=source_url,
            source_type=cam.source_type.value if hasattr(cam.source_type, 'value') else str(cam.source_type),
            username=cam.username or "",
            password=decrypt_secret(cam.encrypted_password) or "",
            ai_enabled=cam.ai_enabled,
            detection_config=cam.detection_config,
        )

    jpeg = None
    for _ in range(25):  # Wait up to 2.5s for first frame
        frame_info = video_stream_service._streams.get(camera_id)
        source_buffer = (
            frame_info.fire_smoke_buffer if (variant == "fire_smoke" and frame_info and frame_info.fire_smoke_buffer)
            else frame_info.buffer if frame_info else None
        )
        jpeg = source_buffer.latest() if source_buffer else None
        if jpeg:
            break
        await asyncio.sleep(0.1)

    if not jpeg:
        jpeg = video_stream_service._placeholder_frame(camera_id)

    return Response(content=jpeg, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_source_url(cam: Camera) -> str:
    """Resolve the correct stream URL for a camera based on its source type."""
    from pathlib import Path
    from app.config import settings

    st_val = str(cam.source_type.value if hasattr(cam.source_type, 'value') else cam.source_type).lower()

    if st_val == "webcam":
        return "0"
    elif st_val == "rtsp":
        # Check if rtsp_url is actually a local video file path
        if cam.rtsp_url and (cam.rtsp_url.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) or Path(cam.rtsp_url).exists()):
            p = Path(cam.rtsp_url)
            if p.exists():
                return str(p.resolve())
        return cam.rtsp_url or ""
    elif st_val == "network":
        return cam.network_url or cam.rtsp_url or ""
    elif st_val == "nvr":
        if cam.nvr_address:
            ch = cam.nvr_channel or 1
            return f"rtsp://{cam.nvr_address}/channel={ch}"
        return cam.rtsp_url or ""
    elif st_val in ("upload", "file"):
        path_str = cam.upload_path or cam.rtsp_url or cam.network_url or ""
        if path_str:
            raw_path = Path(path_str)
            if raw_path.exists():
                return str(raw_path.resolve())

            storage_path = Path(settings.STORAGE_PATH)
            candidates = [
                storage_path / "videos" / "uploaded_cameras" / raw_path.name,
                storage_path / "videos" / raw_path.name,
                storage_path / "training_samples" / raw_path.name,
                storage_path / "recordings" / raw_path.name,
                storage_path / raw_path.name,
                Path("e:/MajorProject_27/backend/storage/videos/uploaded_cameras") / raw_path.name,
                Path("e:/MajorProject_27/backend/storage/training_samples") / raw_path.name,
                Path("e:/MajorProject_27/storage/videos/uploaded_cameras") / raw_path.name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate.resolve())

            # Check inside subfolders of training_samples
            training_dir = storage_path / "training_samples"
            if training_dir.exists():
                for sub in training_dir.rglob(f"*{raw_path.name}*"):
                    if sub.is_file():
                        return str(sub.resolve())
        return path_str
    return cam.upload_path or cam.rtsp_url or ""
