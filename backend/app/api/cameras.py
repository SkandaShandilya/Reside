from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
from app.dependencies import get_db, get_current_user, require_admin, require_guard
from app.models.user import User, UserRole
from app.models.camera import Camera, CameraSourceType
from app.models.camera_zone import CameraZone
from app.schemas.camera import (
    CameraCreate, CameraUpdate, CameraResponse,
    CameraTestRequest, CameraTestResponse,
    CameraZoneCreate, CameraZoneResponse,
)
from app.utils.security import encrypt_secret, decrypt_secret

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.post("/upload-video", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def upload_camera_video(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    resolution: Optional[str] = Form("1920x1080"),
    fps: Optional[str] = Form("30"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Upload a video file to create a dedicated camera video source."""
    from pathlib import Path
    import shutil
    import uuid

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    from app.config import settings
    upload_dir = Path(settings.STORAGE_PATH) / "videos" / "uploaded_cameras"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_filename = f"cam_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = upload_dir / saved_filename

    # Stream-write in 1 MB chunks to avoid loading entire file into RAM
    try:
        with dest_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as write_err:
        # Clean up partial file on error
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {write_err}",
        )

    try:
        parsed_fps = int(fps) if fps and str(fps).isdigit() else 30
    except ValueError:
        parsed_fps = 30

    db_camera = Camera(
        name=name or "Uploaded Camera",
        location=location or "Local Video Feed",
        description=description or f"Uploaded video file: {file.filename}",
        source_type=CameraSourceType.UPLOAD,
        upload_path=str(dest_path.resolve()),
        resolution=resolution or "1920x1080",
        fps=parsed_fps,
        ai_processing_fps=30,
        is_online=True,  # Video file on disk is online & active
        is_active=True,
        ai_enabled=True,
        recording_enabled=True,
        motion_detection=True,
        created_by=current_user.id,
    )

    db.add(db_camera)
    await db.flush()
    await db.refresh(db_camera)

    # Kick off the full-timeline analysis (Phase 2) so the entire uploaded
    # file gets scanned start-to-finish for events, independent of whether
    # anyone ever opens the live-style MJPEG preview of it. Fire-and-forget
    # background thread — never blocks this response. A duplicate/second
    # trigger (e.g. via POST /{id}/analyze) is a no-op while one is already
    # running for this camera (see video_timeline_processor.is_running).
    from ai_service.video_timeline_processor import start_full_timeline_analysis
    start_full_timeline_analysis(
        camera_id=db_camera.id,
        camera_name=db_camera.name,
        camera_location=db_camera.location,
        video_path=str(dest_path.resolve()),
    )

    return CameraResponse(
        id=db_camera.id, name=db_camera.name, location=db_camera.location,
        description=db_camera.description, source_type=db_camera.source_type.value,
        resolution=db_camera.resolution, fps=db_camera.fps,
        ai_processing_fps=db_camera.ai_processing_fps, is_online=db_camera.is_online,
        is_active=db_camera.is_active, ai_enabled=db_camera.ai_enabled,
        recording_enabled=db_camera.recording_enabled, motion_detection=db_camera.motion_detection,
        retention_days=db_camera.retention_days, detection_config=db_camera.detection_config,
        alert_confidence_threshold=db_camera.alert_confidence_threshold,
        cooldown_seconds=db_camera.cooldown_seconds,
        created_at=db_camera.created_at, updated_at=db_camera.updated_at,
        has_credentials=False,
    )


@router.get("", response_model=list[CameraResponse])

async def list_cameras(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Camera).order_by(Camera.created_at.desc()))
    cameras = result.scalars().all()
    return [
        CameraResponse(
            id=c.id, name=c.name, location=c.location, description=c.description,
            source_type=c.source_type.value, resolution=c.resolution, fps=c.fps,
            ai_processing_fps=c.ai_processing_fps, is_online=c.is_online,
            is_active=c.is_active, ai_enabled=c.ai_enabled,
            recording_enabled=c.recording_enabled, motion_detection=c.motion_detection,
            retention_days=c.retention_days, detection_config=c.detection_config,
            alert_confidence_threshold=c.alert_confidence_threshold,
            cooldown_seconds=c.cooldown_seconds,
            created_at=c.created_at, updated_at=c.updated_at,
            has_credentials=bool(c.username),
        )
        for c in cameras
    ]


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera: CameraCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    db_camera = Camera(
        name=camera.name,
        location=camera.location,
        description=camera.description,
        source_type=CameraSourceType(camera.source_type),
        rtsp_url=camera.rtsp_url,
        network_url=camera.network_url,
        nvr_address=camera.nvr_address,
        nvr_channel=camera.nvr_channel,
        nvr_stream=camera.nvr_stream,
        username=camera.username,
        encrypted_password=encrypt_secret(camera.password),
        resolution=camera.resolution,
        fps=camera.fps,
        ai_processing_fps=camera.ai_processing_fps,
        ai_enabled=camera.ai_enabled,
        recording_enabled=camera.recording_enabled,
        motion_detection=camera.motion_detection,
        retention_days=camera.retention_days,
        detection_config=camera.detection_config or {},
        alert_confidence_threshold=camera.alert_confidence_threshold,
        cooldown_seconds=camera.cooldown_seconds,
        is_online=True if camera.source_type in (CameraSourceType.UPLOAD, "upload", "webcam") else False,
        is_active=True,
        created_by=current_user.id,
    )
    db.add(db_camera)
    await db.flush()
    await db.refresh(db_camera)
    return CameraResponse(
        id=db_camera.id, name=db_camera.name, location=db_camera.location,
        description=db_camera.description, source_type=db_camera.source_type.value,
        resolution=db_camera.resolution, fps=db_camera.fps,
        ai_processing_fps=db_camera.ai_processing_fps, is_online=db_camera.is_online,
        is_active=db_camera.is_active, ai_enabled=db_camera.ai_enabled,
        recording_enabled=db_camera.recording_enabled, motion_detection=db_camera.motion_detection,
        retention_days=db_camera.retention_days, detection_config=db_camera.detection_config,
        alert_confidence_threshold=db_camera.alert_confidence_threshold,
        cooldown_seconds=db_camera.cooldown_seconds,
        created_at=db_camera.created_at, updated_at=db_camera.updated_at,
        has_credentials=bool(db_camera.username),
    )


@router.get("/{camera_id}", response_model=CameraResponse)


async def get_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return CameraResponse(
        id=camera.id, name=camera.name, location=camera.location,
        description=camera.description, source_type=camera.source_type.value,
        resolution=camera.resolution, fps=camera.fps,
        ai_processing_fps=camera.ai_processing_fps, is_online=camera.is_online,
        is_active=camera.is_active, ai_enabled=camera.ai_enabled,
        recording_enabled=camera.recording_enabled, motion_detection=camera.motion_detection,
        retention_days=camera.retention_days, detection_config=camera.detection_config,
        alert_confidence_threshold=camera.alert_confidence_threshold,
        cooldown_seconds=camera.cooldown_seconds,
        created_at=camera.created_at, updated_at=camera.updated_at,
        has_credentials=bool(camera.username),
    )


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera_update: CameraUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    update_data = camera_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)
    await db.flush()
    await db.refresh(camera)

    if "ai_enabled" in update_data:
        from app.services.video_stream_service import video_stream_service
        video_stream_service.set_ai_enabled(camera_id, camera.ai_enabled)

    if "detection_config" in update_data:
        # Fire/Smoke toggles take effect on an already-running stream
        # immediately, same as ai_enabled above — no restart required.
        from app.services.video_stream_service import video_stream_service
        dc = camera.detection_config or {}
        video_stream_service.set_fire_smoke_enabled(
            camera_id,
            fire_enabled=dc.get("fire_enabled", True),
            smoke_enabled=dc.get("smoke_enabled", True),
        )

    return CameraResponse(
        id=camera.id, name=camera.name, location=camera.location,
        description=camera.description, source_type=camera.source_type.value,
        resolution=camera.resolution, fps=camera.fps,
        ai_processing_fps=camera.ai_processing_fps, is_online=camera.is_online,
        is_active=camera.is_active, ai_enabled=camera.ai_enabled,
        recording_enabled=camera.recording_enabled, motion_detection=camera.motion_detection,
        retention_days=camera.retention_days, detection_config=camera.detection_config,
        alert_confidence_threshold=camera.alert_confidence_threshold,
        cooldown_seconds=camera.cooldown_seconds,
        created_at=camera.created_at, updated_at=camera.updated_at,
        has_credentials=bool(camera.username),
    )


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    await db.delete(camera)
    await db.flush()

    # The background capture/AI thread for this camera doesn't know its row
    # is gone and keeps running otherwise — it then keeps trying to insert
    # SecurityEvent/Alert rows referencing a camera_id that no longer
    # exists, which fails outright once FK enforcement is on (and silently
    # produced orphaned rows before that). Stop it explicitly on delete.
    from app.services.video_stream_service import video_stream_service
    video_stream_service.stop_stream(camera_id)


@router.post("/{camera_id}/test", response_model=CameraTestResponse)
async def test_camera_connection(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    from app.services.video_stream_service import video_stream_service
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Resolve source URL
    source_url = ""
    if camera.source_type == CameraSourceType.RTSP:
        source_url = camera.rtsp_url or ""
    elif camera.source_type == CameraSourceType.NETWORK:
        source_url = camera.network_url or camera.rtsp_url or ""
    elif camera.source_type == CameraSourceType.WEBCAM:
        source_url = "0"
    elif camera.source_type == CameraSourceType.UPLOAD:
        source_url = camera.upload_path or ""
    elif camera.source_type == CameraSourceType.NVR:
        source_url = f"rtsp://{camera.nvr_address}/channel={camera.nvr_channel or 1}"

    test_result = await video_stream_service.test_connection(
        source_url=source_url,
        source_type=camera.source_type.value,
        username=camera.username or "",
        password=decrypt_secret(camera.encrypted_password) or "",
        timeout=8.0,
    )

    # Update camera online status
    camera.is_online = test_result["success"]
    await db.flush()

    return CameraTestResponse(
        success=test_result["success"],
        message="Connection successful" if test_result["success"] else f"Connection failed: {test_result.get('error', 'Unknown error')}",
        resolution=test_result.get("resolution") or camera.resolution,
        fps=test_result.get("fps") or float(camera.fps),
    )


# ---- Uploaded-video full-timeline analysis (Phase 2) ----
# Runs automatically once on upload (see upload_camera_video above); these
# let a Guard/Admin re-trigger it (e.g. after re-uploading, or if it failed)
# and poll its progress. Only meaningful for UPLOAD-source cameras — live
# RTSP/webcam/NVR sources are analyzed continuously while streamed instead
# (see backend/app/services/video_stream_service.py), never via this path.

@router.post("/{camera_id}/analyze")
async def trigger_full_timeline_analysis(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    from ai_service.video_timeline_processor import start_full_timeline_analysis

    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    if camera.source_type != CameraSourceType.UPLOAD or not camera.upload_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Full-timeline analysis is only available for uploaded video cameras.",
        )

    started = start_full_timeline_analysis(
        camera_id=camera.id,
        camera_name=camera.name,
        camera_location=camera.location,
        video_path=camera.upload_path,
    )
    return {
        "started": started,
        "message": "Analysis started." if started else "Analysis already running, or the video file was not found.",
    }


@router.get("/{camera_id}/analyze/status")
async def get_full_timeline_analysis_status(
    camera_id: int,
    current_user: User = Depends(require_guard),
):
    from ai_service.video_timeline_processor import get_job_status

    job = get_job_status(camera_id)
    if job is None:
        return {"status": "idle", "progress_percent": 0}
    return {
        "status": job.status,
        "progress_percent": job.progress_percent,
        "current_offset_seconds": job.current_offset_seconds,
        "total_duration_seconds": job.total_duration_seconds,
        "frames_sampled": job.frames_sampled,
        "events_detected": job.events_detected,
        "error": job.error,
    }


# ---- Camera Zones ----

@router.get("/{camera_id}/zones", response_model=list[CameraZoneResponse])
async def list_zones(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(
        select(CameraZone).where(CameraZone.camera_id == camera_id).order_by(CameraZone.created_at)
    )
    return result.scalars().all()


@router.post("/{camera_id}/zones", response_model=CameraZoneResponse, status_code=status.HTTP_201_CREATED)
async def create_zone(
    camera_id: int,
    zone: CameraZoneCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    # Verify camera exists
    cam_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    db_zone = CameraZone(
        camera_id=camera_id,
        name=zone.name,
        zone_type=zone.zone_type,
        polygon_points=zone.polygon_points,
        detection_rules=zone.detection_rules or {},
        time_restrictions=zone.time_restrictions,
        alert_severity=zone.alert_severity,
        description=zone.description,
    )
    db.add(db_zone)
    await db.flush()
    await db.refresh(db_zone)
    await _refresh_live_zones(camera_id, db)
    return db_zone


@router.delete("/{camera_id}/zones/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_zone(
    camera_id: int,
    zone_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(
        select(CameraZone).where(CameraZone.id == zone_id, CameraZone.camera_id == camera_id)
    )
    zone = result.scalar_one_or_none()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    await db.delete(zone)
    await db.flush()
    await _refresh_live_zones(camera_id, db)


async def _refresh_live_zones(camera_id: int, db: AsyncSession) -> None:
    """
    Push the camera's current zone list into any already-running stream so
    a newly drawn/removed zone takes effect immediately, without needing
    to restart the camera's stream.
    """
    from app.services.video_stream_service import video_stream_service

    result = await db.execute(
        select(CameraZone).where(CameraZone.camera_id == camera_id, CameraZone.is_active == True)  # noqa: E712
    )
    zones = result.scalars().all()
    zone_dicts = [
        {
            "name": z.name,
            "zone_type": z.zone_type.value if hasattr(z.zone_type, "value") else str(z.zone_type),
            "polygon_points": z.polygon_points,
        }
        for z in zones
    ]
    video_stream_service.set_zones(camera_id, zone_dicts)
