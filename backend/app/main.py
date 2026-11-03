import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import settings
from app.database import init_db, close_db, migrate_legacy_face_registrations
from app.api import auth, cameras, alerts, incidents, review, models, analytics, notifications, users, stream, settings as settings_api
from app.api import faces, vehicles, vehicle_requests, admin, visitor_requests, profile
from app.models.user import User
from app.websocket import security_alerts
from app.services.video_stream_service import video_stream_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Sentinel Security System...")

    # Register this event loop so the (synchronous, background-thread)
    # camera capture loops can schedule alert DB writes + WebSocket
    # broadcasts onto it — see app/services/alert_dispatcher.py.
    import asyncio
    from app.services import alert_dispatcher
    alert_dispatcher.set_main_loop(asyncio.get_running_loop())

    # Create tables
    await init_db()
    # Ensure storage directories exist
    storage_path = Path(settings.STORAGE_PATH)
    for subdir in ["snapshots", "clips", "recordings", "training_samples", "reviewed_samples"]:
        (storage_path / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("Database initialized, storage ready.")

    # One-time migration: copy any pre-existing single-face User
    # registrations into the new household RegisteredPerson model (see
    # app/database.py's migrate_legacy_face_registrations() docstring).
    migrated_count = await migrate_legacy_face_registrations()
    if migrated_count:
        logger.info(f"Migrated {migrated_count} legacy face registration(s) into RegisteredPerson.")

    # Apply any persisted System Settings to the live detection pipeline
    from app.database import async_session_factory
    from app.api.settings import _get_or_create_settings, apply_settings_live
    async with async_session_factory() as db:
        saved_settings = await _get_or_create_settings(db)
        await db.commit()
        apply_settings_live(saved_settings)
    logger.info(
        f"Applied system settings: AI_ANALYSIS_FPS={saved_settings.ai_analysis_fps}, "
        f"confidence_threshold={saved_settings.confidence_threshold}, "
        f"alert_cooldown_seconds={saved_settings.alert_cooldown_seconds}"
    )
    yield
    logger.info("Shutting down Sentinel Security System...")
    video_stream_service.stop_all()
    await close_db()


app = FastAPI(
    title="Sentinel — Apartment Security System",
    description="Real-Time Suspicious Activity Detection for Apartment Management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Storage access — snapshots/clips/recordings/training data contain
# personally identifiable footage, so this must not be an unauthenticated
# StaticFiles mount (previously anyone could browse/download any file under
# ./storage with no login at all). Any authenticated user may fetch a file;
# path is resolved and checked to stay inside storage_path to block traversal.
#
# Auth uses get_stream_user (see app/api/stream.py), NOT get_current_user:
# this endpoint backs plain <video src="..."> / <img src="..."> tags for
# alert clips/snapshots (e.g. the Fire & Smoke tab, IncidentHistory), and a
# browser cannot attach a custom Authorization header to a media element's
# src — the same reason the MJPEG stream endpoints already accept the JWT
# as a ?token= query param instead. Reusing that existing dependency here
# (rather than duplicating the same header-or-query-param logic) was the
# actual root cause of alert clips never being viewable even when a good
# clip file existed on disk: get_current_user is header-only, so any
# <video>/<img> tag pointed at this route would always get a 401.
storage_path = Path(settings.STORAGE_PATH).resolve()
storage_path.mkdir(parents=True, exist_ok=True)


@app.get("/storage/{file_path:path}")
async def get_storage_file(file_path: str, current_user: User = Depends(stream.get_stream_user)):
    requested = (storage_path / file_path).resolve()
    try:
        requested.relative_to(storage_path)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    if not requested.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(requested)

# Register API routers
app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(alerts.router)
app.include_router(incidents.router)
app.include_router(review.router)
app.include_router(models.router)
app.include_router(analytics.router)
app.include_router(notifications.router)
app.include_router(users.router)
app.include_router(profile.router)
app.include_router(stream.router, prefix="/api")
app.include_router(settings_api.router)
app.include_router(faces.router)
app.include_router(vehicles.router)
app.include_router(vehicle_requests.router)
app.include_router(visitor_requests.router)
app.include_router(admin.router)

# Register WebSocket routers
app.include_router(security_alerts.router)


@app.get("/")
async def root():
    return {
        "service": "Sentinel — Apartment Security System API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health": "/api/health",
        "frontend": "http://localhost:5173",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Sentinel Security System",
        "version": "1.0.0",
    }
