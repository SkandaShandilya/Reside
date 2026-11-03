"""
System Settings API — the AI-processing knobs that are genuinely wired
into the live detection pipeline (see how each field is applied below).
Anything NOT in this model (storage backend, HTTPS, IP restriction, etc.)
is infrastructure-level configuration this project doesn't actually
control at runtime, so it deliberately isn't exposed here rather than
pretending to save it.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.dependencies import get_db, get_current_user, require_admin
from app.models.system_setting import SystemSetting
from app.schemas.system_setting import SystemSettingResponse, SystemSettingUpdate

router = APIRouter(prefix="/api/settings", tags=["Settings"])


async def _get_or_create_settings(db: AsyncSession) -> SystemSetting:
    result = await db.execute(select(SystemSetting).where(SystemSetting.id == 1))
    settings = result.scalar_one_or_none()
    if not settings:
        settings = SystemSetting(id=1)
        db.add(settings)
        await db.flush()
        await db.refresh(settings)
    return settings


@router.get("", response_model=SystemSettingResponse)
async def get_settings(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    return await _get_or_create_settings(db)


@router.put("", response_model=SystemSettingResponse)
async def update_settings(
    body: SystemSettingUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    """Persist the new settings AND push them into the already-running
    detection pipeline immediately — no server restart required."""
    settings = await _get_or_create_settings(db)
    settings.ai_analysis_fps = body.ai_analysis_fps
    settings.confidence_threshold = body.confidence_threshold
    settings.alert_cooldown_seconds = body.alert_cooldown_seconds
    await db.flush()
    await db.refresh(settings)

    apply_settings_live(settings)
    return settings


def apply_settings_live(settings: SystemSetting) -> None:
    """
    Push persisted settings into the live singletons that actually use them.

    Note: mutating AIConfig class attributes alone does NOT reach
    video_stream_service — its capture-loop threads read a plain instance
    attribute (`self.ai_analysis_fps`) that's copied from AIConfig once at
    construction time, not re-read from AIConfig on every loop. Without the
    explicit set_ai_analysis_fps() call below, changing "AI Analysis Rate"
    in System Settings would silently do nothing to already-running streams.
    """
    try:
        from ai_service.config import AIConfig
        AIConfig.AI_ANALYSIS_FPS = settings.ai_analysis_fps
        AIConfig.CONFIDENCE_THRESHOLD = settings.confidence_threshold
        AIConfig.ALERT_COOLDOWN_SECONDS = settings.alert_cooldown_seconds
    except Exception:
        pass

    try:
        import sys
        if "app.services.video_stream_service" in sys.modules:
            from app.services.video_stream_service import video_stream_service
            video_stream_service.set_ai_analysis_fps(settings.ai_analysis_fps)
    except Exception:
        pass

    try:
        import sys
        if "ai_service.inference_engine" in sys.modules:
            from ai_service.inference_engine import inference_engine
            inference_engine.behaviour_engine.confidence_threshold = settings.confidence_threshold
            inference_engine.alert_generator.confidence_threshold = settings.confidence_threshold
            inference_engine.alert_generator.cooldown_seconds = settings.alert_cooldown_seconds
    except Exception:
        pass
