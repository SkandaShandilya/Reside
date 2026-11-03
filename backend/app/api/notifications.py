from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.dependencies import get_db, require_admin
from app.models.user import User
from app.models.notification_setting import NotificationSetting
from app.schemas.notification import (
    NotificationSettingResponse, NotificationSettingCreate,
    NotificationSettingUpdate, NotificationSettingsListResponse,
)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


@router.get("/settings", response_model=NotificationSettingsListResponse)
async def get_notification_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(NotificationSetting).order_by(NotificationSetting.activity_type))
    settings = result.scalars().all()
    return NotificationSettingsListResponse(
        settings=[NotificationSettingResponse.model_validate(s) for s in settings]
    )


@router.post("/settings", response_model=NotificationSettingResponse, status_code=201)
async def create_notification_setting(
    body: NotificationSettingCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    setting = NotificationSetting(
        activity_type=body.activity_type,
        notify_resident=body.notify_resident,
        notify_guard=body.notify_guard,
        notify_admin=body.notify_admin,
        channel=body.channel,
        is_active=body.is_active,
        severity_filter=body.severity_filter,
        message_template=body.message_template,
    )
    db.add(setting)
    await db.flush()
    await db.refresh(setting)
    return NotificationSettingResponse.model_validate(setting)


@router.put("/settings/{setting_id}", response_model=NotificationSettingResponse)
async def update_notification_setting(
    setting_id: int,
    body: NotificationSettingUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(NotificationSetting).where(NotificationSetting.id == setting_id))
    setting = result.scalar_one_or_none()
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(setting, field, value)
    await db.flush()
    await db.refresh(setting)
    return NotificationSettingResponse.model_validate(setting)
