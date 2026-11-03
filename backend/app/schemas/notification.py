from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class NotificationSettingResponse(BaseModel):
    id: int
    activity_type: str
    notify_resident: bool
    notify_guard: bool
    notify_admin: bool
    channel: str
    is_active: bool
    severity_filter: Optional[str] = None
    message_template: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class NotificationSettingUpdate(BaseModel):
    notify_resident: Optional[bool] = None
    notify_guard: Optional[bool] = None
    notify_admin: Optional[bool] = None
    channel: Optional[str] = None
    is_active: Optional[bool] = None
    severity_filter: Optional[str] = None
    message_template: Optional[str] = None


class NotificationSettingCreate(BaseModel):
    activity_type: str
    notify_resident: bool = False
    notify_guard: bool = True
    notify_admin: bool = True
    channel: str = "in_app"
    is_active: bool = True
    severity_filter: Optional[str] = None
    message_template: Optional[str] = None


class NotificationSettingsListResponse(BaseModel):
    settings: List[NotificationSettingResponse]
