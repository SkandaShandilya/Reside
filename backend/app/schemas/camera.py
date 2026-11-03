from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class CameraBase(BaseModel):
    name: str
    location: str
    description: Optional[str] = None
    source_type: str  # rtsp, webcam, upload, network, nvr
    resolution: Optional[str] = "1920x1080"
    fps: int = 30
    ai_processing_fps: int = 30


class CameraCreate(CameraBase):
    rtsp_url: Optional[str] = None
    network_url: Optional[str] = None
    nvr_address: Optional[str] = None
    nvr_channel: Optional[int] = None
    nvr_stream: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ai_enabled: bool = True
    recording_enabled: bool = True
    motion_detection: bool = True
    retention_days: int = 30
    detection_config: Optional[Dict[str, Any]] = None
    alert_confidence_threshold: float = 0.80
    cooldown_seconds: int = 30


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    ai_processing_fps: Optional[int] = None
    ai_enabled: Optional[bool] = None
    recording_enabled: Optional[bool] = None
    motion_detection: Optional[bool] = None
    retention_days: Optional[int] = None
    detection_config: Optional[Dict[str, Any]] = None
    alert_confidence_threshold: Optional[float] = None
    cooldown_seconds: Optional[int] = None
    is_active: Optional[bool] = None


class CameraResponse(CameraBase):
    id: int
    is_online: bool
    is_active: bool
    ai_enabled: bool
    recording_enabled: bool
    motion_detection: bool
    retention_days: int
    detection_config: Optional[Dict[str, Any]] = None
    alert_confidence_threshold: float
    cooldown_seconds: int
    created_at: datetime
    updated_at: datetime
    # Never expose credentials
    has_credentials: bool = False

    class Config:
        from_attributes = True


class CameraTestRequest(BaseModel):
    source_type: str
    rtsp_url: Optional[str] = None
    network_url: Optional[str] = None
    nvr_address: Optional[str] = None
    nvr_channel: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None


class CameraTestResponse(BaseModel):
    success: bool
    message: str
    resolution: Optional[str] = None
    fps: Optional[float] = None


class CameraZoneCreate(BaseModel):
    name: str
    zone_type: str = "restricted"
    polygon_points: List[Dict[str, float]]
    detection_rules: Optional[Dict[str, Any]] = None
    time_restrictions: Optional[Dict[str, Any]] = None
    alert_severity: str = "critical"
    description: Optional[str] = None


class CameraZoneResponse(CameraZoneCreate):
    id: int
    camera_id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
