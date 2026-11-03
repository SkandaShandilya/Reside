from pydantic import BaseModel, computed_field
from typing import Any, Dict, Optional, List
from datetime import datetime

from app.constants import event_category


class IncidentResponse(BaseModel):
    id: int
    camera_id: int
    activity_class: str
    confidence: float
    severity: str
    persons_detected: int
    duration_seconds: Optional[float] = None
    video_clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    model_version: Optional[str] = None
    timestamp: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_confidence: Optional[float] = None
    average_confidence: Optional[float] = None
    # Internal detection detail — e.g. a combined "Fire/Smoke Detected"
    # event's {"event_type": "fire_smoke", "fire_detected": bool,
    # "smoke_detected": bool} (see ai_service/inference_engine.py's
    # combined_metadata) — None for activities that have nothing to
    # preserve here. Existing SecurityEvent.metadata_json column, named
    # `metadata_json` on the Python/ORM side since `metadata` is reserved
    # on SQLAlchemy's declarative base.
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

    @computed_field
    @property
    def category(self) -> str:
        return event_category(self.activity_class)


class IncidentListResponse(BaseModel):
    incidents: List[IncidentResponse]
    total: int
    page: int
    page_size: int
