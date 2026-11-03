from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ReviewItemResponse(BaseModel):
    id: int
    event_id: Optional[int] = None
    alert_id: Optional[int] = None
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    camera_id: int
    camera_name: str
    original_prediction: str
    original_confidence: float
    all_predictions: Optional[str] = None
    verified_label: Optional[str] = None
    reviewer_id: Optional[int] = None
    model_version: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    added_to_training: bool = False
    timestamp: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewVerifyRequest(BaseModel):
    verified_label: str
    add_to_training: bool = True
    notes: Optional[str] = None


class ReviewListResponse(BaseModel):
    items: List[ReviewItemResponse]
    total: int
    page: int
    page_size: int
