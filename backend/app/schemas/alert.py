from pydantic import BaseModel, computed_field
from typing import Optional, List
from datetime import datetime

from app.constants import event_category


class AlertResponse(BaseModel):
    id: int
    event_id: Optional[int] = None
    camera_id: Optional[int] = None
    camera_name: str
    camera_location: str
    activity: str
    confidence: float
    severity: str
    status: str
    persons_detected: int = 0
    duration_seconds: Optional[float] = None
    video_clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    assigned_to: Optional[int] = None
    acknowledged_by: Optional[int] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[int] = None
    resolved_at: Optional[datetime] = None
    notes: Optional[str] = None
    source: str = "ai"
    reported_by: Optional[int] = None
    # Timeline-aware event fields (Phase 2) — None for rows created before
    # this feature, or for sources that don't track a smoothed evidence
    # history (rule-based/resident-report alerts).
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_confidence: Optional[float] = None
    average_confidence: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @computed_field
    @property
    def category(self) -> str:
        """'fire_smoke' or 'behaviour' — derived from `activity`, never a
        stored column (see app/constants.py). Lets the frontend filter
        reliably instead of matching activity name substrings itself."""
        return event_category(self.activity)


class AlertActionRequest(BaseModel):
    notes: Optional[str] = None


class FalsePositiveRequest(BaseModel):
    """Reviewer feedback when marking an alert as a false positive. The AI's
    own prediction is never trusted as ground truth — `correct_label` is the
    reviewer's independent judgment of what was actually happening (or
    "nothing"/"none" for a clean scene), left blank if they don't know."""
    notes: Optional[str] = None
    correct_label: Optional[str] = None
    add_to_training: bool = False


class AlertEscalateRequest(BaseModel):
    assign_to: Optional[int] = None
    notes: Optional[str] = None


class AlertListResponse(BaseModel):
    alerts: List[AlertResponse]
    total: int
    page: int
    page_size: int


class ResidentReportCreate(BaseModel):
    """A Resident personally reporting suspicious activity they observed —
    a human report, not an AI detection. Severity is deliberately not
    settable here: an unverified report shouldn't be able to self-assign
    high severity, so it always enters at "warning" and Guard/Admin escalate
    it after reviewing, same as any other alert."""
    camera_id: Optional[int] = None
    location: Optional[str] = None
    description: str


class AlertCreate(BaseModel):
    """Used internally by the AI service to create alerts."""
    camera_id: int
    camera_name: str
    camera_location: str
    activity: str
    confidence: float
    severity: str = "warning"
    persons_detected: int = 0
    duration_seconds: Optional[float] = None
    video_clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    event_id: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_confidence: Optional[float] = None
    average_confidence: Optional[float] = None
