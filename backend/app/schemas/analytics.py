from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AnalyticsOverview(BaseModel):
    active_cameras: int
    cameras_online: int
    cameras_offline: int
    alerts_today: int
    critical_alerts_today: int
    events_under_review: int
    total_events: int
    false_positive_rate: Optional[float] = None


class AlertsPerDay(BaseModel):
    date: str
    count: int
    critical: int
    warning: int
    info: int


class ActivityDistribution(BaseModel):
    activity: str
    count: int
    percentage: float


class CameraActivityStat(BaseModel):
    camera_id: int
    camera_name: str
    event_count: int
    last_event: Optional[str] = None


class DetectionPerformance(BaseModel):
    model_version: str
    accuracy: float
    f1_score: float
    total_predictions: int
    verified_correct: int
    false_positives: int


class AnalyticsResponse(BaseModel):
    overview: AnalyticsOverview
    alerts_per_day: List[AlertsPerDay]
    activity_distribution: List[ActivityDistribution]
    camera_activity: List[CameraActivityStat]
    detection_performance: List[DetectionPerformance]
