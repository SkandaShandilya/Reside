from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, Date
from datetime import datetime, timedelta, timezone

from app.constants import PROJECT_ACTIVITY_CLASSES
from app.dependencies import get_db, require_guard
from app.models.user import User
from app.models.camera import Camera
from app.models.alert import Alert, AlertStatus
from app.models.security_event import SecurityEvent
from app.models.incident_review import IncidentReview
from app.schemas.analytics import (
    AnalyticsResponse, AnalyticsOverview, AlertsPerDay,
    ActivityDistribution, CameraActivityStat, DetectionPerformance,
)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Determine date window
    if start_date and end_date:
        try:
            range_start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            range_end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        except ValueError:
            range_start = today_start - timedelta(days=days)
            range_end = now
    else:
        range_start = today_start - timedelta(days=days - 1)
        range_end = now

    in_project = Alert.activity.in_(PROJECT_ACTIVITY_CLASSES)

    # Overview stats
    total_cameras = (await db.execute(select(func.count(Camera.id)))).scalar() or 0
    online_cameras = (await db.execute(
        select(func.count(Camera.id)).where(Camera.is_online == True)  # noqa: E712
    )).scalar() or 0

    alerts_today = (await db.execute(
        select(func.count(Alert.id)).where(Alert.created_at >= today_start, in_project)
    )).scalar() or 0

    critical_today = (await db.execute(
        select(func.count(Alert.id)).where(
            Alert.created_at >= today_start,
            Alert.severity == "critical",
            in_project,
        )
    )).scalar() or 0

    under_review = (await db.execute(
        select(func.count(IncidentReview.id)).where(IncidentReview.verified_label.is_(None))
    )).scalar() or 0

    total_events = (await db.execute(
        select(func.count(SecurityEvent.id)).where(
            SecurityEvent.timestamp >= range_start,
            SecurityEvent.timestamp <= range_end,
            SecurityEvent.activity_class.in_(PROJECT_ACTIVITY_CLASSES),
        )
    )).scalar() or 0

    # False positive rate
    total_alerts_all = (await db.execute(
        select(func.count(Alert.id)).where(
            Alert.created_at >= range_start,
            Alert.created_at <= range_end,
            in_project,
        )
    )).scalar() or 0

    false_positives = (await db.execute(
        select(func.count(Alert.id)).where(
            Alert.created_at >= range_start,
            Alert.created_at <= range_end,
            Alert.status == AlertStatus.FALSE_POSITIVE,
            in_project,
        )
    )).scalar() or 0
    fp_rate = (false_positives / total_alerts_all * 100) if total_alerts_all > 0 else 0.0

    overview = AnalyticsOverview(
        active_cameras=total_cameras,
        cameras_online=online_cameras,
        cameras_offline=total_cameras - online_cameras,
        alerts_today=alerts_today,
        critical_alerts_today=critical_today,
        events_under_review=under_review,
        total_events=total_events,
        false_positive_rate=round(fp_rate, 1),
    )

    # Alerts per day (grouped SQL query instead of loop)
    num_days = max(1, min((range_end - range_start).days + 1, 365))
    day_map: dict = {}
    for i in range(num_days):
        d_str = (range_start + timedelta(days=i)).strftime("%Y-%m-%d")
        day_map[d_str] = {"count": 0, "critical": 0, "warning": 0, "info": 0}

    # Query alert counts grouped by date and severity
    date_col = func.date(Alert.created_at)
    grouped_res = await db.execute(
        select(date_col, Alert.severity, func.count(Alert.id))
        .where(Alert.created_at >= range_start, Alert.created_at <= range_end, in_project)
        .group_by(date_col, Alert.severity)
    )

    for row_date, severity, cnt in grouped_res.all():
        d_key = str(row_date)
        if d_key in day_map:
            day_map[d_key]["count"] += cnt
            if severity == "critical":
                day_map[d_key]["critical"] += cnt
            elif severity == "warning":
                day_map[d_key]["warning"] += cnt
            else:
                day_map[d_key]["info"] += cnt

    alerts_per_day = [
        AlertsPerDay(
            date=d_key,
            count=vals["count"],
            critical=vals["critical"],
            warning=vals["warning"],
            info=vals["info"],
        )
        for d_key, vals in sorted(day_map.items())
    ]

    # Activity distribution in time range
    activity_result = await db.execute(
        select(Alert.activity, func.count(Alert.id).label("cnt"))
        .where(Alert.created_at >= range_start, Alert.created_at <= range_end, in_project)
        .group_by(Alert.activity)
        .order_by(func.count(Alert.id).desc())
    )
    activities = activity_result.all()
    total_act_cnt = sum(a.cnt for a in activities) or 1
    activity_dist = [
        ActivityDistribution(
            activity=a.activity,
            count=a.cnt,
            percentage=round(a.cnt / total_act_cnt * 100, 1),
        )
        for a in activities
    ]

    # Camera activity in time range
    cam_result = await db.execute(
        select(Alert.camera_id, Alert.camera_name, func.count(Alert.id).label("cnt"))
        .where(Alert.created_at >= range_start, Alert.created_at <= range_end, in_project)
        .group_by(Alert.camera_id, Alert.camera_name)
        .order_by(func.count(Alert.id).desc())
        .limit(10)
    )
    camera_activity = [
        CameraActivityStat(camera_id=c.camera_id, camera_name=c.camera_name, event_count=c.cnt)
        for c in cam_result.all()
    ]

    return AnalyticsResponse(
        overview=overview,
        alerts_per_day=alerts_per_day,
        activity_distribution=activity_dist,
        camera_activity=camera_activity,
        detection_performance=[],
    )


@router.get("/overview", response_model=AnalyticsOverview)
async def get_analytics_overview(
    days: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Return overview statistics summary."""
    full_analytics = await get_analytics(
        days=days,
        start_date=start_date,
        end_date=end_date,
        db=db,
        current_user=current_user,
    )
    return full_analytics.overview


