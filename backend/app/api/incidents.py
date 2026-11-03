from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional
from app.dependencies import get_db, require_guard
from app.models.user import User
from app.models.security_event import SecurityEvent
from app.schemas.incident import IncidentResponse, IncidentListResponse
from app.constants import FIRE_SMOKE_ACTIVITIES

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


@router.get("", response_model=IncidentListResponse)
async def list_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    camera_id: Optional[int] = None,
    activity: Optional[str] = None,
    severity: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
    category: Optional[str] = Query(None, description="'behaviour' or 'fire_smoke' — see app/constants.py"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    query = select(SecurityEvent)
    count_query = select(func.count(SecurityEvent.id))

    if category == "fire_smoke":
        query = query.where(SecurityEvent.activity_class.in_(FIRE_SMOKE_ACTIVITIES))
        count_query = count_query.where(SecurityEvent.activity_class.in_(FIRE_SMOKE_ACTIVITIES))
    elif category == "behaviour":
        query = query.where(SecurityEvent.activity_class.notin_(FIRE_SMOKE_ACTIVITIES))
        count_query = count_query.where(SecurityEvent.activity_class.notin_(FIRE_SMOKE_ACTIVITIES))

    if camera_id:
        query = query.where(SecurityEvent.camera_id == camera_id)
        count_query = count_query.where(SecurityEvent.camera_id == camera_id)
    if activity:
        query = query.where(SecurityEvent.activity_class == activity)
        count_query = count_query.where(SecurityEvent.activity_class == activity)
    if severity:
        query = query.where(SecurityEvent.severity == severity)
        count_query = count_query.where(SecurityEvent.severity == severity)
    if search:
        query = query.where(SecurityEvent.activity_class.ilike(f"%{search}%"))
        count_query = count_query.where(SecurityEvent.activity_class.ilike(f"%{search}%"))

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(desc(SecurityEvent.timestamp))
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    events = result.scalars().all()

    return IncidentListResponse(
        incidents=[IncidentResponse.model_validate(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/export")
async def export_incidents_csv(
    camera_id: Optional[int] = None,
    activity: Optional[str] = None,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Export security incident logs as downloadable CSV."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    query = select(SecurityEvent)
    if camera_id:
        query = query.where(SecurityEvent.camera_id == camera_id)
    if activity:
        query = query.where(SecurityEvent.activity_class == activity)
    if severity:
        query = query.where(SecurityEvent.severity == severity)
    if search:
        query = query.where(SecurityEvent.activity_class.ilike(f"%{search}%"))

    query = query.order_by(desc(SecurityEvent.timestamp)).limit(2000)
    result = await db.execute(query)
    events = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Incident ID", "Timestamp", "Camera ID", "Activity Class",
        "Confidence", "Severity", "Persons Detected", "Duration (s)", "Model Version",
    ])

    for e in events:
        writer.writerow([
            e.id,
            e.timestamp.isoformat() if e.timestamp else "",
            e.camera_id,
            e.activity_class,
            f"{e.confidence * 100:.1f}%",
            e.severity,
            e.persons_detected,
            f"{e.duration_seconds:.1f}" if e.duration_seconds else "",
            e.model_version or "",
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=security_incidents_report.csv"},
    )

