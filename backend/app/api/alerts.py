from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional
from datetime import datetime, timezone
from app.dependencies import get_db, get_current_user, require_guard, require_admin
from app.models.user import User
from app.models.alert import Alert, AlertStatus
from app.models.camera import Camera
from app.schemas.alert import (
    AlertResponse, AlertActionRequest, AlertEscalateRequest, AlertListResponse,
    ResidentReportCreate, FalsePositiveRequest,
)
from app.models.incident_review import IncidentReview
from app.models.training_sample import TrainingSample, SampleSource
from app.constants import FIRE_SMOKE_ACTIVITIES

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


def _parse_status(value: str) -> AlertStatus:
    """AlertStatus(value) raises a bare ValueError (-> unhandled 500) for an
    invalid/typo'd status string. Convert to a clean 400."""
    try:
        return AlertStatus(value)
    except ValueError:
        valid = ", ".join(s.value for s in AlertStatus)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{value}'. Must be one of: {valid}",
        )


@router.post("/test", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_test_alert(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Dev/verification tool (see test_alert.py at the project root) — exercises
    the EXACT SAME dispatch path a real detection uses (SecurityEvent ->
    Alert -> WebSocket broadcast), so a passing run here is real evidence
    the pipeline works end-to-end, not just that an Alert row can be
    inserted directly. Always clearly marked as a test in `notes` and
    `activity` so it's never confused with a real incident.
    """
    from app.services.alert_dispatcher import _dispatch_alert_async
    from ai_service.alert_generator import AlertEvent

    result = await db.execute(select(Camera).order_by(Camera.id).limit(1))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=400, detail="No camera exists to attach the test alert to.")

    test_event = AlertEvent(
        camera_id=camera.id,
        camera_name=camera.name,
        camera_location=camera.location,
        activity="TEST ALERT",
        confidence=0.95,
        severity="warning",
        persons_detected=1,
        duration_seconds=4.2,
        send_to_review=False,
    )
    await _dispatch_alert_async(test_event, recent_jpeg_frames=[])
    await db.commit()

    result = await db.execute(
        select(Alert).where(Alert.activity == "TEST ALERT").order_by(desc(Alert.created_at)).limit(1)
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=500, detail="Test alert dispatch ran but no Alert row was found afterward.")
    return AlertResponse.model_validate(alert)


@router.post("/report", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def report_suspicious_activity(
    body: ResidentReportCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Let any authenticated user (typically a Resident) report suspicious
    activity they personally observed near a monitored location. This is a
    human report, not an AI detection — it feeds into the exact same
    Alert/incident workflow Guard/Admin already use (list, acknowledge,
    resolve, escalate), so nothing downstream needs a separate code path.
    """
    location = (body.location or "").strip()
    if not location:
        raise HTTPException(status_code=400, detail="Location is required")

    description = body.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="Description is required")

    alert = Alert(
        camera_id=None,
        camera_name="Resident report",
        camera_location=location[:255],
        activity="Resident Reported Activity",
        # Confidence here means "certainty a report was filed", not a claim
        # about the underlying activity — Guard/Admin review and decide.
        confidence=1.0,
        severity="warning",
        status=AlertStatus.UNRESOLVED,
        persons_detected=0,
        notes=description,
        source="resident_report",
        reported_by=current_user.id,
    )
    db.add(alert)
    await db.flush()
    await db.refresh(alert)
    return AlertResponse.model_validate(alert)


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    severity: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    camera_id: Optional[int] = None,
    activity: Optional[str] = None,
    category: Optional[str] = Query(None, description="'behaviour' or 'fire_smoke' — see app/constants.py"),
    reported_by: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = select(Alert)
    count_query = select(func.count(Alert.id))

    if category == "fire_smoke":
        query = query.where(Alert.activity.in_(FIRE_SMOKE_ACTIVITIES))
        count_query = count_query.where(Alert.activity.in_(FIRE_SMOKE_ACTIVITIES))
    elif category == "behaviour":
        query = query.where(Alert.activity.notin_(FIRE_SMOKE_ACTIVITIES))
        count_query = count_query.where(Alert.activity.notin_(FIRE_SMOKE_ACTIVITIES))

    # A Resident may only ever query their OWN reports this way — never
    # trust a role/identity claim from the client, always check server-side.
    is_own_reports_query = reported_by is not None and reported_by == current_user.id
    if reported_by is not None:
        if current_user.role.value == "resident" and not is_own_reports_query:
            raise HTTPException(status_code=403, detail="Cannot view another user's reports")
        query = query.where(Alert.reported_by == reported_by)
        count_query = count_query.where(Alert.reported_by == reported_by)

    if severity:
        query = query.where(Alert.severity == severity)
        count_query = count_query.where(Alert.severity == severity)
    if status_filter:
        query = query.where(Alert.status == _parse_status(status_filter))
        count_query = count_query.where(Alert.status == _parse_status(status_filter))
    if camera_id:
        query = query.where(Alert.camera_id == camera_id)
        count_query = count_query.where(Alert.camera_id == camera_id)
    if activity:
        query = query.where(Alert.activity == activity)
        count_query = count_query.where(Alert.activity == activity)

    # Residents are not security operators: they may never see AI-detected
    # alerts (Behaviour or Fire/Smoke), at any severity — only the reports
    # they personally submitted (already enforced above via
    # is_own_reports_query). A resident who omits reported_by, or passes
    # someone else's id, gets nothing back here.
    if current_user.role.value == "resident" and not is_own_reports_query:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Residents may only view their own submitted reports.",
        )

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(desc(Alert.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    alerts = result.scalars().all()

    return AlertListResponse(
        alerts=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/export")
async def export_alerts_csv(
    severity: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    camera_id: Optional[int] = None,
    activity: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Export alerts log as downloadable CSV file. Guard/Admin only — a
    bulk export of security alerts is exactly the kind of monitoring
    access a Resident must not have."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    query = select(Alert)
    if severity:
        query = query.where(Alert.severity == severity)
    if status_filter:
        query = query.where(Alert.status == _parse_status(status_filter))
    if camera_id:
        query = query.where(Alert.camera_id == camera_id)
    if activity:
        query = query.where(Alert.activity == activity)

    query = query.order_by(desc(Alert.created_at)).limit(2000)
    result = await db.execute(query)
    alerts = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Alert ID", "Created At", "Camera ID", "Camera Name", "Location",
        "Activity", "Severity", "Status", "Persons Detected", "Notes",
    ])

    for a in alerts:
        writer.writerow([
            a.id,
            a.created_at.isoformat() if a.created_at else "",
            a.camera_id,
            a.camera_name,
            a.camera_location,
            a.activity,
            a.severity,
            a.status.value if hasattr(a.status, 'value') else str(a.status),
            a.persons_detected,
            a.notes or "",
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=security_alerts_report.csv"},
    )



@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    # Match list_alerts' restriction: a Resident may only view critical
    # alerts or reports they personally filed, never another camera's
    # non-critical alert or someone else's report, by id or otherwise.
    if current_user.role.value == "resident":
        is_own_report = alert.reported_by == current_user.id
        if alert.severity != "critical" and not is_own_report:
            raise HTTPException(status_code=404, detail="Alert not found")
    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    body: AlertActionRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.now(timezone.utc)
    if body and body.notes:
        alert.notes = (alert.notes or "") + f"\n[ACK] {body.notes}"
    await db.flush()
    await db.refresh(alert)
    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    body: AlertActionRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = AlertStatus.RESOLVED
    alert.resolved_by = current_user.id
    alert.resolved_at = datetime.now(timezone.utc)
    if body and body.notes:
        alert.notes = (alert.notes or "") + f"\n[RESOLVED] {body.notes}"
    await db.flush()
    await db.refresh(alert)
    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/escalate", response_model=AlertResponse)
async def escalate_alert(
    alert_id: int,
    body: AlertEscalateRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = AlertStatus.ESCALATED
    if body and body.assign_to:
        alert.assigned_to = body.assign_to
    if body and body.notes:
        alert.notes = (alert.notes or "") + f"\n[ESCALATED] {body.notes}"
    await db.flush()
    await db.refresh(alert)
    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/false-positive", response_model=AlertResponse)
async def mark_false_positive(
    alert_id: int,
    body: FalsePositiveRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """
    Marking an alert false positive is feedback, not just a status flip: it
    creates/updates the matching IncidentReview so the original AI prediction
    and confidence are preserved for audit, and lets the reviewer record what
    was actually happening (never defaulted from the AI's own guess — an
    unset correct_label just means "reviewer didn't specify", not "AI was
    right"). Optionally queues the reviewer-labeled clip as a training sample.
    """
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = AlertStatus.FALSE_POSITIVE
    if body and body.notes:
        alert.notes = (alert.notes or "") + f"\n[FALSE POSITIVE] {body.notes}"

    review_result = await db.execute(
        select(IncidentReview).where(IncidentReview.alert_id == alert_id)
    )
    review = review_result.scalar_one_or_none()
    if review is None and alert.event_id is not None:
        review_result = await db.execute(
            select(IncidentReview).where(IncidentReview.event_id == alert.event_id)
        )
        review = review_result.scalar_one_or_none()

    correct_label = body.correct_label if body else None
    if review is None:
        review = IncidentReview(
            event_id=alert.event_id,
            alert_id=alert.id,
            video_path=alert.video_clip_path,
            thumbnail_path=alert.thumbnail_path,
            camera_id=alert.camera_id,
            camera_name=alert.camera_name,
            original_prediction=alert.activity,
            original_confidence=alert.confidence,
            timestamp=alert.created_at,
        )
        db.add(review)
    else:
        review.alert_id = alert.id

    if correct_label:
        review.verified_label = correct_label
        review.reviewer_id = current_user.id
        review.reviewed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(review)

    if body and body.add_to_training and correct_label and review.video_path:
        training_sample = TrainingSample(
            video_path=review.video_path,
            activity_label=correct_label,
            source=SampleSource.VERIFIED_CCTV,
            camera_id=review.camera_id,
            reviewer_id=current_user.id,
            review_id=review.id,
            model_version=review.model_version,
        )
        db.add(training_sample)
        review.added_to_training = True
        await db.flush()

    await db.refresh(alert)
    return AlertResponse.model_validate(alert)
