from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional
from datetime import datetime, timezone
from app.dependencies import get_db, get_current_user, require_guard
from app.models.user import User
from app.models.incident_review import IncidentReview
from app.models.training_sample import TrainingSample, SampleSource
from app.schemas.review import ReviewItemResponse, ReviewVerifyRequest, ReviewListResponse

router = APIRouter(prefix="/api/review", tags=["Review Queue"])


@router.get("", response_model=ReviewListResponse)
async def list_review_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    reviewed: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    query = select(IncidentReview)
    count_query = select(func.count(IncidentReview.id))

    if reviewed is not None:
        if reviewed:
            query = query.where(IncidentReview.verified_label.isnot(None))
            count_query = count_query.where(IncidentReview.verified_label.isnot(None))
        else:
            query = query.where(IncidentReview.verified_label.is_(None))
            count_query = count_query.where(IncidentReview.verified_label.is_(None))

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(desc(IncidentReview.timestamp))
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = result.scalars().all()

    return ReviewListResponse(
        items=[ReviewItemResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/{review_id}/verify", response_model=ReviewItemResponse)
async def verify_review_item(
    review_id: int,
    body: ReviewVerifyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(IncidentReview).where(IncidentReview.id == review_id))
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review item not found")

    review.verified_label = body.verified_label
    review.reviewer_id = current_user.id
    review.reviewed_at = datetime.now(timezone.utc)

    # Add to training dataset if requested
    if body.add_to_training and review.video_path:
        training_sample = TrainingSample(
            video_path=review.video_path,
            activity_label=body.verified_label,
            source=SampleSource.VERIFIED_CCTV,
            camera_id=review.camera_id,
            reviewer_id=current_user.id,
            review_id=review.id,
            model_version=review.model_version,
        )
        db.add(training_sample)
        review.added_to_training = True

    await db.flush()
    await db.refresh(review)
    return ReviewItemResponse.model_validate(review)
