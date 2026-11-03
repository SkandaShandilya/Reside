"""
Vehicle approval requests — ported from the migration's
`backend/src/controllers/vehicleRequest.controller.js`. A guard raises
a request for an unregistered visitor plate; the assigned resident
approves or rejects it.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.dependencies import get_db, get_current_user, require_guard
from app.models.user import User, UserRole
from app.models.vehicle import Vehicle, VehicleType
from app.models.vehicle_request import VehicleRequest, VehicleRequestStatus
from app.schemas.vehicle import (
    VehicleRequestCreate, VehicleRequestRespond,
    VehicleRequestResponse, VehicleRequestListResponse, VehicleRequestUserSummary,
)
from app.utils.plate import normalize_plate_number

router = APIRouter(prefix="/api/vehicle-requests", tags=["Vehicle Requests"])


@router.get("/residents-list")
async def list_active_residents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Minimal active-resident list for the guard's request-target
    selector — the existing GET /api/users is admin-only, so this is a
    small, additive, guard-accessible equivalent scoped to just what
    the vehicle-request flow needs (id, name, apartment)."""
    result = await db.execute(
        select(User).where(User.role == UserRole.RESIDENT, User.is_active == True)  # noqa: E712
        .order_by(User.full_name)
    )
    residents = result.scalars().all()
    return {
        "success": True,
        "residents": [
            {"id": r.id, "full_name": r.full_name, "apartment_number": r.apartment_number}
            for r in residents
        ],
    }


async def _to_response(db: AsyncSession, req: VehicleRequest) -> VehicleRequestResponse:
    guard_result = await db.execute(select(User).where(User.id == req.guard_id))
    resident_result = await db.execute(select(User).where(User.id == req.resident_id))
    guard = guard_result.scalar_one_or_none()
    resident = resident_result.scalar_one_or_none()
    return VehicleRequestResponse(
        id=req.id,
        plate_number=req.plate_number,
        guard=VehicleRequestUserSummary(
            id=req.guard_id,
            full_name=guard.full_name if guard else "Unknown",
            apartment_number=None,
        ),
        resident=VehicleRequestUserSummary(
            id=req.resident_id,
            full_name=resident.full_name if resident else "Unknown",
            apartment_number=resident.apartment_number if resident else None,
        ),
        apartment_number=req.apartment_number,
        visitor_name=req.visitor_name,
        purpose=req.purpose,
        status=req.status.value if hasattr(req.status, "value") else req.status,
        responded_at=req.responded_at,
        guard_notes=req.guard_notes,
        resident_notes=req.resident_notes,
        created_at=req.created_at,
    )


@router.post("", response_model=VehicleRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_request(
    payload: VehicleRequestCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Guard creates a visitor vehicle approval request. Mirrors
    createRequest(): validates the target resident exists/active,
    defaults apartment_number from the resident's profile."""
    if not payload.plate_number or not payload.resident_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="plate_number and resident_id are required.",
        )

    resident_result = await db.execute(
        select(User).where(
            User.id == payload.resident_id,
            User.role == UserRole.RESIDENT,
            User.is_active == True,  # noqa: E712
        )
    )
    resident = resident_result.scalar_one_or_none()
    if not resident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resident not found or inactive.")

    apartment_number = payload.apartment_number or resident.apartment_number

    request = VehicleRequest(
        plate_number=normalize_plate_number(payload.plate_number),
        guard_id=current_user.id,
        resident_id=payload.resident_id,
        apartment_number=apartment_number or "",
        visitor_name=payload.visitor_name or "",
        purpose=payload.purpose or "",
        guard_notes=payload.guard_notes or "",
    )
    db.add(request)
    await db.flush()
    await db.refresh(request)

    return await _to_response(db, request)


@router.get("/guard", response_model=VehicleRequestListResponse)
async def get_guard_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Guard views their own requests; admin sees all. Mirrors
    getGuardRequests()."""
    query = select(VehicleRequest)
    if current_user.role not in (UserRole.ADMIN, UserRole.SUPER_ADMIN):
        query = query.where(VehicleRequest.guard_id == current_user.id)
    query = query.order_by(VehicleRequest.created_at.desc()).limit(50)

    result = await db.execute(query)
    requests = result.scalars().all()
    return VehicleRequestListResponse(
        count=len(requests), requests=[await _to_response(db, r) for r in requests]
    )


@router.get("/residents", response_model=VehicleRequestListResponse)
async def get_resident_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resident views pending requests assigned to them. Mirrors
    getResidentRequests()."""
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Residents only.")

    result = await db.execute(
        select(VehicleRequest)
        .where(VehicleRequest.resident_id == current_user.id)
        .order_by(VehicleRequest.created_at.desc())
        .limit(50)
    )
    requests = result.scalars().all()
    return VehicleRequestListResponse(
        count=len(requests), requests=[await _to_response(db, r) for r in requests]
    )


@router.get("/my", response_model=VehicleRequestListResponse)
async def get_my_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Any authenticated user views their related requests. Mirrors
    getMyRequests(): guard -> their created requests, resident -> theirs,
    admin -> all."""
    query = select(VehicleRequest)
    if current_user.role == UserRole.GUARD:
        query = query.where(VehicleRequest.guard_id == current_user.id)
    elif current_user.role == UserRole.RESIDENT:
        query = query.where(VehicleRequest.resident_id == current_user.id)
    query = query.order_by(VehicleRequest.created_at.desc()).limit(50)

    result = await db.execute(query)
    requests = result.scalars().all()
    return VehicleRequestListResponse(
        count=len(requests), requests=[await _to_response(db, r) for r in requests]
    )


@router.patch("/{request_id}/respond")
async def respond_to_request(
    request_id: int,
    payload: VehicleRequestRespond,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resident approves or rejects a request. Mirrors
    respondToRequest(): only the assigned resident may respond, only
    while still pending."""
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Residents only.")

    action = (payload.action or "").lower()
    is_approve = action in ("approve", "approved")
    is_reject = action in ("reject", "rejected")
    if not is_approve and not is_reject:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="action must be 'approve' or 'reject'.")

    result = await db.execute(select(VehicleRequest).where(VehicleRequest.id == request_id))
    request = result.scalar_one_or_none()
    if not request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vehicle request not found.")

    if request.resident_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to respond to this request.",
        )
    if request.status != VehicleRequestStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This request has already been {request.status.value}.",
        )

    request.status = VehicleRequestStatus.APPROVED if is_approve else VehicleRequestStatus.REJECTED
    request.responded_at = datetime.now(timezone.utc)
    request.resident_notes = payload.resident_notes or ""

    if is_approve:
        # Persist the approval as a "visitor parking" vehicle record so the
        # plate is recognized as ALLOWED the next time the guard scans it —
        # but only ONCE: POST /api/vehicles/verify-plate deactivates this
        # row (is_active=False) the instant it matches, so a later scan of
        # the same plate falls through to REQUIRES_APPROVAL again and needs
        # a fresh VehicleRequest. Permanent access (never auto-deactivated)
        # is only ever created by the resident themselves via
        # POST /api/vehicles/register (source="resident").
        # plate_number is unique — a previously removed (inactive) record for
        # this plate must be reactivated, not re-inserted (which used to fail
        # with a 500 and left the request impossible to approve).
        existing_rows = (await db.execute(
            select(Vehicle).where(Vehicle.plate_number == request.plate_number)
        )).scalars().all()
        if not any(v.is_active for v in existing_rows):
            if existing_rows:
                v = existing_rows[0]
                v.is_active = True
                v.owner_name = request.visitor_name or "Visitor"
                v.apartment_number = request.apartment_number
                v.resident_id = request.resident_id
                v.source = "visitor"
                v.source_request_id = request.id
            else:
                db.add(Vehicle(
                    plate_number=request.plate_number,
                    owner_name=request.visitor_name or "Visitor",
                    apartment_number=request.apartment_number,
                    vehicle_type=VehicleType.CAR,
                    resident_id=request.resident_id,
                    source="visitor",
                    source_request_id=request.id,
                ))

    await db.flush()
    await db.refresh(request)

    entry_status = "ALLOWED" if request.status == VehicleRequestStatus.APPROVED else "DENIED"
    return {
        "success": True,
        "message": f"Vehicle entry {entry_status}.",
        "entry_status": entry_status,
        "request": await _to_response(db, request),
    }
