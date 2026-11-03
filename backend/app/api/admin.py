"""
Admin-only Resident & Visitor Directory — an aggregated, read-only view
over data the system already stores. Reuses the existing User (resident
profile + face-registration status), Vehicle (registered resident
vehicles), and VehicleRequest (the existing guard-initiated visitor
record: name + plate + approval status) models exactly as they are; no
new tables, no duplicate visitor model. Three queries total regardless
of resident count (residents, then their vehicles, then their visitor
requests, each fetched in bulk and grouped in Python), not one query
per resident.
"""
from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.dependencies import get_db, require_admin
from app.models.user import User, UserRole
from app.models.vehicle import Vehicle
from app.models.vehicle_request import VehicleRequest
from app.models.registered_person import RegisteredPerson
from app.schemas.admin import (
    AdminResidentDirectoryResponse, AdminResidentSummary,
    AdminVehicleSummary, AdminVisitorSummary, AdminHouseholdMemberSummary,
)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/residents", response_model=AdminResidentDirectoryResponse)
async def get_resident_directory(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    residents_result = await db.execute(
        select(User).where(User.role == UserRole.RESIDENT).order_by(User.full_name)
    )
    residents = residents_result.scalars().all()
    resident_ids = [r.id for r in residents]

    household_by_resident: dict[int, list[RegisteredPerson]] = defaultdict(list)
    if resident_ids:
        household_result = await db.execute(
            select(RegisteredPerson)
            .where(RegisteredPerson.resident_id.in_(resident_ids), RegisteredPerson.is_active == True)  # noqa: E712
            .order_by(RegisteredPerson.created_at)
        )
        for p in household_result.scalars().all():
            household_by_resident[p.resident_id].append(p)

    vehicles_by_resident: dict[int, list[Vehicle]] = defaultdict(list)
    if resident_ids:
        vehicles_result = await db.execute(
            select(Vehicle).where(Vehicle.resident_id.in_(resident_ids)).order_by(Vehicle.created_at.desc())
        )
        for v in vehicles_result.scalars().all():
            vehicles_by_resident[v.resident_id].append(v)

    visitors_by_resident: dict[int, list[tuple[VehicleRequest, str]]] = defaultdict(list)
    if resident_ids:
        requests_result = await db.execute(
            select(VehicleRequest, User.full_name)
            .join(User, User.id == VehicleRequest.guard_id)
            .where(VehicleRequest.resident_id.in_(resident_ids))
            .order_by(VehicleRequest.created_at.desc())
        )
        for vr, guard_name in requests_result.all():
            visitors_by_resident[vr.resident_id].append((vr, guard_name))

    summaries = []
    for r in residents:
        summaries.append(AdminResidentSummary(
            id=r.id,
            full_name=r.full_name,
            email=r.email,
            apartment_number=r.apartment_number,
            phone=r.phone,
            is_active=r.is_active,
            face_registered=r.face_registered,
            face_registered_at=r.face_registered_at,
            created_at=r.created_at,
            household_members=[
                AdminHouseholdMemberSummary(
                    id=p.id, full_name=p.full_name, phone=p.phone,
                    relationship=p.relationship.value if hasattr(p.relationship, "value") else p.relationship,
                    face_registered=p.face_registered, created_at=p.created_at,
                )
                for p in household_by_resident.get(r.id, [])
            ],
            vehicles=[
                AdminVehicleSummary(
                    id=v.id, plate_number=v.plate_number,
                    vehicle_type=v.vehicle_type.value if hasattr(v.vehicle_type, "value") else v.vehicle_type,
                    is_active=v.is_active, created_at=v.created_at,
                )
                for v in vehicles_by_resident.get(r.id, [])
            ],
            visitors=[
                AdminVisitorSummary(
                    id=vr.id, visitor_name=vr.visitor_name or "Unnamed visitor",
                    plate_number=vr.plate_number, purpose=vr.purpose,
                    status=vr.status.value if hasattr(vr.status, "value") else vr.status,
                    guard_name=guard_name, created_at=vr.created_at,
                    responded_at=vr.responded_at,
                )
                for vr, guard_name in visitors_by_resident.get(r.id, [])
            ],
        ))

    return AdminResidentDirectoryResponse(count=len(summaries), residents=summaries)
