"""
Vehicle Number-Plate OCR + registration — ported from the migration's
`backend/src/controllers/vehicle.controller.js`. OCR pipeline itself
(PaddleOCR, multi-stage detection/scoring) lives entirely in
face-service and is called unchanged via
app/services/face_service_client.verify_plate(); this module only owns
the business logic: normalize -> lookup -> ALLOWED/REQUIRES_APPROVAL,
plus vehicle CRUD.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.dependencies import get_db, get_current_user, require_guard, require_any_authenticated
from app.models.user import User, UserRole
from app.models.vehicle import Vehicle, VehicleType
from app.models.vehicle_request import VehicleRequest, VehicleRequestStatus
from app.schemas.vehicle import VehicleResponse, VehicleListResponse, PlateVerifyResponse
from app.services.face_service_client import verify_plate as ocr_verify_plate
from app.utils.plate import normalize_plate_number, is_valid_indian_plate

router = APIRouter(prefix="/api/vehicles", tags=["Vehicles"])


def _vehicle_to_response(v: Vehicle) -> VehicleResponse:
    return VehicleResponse(
        id=v.id,
        plate_number=v.plate_number,
        owner_name=v.owner_name,
        apartment_number=v.apartment_number,
        vehicle_type=v.vehicle_type.value if hasattr(v.vehicle_type, "value") else v.vehicle_type,
        is_active=v.is_active,
        source=v.source,
        created_at=v.created_at,
    )


@router.post("/verify-plate", response_model=PlateVerifyResponse)
async def verify_vehicle_plate(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Guard/admin uploads a vehicle image -> OCR -> registered-vehicle
    lookup. Mirrors vehicle.controller.js's verifyVehiclePlate()
    exactly, including the OCR_FAILED / ALLOWED / REQUIRES_APPROVAL
    entry_status values."""
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please upload a vehicle image.",
        )

    ocr_result = await ocr_verify_plate(file.filename, content, file.content_type)

    if not ocr_result["plate_detected"] or not ocr_result["detected_number"]:
        return PlateVerifyResponse(
            success=True,
            plate_detected=False,
            detected_number=None,
            confidence=ocr_result.get("confidence", 0.0),
            raw_texts=ocr_result.get("raw_texts", []),
            registered=False,
            vehicle=None,
            entry_status="OCR_FAILED",
            message=ocr_result.get("message") or "Could not read the number plate. Please try again with a clearer image.",
        )

    normalized_plate = normalize_plate_number(ocr_result["detected_number"])
    is_valid_format = is_valid_indian_plate(normalized_plate)

    result = await db.execute(
        select(Vehicle).where(Vehicle.plate_number == normalized_plate, Vehicle.is_active == True)  # noqa: E712
    )
    registered_vehicle = result.scalar_one_or_none()

    if registered_vehicle is not None and registered_vehicle.source == "visitor":
        # Response data must be captured BEFORE the delete below, since
        # _vehicle_to_response() is only called after it.
        vehicle_response = _vehicle_to_response(registered_vehicle)
        # One-time pass from an approved VehicleRequest — DELETE it
        # outright now, atomically (DELETE ... WHERE is_active=1), rather
        # than just deactivating it: once this plate has entered once,
        # nothing about it should be left behind to match against on a
        # later visit, so a later scan of the same plate requires a fresh
        # approval instead of standing ALLOWED forever. The WHERE guards
        # against two concurrent verify-plate calls both claiming the
        # same pass; only the one whose DELETE actually removes a row wins.
        consume_result = await db.execute(
            delete(Vehicle)
            .where(Vehicle.id == registered_vehicle.id, Vehicle.is_active == True)  # noqa: E712
        )
        if consume_result.rowcount != 1:
            registered_vehicle = None  # lost the race — treat as not currently registered
        else:
            if registered_vehicle.source_request_id is not None:
                await db.execute(
                    update(VehicleRequest)
                    .where(
                        VehicleRequest.id == registered_vehicle.source_request_id,
                        VehicleRequest.status == VehicleRequestStatus.APPROVED,
                    )
                    .values(status=VehicleRequestStatus.USED, consumed_at=datetime.now(timezone.utc))
                )
            await db.flush()
            return PlateVerifyResponse(
                success=True,
                plate_detected=True,
                detected_number=normalized_plate,
                confidence=ocr_result["confidence"],
                raw_texts=ocr_result["raw_texts"],
                is_valid_format=is_valid_format,
                registered=True,
                vehicle=vehicle_response,
                entry_status="ALLOWED",
                message="Registered vehicle — entry allowed.",
            )

    if registered_vehicle:
        # source == "resident": permanent, matches every scan, never consumed.
        return PlateVerifyResponse(
            success=True,
            plate_detected=True,
            detected_number=normalized_plate,
            confidence=ocr_result["confidence"],
            raw_texts=ocr_result["raw_texts"],
            is_valid_format=is_valid_format,
            registered=True,
            vehicle=_vehicle_to_response(registered_vehicle),
            entry_status="ALLOWED",
            message="Registered vehicle — entry allowed.",
        )

    # Not currently registered — distinguish "never requested" from "this
    # exact plate already used its one-time pass" so the guard/resident
    # aren't misled into thinking it's still standing approval. The Vehicle
    # row itself is gone (deleted on consumption above), but the plate
    # number isn't sensitive biometric data, so VehicleRequest history is
    # kept and is enough to tell the two cases apart.
    previously_used = (await db.execute(
        select(VehicleRequest.id).where(
            VehicleRequest.plate_number == normalized_plate,
            VehicleRequest.status == VehicleRequestStatus.USED,
        ).limit(1)
    )).scalar_one_or_none() is not None

    return PlateVerifyResponse(
        success=True,
        plate_detected=True,
        detected_number=normalized_plate,
        confidence=ocr_result["confidence"],
        raw_texts=ocr_result["raw_texts"],
        is_valid_format=is_valid_format,
        registered=False,
        vehicle=None,
        entry_status="REQUIRES_APPROVAL",
        message=(
            "Previous approval has been used. Please request new resident approval."
            if previously_used
            else "Vehicle not registered. Please select a resident for visitor approval."
        ),
    )


@router.post("", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
@router.post("/register", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
async def register_vehicle(
    plate_number: str = Form(...),
    owner_name: str | None = Form(None),
    apartment_number: str | None = Form(None),
    vehicle_type: str = Form("car"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """Register a vehicle to an apartment. Mirrors
    vehicle.controller.js's registerVehicle(): defaults owner_name/
    apartment_number from the logged-in resident's profile, rejects
    duplicates."""
    if not owner_name and current_user.role == UserRole.RESIDENT:
        owner_name = current_user.full_name
    if not apartment_number and current_user.role == UserRole.RESIDENT:
        apartment_number = current_user.apartment_number

    if not plate_number or not owner_name or not apartment_number:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="plate_number, owner_name, and apartment_number are required.",
        )

    normalized = normalize_plate_number(plate_number)
    if not normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid plate number provided.")

    existing = await db.execute(select(Vehicle).where(Vehicle.plate_number == normalized))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vehicle with plate {normalized} is already registered.",
        )

    try:
        vtype = VehicleType(vehicle_type.lower())
    except ValueError:
        vtype = VehicleType.CAR

    vehicle = Vehicle(
        plate_number=normalized,
        owner_name=owner_name,
        apartment_number=apartment_number,
        vehicle_type=vtype,
        resident_id=current_user.id if current_user.role == UserRole.RESIDENT else None,
    )
    db.add(vehicle)
    await db.flush()
    await db.refresh(vehicle)

    return _vehicle_to_response(vehicle)


@router.get("/my", response_model=VehicleListResponse)
async def get_my_vehicles(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mirrors vehicle.controller.js's getMyVehicles(): residents see
    only their own; guards/admins see all active vehicles."""
    query = select(Vehicle).where(Vehicle.is_active == True)  # noqa: E712
    if current_user.role == UserRole.RESIDENT:
        query = query.where(Vehicle.resident_id == current_user.id)
    query = query.order_by(Vehicle.created_at.desc())

    result = await db.execute(query)
    vehicles = result.scalars().all()
    return VehicleListResponse(count=len(vehicles), vehicles=[_vehicle_to_response(v) for v in vehicles])


@router.delete("/{vehicle_id}")
async def delete_vehicle(
    vehicle_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Deactivate (soft-delete) a vehicle. Mirrors deleteVehicle():
    only the owning resident or admin/guard may remove it."""
    result = await db.execute(select(Vehicle).where(Vehicle.id == vehicle_id))
    vehicle = result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vehicle not found.")

    if (
        current_user.role == UserRole.RESIDENT
        and vehicle.resident_id is not None
        and vehicle.resident_id != current_user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to delete this vehicle.",
        )

    vehicle.is_active = False
    await db.flush()

    return {"success": True, "message": "Vehicle removed successfully."}
