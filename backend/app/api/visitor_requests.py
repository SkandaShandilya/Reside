"""
Unregistered-visitor approval requests. When the existing face
verification (POST /api/faces/verify, unchanged) finds NO match, the
guard can capture the visitor's photo and send it to a resident, who
approves or rejects it. Follows the same lifecycle/pattern as
vehicle_requests.py. An unrecognized face is never auto-approved — only
the resident's explicit APPROVE changes the status.

Approval grants the visitor exactly ONE future match at the gate, not
permanent access — see _register_approved_visitor()/RegisteredPerson.
source below. Permanent access is only ever created by the resident
themselves via POST /api/faces/register.
"""
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.dependencies import get_db, get_current_user, require_guard
from app.models.user import User, UserRole
from app.models.visitor_face_request import VisitorFaceRequest, VisitorFaceRequestStatus
from app.models.registered_person import RegisteredPerson, PersonRelationship
from app.utils.face_similarity import is_valid_embedding, cosine_similarity
from app.services.face_service_client import detect_and_embed, FaceServiceError
from app.utils.security import decode_access_token

router = APIRouter(prefix="/api/visitor-requests", tags=["Visitor Face Requests"])

REQUEST_TTL = timedelta(minutes=10)
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class VisitorRespond(BaseModel):
    action: str
    resident_notes: str | None = None


async def _has_face(filename: str | None, content: bytes, content_type: str | None) -> bool:
    """True if the existing face-service detects a usable face. A 4xx from
    face-service ('no face found') means False; a service outage (503)
    propagates so the guard sees a real error instead of a false 'no face'."""
    try:
        await detect_and_embed(filename or "visitor.jpg", content, content_type or "image/jpeg")
        return True
    except FaceServiceError as e:
        if e.status_code == status.HTTP_400_BAD_REQUEST:
            return False
        raise


def _expire_if_stale(req: VisitorFaceRequest) -> None:
    if req.status == VisitorFaceRequestStatus.PENDING:
        created = req.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created is not None and datetime.now(timezone.utc) - created > REQUEST_TTL:
            req.status = VisitorFaceRequestStatus.EXPIRED


async def _to_response(db: AsyncSession, req: VisitorFaceRequest) -> dict:
    guard = (await db.execute(select(User).where(User.id == req.guard_id))).scalar_one_or_none()
    resident = (await db.execute(select(User).where(User.id == req.resident_id))).scalar_one_or_none()
    return {
        "id": req.id,
        "status": req.status.value,
        "image_url": f"/api/visitor-requests/{req.id}/image",
        "visitor_name": req.visitor_name,
        "location": req.location,
        "guard": {"id": req.guard_id, "full_name": guard.full_name if guard else "Unknown"},
        "resident": {
            "id": req.resident_id,
            "full_name": resident.full_name if resident else "Unknown",
            "apartment_number": resident.apartment_number if resident else None,
        },
        "resident_notes": req.resident_notes,
        "responded_at": req.responded_at,
        "created_at": req.created_at,
    }


@router.post("/validate-face")
async def validate_face(
    file: UploadFile = File(...),
    current_user: User = Depends(require_guard),
):
    """Checks a captured frame contains a detectable face, using the
    existing face-service. No embedding is stored or compared."""
    content = await file.read()
    if not content:
        return {"success": True, "face_detected": False}
    return {"success": True, "face_detected": await _has_face(file.filename, content, file.content_type)}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_request(
    file: UploadFile = File(...),
    resident_id: int = Form(...),
    location: str = Form("Main Gate"),
    visitor_name: str = Form(""),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="No image provided.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")

    resident = (await db.execute(
        select(User).where(
            User.id == resident_id, User.role == UserRole.RESIDENT, User.is_active == True,  # noqa: E712
        )
    )).scalar_one_or_none()
    if not resident:
        raise HTTPException(status_code=404, detail="Unable to reach the resident.")

    # Never forward an image with no usable face to the resident.
    if not await _has_face(file.filename, content, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="No face detected. Please position the visitor inside the frame and try again.",
        )

    rel_dir = Path("visitor_requests")
    abs_dir = Path(settings.STORAGE_PATH) / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    (abs_dir / filename).write_bytes(content)

    req = VisitorFaceRequest(
        guard_id=current_user.id,
        resident_id=resident_id,
        image_path=(rel_dir / filename).as_posix(),
        location=(location or "Main Gate").strip()[:255] or "Main Gate",
        visitor_name=(visitor_name or "").strip()[:120] or None,
    )
    db.add(req)
    await db.flush()
    await db.refresh(req)
    return await _to_response(db, req)


@router.get("/guard")
async def list_guard_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    query = select(VisitorFaceRequest)
    if current_user.role not in (UserRole.ADMIN, UserRole.SUPER_ADMIN):
        query = query.where(VisitorFaceRequest.guard_id == current_user.id)
    rows = (await db.execute(query.order_by(VisitorFaceRequest.created_at.desc()).limit(50))).scalars().all()
    for r in rows:
        _expire_if_stale(r)
    return {"success": True, "requests": [await _to_response(db, r) for r in rows]}


@router.get("/resident")
async def list_resident_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(status_code=403, detail="Residents only.")
    rows = (await db.execute(
        select(VisitorFaceRequest)
        .where(VisitorFaceRequest.resident_id == current_user.id)
        .order_by(VisitorFaceRequest.created_at.desc()).limit(50)
    )).scalars().all()
    for r in rows:
        _expire_if_stale(r)
    return {"success": True, "requests": [await _to_response(db, r) for r in rows]}


@router.get("/{request_id}/image")
async def get_request_image(
    request_id: int,
    request: Request,
    token: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """The captured visitor photo. Accepts the JWT as a Bearer header or
    ?token= (an <img> tag can't send headers). Only the guard who sent the
    request, the resident it is addressed to, and admins may view it — the
    generic /storage route deliberately refuses residents."""
    raw = token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        raw = auth[7:]
    payload = decode_access_token(raw) if raw else None
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    user = (await db.execute(select(User).where(User.id == int(payload["sub"])))).scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    req = (await db.execute(select(VisitorFaceRequest).where(VisitorFaceRequest.id == request_id))).scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Visitor request not found.")
    if user.role not in (UserRole.ADMIN, UserRole.SUPER_ADMIN) and user.id not in (req.guard_id, req.resident_id):
        raise HTTPException(status_code=403, detail="Not authorized.")
    base = Path(settings.STORAGE_PATH).resolve()
    path = (base / req.image_path).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="Image not found.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{request_id}")
async def get_request(
    request_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    req = (await db.execute(select(VisitorFaceRequest).where(VisitorFaceRequest.id == request_id))).scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Visitor request not found.")
    is_admin = current_user.role in (UserRole.ADMIN, UserRole.SUPER_ADMIN)
    if not is_admin and current_user.id not in (req.guard_id, req.resident_id):
        raise HTTPException(status_code=403, detail="Not authorized.")
    _expire_if_stale(req)
    return await _to_response(db, req)


@router.patch("/{request_id}/respond")
async def respond_to_request(
    request_id: int,
    payload: VisitorRespond,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(status_code=403, detail="Residents only.")
    action = (payload.action or "").lower()
    is_approve = action in ("approve", "approved")
    if not is_approve and action not in ("reject", "rejected"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'.")

    req = (await db.execute(select(VisitorFaceRequest).where(VisitorFaceRequest.id == request_id))).scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Visitor request not found.")
    if req.resident_id != current_user.id:
        raise HTTPException(status_code=403, detail="You are not authorized to respond to this request.")
    _expire_if_stale(req)
    if req.status != VisitorFaceRequestStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"This request is already {req.status.value}.")

    req.status = VisitorFaceRequestStatus.APPROVED if is_approve else VisitorFaceRequestStatus.REJECTED
    req.responded_at = datetime.now(timezone.utc)
    req.resident_notes = payload.resident_notes or ""
    await db.flush()
    await db.refresh(req)

    registered = False
    if is_approve:
        registered = await _register_approved_visitor(db, req)
    return {"success": True, "face_registered": registered, "request": await _to_response(db, req)}


async def _register_approved_visitor(db: AsyncSession, req: VisitorFaceRequest) -> bool:
    """When a resident accepts a visitor, grant that face exactly ONE
    future match at the gate — NOT permanent household access. Permanent
    access is only ever created via the resident's own explicit
    POST /api/faces/register. This writes a RegisteredPerson row with
    source="visitor"; POST /api/faces/verify deactivates it (is_active
    = False) the instant it's matched, so a second attempt with the same
    face falls through to "no match" and the guard must raise a fresh
    VisitorFaceRequest. Best-effort: the approval itself never fails
    because this step couldn't complete (e.g. face-service down) — it
    just reports face_registered=False."""
    try:
        path = Path(settings.STORAGE_PATH) / req.image_path
        result = await detect_and_embed(path.name, path.read_bytes(), "image/jpeg")
        embedding = result["embedding"]
        valid, _ = is_valid_embedding(embedding)
        if not valid:
            return False

        existing = (await db.execute(
            select(RegisteredPerson).where(
                RegisteredPerson.resident_id == req.resident_id,
                RegisteredPerson.is_active == True,  # noqa: E712
                RegisteredPerson.face_registered == True,  # noqa: E712
            )
        )).scalars().all()
        for person in existing:
            if person.face_embedding:
                similarity, ok, _ = cosine_similarity(embedding, person.face_embedding)
                if ok and similarity >= settings.FACE_MATCH_THRESHOLD:
                    # Already has a standing pass (household member, or an
                    # unused one-time visitor pass from an earlier approval)
                    # — don't stack a second one.
                    return True

        stamp = datetime.now().strftime("%d %b %Y %H:%M")
        db.add(RegisteredPerson(
            resident_id=req.resident_id,
            full_name=req.visitor_name or f"Approved Visitor ({stamp})",
            phone=None,
            relationship=PersonRelationship.OTHER,
            face_embedding=embedding,
            face_registered=True,
            face_registered_at=datetime.now(timezone.utc),
            source="visitor",
            source_request_id=req.id,
        ))
        await db.flush()
        return True
    except Exception:
        return False


@router.post("/{request_id}/cancel")
async def cancel_request(
    request_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    req = (await db.execute(select(VisitorFaceRequest).where(VisitorFaceRequest.id == request_id))).scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Visitor request not found.")
    if req.guard_id != current_user.id and current_user.role not in (UserRole.ADMIN, UserRole.SUPER_ADMIN):
        raise HTTPException(status_code=403, detail="Not authorized.")
    if req.status == VisitorFaceRequestStatus.PENDING:
        req.status = VisitorFaceRequestStatus.EXPIRED
        await db.flush()
    return {"success": True, "request": await _to_response(db, req)}
