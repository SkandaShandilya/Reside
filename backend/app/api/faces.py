"""
Face Verification & Household Registration — ported from the
migration's `backend/src/controllers/face.controller.js`, adapted from
Express+Mongoose to FastAPI+SQLAlchemy, then extended so a Resident's
account (household) can register MULTIPLE people (self, spouse,
child, parent, other) rather than a single face per login — each
person gets their own RegisteredPerson row and own embedding; nothing
is ever combined into User.face_embedding (see app/models/user.py and
app/models/registered_person.py's docstrings for why, and
app/database.py's migrate_legacy_face_registrations() for how
pre-existing single-face User data was carried forward, unchanged).

Calls out to the separate face-service process for the actual
InsightFace inference (app/services/face_service_client.py) — a
face-service outage surfaces here as a clean 503, never a crash, and
cannot affect ai_service (Behaviour/Fire-Smoke), which this module
never imports from. Thresholds/similarity logic
(app/utils/face_similarity.py, FACE_MATCH_THRESHOLD) are unchanged.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.dependencies import get_db, get_current_user, require_guard
from app.models.user import User, UserRole
from app.models.registered_person import RegisteredPerson, PersonRelationship
from app.models.visitor_face_request import VisitorFaceRequest, VisitorFaceRequestStatus
from app.schemas.person import (
    PersonSummary, PersonRegisterResponse, PersonListResponse,
    HouseholdFaceVerifyResponse, HouseholdMatchResident,
)
from app.services.face_service_client import detect_and_embed, FaceServiceError
from app.utils.face_similarity import is_valid_embedding, cosine_similarity
from app.utils.phone import normalize_phone, is_valid_phone
from app.config import settings

router = APIRouter(prefix="/api/faces", tags=["Face Verification"])


def _to_summary(person: RegisteredPerson) -> PersonSummary:
    return PersonSummary(
        id=person.id,
        full_name=person.full_name,
        phone=person.phone,
        relationship=person.relationship.value if hasattr(person.relationship, "value") else person.relationship,
        face_registered=person.face_registered,
        face_registered_at=person.face_registered_at,
        is_active=person.is_active,
        created_at=person.created_at,
    )


@router.post("/register", response_model=PersonRegisterResponse)
async def register_household_member(
    file: UploadFile = File(...),
    full_name: str = Form(...),
    phone: str = Form(...),
    relationship: str = Form("self"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resident registers a household member's face (themselves, or a
    family member living in the same apartment) — every call creates a
    NEW RegisteredPerson row under the authenticated resident's
    household; `current_user` (from the JWT) is always the owner, never
    a client-supplied id, so a Resident can only ever add to their own
    household. Duplicate protection now means "this exact face isn't
    already registered in THIS household" (embedding-similarity check
    against the resident's existing people), not "you can only call
    this once" — unlimited household members are allowed, but the same
    person can't be added twice."""
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only residents can register household members.",
        )

    full_name = (full_name or "").strip()
    if not full_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Full name is required.")

    normalized_phone = normalize_phone(phone)
    if not is_valid_phone(normalized_phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid phone number (7-15 digits, optionally starting with +).",
        )

    try:
        relationship_value = PersonRelationship((relationship or "other").strip().lower())
    except ValueError:
        valid = ", ".join(r.value for r in PersonRelationship)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid relationship '{relationship}'. Must be one of: {valid}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please upload a face image.",
        )

    try:
        result = await detect_and_embed(file.filename, content, file.content_type)
    except FaceServiceError:
        raise

    embedding = result["embedding"]
    valid, reason = is_valid_embedding(embedding)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Received invalid face embedding: {reason}",
        )

    # Duplicate check — only ever against THIS resident's own household
    # (never another household's members, and never weakening
    # FACE_MATCH_THRESHOLD to do it).
    existing_result = await db.execute(
        select(RegisteredPerson).where(
            RegisteredPerson.resident_id == current_user.id,
            RegisteredPerson.is_active == True,  # noqa: E712
            RegisteredPerson.face_registered == True,  # noqa: E712
        )
    )
    for existing_person in existing_result.scalars().all():
        if not existing_person.face_embedding:
            continue
        similarity, sim_valid, _ = cosine_similarity(embedding, existing_person.face_embedding)
        if sim_valid and similarity >= settings.FACE_MATCH_THRESHOLD:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This face is already registered.",
            )

    person = RegisteredPerson(
        resident_id=current_user.id,
        full_name=full_name,
        phone=normalized_phone,
        relationship=relationship_value,
        face_embedding=embedding,
        face_registered=True,
        face_registered_at=datetime.now(timezone.utc),
    )
    db.add(person)
    await db.flush()
    await db.refresh(person)

    return PersonRegisterResponse(
        success=True,
        message=f"{full_name} registered successfully.",
        person=_to_summary(person),
    )


@router.get("/household", response_model=PersonListResponse)
async def list_household(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resident views their own household's registered people —
    scoped to resident_id == current_user.id, so this can never return
    another household's members regardless of any query parameter
    (there isn't one to manipulate)."""
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only residents can view a household's registered people.",
        )
    result = await db.execute(
        select(RegisteredPerson)
        .where(RegisteredPerson.resident_id == current_user.id, RegisteredPerson.is_active == True)  # noqa: E712
        .order_by(RegisteredPerson.created_at)
    )
    people = result.scalars().all()
    return PersonListResponse(count=len(people), people=[_to_summary(p) for p in people])


@router.delete("/household/{person_id}")
async def remove_household_member(
    person_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Soft-deletes (is_active=False) a household member — mirrors
    Vehicle's soft-delete pattern. 404 (never 403) for a person that
    doesn't exist OR belongs to a different household, so a Resident
    can't even confirm another household's member ids exist."""
    if current_user.role != UserRole.RESIDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only residents can manage a household's registered people.",
        )
    result = await db.execute(
        select(RegisteredPerson).where(
            RegisteredPerson.id == person_id,
            RegisteredPerson.resident_id == current_user.id,
        )
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Household member not found.")

    person.is_active = False
    await db.flush()
    return {"success": True, "message": f"{person.full_name} removed from your household."}


@router.post("/verify", response_model=HouseholdFaceVerifyResponse)
async def verify_face(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Guard/admin verifies a visitor's face against ALL active
    registered household members (not just one face per account) —
    best-match-wins cosine similarity across every household, then
    threshold comparison (FACE_MATCH_THRESHOLD, default 0.50 —
    unchanged). Returns the matched person plus which household
    (resident/apartment) they belong to; never the raw embedding."""
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please upload a face image.",
        )

    result = await detect_and_embed(file.filename, content, file.content_type)
    visitor_embedding = result["embedding"]

    valid, reason = is_valid_embedding(visitor_embedding)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Received invalid face embedding: {reason}",
        )

    people_result = await db.execute(
        select(RegisteredPerson, User)
        .join(User, User.id == RegisteredPerson.resident_id)
        .where(
            RegisteredPerson.is_active == True,  # noqa: E712
            RegisteredPerson.face_registered == True,  # noqa: E712
            User.is_active == True,  # noqa: E712
        )
    )
    rows = people_result.all()

    threshold = settings.FACE_MATCH_THRESHOLD
    best_similarity = -1.0

    # Every candidate clearing the threshold, best-first — not just the
    # single best — so that if the top candidate is a one-time visitor
    # pass that loses the atomic-consume race below (another concurrent
    # verify already claimed it), we fall through to the next-best match
    # instead of silently reporting "no match" despite a real candidate.
    candidates: list[tuple[float, RegisteredPerson, User]] = []
    for person, resident in rows:
        if not person.face_embedding:
            continue
        similarity, sim_valid, _ = cosine_similarity(visitor_embedding, person.face_embedding)
        if not sim_valid:
            continue
        best_similarity = max(best_similarity, similarity)
        if similarity >= threshold:
            candidates.append((similarity, person, resident))
    candidates.sort(key=lambda c: c[0], reverse=True)

    rounded_similarity = round(best_similarity, 2) if best_similarity >= 0 else 0.0

    for similarity, person, resident in candidates:
        # Build the response data BEFORE any delete below — person/resident
        # are already-fetched ORM objects (expire_on_commit=False on the
        # session factory), so reading their attributes afterward is safe,
        # but doing it up front keeps the control flow obvious.
        person_summary = _to_summary(person)
        resident_summary = HouseholdMatchResident(
            id=resident.id, full_name=resident.full_name, apartment_number=resident.apartment_number,
        )

        if person.source == "visitor":
            # One-time visitor pass (granted by a resident's approval of a
            # VisitorFaceRequest, not self-registered) — DELETE it outright
            # now, atomically (DELETE ... WHERE is_active=1), rather than
            # just deactivating it: once the visitor has been let in once,
            # nothing about their face should be left behind to compare
            # against on a later visit, so a second gate attempt with the
            # same face finds no match and must go through a fresh
            # approval. The WHERE guards against two concurrent verify
            # calls both claiming the same pass; only the one whose DELETE
            # actually removes a row wins, the other falls through to the
            # next candidate (or no-match).
            result = await db.execute(
                delete(RegisteredPerson)
                .where(RegisteredPerson.id == person.id, RegisteredPerson.is_active == True)  # noqa: E712
            )
            if result.rowcount != 1:
                continue  # lost the race (or already consumed) — try the next candidate
            if person.source_request_id is not None:
                await db.execute(
                    update(VisitorFaceRequest)
                    .where(
                        VisitorFaceRequest.id == person.source_request_id,
                        VisitorFaceRequest.status == VisitorFaceRequestStatus.APPROVED,
                    )
                    .values(status=VisitorFaceRequestStatus.USED, consumed_at=datetime.now(timezone.utc))
                )
            await db.flush()
        # source == "household": permanent, matches every time, never consumed.
        return HouseholdFaceVerifyResponse(
            success=True,
            matched=True,
            similarity=round(similarity, 2),
            person=person_summary,
            resident=resident_summary,
        )

    # No active candidate won. Note: a visitor whose one-time pass was
    # already consumed is now indistinguishable from a visitor never seen
    # before — the embedding was deleted along with the row above, by
    # design, rather than kept around just to detect this case.
    return HouseholdFaceVerifyResponse(
        success=True,
        matched=False,
        similarity=rounded_similarity,
        message="No registered household member matched.",
    )
