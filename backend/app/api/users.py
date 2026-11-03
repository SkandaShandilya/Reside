from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.models.alert import Alert
from app.models.camera import Camera
from app.models.incident_review import IncidentReview
from app.models.training_sample import TrainingSample
from app.models.audit_log import AuditLog
from app.models.vehicle import Vehicle
from app.dependencies import get_db, require_admin
from app.models.user import User, UserRole
from app.schemas.user import UserResponse, UserCreate, UserUpdate
from app.utils.security import hash_password

router = APIRouter(prefix="/api/users", tags=["Users"])


def _parse_role(value: str) -> UserRole:
    """UserRole(value) raises a bare ValueError (-> unhandled 500) for any
    invalid string, e.g. a typo'd role name. Convert to a clean 400."""
    try:
        return UserRole(value)
    except ValueError:
        valid = ", ".join(r.value for r in UserRole)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{value}'. Must be one of: {valid}",
        )


@router.get("", response_model=list[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return [
        UserResponse(
            id=u.id, email=u.email, full_name=u.full_name,
            role=u.role.value, is_active=u.is_active,
            last_login=u.last_login, created_at=u.created_at,
            apartment_number=u.apartment_number, phone=u.phone,
        )
        for u in users
    ]


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role=_parse_role(body.role),
        apartment_number=body.apartment_number,
        phone=body.phone,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        role=user.role.value, is_active=user.is_active,
        last_login=user.last_login, created_at=user.created_at,
        apartment_number=user.apartment_number, phone=user.phone,
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    update_data = body.model_dump(exclude_unset=True)
    new_email = update_data.get("email")
    if new_email and new_email != user.email:
        dup = await db.execute(select(User).where(User.email == new_email, User.id != user_id))
        if dup.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already registered")
    if user.id == current_user.id and (
        update_data.get("is_active") is False
        or ("role" in update_data and update_data["role"] != user.role.value)
    ):
        raise HTTPException(status_code=400, detail="You cannot deactivate or change the role of your own account")
    password = update_data.pop("password", None)
    if password:
        user.hashed_password = hash_password(password)
    for field, value in update_data.items():
        if field == "role":
            value = _parse_role(value)
        elif field == "email" and not value:
            continue
        setattr(user, field, value)
    await db.flush()
    await db.refresh(user)
    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        role=user.role.value, is_active=user.is_active,
        last_login=user.last_login, created_at=user.created_at,
        apartment_number=user.apartment_number, phone=user.phone,
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Detach nullable references so history (alerts, reviews, ...) is kept.
    refs = [
        (Alert, "assigned_to"), (Alert, "acknowledged_by"), (Alert, "resolved_by"),
        (Alert, "reported_by"), (Camera, "created_by"), (IncidentReview, "reviewer_id"),
        (TrainingSample, "reviewer_id"), (AuditLog, "user_id"), (Vehicle, "resident_id"),
    ]
    for model, col in refs:
        if hasattr(model, col):
            await db.execute(update(model).where(getattr(model, col) == user_id).values({col: None}))
    await db.delete(user)
