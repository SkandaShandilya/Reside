import io
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_current_user, get_db
from app.models.user import User, UserRole
from app.schemas.user import UserResponse
from app.utils.avatar import avatar_url_for
from app.utils.security import decode_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/profile", tags=["Profile"])

MAX_AVATAR_BYTES = 5 * 1024 * 1024
AVATAR_DIR = "avatars"


class ProfileUpdate(BaseModel):
    full_name: str | None = None
    phone: str | None = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def _to_response(u: User) -> UserResponse:
    return UserResponse(
        id=u.id, email=u.email, full_name=u.full_name, role=u.role.value,
        is_active=u.is_active, last_login=u.last_login, created_at=u.created_at,
        apartment_number=u.apartment_number, phone=u.phone,
        face_registered=u.face_registered, avatar_url=avatar_url_for(u),
    )


def _delete_avatar_file(user: User) -> None:
    if not user.avatar_path:
        return
    base = Path(settings.STORAGE_PATH).resolve()
    path = (base / user.avatar_path).resolve()
    try:
        path.relative_to(base)
        path.unlink(missing_ok=True)
    except (ValueError, OSError):
        pass


@router.put("", response_model=UserResponse)
async def update_profile(
    body: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Self-service edit. Email, role and apartment are admin-managed and
    deliberately not editable here."""
    if body.full_name is not None:
        name = body.full_name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        current_user.full_name = name[:255]
    if body.phone is not None:
        current_user.phone = body.phone.strip()[:20] or None
    await db.flush()
    await db.refresh(current_user)
    return _to_response(current_user)


@router.post("/password")
async def change_password(
    body: PasswordChange,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    if body.new_password == body.current_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current one")
    current_user.hashed_password = hash_password(body.new_password)
    await db.flush()
    return {"success": True}


@router.post("/avatar", response_model=UserResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_AVATAR_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 5MB)")
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not a valid image")

    # Normalise: square centre-crop, 256px JPEG (also strips EXIF/metadata).
    img = img.convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((256, 256))

    rel = Path(AVATAR_DIR) / f"{current_user.id}_{uuid.uuid4().hex[:8]}.jpg"
    abs_path = Path(settings.STORAGE_PATH) / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(abs_path, "JPEG", quality=88)

    _delete_avatar_file(current_user)
    current_user.avatar_path = rel.as_posix()
    await db.flush()
    await db.refresh(current_user)
    return _to_response(current_user)


@router.delete("/avatar", response_model=UserResponse)
async def remove_avatar(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _delete_avatar_file(current_user)
    current_user.avatar_path = None
    await db.flush()
    await db.refresh(current_user)
    return _to_response(current_user)


@router.get("/avatar/{user_id}")
async def get_avatar(
    user_id: int,
    request: Request,
    token: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Profile picture. Accepts the JWT as Bearer or ?token= (<img> can't
    send headers). Anyone may see their own; guards/admins may see others."""
    raw = token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        raw = auth[7:]
    payload = decode_access_token(raw) if raw else None
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    viewer = (await db.execute(select(User).where(User.id == int(payload["sub"])))).scalar_one_or_none()
    if not viewer or not viewer.is_active:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if viewer.id != user_id and viewer.role == UserRole.RESIDENT:
        raise HTTPException(status_code=403, detail="Not authorized.")
    owner = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not owner or not owner.avatar_path:
        raise HTTPException(status_code=404, detail="No profile picture.")
    base = Path(settings.STORAGE_PATH).resolve()
    path = (base / owner.avatar_path).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
