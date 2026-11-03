import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet, InvalidToken
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Reversible encryption for camera credentials ────────────────────────
# Camera RTSP/NVR passwords must be usable to actually connect (unlike
# login passwords, which are one-way hashed via pwd_context above) — they
# were previously stored as plain text in the `encrypted_password` column
# despite its name (backend/app/api/cameras.py had a literal
# "# TODO: encrypt in production" on that assignment). Fernet key is
# derived from the app's existing JWT_SECRET_KEY via SHA-256 rather than
# requiring a brand-new required config value this project doesn't have
# yet — anyone who already has JWT_SECRET_KEY can forge auth tokens
# anyway, so this doesn't introduce a new trust boundary, but it does stop
# a plain DB read/backup/leak from handing over camera passwords in the
# clear, which was the actual failure mode being fixed.
_fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(settings.JWT_SECRET_KEY.encode()).digest()))


def encrypt_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return _fernet.encrypt(value.encode()).decode()


def decrypt_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    try:
        return _fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        # Pre-existing plaintext row from before this fix, or corrupted
        # data — fail open to the raw stored value rather than crashing a
        # camera connection that previously worked, but this is exactly
        # the legacy state new writes are meant to stop producing.
        return value


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None
