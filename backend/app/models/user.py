import enum
from sqlalchemy import String, Boolean, Enum, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class UserRole(str, enum.Enum):
    RESIDENT = "resident"
    GUARD = "guard"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.RESIDENT,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    apartment_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    avatar_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Face Verification (see api/faces.py, face-service/) — additive,
    # all nullable so every existing row/user is unaffected. Storing the
    # raw 512-d InsightFace embedding as JSON (a plain list[float]),
    # matching the original migration's representation 1:1 rather than
    # inventing a new one. Never selected by default in a bare `select
    # (User)` — see get_residents_with_face_embeddings() in
    # api/faces.py, which explicitly loads this column only where the
    # verification comparison actually needs it (mirrors the migration's
    # Mongoose `select: false` intent, adapted to SQLAlchemy, which has
    # no direct equivalent — callers must simply not request it in a
    # response schema unless needed).
    face_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    face_registered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    face_registered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', role='{self.role.value}')>"
