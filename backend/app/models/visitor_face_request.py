import enum
from sqlalchemy import String, Text, Enum, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class VisitorFaceRequestStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    # The resident's approval granted exactly one entry, and that entry
    # has now happened (POST /api/faces/verify matched the one-time pass
    # this request created and consumed it) — terminal, same as
    # REJECTED/EXPIRED: never re-usable.
    USED = "used"


class VisitorFaceRequest(Base, TimestampMixin):
    """
    A guard-initiated approval request for a visitor whose face did NOT
    match any registered household member. Carries the captured visitor
    photo so the resident can see who is at the gate before deciding.
    Mirrors VehicleRequest's pending/approved/rejected lifecycle; the
    resident's decision is the only thing that ever authorizes entry.
    """
    __tablename__ = "visitor_face_requests"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    guard_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    resident_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    visitor_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    location: Mapped[str] = mapped_column(String(255), default="Main Gate", nullable=False)
    status: Mapped[VisitorFaceRequestStatus] = mapped_column(
        Enum(VisitorFaceRequestStatus, name="visitor_face_request_status"),
        nullable=False,
        default=VisitorFaceRequestStatus.PENDING,
    )
    responded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resident_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Set when status transitions APPROVED -> USED (the one-time pass was
    # actually consumed at the gate), distinct from responded_at (when the
    # resident approved it).
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_visitor_face_requests_resident_status", "resident_id", "status"),
        Index("ix_visitor_face_requests_guard_status", "guard_id", "status"),
    )
