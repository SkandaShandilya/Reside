import enum
from sqlalchemy import String, Text, Enum, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class VehicleRequestStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    # The resident's approval granted exactly one entry, and that entry
    # has now happened (POST /api/vehicles/verify-plate matched the
    # one-time pass this request created and consumed it) — terminal,
    # same as REJECTED: never re-usable.
    USED = "used"


class VehicleRequest(Base, TimestampMixin):
    """
    A guard-initiated visitor-vehicle approval request, awaiting a
    resident's approve/reject response — ported from the migration's
    Mongoose `VehicleRequest` model, additive to the schema.
    """
    __tablename__ = "vehicle_requests"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    plate_number: Mapped[str] = mapped_column(String(20), nullable=False)
    guard_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    resident_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    apartment_number: Mapped[str] = mapped_column(String(50), nullable=False)
    visitor_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    purpose: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    status: Mapped[VehicleRequestStatus] = mapped_column(
        Enum(VehicleRequestStatus, name="vehicle_request_status"),
        nullable=False,
        default=VehicleRequestStatus.PENDING,
    )
    responded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    guard_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    resident_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Set when status transitions APPROVED -> USED (the one-time pass was
    # actually consumed at the gate), distinct from responded_at (when the
    # resident approved it).
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_vehicle_requests_resident_status", "resident_id", "status"),
        Index("ix_vehicle_requests_guard_status", "guard_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<VehicleRequest(id={self.id}, plate_number='{self.plate_number}', status='{self.status.value}')>"
