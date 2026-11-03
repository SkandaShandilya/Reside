import enum
from sqlalchemy import String, Boolean, Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base, TimestampMixin


class VehicleType(str, enum.Enum):
    CAR = "car"
    BIKE = "bike"
    OTHER = "other"


class Vehicle(Base, TimestampMixin):
    """
    A registered resident vehicle — additive to the schema, ported from
    the migration's Mongoose `Vehicle` model. `plate_number` is stored
    already-normalized (uppercase, alphanumeric only — see
    app/utils/plate.py's normalize_plate_number(), applied by the caller
    before insert/lookup, same as the original's pre-save hook).
    """
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    plate_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    owner_name: Mapped[str] = mapped_column(String(100), nullable=False)
    apartment_number: Mapped[str] = mapped_column(String(50), nullable=False)
    vehicle_type: Mapped[VehicleType] = mapped_column(
        Enum(VehicleType, name="vehicle_type"),
        nullable=False,
        default=VehicleType.CAR,
    )
    # Nullable — a vehicle can be registered by a guard/admin on a
    # resident's behalf without an owning User account tied to it,
    # matching the migration's `resident: default null`.
    resident_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # "resident" — self-registered by the resident via My Vehicles.
    # "visitor" — created automatically when a resident approves a
    # guard-raised VehicleRequest, so the same plate is recognized as
    # ALLOWED on future guard scans without re-approval.
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="resident")
    # Traceability back to the VehicleRequest this one-time pass was
    # created from (NULL for source="resident" rows). Lets
    # POST /api/vehicles/verify-plate flip the originating request's
    # status to USED the moment the pass is consumed.
    source_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicle_requests.id", ondelete="SET NULL"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Vehicle(id={self.id}, plate_number='{self.plate_number}')>"
