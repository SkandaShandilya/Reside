import enum
from sqlalchemy import String, Boolean, Enum, ForeignKey, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base, TimestampMixin


class ZoneType(str, enum.Enum):
    RESTRICTED = "restricted"
    MONITORING = "monitoring"
    NORMAL = "normal"
    # A tracked person dwelling inside an ASSET zone for
    # AIConfig.VANDALISM_MIN_SECONDS triggers a rule-based Vandalism event
    # (see ai_service/behaviour_engine.py) — distinct from RESTRICTED, which
    # alerts instantly on entry regardless of dwell time.
    ASSET = "asset"


class CameraZone(Base, TimestampMixin):
    __tablename__ = "camera_zones"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    zone_type: Mapped[ZoneType] = mapped_column(
        Enum(ZoneType, name="zone_type"),
        nullable=False,
        default=ZoneType.NORMAL,
    )
    # Polygon points stored as list of {x, y} coordinates (normalized 0-1)
    polygon_points: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Detection rules for this zone
    detection_rules: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    # Time restrictions (e.g., {"start": "22:00", "end": "06:00", "days": ["mon","tue",...]})
    time_restrictions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    alert_severity: Mapped[str] = mapped_column(String(20), default="critical", nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return f"<CameraZone(id={self.id}, name='{self.name}', type='{self.zone_type.value}')>"