import enum
from sqlalchemy import String, Float, Enum, ForeignKey, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class AlertStatus(str, enum.Enum):
    UNRESOLVED = "unresolved"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FALSE_POSITIVE = "false_positive"


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("security_events.id", ondelete="SET NULL"), nullable=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=True, index=True)
    camera_name: Mapped[str] = mapped_column(String(255), nullable=False)
    camera_location: Mapped[str] = mapped_column(String(255), nullable=False)
    activity: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="warning")
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status"),
        nullable=False,
        default=AlertStatus.UNRESOLVED,
    )
    persons_detected: Mapped[int] = mapped_column(default=0, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    assigned_to: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    acknowledged_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # "ai" (default, from the behaviour engine) or "resident_report" (a
    # Resident personally reporting something they observed).
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="ai")
    reported_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # Timeline-aware event fields (Phase 2), mirroring SecurityEvent's —
    # nullable so existing rows and non-lifecycle-tracked alert sources
    # (rule-based behaviour_engine detections, fire/smoke, resident reports)
    # stay valid with NULL rather than a fabricated value.
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    peak_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, activity='{self.activity}', status='{self.status.value}')>"
