import enum
from sqlalchemy import String, Float, Integer, ForeignKey, JSON, DateTime, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class SeverityLevel(str, enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class SecurityEvent(Base, TimestampMixin):
    __tablename__ = "security_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    activity_class: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[SeverityLevel] = mapped_column(
        Enum(SeverityLevel, name="severity_level"),
        nullable=False,
        default=SeverityLevel.INFO,
    )
    persons_detected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    # Timeline-aware event fields (nullable — existing rows predate these
    # and stay valid; only events created/updated via the state-machine
    # lifecycle in ai_service/inference_engine.py + alert_dispatcher.py
    # populate them). `timestamp` above is unchanged and keeps meaning
    # "when this row was recorded" for existing consumers; `start_time` is
    # the actual moment the underlying activity began, which for an
    # edge-triggered event is the same moment but is tracked separately so
    # a future event type where they diverge doesn't need a schema change.
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    peak_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<SecurityEvent(id={self.id}, activity='{self.activity_class}', confidence={self.confidence})>"
