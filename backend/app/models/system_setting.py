from sqlalchemy import Integer, Float
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base, TimestampMixin


class SystemSetting(Base, TimestampMixin):
    """
    Single-row table (id is always 1) holding the system-wide AI settings
    that are genuinely wired into the live detection pipeline. Only fields
    that actually change runtime behavior belong here — see
    app/api/settings.py for how each one is applied.
    """
    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    ai_analysis_fps: Mapped[int] = mapped_column(Integer, nullable=False, default=6)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.80)
    alert_cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=30)

    def __repr__(self) -> str:
        return f"<SystemSetting(ai_analysis_fps={self.ai_analysis_fps}, confidence_threshold={self.confidence_threshold})>"
