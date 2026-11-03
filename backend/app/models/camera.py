import enum
from sqlalchemy import String, Boolean, Enum, Integer, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base, TimestampMixin


class CameraSourceType(str, enum.Enum):
    RTSP = "rtsp"
    WEBCAM = "webcam"
    UPLOAD = "upload"
    NETWORK = "network"
    NVR = "nvr"


class Camera(Base, TimestampMixin):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Source configuration
    source_type: Mapped[CameraSourceType] = mapped_column(
        Enum(CameraSourceType, name="camera_source_type"),
        nullable=False,
    )
    rtsp_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    network_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    nvr_address: Mapped[str | None] = mapped_column(String(500), nullable=True)
    nvr_channel: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nvr_stream: Mapped[str | None] = mapped_column(String(100), nullable=True)
    upload_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Credentials (stored backend-side only, never exposed to frontend)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    encrypted_password: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Stream settings
    resolution: Mapped[str | None] = mapped_column(String(20), nullable=True, default="1920x1080")
    fps: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    ai_processing_fps: Mapped[int] = mapped_column(Integer, default=30, nullable=False)

    # Status
    is_online: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Features
    ai_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    recording_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    motion_detection: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Configuration
    retention_days: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    detection_config: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    alert_confidence_threshold: Mapped[float] = mapped_column(Float, default=0.80, nullable=False)
    cooldown_seconds: Mapped[int] = mapped_column(Integer, default=30, nullable=False)

    # Ownership
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name='{self.name}', online={self.is_online})>"
