import enum
from sqlalchemy import String, Boolean, Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base, TimestampMixin


class NotificationChannel(str, enum.Enum):
    IN_APP = "in_app"
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"


class NotificationSetting(Base, TimestampMixin):
    __tablename__ = "notification_settings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    activity_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    notify_resident: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notify_guard: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_admin: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    channel: Mapped[NotificationChannel] = mapped_column(
        Enum(NotificationChannel, name="notification_channel"),
        nullable=False,
        default=NotificationChannel.IN_APP,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    severity_filter: Mapped[str | None] = mapped_column(String(20), nullable=True)
    message_template: Mapped[str | None] = mapped_column(String(500), nullable=True)

    def __repr__(self) -> str:
        return f"<NotificationSetting(activity='{self.activity_type}', channel='{self.channel.value}')>"
