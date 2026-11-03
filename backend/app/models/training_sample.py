import enum
from sqlalchemy import String, Enum, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base, TimestampMixin


class SampleSource(str, enum.Enum):
    DATASET = "dataset"
    VERIFIED_CCTV = "verified_cctv"


class TrainingSample(Base, TimestampMixin):
    __tablename__ = "training_samples"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_path: Mapped[str] = mapped_column(String(500), nullable=False)
    activity_label: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    source: Mapped[SampleSource] = mapped_column(
        Enum(SampleSource, name="sample_source"),
        nullable=False,
        default=SampleSource.DATASET,
    )
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    review_id: Mapped[int | None] = mapped_column(ForeignKey("incident_reviews.id"), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return f"<TrainingSample(id={self.id}, label='{self.activity_label}', source='{self.source.value}')>"
