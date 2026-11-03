from sqlalchemy import String, Float, Integer, Boolean, JSON, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture: Mapped[str] = mapped_column(String(100), nullable=False, default="TimeDist-CNN")
    dataset_name: Mapped[str | None] = mapped_column(String(255), nullable=True, default="UCF Crime + Verified CCTV Dataset")
    weights_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Metrics
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Detailed metrics
    confusion_matrix: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    per_class_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    activity_classes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Training info
    training_samples_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validation_samples_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    training_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Status
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelVersion(version='{self.version}', deployed={self.is_deployed})>"
