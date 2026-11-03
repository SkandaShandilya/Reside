from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class ModelVersionResponse(BaseModel):
    id: int
    version: str
    name: str
    architecture: str
    dataset_name: Optional[str] = "UCF Crime + Verified CCTV Dataset"
    accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    confusion_matrix: Optional[Dict[str, Any]] = None
    per_class_metrics: Optional[Dict[str, Any]] = None
    activity_classes: Optional[List[str]] = None
    training_samples_count: Optional[int] = None
    validation_samples_count: Optional[int] = None
    epochs: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    is_deployed: bool
    is_baseline: bool
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ModelTrainRequest(BaseModel):
    name: Optional[str] = None
    dataset_name: Optional[str] = "Selected Datasets Ensemble"
    selected_datasets: List[str] = ["ucf_crime", "cctv_action", "verified_cctv"]
    architecture: str = "TimeDist-CNN"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    notes: Optional[str] = None




class ModelDeployRequest(BaseModel):
    notes: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[ModelVersionResponse]
    deployed_model: Optional[ModelVersionResponse] = None
