"""
Isolated YOLOv8 training script for the new Fire+Smoke object detector.

Trains a fresh yolov8n.pt base on the combined dataset (see
prepare_fire_smoke_yolo_dataset.py + fire_smoke_combined.yaml) and saves
weights to backend/storage/models/fire_smoke_yolov8/ — a directory
completely separate from the existing suspicious_activity_*.pt and
firesmoke_model_v*.pt checkpoints, which this script never reads or
writes.

Run: python -m ai_service.training.train_fire_smoke_yolo
"""
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "E:/MajorProject_27/backend/storage/datasets/fire_smoke_prepared/fire_smoke_combined.yaml"
BASE_WEIGHTS = "E:/MajorProject_27/backend/yolov8n.pt"
PROJECT_DIR = "E:/MajorProject_27/backend/storage/models/fire_smoke_yolov8"
RUN_NAME = "train"


def main():
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=15,
        device=0,
    )


if __name__ == "__main__":
    main()
