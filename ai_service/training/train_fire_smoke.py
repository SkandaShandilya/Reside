"""
Fire/Smoke Neural Classifier — Training Script.

Trains ai_service.fire_smoke_model.FireSmokeCNN (a dedicated, single-frame
image classifier) on the dataset ingested by backend/scripts/
ingest_smoke_fire.py — completely separate from ai_service/training/
fine_tune.py (the suspicious-behaviour CNN's trainer, different classes,
different architecture, different data, different checkpoints).

Usage:
    python -m ai_service.training.train_fire_smoke
"""
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai_service.fire_smoke_model import create_fire_smoke_model
from ai_service.training.dataset_fire_smoke import FireSmokeImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["Normal", "Fire", "Smoke"]
IMAGE_SIZE = (224, 224)

# Real image sources only — the original Smoke/ folder mixes 50 SYNTHETIC
# CGI plume clips ("smoke_sim_*") in with the 3000 real Kaggle smoke photos
# ("yolosmoke_*") ingested alongside them; synthetic data is explicitly
# excluded here (never used for training this or any other model in this
# project) by filtering on filename prefix rather than folder alone.
SOURCE_DIRS = {
    "Fire": ("Fire", None),                      # no prefix filter — all real (curated + yolo-ingested)
    "Smoke": ("Smoke", "yolosmoke_"),             # ONLY the real, YOLO-ingested smoke photos
    "Normal": ("Normal_FireSmoke", "yolonormal_"),
}


def _gather_samples(storage_dir: Path) -> List[Dict[str, str]]:
    samples = []
    for label, (folder_name, prefix_filter) in SOURCE_DIRS.items():
        folder = storage_dir / folder_name
        if not folder.exists():
            logger.warning(f"{folder} does not exist — skipping class '{label}'.")
            continue
        files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            and (prefix_filter is None or f.name.startswith(prefix_filter))
        ]
        logger.info(f"Class '{label}': {len(files)} real images found in {folder}.")
        samples.extend({"path": str(f), "label": label} for f in files)
    return samples


def _compute_metrics(preds: List[int], targets: List[int], classes: List[str]) -> Tuple[float, float, float, float, Dict, List[List[int]]]:
    """Real predictions vs. real targets only — never fabricated (matches
    ai_service/training/fine_tune.py's _compute_metrics convention)."""
    n = len(classes)
    confusion = [[0] * n for _ in range(n)]
    for p, t in zip(preds, targets):
        confusion[t][p] += 1

    correct = sum(p == t for p, t in zip(preds, targets))
    acc = round(correct / len(preds), 3) if preds else 0.0

    per_class = {}
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    for i, cls_name in enumerate(classes):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n) if r != i)
        fn = sum(confusion[i][c] for c in range(n) if c != i)
        p_val = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        r_val = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1_val = round(2 * p_val * r_val / (p_val + r_val), 3) if (p_val + r_val) > 0 else 0.0
        per_class[cls_name] = {"precision": p_val, "recall": r_val, "f1": f1_val}
        total_p += p_val
        total_r += r_val
        total_f1 += f1_val

    avg_p = round(total_p / n, 3)
    avg_r = round(total_r / n, 3)
    avg_f1 = round(total_f1 / n, 3)
    return acc, avg_p, avg_r, avg_f1, per_class, confusion


def run_training(
    version: str = "1.0.0",
    epochs: int = 6,
    lr: float = 1e-4,
    batch_size: int = 16,
    checkpoint_dir: str = None,
) -> Dict:
    start_time = time.time()
    backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
    storage_dir = backend_dir / "storage" / "training_samples"
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else backend_dir / "storage" / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_samples = _gather_samples(storage_dir)
    if not all_samples:
        raise RuntimeError("No Fire/Smoke/Normal training images found — run backend/scripts/ingest_smoke_fire.py first.")

    # Stratified 80/20 split by class, shuffled — each sample here is
    # already one independent image (no video-window leakage concern the
    # way the behaviour CNN has).
    rng = random.Random(42)
    by_class: Dict[str, list] = {}
    for s in all_samples:
        by_class.setdefault(s["label"], []).append(s)

    train_samples, val_samples = [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        cut = max(1, int(len(items) * 0.8))
        train_samples.extend(items[:cut])
        val_samples.extend(items[cut:] or items[:1])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    logger.info(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    train_ds = FireSmokeImageDataset(train_samples, image_size=IMAGE_SIZE, class_to_idx=class_to_idx)
    val_ds = FireSmokeImageDataset(val_samples, image_size=IMAGE_SIZE, class_to_idx=class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_fire_smoke_model(num_classes=len(CLASSES))
    model.to(device)

    class_counts = [0] * len(CLASSES)
    for s in train_samples:
        class_counts[class_to_idx[s["label"]]] += 1
    class_weights = torch.tensor(
        [1.0 / max(c, 1) for c in class_counts], dtype=torch.float32, device=device
    )
    class_weights = class_weights * (len(CLASSES) / class_weights.sum())
    logger.info(f"Class counts (train): {dict(zip(CLASSES, class_counts))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(len(train_loader), 1)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    acc, precision, recall, f1, per_class, confusion = _compute_metrics(all_preds, all_targets, CLASSES)
    logger.info(f"Validation accuracy: {acc:.3f}  F1: {f1:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}")
    logger.info(f"Per-class: {per_class}")
    logger.info(f"Confusion matrix (rows=actual, cols=predicted, order={CLASSES}): {confusion}")

    checkpoint_path = checkpoint_dir / f"firesmoke_model_v{version}.pt"
    torch.save({
        "version": version,
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "accuracy": acc,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "trained_at": datetime.utcnow().isoformat(),
        "training_samples_count": len(train_samples),
        "validation_samples_count": len(val_samples),
        "epochs": epochs,
    }, checkpoint_path)

    duration = round(time.time() - start_time, 1)
    logger.info(f"Training completed in {duration}s. Saved checkpoint: {checkpoint_path}")

    return {
        "version": version,
        "weights_path": str(checkpoint_path),
        "classes": CLASSES,
        "accuracy": acc,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "training_samples_count": len(train_samples),
        "validation_samples_count": len(val_samples),
        "epochs": epochs,
        "training_duration_seconds": duration,
    }


if __name__ == "__main__":
    import json
    result = run_training()
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_matrix"}, indent=2))
    print("Confusion matrix:", result["confusion_matrix"])
