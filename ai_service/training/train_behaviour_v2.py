"""
Behaviour Model v2 — Training FROM SCRATCH on ONLY:
  backend/storage/datasets/RWF-2000-dataset
  backend/storage/datasets/ucf-dataset

Deliberately a SEPARATE script from fine_tune.py, not a modification of it:
fine_tune.py (a) always tries to load a prior checkpoint's weights and
continue training them, and (b) auto-folds in backend/storage/
training_samples/ (the OLD behaviour dataset, outside the two datasets this
task is restricted to). Both are exactly what this task forbids, so reusing
fine_tune.py's code path — even with those two things disabled by a flag —
risked one of them silently reactivating in the future. fine_tune.py itself
is left completely untouched and still available for its original purpose
(admin-triggered fine-tuning from the human-verified review queue).

Classes: Normal, Fighting ONLY. Kicking/Snatching/Falling are not trained —
see ai_service/training/build_behaviour_manifest.py's docstring and this
project's audit: neither allowed dataset contains a single legitimate
example of any of the three, and fabricating labels for them was explicitly
ruled out.

Training flow (per spec): dataset audit -> class filtering -> video-level
split -> temporal window generation -> training -> validation (per-epoch,
best-checkpoint selection) -> final test (held out, touched exactly once).
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ai_service.config import AIConfig
from ai_service.activity_model import create_model
from ai_service.training.dataset import ActivityVideoDataset

MANIFEST_PATH = Path("ai_service/training/behaviour_manifest.json")
CLASSES = ["Normal", "Fighting"]


def _compute_metrics(preds: List[int], targets: List[int], classes: List[str]) -> Tuple[float, float, float, float, Dict, List[List[int]]]:
    """Real predictions vs. real targets only — never fabricated (same
    convention as fine_tune.py / train_fire_smoke.py's _compute_metrics)."""
    n = len(classes)
    confusion = [[0] * n for _ in range(n)]
    if not preds or not targets or len(preds) != len(targets):
        zero_per_class = {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for c in classes}
        return 0.0, 0.0, 0.0, 0.0, zero_per_class, confusion

    for p, t in zip(preds, targets):
        confusion[t][p] += 1

    correct = sum(p == t for p, t in zip(preds, targets))
    acc = round(correct / len(preds), 3)

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


def _evaluate(model, loader, device) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())
    return all_preds, all_targets


def run_training(
    version: str = "16.0.0",
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 8,
    checkpoint_dir: str = "backend/storage/models",
) -> Dict[str, Any]:
    start_time = time.time()

    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available — cannot train (no fabricated-metrics fallback for this script).")

    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"{MANIFEST_PATH} not found — run build_behaviour_manifest.py first.")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    train_samples = manifest["train"]
    val_samples = manifest["val"]
    test_samples = manifest["test"]

    logger.info(f"Train={len(train_samples)} Val={len(val_samples)} Test={len(test_samples)}")
    logger.info(f"Manifest audit: {json.dumps(manifest['audit'])}")

    train_ds = ActivityVideoDataset(train_samples, sequence_length=AIConfig.SEQUENCE_LENGTH, class_to_idx=class_to_idx)
    val_ds = ActivityVideoDataset(val_samples, sequence_length=AIConfig.SEQUENCE_LENGTH, class_to_idx=class_to_idx)
    test_ds = ActivityVideoDataset(test_samples, sequence_length=AIConfig.SEQUENCE_LENGTH, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # FROM SCRATCH: ImageNet-pretrained ResNet-18 backbone (create_model's
    # own default), but NO prior suspicious_activity_*.pt weights loaded —
    # deliberately, per the task's "no old checkpoint fine-tuning" rule.
    model = create_model("TimeDist-CNN", num_classes=len(CLASSES), sequence_length=AIConfig.SEQUENCE_LENGTH)
    model.to(device)

    class_counts = [0] * len(CLASSES)
    for s in train_samples:
        class_counts[class_to_idx[s["label"]]] += 1
    class_weights = torch.tensor([1.0 / max(c, 1) for c in class_counts], dtype=torch.float32, device=device)
    class_weights = class_weights * (len(CLASSES) / class_weights.sum())
    logger.info(f"Class counts (train): {dict(zip(CLASSES, class_counts))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state_dict = None
    best_epoch = -1
    per_epoch_log = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)

        val_preds, val_targets = _evaluate(model, val_loader, device)
        val_acc, val_p, val_r, val_f1, _, _ = _compute_metrics(val_preds, val_targets, CLASSES)
        logger.info(f"Epoch [{epoch+1}/{epochs}] loss={avg_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")
        per_epoch_log.append({"epoch": epoch + 1, "loss": round(avg_loss, 4), "val_accuracy": val_acc, "val_f1": val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state_dict is None:
        raise RuntimeError("Training produced no evaluable epoch — cannot select a best checkpoint.")

    logger.info(f"Best epoch: {best_epoch} (val_accuracy={best_val_acc:.3f}) — restoring its weights for final test.")
    model.load_state_dict(best_state_dict)

    # FINAL TEST — the held-out 15% never used for training or per-epoch
    # model selection, touched exactly once, here.
    test_preds, test_targets = _evaluate(model, test_loader, device)
    test_acc, test_p, test_r, test_f1, per_class, confusion = _compute_metrics(test_preds, test_targets, CLASSES)
    logger.info(f"FINAL TEST — accuracy={test_acc:.3f} f1={test_f1:.3f} precision={test_p:.3f} recall={test_r:.3f}")
    logger.info(f"Per-class: {per_class}")
    logger.info(f"Confusion matrix (rows=actual, cols=predicted, order={CLASSES}): {confusion}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(Path(checkpoint_dir) / f"suspicious_activity_v{version}.pt")
    torch.save({
        "version": version,
        "model_state_dict": best_state_dict,
        "activity_classes": CLASSES,
        "class_to_index": class_to_idx,
        "sequence_length": AIConfig.SEQUENCE_LENGTH,
        "image_size": list(AIConfig.IMAGE_SIZE),
        "normalization": "0-1 scale, RGB, no mean/std subtraction (matches ActivityVideoDataset._process_frames)",
        "architecture": "Time-Distributed CNN (ResNet-18 + BiLSTM)",
        "dataset_sources": ["RWF-2000-dataset", "ucf-dataset"],
        "dataset_version": "project-level 70/15/15 video-level split, seed=42 (see build_behaviour_manifest.py)",
        "accuracy": test_acc,
        "val_accuracy": best_val_acc,
        "precision_score": test_p,
        "recall_score": test_r,
        "f1_score": test_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "best_epoch": best_epoch,
        "per_epoch_log": per_epoch_log,
        "training_samples_count": len(train_samples),
        "validation_samples_count": len(val_samples),
        "test_samples_count": len(test_samples),
        "dataset_audit": manifest["audit"],
        "epochs": epochs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Trained from scratch (ImageNet backbone only, no prior "
            "suspicious_activity_*.pt weights loaded) on ONLY RWF-2000-dataset "
            "and ucf-dataset. Normal+Fighting only -- Kicking/Snatching/Falling "
            "dropped: zero legitimate examples in either allowed dataset."
        ),
    }, checkpoint_path)

    duration = round(time.time() - start_time, 1)
    logger.info(f"Training complete in {duration}s. Saved: {checkpoint_path}")

    return {
        "version": version,
        "weights_path": checkpoint_path,
        "accuracy": test_acc,
        "val_accuracy": best_val_acc,
        "precision_score": test_p,
        "recall_score": test_r,
        "f1_score": test_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "best_epoch": best_epoch,
        "training_samples_count": len(train_samples),
        "validation_samples_count": len(val_samples),
        "test_samples_count": len(test_samples),
        "epochs": epochs,
        "training_duration_seconds": duration,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result = run_training()
    print(json.dumps(result, indent=2))
