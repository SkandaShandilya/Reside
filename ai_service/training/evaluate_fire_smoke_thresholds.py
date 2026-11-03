"""
Fire/Smoke Threshold Analysis — evaluates the FireSmokeDetector (HSV +
optical-flow, see ai_service/detector.py) against labeled real footage at a
range of candidate confidence thresholds, and saves per-threshold
precision/recall/F1/false-positive/false-negative CSVs.

IMPORTANT — this evaluates the CV DETECTOR's raw per-frame confidence
output, not a trained neural network: there is no separate "trained model"
to hold out a validation split from in the ML sense. Fire/Smoke/Normal are
split at the SOURCE-FILE level (whole videos/images never split across
categories) so no near-duplicate frames from the same source leak across
categories, honoring the same source-level-split principle a trained model
would need.

Usage:
    python -m ai_service.training.evaluate_fire_smoke_thresholds
"""
import csv
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ai_service.detector import FireSmokeDetector

CANDIDATE_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TARGET_WIDTH = 960
MAX_FRAMES_PER_VIDEO = 30
MAX_IMAGES_PER_CLASS = 300


def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (TARGET_WIDTH, int(h * TARGET_WIDTH / w)))


def _collect_confidences(paths: List[Path], target_class: str) -> List[float]:
    """Run the detector across sampled frames of these files and return the
    raw confidence for `target_class` on each frame (0.0 when absent)."""
    confidences = []
    for path in paths:
        detector = FireSmokeDetector(fire_threshold=0.0, smoke_threshold=0.0)  # no internal gate — we sweep thresholds ourselves
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            # FireSmokeDetector needs 6-of-8 frames of temporal-vote history
            # before it EVER reports a nonzero confidence (see detector.py)
            # — a single still image would silently read 0.0 forever
            # without priming that buffer. Repeating an identical frame is
            # a legitimate way to do that for a static image (maximally
            # "consistent" by definition, not fabricated evidence) — only
            # the final, fully-primed call is recorded as this image's
            # sample, the 7 warm-up calls are discarded.
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            resized = _resize(frame)
            dets = []
            for _ in range(8):
                dets, _ = detector.detect(resized)
            conf = max((d["confidence"] for d in dets if d["class_name"] == target_class), default=0.0)
            confidences.append(conf)
            continue

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES_PER_VIDEO
        stride = max(1, total // MAX_FRAMES_PER_VIDEO)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = _resize(frame)
            dets, _ = detector.detect(resized)
            if idx % stride == 0:
                conf = max((d["confidence"] for d in dets if d["class_name"] == target_class), default=0.0)
                confidences.append(conf)
            idx += 1
        cap.release()
    return confidences


def _metrics_at_threshold(pos_confs: List[float], neg_confs: List[float], threshold: float) -> dict:
    tp = sum(1 for c in pos_confs if c >= threshold)
    fn = len(pos_confs) - tp
    fp = sum(1 for c in neg_confs if c >= threshold)
    tn = len(neg_confs) - fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold, "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
    }


def evaluate(storage_dir: Path, seed: int = 42):
    rng = random.Random(seed)

    fire_dir = storage_dir / "Fire"
    fire_images_dir = storage_dir / "Fire_Images"
    smoke_dir = storage_dir / "Smoke"
    normal_dir = storage_dir / "Normal"

    fire_files = list(fire_dir.glob("*.*")) if fire_dir.exists() else []
    fire_files += list(fire_images_dir.glob("*.*")) if fire_images_dir.exists() else []
    rng.shuffle(fire_files)
    fire_files = fire_files[:MAX_IMAGES_PER_CLASS]

    smoke_files = list(smoke_dir.glob("*.*")) if smoke_dir.exists() else []
    rng.shuffle(smoke_files)

    normal_files = list(normal_dir.glob("*.*")) if normal_dir.exists() else []
    rng.shuffle(normal_files)
    normal_files = normal_files[:40]  # enough frames once sampled per-video

    print(f"Fire positives: {len(fire_files)} files | Smoke positives: {len(smoke_files)} files | "
          f"Normal negatives: {len(normal_files)} files")

    fire_pos_confs = _collect_confidences(fire_files, "fire")
    fire_neg_confs = _collect_confidences(normal_files, "fire") + _collect_confidences(smoke_files, "fire")
    smoke_pos_confs = _collect_confidences(smoke_files, "smoke")
    smoke_neg_confs = _collect_confidences(normal_files, "smoke") + _collect_confidences(fire_files, "smoke")

    fire_rows = [_metrics_at_threshold(fire_pos_confs, fire_neg_confs, t) for t in CANDIDATE_THRESHOLDS]
    smoke_rows = [_metrics_at_threshold(smoke_pos_confs, smoke_neg_confs, t) for t in CANDIDATE_THRESHOLDS]

    fieldnames = ["threshold", "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives", "true_negatives"]
    for name, rows in [("fire_threshold_analysis.csv", fire_rows), ("smoke_threshold_analysis.csv", smoke_rows)]:
        out_path = storage_dir.parent / "models" / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {out_path}")

    return fire_rows, smoke_rows


if __name__ == "__main__":
    search_dirs = [
        Path("./backend/storage/training_samples"),
        Path("./storage/training_samples"),
    ]
    storage_dir = next((d for d in search_dirs if d.exists()), None)
    if storage_dir is None:
        raise FileNotFoundError("storage/training_samples not found — run from the project root or backend/.")

    fire_rows, smoke_rows = evaluate(storage_dir)

    print(f"\n{'FIRE':^70}")
    print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    for r in fire_rows:
        print(f"{r['threshold']:>10.2f} | {r['precision']:>9.3f} | {r['recall']:>7.3f} | {r['f1']:>6.3f} | "
              f"{r['true_positives']:>4} | {r['false_positives']:>4} | {r['false_negatives']:>4}")

    print(f"\n{'SMOKE':^70}")
    print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    for r in smoke_rows:
        print(f"{r['threshold']:>10.2f} | {r['precision']:>9.3f} | {r['recall']:>7.3f} | {r['f1']:>6.3f} | "
              f"{r['true_positives']:>4} | {r['false_positives']:>4} | {r['false_negatives']:>4}")
