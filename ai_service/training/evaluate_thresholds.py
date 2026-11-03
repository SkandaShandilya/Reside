"""
Threshold Calibration — evaluates the currently-deployed activity classifier
checkpoint on the VALIDATION split only, at a range of candidate confidence
thresholds, and reports precision/recall/F1/false-positive/false-negative
counts for each. This is how ACTIVITY_CONFIRM_THRESHOLD / ALERT_THRESHOLD
should actually be chosen — never by eyeballing one video, and never by
lowering a threshold until more alerts appear.

Uses the SAME train/val split logic as fine_tune.py (stratified 80/20 per
class) so the validation set here matches what the checkpoint itself was
validated against — this script does not re-train anything.

Usage:
    python -m ai_service.training.evaluate_thresholds
    python -m ai_service.training.evaluate_thresholds --checkpoint storage/models/suspicious_activity_v3.0.0.pt
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ai_service.config import AIConfig
from ai_service.activity_model import create_model
from ai_service.training.dataset import ActivityVideoDataset

CANDIDATE_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


class LeakageError(RuntimeError):
    """Raised when a leakage-free validation split isn't available — this
    script refuses to report numbers it can't back, rather than silently
    evaluating on data the checkpoint may have already been trained on."""


def _gather_validation_samples_from_checkpoint(checkpoint_ckpt: dict) -> List[Dict[str, Any]]:
    """
    The ONLY leakage-free path: fine_tune.py now saves the exact held-out
    file list into the checkpoint (`validation_video_paths`). Re-score
    exactly those files, nothing reconstructed or guessed.
    """
    paths = checkpoint_ckpt.get("validation_video_paths")
    if not paths:
        raise LeakageError(
            "This checkpoint was saved before validation_video_paths was added "
            "(see fine_tune.py) — there is no way to recover its exact held-out "
            "split after the fact, and re-deriving an approximate split risks "
            "silently scoring on data the model was trained on (this is exactly "
            "what produced an implausible ~0.98 F1 against a model whose "
            "checkpoint reports 0.807 training-time validation accuracy). "
            "Re-run fine-tuning to produce a checkpoint with a saved split, "
            "then evaluate that one."
        )
    class_to_idx = {c: i for i, c in enumerate(checkpoint_ckpt.get("activity_classes") or AIConfig.DEFAULT_ACTIVITY_CLASSES)}
    folder_label_aliases = {"Fire_Images": "Fire"}
    samples = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        label = folder_label_aliases.get(path.parent.name, path.parent.name)
        if label not in class_to_idx:
            continue
        samples.append({"video_path": str(path), "label": label})
    return samples


def _load_checkpoint(checkpoint_path: str = None):
    if checkpoint_path:
        path = Path(checkpoint_path)
    else:
        search_dirs = [Path("./backend/storage/models"), Path("./storage/models")]
        model_dir = next((d for d in search_dirs if d.exists()), None)
        if model_dir is None:
            raise FileNotFoundError("No storage/models directory found.")
        candidates = sorted(model_dir.glob("suspicious_activity_v*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found to evaluate.")
        # Same "highest validated accuracy" rule inference_engine.py uses —
        # evaluate whatever is actually deployed, not an arbitrary file.
        best_path, best_acc = None, -1.0
        for p in candidates:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            if ckpt.get("validation_samples_count", 0) >= 100 and ckpt.get("accuracy", -1) > best_acc:
                best_path, best_acc = p, ckpt["accuracy"]
        path = best_path or candidates[-1]

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    classes = ckpt.get("activity_classes") or AIConfig.DEFAULT_ACTIVITY_CLASSES
    model = create_model("TimeDist-CNN", num_classes=len(classes), sequence_length=AIConfig.SEQUENCE_LENGTH)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes, str(path), ckpt


def evaluate(checkpoint_path: str = None, output_path: str = "storage/models/threshold_evaluation.json"):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run threshold evaluation.")

    model, classes, checkpoint_used, ckpt = _load_checkpoint(checkpoint_path)
    val_samples = _gather_validation_samples_from_checkpoint(ckpt)
    if not val_samples:
        raise RuntimeError(f"Checkpoint {checkpoint_used}'s saved validation files no longer exist on disk.")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    val_ds = ActivityVideoDataset(val_samples, sequence_length=AIConfig.SEQUENCE_LENGTH, class_to_idx=class_to_idx)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Collect (top_confidence, predicted_idx, true_idx) once — reused across
    # every candidate threshold instead of re-running inference per threshold.
    records = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            probs = torch.softmax(model(inputs), dim=1)
            top_conf, top_idx = probs.max(dim=1)
            for c, p, t in zip(top_conf.tolist(), top_idx.tolist(), targets.tolist()):
                records.append((c, p, t))

    normal_idx = class_to_idx.get("Normal")
    results = {}
    for thresh in CANDIDATE_THRESHOLDS:
        tp = fp = fn = tn = 0
        for conf, pred, true in records:
            true_is_abnormal = true != normal_idx
            # "Alertable" per this threshold: confident enough AND not predicting Normal.
            predicted_abnormal = conf >= thresh and pred != normal_idx
            correct_class = pred == true
            if true_is_abnormal and predicted_abnormal and correct_class:
                tp += 1
            elif true_is_abnormal and not predicted_abnormal:
                fn += 1
            elif true_is_abnormal and predicted_abnormal and not correct_class:
                fp += 1  # confident, but the wrong abnormal class
            elif not true_is_abnormal and predicted_abnormal:
                fp += 1  # false alarm on an actually-normal clip
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[thresh] = {
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        }

    output = {
        "checkpoint": checkpoint_used,
        "classes": classes,
        "validation_samples": len(val_samples),
        "thresholds": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate activity-confirmation thresholds on the validation split.")
    parser.add_argument("--checkpoint", default=None, help="Path to a specific .pt checkpoint (default: best deployed one).")
    parser.add_argument("--output", default="storage/models/threshold_evaluation.json")
    args = parser.parse_args()

    out = evaluate(args.checkpoint, args.output)
    print(f"\nEvaluated {out['checkpoint']} on {out['validation_samples']} validation samples.\n")
    print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    for t, m in out["thresholds"].items():
        print(f"{t:>10.2f} | {m['precision']:>9.3f} | {m['recall']:>7.3f} | {m['f1']:>6.3f} | "
              f"{m['true_positives']:>4} | {m['false_positives']:>4} | {m['false_negatives']:>4}")
    print(f"\nFull results saved to {args.output}")
