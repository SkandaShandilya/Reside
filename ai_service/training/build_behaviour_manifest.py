"""
Behaviour Dataset Manifest Builder — trains ONLY on:
  backend/storage/datasets/RWF-2000-dataset
  backend/storage/datasets/ucf-dataset

Per the task's explicit rule: do NOT combine each dataset's own train/val/
test directories as-is (their split definitions are dataset-specific, and
UCF's frame-extraction pipeline that produced this project's copy is
unrelated to RWF-2000's). Instead this pools every video from BOTH
directories of each dataset into one list per class, then performs a single,
clean, project-level 70/15/15 split at the ORIGINAL VIDEO level (never at
the generated-sample/window level, which would leak).

Two source formats:
  - RWF-2000: real .avi video files (train/Fight, train/NonFight, val/Fight,
    val/NonFight -- all four pooled).
  - ucf-dataset: pre-extracted PNG frames, one file per original video per
    sampled frame (stride 10), named "{video_id}_{frame_index}.png" (Train/
    Fighting, Train/NormalVideos, Test/Fighting, Test/NormalVideos -- all
    four pooled). Frames sharing the same "{video_id}" prefix are one
    original video and must never be split across train/val/test.

Only Fighting and Normal are included -- see the audit: neither dataset
contains any Kicking, Snatching, or Falling examples (checked directly,
twice, against the full directory tree). Building those classes from this
data would mean fabricating labels, which this project's rules explicitly
forbid.

Label mapping (transparent, not blind):
  RWF-2000 "Fight"        -> "Fighting"  (real fight videos, real label)
  RWF-2000 "NonFight"     -> "Normal"    (real non-violent videos, real label)
  ucf-dataset "Fighting"      -> "Fighting"  (real fight videos, real label)
  ucf-dataset "NormalVideos"  -> "Normal"    (real normal videos, real label)
No other ucf-dataset folder (RoadAccidents, Shooting, Vandalism) is used --
none of them represent Fighting or Normal, and none represent a currently
trainable class.
"""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

RWF_DIR = Path("backend/storage/datasets/RWF-2000-dataset")
UCF_DIR = Path("backend/storage/datasets/ucf-dataset")
OUT_PATH = Path("ai_service/training/behaviour_manifest.json")

SPLIT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is the remainder (0.15)


def _collect_rwf() -> Dict[str, List[dict]]:
    """Pool RWF-2000's train/ and val/ directories together -- see module
    docstring for why we don't keep the dataset's own split."""
    by_class: Dict[str, List[dict]] = defaultdict(list)
    mapping = {"Fight": "Fighting", "NonFight": "Normal"}
    for split_dir in ("train", "val"):
        for rwf_class, label in mapping.items():
            folder = RWF_DIR / split_dir / rwf_class
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in (".avi", ".mp4", ".mkv"):
                    by_class[label].append({
                        "type": "video",
                        "video_path": str(f.resolve()),
                        "label": label,
                        "source_id": f"rwf:{f.name}",
                    })
    return by_class


def _collect_ucf() -> Dict[str, List[dict]]:
    """Pool ucf-dataset's Train/ and Test/ directories together -- see
    module docstring. Frames are grouped by their video-id prefix (the
    filename minus the trailing "_<frame_index>.png")."""
    by_class: Dict[str, List[dict]] = defaultdict(list)
    mapping = {"Fighting": "Fighting", "NormalVideos": "Normal"}
    for split_dir in ("Train", "Test"):
        for ucf_class, label in mapping.items():
            folder = UCF_DIR / split_dir / ucf_class
            if not folder.exists():
                continue
            groups: Dict[str, List[Path]] = defaultdict(list)
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() == ".png":
                    # "Fighting002_x264_1230.png" -> prefix "Fighting002_x264", index 1230
                    stem = f.stem
                    idx = stem.rfind("_")
                    if idx == -1:
                        continue
                    prefix = stem[:idx]
                    try:
                        frame_idx = int(stem[idx + 1:])
                    except ValueError:
                        continue
                    groups[prefix].append((frame_idx, f))
            for prefix, frames in groups.items():
                frames.sort(key=lambda t: t[0])
                by_class[label].append({
                    "type": "frames",
                    "frame_paths": [str(p.resolve()) for _, p in frames],
                    "label": label,
                    "source_id": f"ucf:{split_dir}:{prefix}",
                })
    return by_class


def _split_video_level(items: List[dict], seed: int) -> Dict[str, List[dict]]:
    """70/15/15 split by source_id (one original video = one unit), never
    by generated sample -- items here are already one-per-original-video."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def build_manifest() -> dict:
    rwf = _collect_rwf()
    ucf = _collect_ucf()

    all_by_class: Dict[str, List[dict]] = defaultdict(list)
    for label, items in rwf.items():
        all_by_class[label].extend(items)
    for label, items in ucf.items():
        all_by_class[label].extend(items)

    manifest = {"train": [], "val": [], "test": [], "audit": {}}
    for label, items in sorted(all_by_class.items()):
        split = _split_video_level(items, seed=SPLIT_SEED)
        for part in ("train", "val", "test"):
            manifest[part].extend(split[part])

        rwf_count = len(rwf.get(label, []))
        ucf_count = len(ucf.get(label, []))
        manifest["audit"][label] = {
            "rwf_2000_source_videos": rwf_count,
            "ucf_dataset_source_videos": ucf_count,
            "total_source_videos": len(items),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        }

    return manifest


if __name__ == "__main__":
    manifest = build_manifest()
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["audit"], indent=2))
    print(f"\nTotal train={len(manifest['train'])} val={len(manifest['val'])} test={len(manifest['test'])}")
    print(f"Manifest written to {OUT_PATH}")
