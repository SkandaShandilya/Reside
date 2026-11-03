"""
One-time, isolated data-prep step for the combined Fire+Smoke YOLOv8
dataset (see ai_service/training/train_fire_smoke_yolo.py).

Two source datasets have DIFFERENT class orderings:
  - fire-smoke.v1i.yolov8:      names: ['fire', 'smoke']  -> 0=fire, 1=smoke
  - Smoke-Fire-Detection-YOLO:  names: ['smoke', 'fire']  -> 0=smoke, 1=fire

Naively pointing YOLO at both with a single names=['fire','smoke'] map
would silently swap every fire/smoke label in dataset 2. This script
does NOT touch/move/rename any original image or label file — it
writes REMAPPED COPIES of dataset 2's label .txt files (class id
0<->1 swapped) into a new, isolated directory, and creates NTFS
directory junctions (not copies) to the original image folders so
disk usage/setup time stays minimal while YOLO's default
"labels" <-> "images" sibling-folder convention still resolves
correctly for the new label files.

Run once: python -m ai_service.training.prepare_fire_smoke_yolo_dataset
"""
import subprocess
from pathlib import Path

ROOT = Path("E:/MajorProject_27/backend/storage/datasets/Fire-and-Smoke-detection(YOLOv8)")
DATASET1 = ROOT / "fire-smoke.v1i.yolov8"          # already fire=0, smoke=1 — used as-is
DATASET2 = ROOT / "Smoke-Fire-Detection-YOLO" / "data"  # smoke=0, fire=1 — needs remap

PREPARED = Path("E:/MajorProject_27/backend/storage/datasets/fire_smoke_prepared/dataset2_fixed")

# dataset2's split dir name -> prepared split dir name
SPLITS = {"train": "train", "val": "val", "test": "test"}


def remap_labels(src_labels_dir: Path, dst_labels_dir: Path) -> int:
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_file in src_labels_dir.glob("*.txt"):
        lines_out = []
        for line in src_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            # swap: dataset2's 0(smoke)->1(smoke in combined map), 1(fire)->0(fire)
            new_cls = 1 - cls if cls in (0, 1) else cls
            parts[0] = str(new_cls)
            lines_out.append(" ".join(parts))
        (dst_labels_dir / src_file.name).write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
        count += 1
    return count


def make_junction(target: Path, link: Path) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mklink failed for {link} -> {target}: {result.stderr or result.stdout}")


def main():
    assert DATASET1.exists(), f"Dataset 1 not found: {DATASET1}"
    assert DATASET2.exists(), f"Dataset 2 not found: {DATASET2}"

    for split in SPLITS:
        src_images = DATASET2 / split / "images"
        src_labels = DATASET2 / split / "labels"
        dst_images = PREPARED / split / "images"
        dst_labels = PREPARED / split / "labels"

        assert src_images.exists(), f"missing {src_images}"
        assert src_labels.exists(), f"missing {src_labels}"

        n = remap_labels(src_labels, dst_labels)
        make_junction(src_images.resolve(), dst_images)
        print(f"[{split}] remapped {n} label files, junction {dst_images} -> {src_images}")

    print("Dataset 2 remap complete. Originals untouched.")


if __name__ == "__main__":
    main()
