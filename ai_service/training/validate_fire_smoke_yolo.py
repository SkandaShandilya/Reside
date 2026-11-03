"""
Post-training validation for the new YOLOv8 fire/smoke detector.

1. Runs YOLO's own val() against the held-out val split of the combined
   dataset (real photos, honestly measured precision/recall/mAP).
2. Runs the LIVE detector class (ai_service/fire_smoke_yolo_detector.py)
   against a sample of existing non-fire uploaded camera footage as a
   false-positive check — these clips are ordinary surveillance content
   (people/behaviour scenes), not fire/smoke, so any sustained
   CONFIRMED fire/smoke state on them is a real false-positive signal.

Run: python -m ai_service.training.validate_fire_smoke_yolo
"""
import glob
import json
from pathlib import Path

import cv2

DATA_YAML = "E:/MajorProject_27/backend/storage/datasets/fire_smoke_prepared/fire_smoke_combined.yaml"
UPLOADED_DIR = "E:/MajorProject_27/backend/storage/videos/uploaded_cameras"
SAMPLE_VIDEOS = 5
FRAMES_PER_VIDEO = 150


def run_val_metrics():
    from ultralytics import YOLO
    from ai_service.fire_smoke_yolo_detector import YOLOFireSmokeDetector

    d = YOLOFireSmokeDetector()
    assert d.model is not None, "No trained checkpoint found — run train_fire_smoke_yolo.py first."
    metrics = d.model.val(data=DATA_YAML, split="val")
    print("\n=== Held-out validation split metrics (real photos) ===")
    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision:{metrics.box.mp:.3f}")
    print(f"Recall:   {metrics.box.mr:.3f}")
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def run_false_positive_check():
    from ai_service.fire_smoke_yolo_detector import YOLOFireSmokeDetector
    from ai_service.fire_smoke_state import CameraFireSmokeState, CONFIRMED

    videos = sorted(glob.glob(str(Path(UPLOADED_DIR) / "*.mp4")))[:SAMPLE_VIDEOS]
    results = []
    for video_path in videos:
        detector = YOLOFireSmokeDetector()
        state = CameraFireSmokeState()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        confirmed_fire_frames = 0
        confirmed_smoke_frames = 0
        n = 0
        while n < FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (960, int(h * 960 / w)))
            dets, _ = detector.detect(resized)
            fire_conf = max((x["confidence"] for x in dets if x["class_name"] == "fire"), default=0.0)
            smoke_conf = max((x["confidence"] for x in dets if x["class_name"] == "smoke"), default=0.0)
            fr = state.fire.update(fire_conf, n / fps)
            sr = state.smoke.update(smoke_conf, n / fps)
            if fr.state == CONFIRMED:
                confirmed_fire_frames += 1
            if sr.state == CONFIRMED:
                confirmed_smoke_frames += 1
            n += 1
        cap.release()
        results.append({
            "video": Path(video_path).name, "frames": n,
            "confirmed_fire_frames": confirmed_fire_frames,
            "confirmed_smoke_frames": confirmed_smoke_frames,
        })
        print(f"{Path(video_path).name}: {n} frames, CONFIRMED fire={confirmed_fire_frames}, CONFIRMED smoke={confirmed_smoke_frames}")
    return results


def main():
    val_metrics = run_val_metrics()
    fp_results = run_false_positive_check()
    out = {"val_metrics": val_metrics, "false_positive_check": fp_results}
    Path("E:/MajorProject_27/fire_smoke_yolo_validation_report.json").write_text(json.dumps(out, indent=2))
    print("\nSaved fire_smoke_yolo_validation_report.json")


if __name__ == "__main__":
    main()
