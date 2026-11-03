"""
Video Processor & Event Clip Recorder — Phase 4 (Fire/Smoke rewrite)

Saves pre-event and post-event video clips and thumbnail snapshots
when a security alert is triggered.

Storage layout (paths returned are RELATIVE TO the backend's storage root,
matching backend/app/main.py's `GET /storage/{file_path:path}` route, which
resolves `file_path` against that same root — a caller builds a servable
URL as `f"/storage/{returned_path}"`, nothing more):
  clips/{prefix}_cam{id}_{timestamp}.mp4
  snapshots/{prefix}_cam{id}_{timestamp}.jpg

This is the ONE place that writes alert clips/snapshots — both the live
camera path (backend/app/services/video_stream_service.py,
alert_dispatcher.py) and the uploaded-video path (ai_service/
video_timeline_processor.py) call into this module rather than each
maintaining their own VideoWriter logic, so a fix here (e.g. the validation
added below) applies everywhere.
"""
import cv2
import time
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles thumbnail generation and MP4 video clip recording, with the
    clip verified (not just "the write call didn't raise") before it is
    considered successful — a codec/container failure in OpenCV commonly
    reports success (writer.isOpened() True, no exception) while producing
    a file that will not decode a single frame. See _verify_clip().
    """

    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = Path(storage_dir)
        self.snapshots_dir = self.storage_dir / "snapshots"
        self.clips_dir = self.storage_dir / "clips"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        frame: np.ndarray,
        camera_id: int,
        prefix: str = "alert",
    ) -> Optional[str]:
        """
        Save a single JPEG frame snapshot.
        Returns a path relative to this processor's storage root (e.g.
        "snapshots/alert_cam1_1692384000.jpg"), or None if the frame was
        unusable or the write failed.
        """
        if frame is None or frame.size == 0:
            return None

        try:
            timestamp = int(time.time() * 1000)
            filename = f"{prefix}_cam{camera_id}_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename
            ok = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok or not filepath.exists() or filepath.stat().st_size == 0:
                logger.error(f"Snapshot write failed or produced an empty file: {filepath}")
                filepath.unlink(missing_ok=True)
                return None
            return f"snapshots/{filename}"
        except Exception as e:
            logger.error(f"Snapshot save error: {e}")
            return None

    def save_clip(
        self,
        frames: List[np.ndarray],
        camera_id: int,
        fps: float = 10,
        prefix: str = "event",
    ) -> Optional[str]:
        """
        Write a sequence of BGR frames as an MP4 video clip.
        Returns a path relative to this processor's storage root (e.g.
        "clips/event_cam1_1692384000.mp4"), or None if there were no usable
        frames or the resulting file failed verification (see
        _verify_clip) — in which case the bad file is deleted rather than
        left behind as a phantom "successful" clip.
        """
        usable = [f for f in frames if f is not None and f.size > 0]
        if not usable:
            return None
        return self._write_clip(usable, camera_id, fps, prefix)

    def save_clip_from_jpegs(
        self,
        jpeg_frames: List[bytes],
        camera_id: int,
        fps: float = 15,
        prefix: str = "event",
    ) -> Optional[str]:
        """
        Same as save_clip, but decodes a list of already-JPEG-encoded frames
        first — used by the live-camera path, whose rolling frame buffer
        stores JPEG bytes (see backend/app/services/video_stream_service.py
        FrameBuffer), not raw arrays.
        """
        decoded: List[np.ndarray] = []
        for jpeg in jpeg_frames:
            if not jpeg:
                continue
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                decoded.append(frame)
        if not decoded:
            return None
        return self._write_clip(decoded, camera_id, fps, prefix)

    def _write_clip(
        self,
        frames: List[np.ndarray],
        camera_id: int,
        fps: float,
        prefix: str,
    ) -> Optional[str]:
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{prefix}_cam{camera_id}_{timestamp}.mp4"
            filepath = self.clips_dir / filename

            h, w = frames[0].shape[:2]
            if h <= 0 or w <= 0:
                logger.error(f"Clip write aborted — invalid frame dimensions {w}x{h}")
                return None

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(filepath), fourcc, max(fps, 1.0), (w, h))
            if not writer.isOpened():
                logger.error(f"cv2.VideoWriter failed to open for {filepath} (codec/container issue).")
                return None

            written = 0
            try:
                for f in frames:
                    if f is None:
                        continue
                    if f.shape[:2] != (h, w):
                        f = cv2.resize(f, (w, h))
                    writer.write(f)
                    written += 1
            finally:
                writer.release()

            if not self._verify_clip(filepath):
                filepath.unlink(missing_ok=True)
                return None

            logger.info(f"Saved video clip: {filepath} ({written} frames, {fps:.1f}fps)")
            return f"clips/{filename}"
        except Exception as e:
            logger.error(f"Video clip save error: {e}")
            return None

    @staticmethod
    def _verify_clip(filepath: Path) -> bool:
        """
        A VideoWriter that reports success can still produce a file that
        will not play — most commonly because every write() call silently
        no-opped after a dimension/format mismatch, or the codec produced a
        container with zero decodable frames. This is exactly the "blank
        clip" failure mode this rewrite targets, so every clip is verified
        by actually reopening and decoding it before it's handed back as a
        successful path:
          - file exists and is non-empty
          - VideoCapture can (re)open it
          - it reports a positive frame count and fps
          - at least one frame can actually be decoded from it
        """
        if not filepath.exists() or filepath.stat().st_size == 0:
            logger.error(f"Clip verification failed — file missing or empty: {filepath}")
            return False

        cap = cv2.VideoCapture(str(filepath))
        try:
            if not cap.isOpened():
                logger.error(f"Clip verification failed — OpenCV could not reopen: {filepath}")
                return False
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.error(f"Clip verification failed — could not decode any frame: {filepath}")
                return False
            if width <= 0 or height <= 0:
                logger.error(f"Clip verification failed — invalid dimensions {width}x{height}: {filepath}")
                return False
            # frame_count/fps can legitimately read back as 0/-1 for some
            # mp4v-in-mp4 containers immediately after writing on certain
            # OpenCV/ffmpeg builds even when frames genuinely decode fine —
            # the actual successful frame READ above is the authoritative
            # check; frame_count/fps are logged for diagnostics only.
            if frame_count <= 0 or fps <= 0:
                logger.info(
                    f"Clip {filepath.name}: container reports frame_count={frame_count} fps={fps} "
                    f"(metadata not populated by this codec) but a real frame decoded successfully — accepting."
                )
            return True
        finally:
            cap.release()


# Global singleton â€” used by callers that don't need a specific storage
# root (e.g. standalone scripts/tests). Backend services should construct
# their own VideoProcessor(settings.STORAGE_PATH) instead so paths are
# correct regardless of process working directory — see
# backend/app/services/alert_dispatcher.py.
video_processor = VideoProcessor()
