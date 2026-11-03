"""
Activity Dataset Loader — Phase 5

PyTorch Dataset loader for video activity recognition datasets.
Handles loading frame sequences from video clips or image frame folders,
applying spatial/temporal augmentations, and formatting tensors for model training.

Input format:
  Sequence of video frames normalized to shape (sequence_length, 3, 224, 224).
"""
import os
import cv2
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass
    torch = None
from typing import List, Tuple, Optional, Dict, Any


class ActivityVideoDataset(Dataset):
    """
    Dataset loader for suspicious activity video clips.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        sequence_length: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        class_to_idx: Optional[Dict[str, int]] = None,
        transform=None,
    ):
        """
        Args:
            samples: List of sample dicts containing {"video_path": str, "label": str}
            sequence_length: Number of frames per video clip
            image_size: Target frame size (width, height)
            class_to_idx: Mapping of activity label to integer index
        """
        self.samples = samples
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.transform = transform

        if class_to_idx is None:
            labels = sorted(list({s["label"] for s in samples}))
            self.class_to_idx = {label: i for i, label in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        sample = self.samples[idx]
        label_name = sample["label"]
        label_idx = self.class_to_idx.get(label_name, 0)

        if sample.get("frame_paths"):
            # An ORDERED sequence of individual frame images belonging to
            # one original video (see build_behaviour_manifest.py) -- e.g.
            # ucf-dataset's pre-extracted PNGs, sampled every Nth original
            # frame. Distinct from _load_image_as_clip below, which repeats
            # a SINGLE static image -- this reads real temporal evidence
            # across the video's actual (sparser) extracted frames.
            frames = self._load_frame_sequence(sample["frame_paths"])
        else:
            video_path = sample["video_path"]
            ext = os.path.splitext(video_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg"):
                frames = self._load_image_as_clip(video_path)
            else:
                frames = self._load_video_frames(
                    video_path,
                    window_start_sec=sample.get("window_start_sec"),
                    window_end_sec=sample.get("window_end_sec"),
                )
        tensor = self._process_frames(frames)

        return tensor, label_idx

    @staticmethod
    def _probe_duration_seconds(video_path: str) -> Optional[float]:
        """Returns the video's duration in seconds, or None if it can't be
        determined (missing file, unreadable header, zero fps/frame count)."""
        if not os.path.exists(video_path):
            return None
        cap = cv2.VideoCapture(video_path)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if total <= 0 or fps <= 0:
            return None
        return total / fps

    @classmethod
    def build_windowed_samples(
        cls,
        samples: List[Dict[str, Any]],
        window_seconds: float = 4.0,
        stride_seconds: float = 2.0,
        short_clip_max_seconds: float = 6.0,
        max_windows_per_video: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Expand {"video_path", "label"} samples into training samples with an
        added temporal window (see AIConfig.TRAIN_WINDOW_* for the deployed
        defaults, and the module docstring for why this matters).

        A video longer than `short_clip_max_seconds` is assumed to be an
        untrimmed/long recording that may contain the labeled event in only
        part of its duration, so it is split into multiple overlapping
        `window_seconds`-long windows (stride `stride_seconds` apart, capped
        at `max_windows_per_video`) — each becomes its own sample carrying
        "window_start_sec"/"window_end_sec" alongside the original video
        path and label, instead of spreading `sequence_length` frames evenly
        across the ENTIRE file (which silently diluted or entirely missed a
        brief event inside a long video — the exact bug this replaces).
        Videos at or under the threshold are assumed to already BE the event
        (e.g. a pre-trimmed dataset clip) and pass through unchanged, since
        windowing an already-short clip would just create near-duplicate
        samples with no new information.

        Every generated window keeps a "source_video" field. Callers MUST
        split into train/validation by that field (i.e. before calling this,
        on the original un-windowed sample list), never by generated sample
        — otherwise two windows of the same source video could land on
        opposite sides of the split and leak information across it.
        """
        expanded: List[Dict[str, Any]] = []
        for original in samples:
            video_path = original["video_path"]
            s = dict(original)
            s.setdefault("source_video", video_path)

            ext = os.path.splitext(video_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg"):
                expanded.append(s)
                continue

            duration = cls._probe_duration_seconds(video_path)
            if duration is None or duration <= short_clip_max_seconds:
                expanded.append(s)
                continue

            starts: List[float] = []
            t = 0.0
            while t + window_seconds <= duration and len(starts) < max_windows_per_video:
                starts.append(t)
                t += stride_seconds
            if not starts:
                starts = [0.0]

            for w_start in starts:
                w = dict(s)
                w["window_start_sec"] = w_start
                w["window_end_sec"] = min(w_start + window_seconds, duration)
                expanded.append(w)

        return expanded

    def _load_frame_sequence(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Load real temporal evidence from an ORDERED list of individual frame
        image paths (already sorted by original frame index by the caller —
        see build_behaviour_manifest.py). Evenly samples sequence_length
        indices across whatever frames are actually available for this
        video (the same np.linspace-based even-sampling _load_video_frames
        uses for real video files), rather than requiring exactly
        sequence_length frames to exist.
        """
        n = len(frame_paths)
        if n == 0:
            dummy = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            return [dummy] * self.sequence_length

        indices = np.linspace(0, n - 1, min(self.sequence_length, n), dtype=int)
        frames: List[np.ndarray] = []
        last_good: Optional[np.ndarray] = None
        for i in indices:
            img = cv2.imread(frame_paths[int(i)])
            if img is None:
                if last_good is not None:
                    frames.append(last_good.copy())
                    continue
                img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, self.image_size)
            frames.append(img)
            last_good = img

        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8))

        return frames[: self.sequence_length]

    def _load_image_as_clip(self, image_path: str) -> List[np.ndarray]:
        """
        Represent a single still image as a static clip (the same frame
        repeated sequence_length times). Feeding an image path straight to
        cv2.VideoCapture (as the video path) silently fails — it opens with
        0 frames and used to fall through to all-black dummy frames, which
        would have injected thousands of zero-signal "samples" into
        training without any error. Loading it as an image directly avoids
        that.
        """
        img = cv2.imread(image_path)
        if img is None:
            dummy = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            return [dummy] * self.sequence_length
        img = cv2.resize(img, self.image_size)
        return [img] * self.sequence_length

    def _load_video_frames(
        self,
        video_path: str,
        window_start_sec: Optional[float] = None,
        window_end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Read evenly spaced frames from a video file. If `window_start_sec`/
        `window_end_sec` are given (see `build_windowed_samples`), frames are
        sampled evenly only within that sub-range of the video instead of
        across its entire duration — otherwise behaves exactly as before.
        """
        frames = []
        if not os.path.exists(video_path):
            # Return synthetic dummy frames if video file doesn't exist
            dummy = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            return [dummy] * self.sequence_length

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        if total_frames <= 0:
            cap.release()
            dummy = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            return [dummy] * self.sequence_length

        if window_start_sec is not None and window_end_sec is not None and fps > 0:
            start_idx = max(0, int(window_start_sec * fps))
            end_idx = min(total_frames - 1, int(window_end_sec * fps))
            if end_idx <= start_idx:
                end_idx = min(total_frames - 1, start_idx + self.sequence_length)
        else:
            start_idx, end_idx = 0, total_frames - 1

        # Select frame indices evenly across the (possibly windowed) range
        indices = np.linspace(start_idx, end_idx, self.sequence_length, dtype=int)
        current_idx = 0

        for target_idx in indices:
            while current_idx < target_idx:
                cap.grab()
                current_idx += 1
            ret, frame = cap.read()
            current_idx += 1
            if ret:
                frame = cv2.resize(frame, self.image_size)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    dummy = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
                    frames.append(dummy)

        cap.release()

        # Pad if needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return frames[: self.sequence_length]

    def _process_frames(self, frames: List[np.ndarray]) -> Any:
        """Convert list of BGR numpy frames to (sequence_length, 3, 224, 224) float tensor."""
        processed = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            norm = rgb.astype(np.float32) / 255.0
            chw = np.transpose(norm, (2, 0, 1))
            processed.append(chw)

        tensor = torch.from_numpy(np.array(processed))  # (seq, 3, 224, 224)
        return tensor
