"""
Object Tracker — Phase 3

Provides multi-person centroid & IOU tracking across video frames.
Assigns consistent Track IDs (e.g. ID #1, ID #2) to detected individuals.

Features:
- Track ID assignment and persistence across frame drops
- Trajectory path history (centroid coordinates over time)
- Speed / movement vector calculation (useful for running/rapid movement detection)
- Dwell time / loitering duration tracking per person
"""
import math
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import deque


class Track:
    """Represents a tracked person over time."""

    # How far back (seconds) we keep position history — must be AT LEAST
    # AIConfig.LOITERING_SECONDS (180s), otherwise wander_radius below only
    # ever reflects the trailing HISTORY_WINDOW_SECONDS of movement: a
    # person who drifted outside the loiter radius early on and wandered
    # back would incorrectly read as having stayed put the whole time,
    # since the evidence of that excursion would already have been pruned
    # out of history. Not read from AIConfig directly (this module has no
    # dependency on it) — kept in sync by hand, with margin.
    HISTORY_WINDOW_SECONDS = 200.0
    # Window used for the *instantaneous* speed estimate, so a person who
    # stopped moving reads as slow immediately rather than being averaged
    # against their whole time on screen.
    SPEED_WINDOW_SECONDS = 1.0
    # How much recent history the overlay's trajectory tail draws — a
    # genuinely TIME-windowed slice (not a fixed sample count), so it means
    # the same ~2 real seconds regardless of analysis cadence (6fps live vs
    # 8fps uploaded video).
    TRAIL_WINDOW_SECONDS = 2.0

    def __init__(self, track_id: int, initial_bbox: List[int], initial_center: Tuple[int, int]):
        self.track_id = track_id
        self.bbox = initial_bbox  # [x1, y1, x2, y2]
        self.center = initial_center
        self.history: deque = deque()  # (center, timestamp), pruned by age not count
        self.history.append((initial_center, time.time()))
        # (aspect_ratio h/w, timestamp) — separate from `history` (centers
        # only) so Fighting's Gate 2 can measure how rapidly a track's own
        # bbox SHAPE is changing (grappling/falling), not just how far its
        # centroid has moved — see aspect_ratio_jitter().
        self.aspect_history: deque = deque()
        self.aspect_history.append((self._aspect_ratio(initial_bbox), time.time()))

        self.first_seen = time.time()
        self.last_seen = time.time()
        self.missed_frames = 0

    @staticmethod
    def _aspect_ratio(bbox: List[int]) -> float:
        w = max(bbox[2] - bbox[0], 1e-6)
        h = max(bbox[3] - bbox[1], 1e-6)
        return h / w

    def update(self, bbox: List[int], center: Tuple[int, int]):
        self.bbox = bbox
        self.center = center
        self.last_seen = time.time()
        self.missed_frames = 0
        self.history.append((center, self.last_seen))
        cutoff = self.last_seen - self.HISTORY_WINDOW_SECONDS
        while len(self.history) > 1 and self.history[0][1] < cutoff:
            self.history.popleft()

        self.aspect_history.append((self._aspect_ratio(bbox), self.last_seen))
        while len(self.aspect_history) > 1 and self.aspect_history[0][1] < cutoff:
            self.aspect_history.popleft()

    def aspect_ratio_jitter(self, window_seconds: float = 1.0) -> float:
        """
        Max - min bbox aspect ratio (height/width) within the last
        `window_seconds` — a real fluctuation measure (rapid oscillation),
        not just a start-vs-end delta, which could miss a jitter that
        happens to return to its starting value. Used to catch
        grappling/falling motion that doesn't necessarily involve much
        centroid displacement (e.g. two people wrestling in place).
        """
        if len(self.aspect_history) < 2:
            return 0.0
        t_last = self.aspect_history[-1][1]
        cutoff = t_last - window_seconds
        vals = [a for a, t in self.aspect_history if t >= cutoff]
        if len(vals) < 2:
            return 0.0
        return max(vals) - min(vals)

    @property
    def dwell_time(self) -> float:
        """Return total duration (in seconds) this person has been tracked."""
        return self.last_seen - self.first_seen

    @property
    def speed(self) -> float:
        """
        Instantaneous movement speed (pixels/sec), measured over the last
        ~1s of motion rather than the whole track lifetime — otherwise a
        person who walked then stood still still reads as "fast" for
        minutes, and someone who wandered back to their start point reads
        as motionless even if they never stopped moving.
        """
        if len(self.history) < 2:
            return 0.0
        (_, t_last) = self.history[-1]
        window_start = self.history[0]
        for pt, t in self.history:
            if t_last - t <= self.SPEED_WINDOW_SECONDS:
                window_start = (pt, t)
                break
        (c1, t1) = window_start
        (c2, t2) = self.history[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        dist = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
        return dist / dt

    def recent_trail(self, window_seconds: float = TRAIL_WINDOW_SECONDS) -> List[Tuple[int, int]]:
        """
        Centroid points from the last `window_seconds` of this track's
        history — genuinely TIME-windowed (not a fixed sample count), for
        the overlay's fading trajectory tail. Oldest-to-newest order.
        Separate from the full `history` deque (which Fighting/Snatching's
        trailing-WINDOW-count gates in behaviour_engine.py deliberately
        index into by sample count, not time) — this exists so the overlay
        can draw a fixed, meaningful real-time span without changing that
        other consumer's semantics.
        """
        if not self.history:
            return []
        t_last = self.history[-1][1]
        cutoff = t_last - window_seconds
        return [pt for pt, t in self.history if t >= cutoff]

    def recent_trail_timed(self, window_seconds: float = TRAIL_WINDOW_SECONDS) -> List[Tuple[Tuple[int, int], float]]:
        """
        Same time-windowed slice as recent_trail(), but keeping each
        point's real timestamp — needed by consumers that must compute
        actual velocity (distance/dt) between samples, where dt cannot be
        assumed uniform (analysis cadence varies: 6fps live vs 8fps
        uploaded, plus real frame-to-frame jitter). Oldest-to-newest order.
        """
        if not self.history:
            return []
        t_last = self.history[-1][1]
        cutoff = t_last - window_seconds
        return [(pt, t) for pt, t in self.history if t >= cutoff]

    def position_seconds_ago(self, seconds: float) -> Optional[Tuple[int, int]]:
        """
        The centroid position closest to (but not more recent than)
        `seconds` before this track's most recent update — used for the
        overlay's directional arrow anchor ("where were they ~1s ago").
        Genuinely time-based, unlike a fixed sample-index lookback (e.g.
        "5 samples back"), which means a different real duration depending
        on analysis FPS (0.83s at 6fps live vs. 0.625s at 8fps uploaded).
        """
        if not self.history:
            return None
        t_last = self.history[-1][1]
        target = t_last - seconds
        candidate = self.history[0][0]
        for pt, t in self.history:
            if t <= target:
                candidate = pt
            else:
                break
        return candidate

    @property
    def wander_radius(self) -> float:
        """
        Bounding-box diagonal of everywhere this person has stood across
        their retained history. Small radius + long dwell time = genuinely
        stationary (loitering); large radius means they've just been
        walking around the scene the whole time.
        """
        if len(self.history) < 2:
            return 0.0
        xs = [pt[0] for pt, _ in self.history]
        ys = [pt[1] for pt, _ in self.history]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def roaming_radius(self, window_seconds: float) -> float:
        """
        Same bounding-box-diagonal confinement measure as wander_radius,
        but over a genuinely TIME-WINDOWED trailing slice of history
        instead of everything retained. Needed for loitering's pacing
        check: someone who walked 300px to reach a corner and then paced
        in a tight loop there for 2 minutes should have that initial
        approach NOT count against their confinement — wander_radius's
        whole-retained-history diagonal would incorrectly include that
        approach for as long as it stays within HISTORY_WINDOW_SECONDS;
        this only looks at the trailing window_seconds, so confinement is
        judged on recent behavior only.
        """
        if not self.history:
            return 0.0
        t_last = self.history[-1][1]
        cutoff = t_last - window_seconds
        recent = [pt for pt, t in self.history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        # np.ptp (peak-to-peak = max-min) over a vectorized array slice,
        # per this project's explicit request, in place of the previous
        # max()/min() over Python list comprehensions. Measured note: on
        # real profiling of this exact call site (see the RTX 5070/
        # Loitering+Overcrowding latency work), this was never actually
        # the bottleneck — typical windows hold well under 100 points, and
        # numpy's per-call array-construction overhead is comparable to
        # or larger than Python's built-in max()/min() at that size. Kept
        # here because it's correctness-preserving and explicitly
        # requested, not because it measurably changed this function's
        # cost.
        pts = np.asarray(recent, dtype=np.float64)
        dx, dy = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        return float(math.hypot(dx, dy))


class PersonTracker:
    """
    Centroid-distance + IoU Multi-Person Object Tracker.
    """

    @staticmethod
    def _iou(bbox_a: List[int], bbox_b: List[int]) -> float:
        """Intersection-over-union of two [x1,y1,x2,y2] boxes, 0 if they
        don't overlap at all."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def __init__(self, max_distance: float = 80.0, max_missed_seconds: float = 2.75):
        self.max_distance = max_distance  # Maximum pixel distance to match a track
        # TIME-based occlusion/loss tolerance (NOT a frame count) — a track
        # survives up to this many seconds of not being matched before it's
        # dropped, regardless of how often update() is actually called.
        # Replaces the old raw missed-FRAME counter (max_missed=15), which
        # meant a wildly different real-world duration depending on the
        # caller's analysis cadence (2.5s at AI_ANALYSIS_FPS=6, but only
        # ~0.5s at a slower/degraded rate, or way more at a faster one) —
        # same "express temporal tolerances in seconds, not frames" fix
        # already applied elsewhere in this project (fire/smoke grace
        # periods, RUNNING_MIN_SECONDS, etc).
        self.max_missed_seconds = max_missed_seconds
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        # track_ids evicted by the MOST RECENT update() call (time-based
        # eviction, not this call's detections) — a side-channel output
        # rather than a return-signature change, so existing callers of
        # update() are unaffected. Consumers that need to react to a
        # person truly leaving frame (e.g. Loitering's Re-ID memory —
        # capture their last-known state before it's gone) read this
        # right after calling update().
        self.last_evicted_track_ids: List[int] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new frame detections.

        Args:
            detections: List of detection dicts with "bbox" and "center" keys.

        Returns:
            Updated detections list with added "track_id", "dwell_time", and "speed" fields.
        """
        now = time.time()
        updated_detections = []
        detection_matched = [False] * len(detections)
        track_matched: Dict[int, bool] = {tid: False for tid in self.tracks}

        # GLOBAL greedy matching (best pairs across ALL tracks/detections
        # first), not per-track greedy — the previous version let whichever
        # track happened to be iterated first "steal" the nearest detection
        # even when it more rightfully belonged to a different track, a
        # real source of ID swaps whenever two people were close together.
        # Cost combines centroid distance (works even when boxes don't
        # overlap, e.g. fast motion) with IoU (a much stronger identity
        # signal than raw distance specifically when two people ARE close
        # together/overlapping — reduces ID flipping in exactly that case).
        # High IoU discounts the effective distance cost; zero IoU falls
        # back to pure distance, so fast-moving non-overlapping tracks still
        # match correctly.
        candidates: List[Tuple[float, int, int]] = []  # (cost, track_id, det_idx)
        for track_id, track in self.tracks.items():
            for i, det in enumerate(detections):
                dist = math.sqrt(
                    (det["center"][0] - track.center[0]) ** 2 + (det["center"][1] - track.center[1]) ** 2
                )
                if dist > self.max_distance:
                    continue
                iou = self._iou(track.bbox, det["bbox"])
                cost = dist * (1.0 - 0.5 * iou)
                candidates.append((cost, track_id, i))
        candidates.sort(key=lambda c: c[0])

        for cost, track_id, i in candidates:
            if track_matched.get(track_id) or detection_matched[i]:
                continue
            track = self.tracks[track_id]
            det = detections[i]
            track.update(det["bbox"], det["center"])
            detection_matched[i] = True
            track_matched[track_id] = True

            det_copy = dict(det)
            det_copy["track_id"] = track.track_id
            det_copy["dwell_time"] = round(track.dwell_time, 1)
            det_copy["speed"] = round(track.speed, 1)
            det_copy["wander_radius"] = round(track.wander_radius, 1)
            det_copy["history"] = [pt[0] for pt in list(track.history)]
            # Time-windowed (not sample-count) fields specifically for the
            # overlay's trajectory tail/velocity arrow — see recent_trail()/
            # position_seconds_ago()'s docstrings.
            det_copy["trail"] = track.recent_trail()
            det_copy["position_1s_ago"] = track.position_seconds_ago(1.0)
            det_copy["aspect_jitter_1s"] = round(track.aspect_ratio_jitter(1.0), 3)
            # Timestamped trail (point, timestamp) pairs, needed by
            # Fighting's Gate 2 constant-velocity check to compute real
            # per-segment velocity (distance/dt) rather than assuming
            # uniform sampling — see recent_trail_timed()'s docstring.
            det_copy["trail_timed_1s"] = track.recent_trail_timed(1.0)
            # Loitering's demo (10s), dwell (45s), and pacing (120s) rules
            # each need a genuinely time-windowed confinement radius (see
            # roaming_radius()'s docstring) — window lengths hardcoded here
            # to match config.py's LOITERING_DWELL_TIME_SECONDS/
            # LOITERING_DWELL_SECONDS/LOITERING_PACING_WINDOW_SECONDS
            # defaults, same "kept in sync by hand" tradeoff as
            # position_1s_ago/aspect_jitter_1s above (this module
            # deliberately has no AIConfig dependency).
            det_copy["roaming_radius_10s"] = round(track.roaming_radius(10.0), 1)
            det_copy["roaming_radius_45s"] = round(track.roaming_radius(45.0), 1)
            det_copy["roaming_radius_120s"] = round(track.roaming_radius(120.0), 1)
            det_copy["is_new_track"] = False
            updated_detections.append(det_copy)

        # Unmatched tracks: evict only after max_missed_seconds of real time
        # unmatched, not a frame count — see max_missed_seconds's docstring.
        self.last_evicted_track_ids = []
        for track_id, track in list(self.tracks.items()):
            if track_matched.get(track_id):
                continue
            track.missed_frames += 1
            if (now - track.last_seen) > self.max_missed_seconds:
                del self.tracks[track_id]
                self.last_evicted_track_ids.append(track_id)

        # Create new tracks for unmatched detections
        for i, matched in enumerate(detection_matched):
            if not matched:
                det = detections[i]
                new_track = Track(self.next_id, det["bbox"], det["center"])
                self.tracks[self.next_id] = new_track

                det_copy = dict(det)
                det_copy["track_id"] = self.next_id
                det_copy["dwell_time"] = 0.0
                det_copy["speed"] = 0.0
                det_copy["wander_radius"] = 0.0
                det_copy["history"] = [det["center"]]
                det_copy["trail"] = [det["center"]]
                det_copy["position_1s_ago"] = det["center"]
                det_copy["aspect_jitter_1s"] = 0.0
                det_copy["trail_timed_1s"] = [(det["center"], now)]
                det_copy["roaming_radius_10s"] = 0.0
                det_copy["roaming_radius_45s"] = 0.0
                det_copy["roaming_radius_120s"] = 0.0
                det_copy["is_new_track"] = True
                updated_detections.append(det_copy)

                self.next_id += 1

        return updated_detections

    def get_track_count(self) -> int:
        """Return number of currently active tracks."""
        return len(self.tracks)

    def reset(self):
        """Reset all tracking state."""
        self.tracks.clear()
        self.next_id = 1
