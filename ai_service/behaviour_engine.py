"""
Behaviour Engine — Rule-based behaviour analysis.

There is currently no trained weights checkpoint for the neural activity
classifier (see ai_service/activity_model.py) that has been validated on
real held-out data — the fine-tuning pipeline (ai_service/training/fine_tune.py)
has fabricated-metrics fallback paths that produce plausible-looking but
made-up accuracy numbers whenever training data is sparse, with nothing
distinguishing a real run from a fake one. So this engine intentionally
only reports activities it can verify deterministically from tracked
positions:
1. Crowd formation (person count threshold, sustained continuously)
2. Restricted-zone entry (point-in-polygon)
3. Loitering (a specific tracked person dwelling in a small area for a
   sustained time — not just "someone has been on screen a while")
4. Running (a specific tracked person's instantaneous speed, from
   tracker.py's Track.speed, sustained above a threshold)
5. Vandalism: a HYBRID gate, not dwell-time alone (dwell-time-only was
   found to false-positive on ordinary asset-zone use, e.g. someone
   searching for their keys near a shelf) — requires a tracked person's
   centroid inside an "asset"-type zone AND the CNN's own Kicking or
   Fighting probability (an aggressive-motion signal; there is no trained
   "Vandalism" class — see ai_service/config.py's DEFAULT_ACTIVITY_CLASSES
   docstring) sustained together for a minimum duration.
6. Falling/Snatching CNN-gating helpers: geometric/kinematic pre-conditions
   (aspect ratio, motionlessness, contact + departure velocity) that must
   hold before the CNN's Falling/Snatching probability is allowed through
   to the activity state machine at all — see check_falling_candidate() /
   check_snatch_divergence() and their call sites in inference_engine.py.

Do NOT reintroduce single-frame heuristics (e.g. "two people close together
+ fast movement = Fighting", "wide bounding box = Falling") with hardcoded
confidence values — these have no evidentiary basis and produce constant
false positives on completely normal activity (people talking, bending
over, jogging). If real Fighting/Falling/Snatching detection is wanted,
it needs an actually-validated trained model gated behind this project's
human-in-the-loop review threshold (see AIConfig.CONFIDENCE_THRESHOLD),
never a direct live alert from a low-confidence guess.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime
import time
import math
import numpy as np
from ai_service import reid


def _bbox_proximate(b1: List[float], b2: List[float], tolerance_px: float) -> bool:
    """True if two [x1,y1,x2,y2] boxes intersect, or come within
    `tolerance_px` of each other — an expanded-rect test, not a strict
    intersection, since 2D camera angles can render two people who are
    genuinely in contact as slightly apart in the frame. Shared by the
    Fighting proximity/motion gate and the Snatching contact gate below
    (both live in inference_engine.py / this module respectively)."""
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    return not (
        x2a + tolerance_px < x1b
        or x1a - tolerance_px > x2b
        or y2a + tolerance_px < y1b
        or y1a - tolerance_px > y2b
    )


def _persons_in_contact(tracked_persons: List[Dict[str, Any]], tolerance_px: float) -> bool:
    """True if any two tracked person boxes are within `tolerance_px` of
    each other. Used to spatially gate the neural Fighting class — see
    call site in inference_engine.py's analyze()."""
    boxes = [p["bbox"] for p in tracked_persons if p.get("bbox")]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _bbox_proximate(boxes[i], boxes[j], tolerance_px):
                return True
    return False


def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes, 0 if they don't
    overlap at all. Standalone here (not imported from tracker.py) to keep
    the two modules decoupled — same small-duplication tradeoff already
    made for _bbox_proximate above."""
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


def _fighting_gate1_proximity_ok(bbox_a: List[float], bbox_b: List[float], proximity_multiplier: float, iou_min: float) -> bool:
    """GATE 1 — scale-invariant proximity (see ai_service/config.py's
    FIGHTING_PROXIMITY_MULTIPLIER/FIGHTING_IOU_MIN): passes if centroid
    distance <= proximity_multiplier * the pair's average bbox WIDTH, OR
    the boxes overlap beyond a trivial amount (IoU > iou_min). Both scale
    with how large the pair appears in frame — a fixed pixel distance
    means something different up close vs. far from the camera."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    w_avg = ((ax2 - ax1) + (bx2 - bx1)) / 2.0
    ca = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    cb = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    if dist <= proximity_multiplier * w_avg:
        return True
    return _iou(bbox_a, bbox_b) > iou_min


def _velocity_segments(
    trail_timed: List[Tuple[Tuple[float, float], float]],
) -> List[Tuple[float, float, float]]:
    """
    Turns a (point, timestamp) trail into a list of per-segment (speed_px_s,
    heading_deg, dt) samples between consecutive points — the raw material
    for judging whether a person's recent motion has been "steady" (walking
    past) or "erratic" (a real scuffle: stopping, pushing, pulling,
    reversing direction). Skips zero/negative-dt segments (duplicate
    timestamps) rather than dividing by zero.
    """
    segments = []
    for (p1, t1), (p2, t2) in zip(trail_timed, trail_timed[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        speed = dist / dt
        heading = math.degrees(math.atan2(dy, dx)) if dist > 0 else None
        segments.append((speed, heading, dt))
    return segments


def _velocity_is_steady(
    trail_timed: List[Tuple[Tuple[float, float], float]],
    angle_change_threshold_deg: float,
    speed_change_ratio_threshold: float,
) -> bool:
    """
    (Superseded — see _fighting_gate2_motion_ok's docstring. Kept as a
    documented, callable building block rather than deleted outright,
    since "constant-velocity rejection" is a real, reusable technique that
    was deliberately tried and walked back for THIS gate specifically, not
    proven wrong in general.)

    True if this person's recent velocity (both speed and heading) has
    stayed roughly constant across the trailing window — the signature of
    someone walking in a straight line at a steady pace, as opposed to a
    real altercation's stopping/pushing/pulling/reversing. Requires at
    least 2 velocity segments (3 trail points) to compare; with fewer,
    there isn't enough evidence to PROVE steadiness, so this returns False
    (erratic-or-unknown), never True — an unproven claim of "steady" must
    not be able to suppress a fast-developing real fight just because the
    track is brand new.
    """
    segments = _velocity_segments(trail_timed)
    if len(segments) < 2:
        return False

    speeds = [s for s, _, _ in segments]
    headings = [h for _, h, _ in segments if h is not None]

    max_speed = max(speeds) if speeds else 0.0
    if max_speed > 0:
        speed_change_ratio = (max(speeds) - min(speeds)) / max_speed
        if speed_change_ratio > speed_change_ratio_threshold:
            return False

    for h1, h2 in zip(headings, headings[1:]):
        delta = abs(h1 - h2) % 360.0
        if delta > 180.0:
            delta = 360.0 - delta
        if delta > angle_change_threshold_deg:
            return False

    return True


def _fighting_gate2_motion_ok(
    person_a: Dict[str, Any], person_b: Dict[str, Any],
    motion_min_px: float,
) -> bool:
    """
    GATE 2 — kinetic motion (see config.py's FIGHTING_MOTION_MIN_PX).
    Ultra-permissive by design: passes whenever the pair's combined
    1-second centroid displacement (using tracker.py's Track.
    position_seconds_ago(1.0), exposed as "position_1s_ago" — the same
    signal a bounding box's own movement shows up in, since the box
    tracks the person) is above a near-zero absolute pixel floor —
    NOT a large or scale-relative threshold. This deliberately does not
    require large displacement, fast speed, or a specific kind of motion:
    slow wrestling, close-quarters grappling, and a shove where physical
    displacement is naturally small must all still reach Gate 3. Rules out
    only two people who are GENUINELY motionless close together (a
    handshake/conversation, or float/pixel-rounding noise on a static
    pair), which Gate 1's proximity check can't distinguish on its own;
    everything else close-quarters is left entirely to Gate 3's neural
    re-inference on the cropped region — P(Fighting)>=0.70 is where the
    real filtering now happens, not here.

    History of this gate's exact rule, kept here because it's directly
    relevant to why it looks this simple:
    1. Originally: combined displacement >= a FRACTION of average bbox
       height, OR bbox aspect-ratio jitter (jitter meant to catch
       grappling/falling that deforms the box without moving the centroid
       much). Jitter removed: real chaotic multi-person footage showed
       YOLO box-merging during close proximity creates fake aspect-ratio
       jitter on its own, making that signal trigger on ordinary
       passing/crossing, not just real grappling.
    2. Tried: replacing displacement with a "constant-velocity rejection"
       — REJECT the pair if both individuals' recent velocity (speed +
       heading) was steady (walking past/beside each other), only PASS on
       erratic acceleration/deceleration/direction change (see
       _velocity_is_steady(), kept but unused). Walked back: measurably
       suppressed some real fights too, since close-quarters/slow-
       developing scuffles can look "steady" to a per-segment velocity
       check for the first second or more.
    3. Tried: combined displacement >= 0.5 * average bbox HEIGHT
       (scale-invariant, like Gate 1). Walked back after real-video
       testing showed this height-relative fraction was ALSO too strict
       for slow, close-quarters struggles — a real fight with naturally
       small physical displacement (grappling, a shove, wrestling at
       close range) could fail to reach even half a body-height of
       combined movement in a second, silently blocking Gate 3 from ever
       getting to judge the actual footage. Replaced with the current
       near-zero absolute floor (FIGHTING_MOTION_MIN_PX) specifically so
       the mechanical gate stops being the bottleneck, and P(Fighting)
       >=0.70 plus Gate 4's hysteresis carry the real detection burden —
       see config.py's FIGHTING_* block for the current rationale.
    """
    def _displacement_1s(p: Dict[str, Any]) -> float:
        pos_1s_ago = p.get("position_1s_ago")
        center = p.get("center")
        if pos_1s_ago is None or center is None:
            return 0.0
        return ((center[0] - pos_1s_ago[0]) ** 2 + (center[1] - pos_1s_ago[1]) ** 2) ** 0.5

    combined_displacement = _displacement_1s(person_a) + _displacement_1s(person_b)
    return combined_displacement > motion_min_px


def find_fighting_pair(
    tracked_persons: List[Dict[str, Any]],
    proximity_multiplier: float,
    iou_min: float,
    motion_min_px: float,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Fighting's Gates 1+2 combined (see ai_service/config.py's FIGHTING_*
    constants for the full 4-gate design) — returns the first pair of
    tracked persons that clears BOTH the scale-invariant proximity gate
    and the kinetic-motion gate, or None if no such pair exists. Requires
    proximity as a precondition: a pair that isn't close at all is never
    a candidate regardless of how much either one is moving elsewhere in
    the frame.

    Returns the actual (person_a, person_b) dicts (not just a bool) so a
    caller can draw the specific evidence this gate found (Gate 3 also
    needs the exact boxes, to crop the pair's combined region) — a guard
    shown a "FIGHTING" alert with no indication of who or why is exactly
    the "looks like a false alarm" failure mode this is meant to prevent.
    """
    boxes = [p for p in tracked_persons if p.get("bbox")]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not _fighting_gate1_proximity_ok(a["bbox"], b["bbox"], proximity_multiplier, iou_min):
                continue
            if _fighting_gate2_motion_ok(a, b, motion_min_px):
                return a, b
    return None


def fighting_pair_still_qualifies(
    person_a: Dict[str, Any],
    person_b: Dict[str, Any],
    proximity_multiplier: float,
    iou_min: float,
    motion_min_px: float,
) -> bool:
    """Gates 1+2 for one SPECIFIC, already-known pair (as opposed to
    find_fighting_pair(), which searches for the first qualifying pair
    among everyone in frame). Needed so a caller can keep tracking the
    SAME pair across frames ("sticky" selection) instead of re-picking
    whichever pair happens to be first in tracked_persons each window —
    in a crowded scene with several people close together, the
    nested-loop "first match" can flip between several different
    genuinely-qualifying candidate pairs from one window to the next
    (real, observed on chaotic multi-person fight footage), which meant
    no single pair_key ever accumulated Gate 4's required continuous
    confirm_seconds even though a real fight was continuously present
    the whole time. Preferring "is my current pair still valid?" over
    "what's the first valid pair this window?" fixes that without
    weakening any of the four gates themselves."""
    if not person_a.get("bbox") or not person_b.get("bbox"):
        return False
    if not _fighting_gate1_proximity_ok(person_a["bbox"], person_b["bbox"], proximity_multiplier, iou_min):
        return False
    return _fighting_gate2_motion_ok(person_a, person_b, motion_min_px)


class FightingPairState:
    """
    GATE 4 — temporal persistence & hysteresis for ONE tracked pair (see
    ai_service/config.py's FIGHTING_CONFIRM_SECONDS/FIGHTING_RECOVERY_*/
    FIGHTING_PROXIMITY_FAIL_SECONDS for the full rationale). Deliberately
    self-contained and separate from the shared ActivityStateMachine —
    same "own dedicated subsystem" pattern as Fire/Smoke's
    FireSmokeStateMachine.

    Only two real states: NORMAL and CONFIRMED. The "a brief dip doesn't
    reset the confirmation" requirement is achieved structurally, not with
    a separate grace-period state: the recovery probability floor
    (FIGHTING_RECOVERY_PROB_THRESHOLD, 0.40) is deliberately LOWER than
    the confirm/Gate-3 floor (FIGHTING_NEURAL_CONFIRM_THRESHOLD, 0.70), so
    a dip anywhere between the two — or even briefly below 0.40, as long
    as it's under FIGHTING_RECOVERY_PROB_SECONDS — never on its own resets
    anything, because nothing is watching that range once CONFIRMED.
    """

    def __init__(self):
        self.state = "NORMAL"
        self._gates_pass_since: Optional[float] = None
        self._pretrigger_gap_since: Optional[float] = None
        self._confirmed_since: Optional[float] = None
        self._proximity_fail_since: Optional[float] = None
        self._low_prob_since: Optional[float] = None
        self._peak = 0.0
        self._sum = 0.0
        self._count = 0

    def _reset(self) -> None:
        self.state = "NORMAL"
        self._gates_pass_since = None
        self._pretrigger_gap_since = None
        self._confirmed_since = None
        self._proximity_fail_since = None
        self._low_prob_since = None
        self._peak = 0.0
        self._sum = 0.0
        self._count = 0

    def update(
        self,
        now: float,
        gates_1_2_pass: bool,
        gate3_prob: float,
        confirm_seconds: float,
        neural_confirm_threshold: float,
        recovery_prob_threshold: float,
        recovery_prob_seconds: float,
        proximity_fail_seconds: float,
        pretrigger_gap_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        gate3_pass = gate3_prob >= neural_confirm_threshold

        if self.state == "NORMAL":
            # A gap that has already lasted >=pretrigger_gap_seconds
            # invalidates the accumulation regardless of which branch
            # this specific call falls into below — checked unconditionally
            # up front so a fail call followed much later by a pass call
            # (skipping any intermediate fail calls that would otherwise
            # have caught the staleness) can't silently skip the check.
            if (
                self._pretrigger_gap_since is not None
                and now - self._pretrigger_gap_since >= pretrigger_gap_seconds
            ):
                self._gates_pass_since = None
                self._pretrigger_gap_since = None

            if gates_1_2_pass and gate3_pass:
                if self._gates_pass_since is None:
                    self._gates_pass_since = now
                self._pretrigger_gap_since = None
                if now - self._gates_pass_since >= confirm_seconds:
                    self.state = "CONFIRMED"
                    self._confirmed_since = now
                    self._gates_pass_since = None
                    self._pretrigger_gap_since = None
                    self._proximity_fail_since = None
                    self._low_prob_since = None
                    self._peak = gate3_prob
                    self._sum = gate3_prob
                    self._count = 1
            elif self._gates_pass_since is not None:
                # Already accumulating toward confirmation — tolerate a
                # brief (<pretrigger_gap_seconds) gap without restarting
                # the timer, mirroring the same "a brief interruption
                # doesn't undo what's already been observed" principle
                # already used below for the post-confirm recovery phase.
                # Real per-frame tracker jitter (a single missed sample
                # from occlusion mid-scuffle) was observed, on real
                # chaotic footage, to reset this timer to zero on almost
                # every single window, which meant a real, continuous
                # fight with a sustained high Gate-3 probability never
                # actually satisfied a STRICTLY unbroken 2.0s. The
                # elapsed time during the gap still counts; only a gap
                # that itself lasts >=pretrigger_gap_seconds actually
                # restarts the trigger timer from scratch.
                if self._pretrigger_gap_since is None:
                    self._pretrigger_gap_since = now
            # else: never started accumulating — nothing to reset.

        elif self.state == "CONFIRMED":
            self._peak = max(self._peak, gate3_prob)
            self._sum += gate3_prob
            self._count += 1

            # find_fighting_pair() (Gates 1+2) returns one combined
            # pass/fail — a pair failing EITHER sub-gate is treated as "no
            # longer in qualifying contact" for this recovery timer.
            if not gates_1_2_pass:
                if self._proximity_fail_since is None:
                    self._proximity_fail_since = now
            else:
                self._proximity_fail_since = None

            if gate3_prob < recovery_prob_threshold:
                if self._low_prob_since is None:
                    self._low_prob_since = now
            else:
                self._low_prob_since = None

            proximity_failed_long_enough = (
                self._proximity_fail_since is not None
                and (now - self._proximity_fail_since) >= proximity_fail_seconds
            )
            prob_low_long_enough = (
                self._low_prob_since is not None
                and (now - self._low_prob_since) >= recovery_prob_seconds
            )
            if proximity_failed_long_enough or prob_low_long_enough:
                self._reset()

        duration = (now - self._confirmed_since) if self._confirmed_since else 0.0
        average = round(self._sum / self._count, 3) if self._count else 0.0
        return {
            "state": self.state,
            "idle": self.state == "NORMAL" and self._gates_pass_since is None,
            "duration_seconds": round(duration, 1),
            "peak_confidence": round(self._peak, 3),
            "average_confidence": average,
        }


def is_running_speed(
    speed_px_s: float,
    bbox: Optional[List[float]],
    body_heights_per_sec: float,
    min_abs_speed_px_s: float,
    fallback_abs_speed_px_s: float,
) -> bool:
    """
    Single source of truth for "is this speed fast enough to count as
    running" — used by BOTH the alert rule (BehaviourEngine._check_rules)
    and the overlay renderer (inference_engine.py), so the two can never
    disagree about which tracks are "running".

    Scale-invariant: expresses the threshold as body-heights-per-second
    (roughly constant across camera distance/zoom for the same real gait)
    rather than a single fixed pixel speed, which means something different
    for a person filling a third of the frame vs. a distant speck. Falls
    back to a bare absolute pixel threshold only if no bbox is available
    (there is nothing to normalize by).
    """
    if bbox is None:
        return speed_px_s >= fallback_abs_speed_px_s

    bbox_height = bbox[3] - bbox[1]
    if bbox_height <= 0:
        return speed_px_s >= fallback_abs_speed_px_s

    if speed_px_s < min_abs_speed_px_s:
        # Floor: guards a tiny/distant/noisy box from trivially satisfying
        # the relative ratio from a few pixels of jitter.
        return False

    return (speed_px_s / bbox_height) >= body_heights_per_sec


@dataclass
class DetectionResult:
    """A single detection result from the AI pipeline."""
    camera_id: int
    activity: str
    confidence: float
    severity: str
    persons_detected: int
    duration_seconds: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.80

    @property
    def needs_review(self) -> bool:
        return 0.50 <= self.confidence < 0.80


def find_overcrowding_clusters(
    tracked_persons: List[Dict[str, Any]], max_radius_px: float,
) -> List[List[Dict[str, Any]]]:
    """
    Groups tracked persons into connected components by centroid
    proximity — two people are directly linked if their centroids are
    within max_radius_px of each other, and a cluster is the transitive
    closure of those links (so five people standing in a rough line, each
    close to their immediate neighbor but not to the far ends of the
    line, still count as ONE group of 5 — matching real crowding topology
    rather than requiring every pair in the group to be mutually close,
    which a real crowd rarely is). Returns every cluster found (including
    singletons), oldest-index-first; callers filter by size themselves.
    """
    people = [p for p in tracked_persons if p.get("center") is not None]
    n = len(people)
    if n == 0:
        return []

    # Vectorized pairwise centroid distances (numpy broadcasting) instead
    # of a Python-level nested loop — real profiling of this exact call
    # site (see the RTX 5070/Loitering+Overcrowding latency work) found
    # the ORIGINAL pure-Python version already cost under 0.15ms even on
    # real footage with a genuine 5-14 person crowd (N is small; O(N^2)
    # at N<=~20 is a few hundred operations either way), so this wasn't
    # fixing a measured bottleneck — kept because it's correctness-
    # preserving, explicitly requested, and scales better for a much
    # larger N than this deployment currently sees.
    centers = np.array([p["center"] for p in people], dtype=np.float64)
    diffs = centers[:, None, :] - centers[None, :, :]
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=-1))
    adjacency = dist_matrix <= max_radius_px
    np.fill_diagonal(adjacency, False)

    visited = np.zeros(n, dtype=bool)
    clusters: List[List[Dict[str, Any]]] = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: List[Dict[str, Any]] = []
        while stack:
            idx = stack.pop()
            component.append(people[idx])
            for neighbor in np.nonzero(adjacency[idx])[0]:
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        clusters.append(component)
    return clusters


def find_largest_overcrowding_cluster(
    tracked_persons: List[Dict[str, Any]], min_count: int, max_radius_px: float,
) -> Optional[List[Dict[str, Any]]]:
    """The largest cluster (by member count) that meets min_count, or None
    if no cluster is large enough this frame. Only the largest is
    reported — matches OvercrowdingState's one-incident-per-camera design
    (see its docstring), so a scene with two separate qualifying clusters
    still raises exactly one alert per window, not one per cluster."""
    clusters = find_overcrowding_clusters(tracked_persons, max_radius_px)
    qualifying = [c for c in clusters if len(c) >= min_count]
    if not qualifying:
        return None
    return max(qualifying, key=len)


class OvercrowdingState:
    """
    Temporal accumulation & hysteresis for Overcrowding — same "gap
    tolerance doesn't reset a real state" state-machine shape as Fighting's
    FightingPairState/update_fighting_incident_state (see that class's
    docstring for the original rationale), adapted for a single boolean
    condition (cluster_present) instead of a probability threshold.

    Deliberately ONE state per camera, not per specific set of track_ids:
    cluster MEMBERSHIP is not a stable identity across frames (people join
    and leave the edges of a crowd constantly; the group is still "the
    same crowd" even as its exact roster shifts slightly window to
    window), so tracking "is there currently a qualifying crowd somewhere
    in this camera" is the meaningful state, not "is this exact set of
    track_ids still clustered" — the same lesson already learned and
    applied for Fighting's per-camera (not per-pair) incident tracking.

    Only two real states: NORMAL and CONFIRMED. The pre-confirm
    accumulation timer tolerates a gap shorter than pretrigger_gap_seconds
    without resetting (elapsed time during the gap still counts); a gap
    of pretrigger_gap_seconds or longer resets it. The SAME gap tolerance
    is reused symmetrically for recovery once CONFIRMED (the task's own
    spec describes it generically as "hysteresis gap tolerance for minor
    tracker dropouts", not confirm-phase-only) — a brief dropout below
    OVERCROWDING_MIN_COUNT doesn't instantly end an already-confirmed
    overcrowding event, but a sustained one does.
    """

    def __init__(self):
        self.state = "NORMAL"
        self._accum_since: Optional[float] = None
        self._pretrigger_gap_since: Optional[float] = None
        self._confirmed_since: Optional[float] = None
        self._peak_count = 0
        self._last_members: List[Dict[str, Any]] = []

    def _reset(self) -> None:
        self.state = "NORMAL"
        self._accum_since = None
        self._pretrigger_gap_since = None
        self._confirmed_since = None
        self._peak_count = 0
        self._last_members = []

    def update(
        self,
        now: float,
        cluster: Optional[List[Dict[str, Any]]],
        dwell_seconds: float,
        pretrigger_gap_seconds: float,
    ) -> Dict[str, Any]:
        cluster_present = cluster is not None
        if cluster_present:
            self._last_members = cluster

        if self.state == "NORMAL":
            # A gap that has already lasted >=pretrigger_gap_seconds
            # invalidates the accumulation regardless of which branch this
            # specific call falls into below — checked unconditionally up
            # front so a dropout followed much later by a re-qualifying
            # cluster (skipping any intermediate "still absent" calls that
            # would otherwise have caught the staleness) can't silently
            # skip the check. Same fix already applied in Fighting's Gate
            # 4 for the identical failure mode.
            if (
                self._pretrigger_gap_since is not None
                and now - self._pretrigger_gap_since >= pretrigger_gap_seconds
            ):
                self._accum_since = None
                self._pretrigger_gap_since = None

            if cluster_present:
                if self._accum_since is None:
                    self._accum_since = now
                self._pretrigger_gap_since = None
                self._peak_count = max(self._peak_count, len(cluster))
                if now - self._accum_since >= dwell_seconds:
                    self.state = "CONFIRMED"
                    self._confirmed_since = now
                    self._pretrigger_gap_since = None
            elif self._accum_since is not None:
                # Already accumulating — tolerate a brief gap without
                # restarting the timer (see class docstring).
                if self._pretrigger_gap_since is None:
                    self._pretrigger_gap_since = now
            # else: never started accumulating — nothing to reset.

        elif self.state == "CONFIRMED":
            if cluster_present:
                self._peak_count = max(self._peak_count, len(cluster))
                self._pretrigger_gap_since = None
            else:
                if self._pretrigger_gap_since is None:
                    self._pretrigger_gap_since = now
                elif now - self._pretrigger_gap_since >= pretrigger_gap_seconds:
                    self._reset()

        return self.snapshot(now)

    def snapshot(self, now: float) -> Dict[str, Any]:
        """Read-only view of the current state — same dict shape as
        update()'s return value, but without advancing anything. Lets a
        caller (the overlay renderer) ask "what's the current Overcrowding
        state for this camera" independently of whichever call actually
        drove update() forward this window (mirrors get_running_track_ids
        /person_is_loitering's "read the already-computed state" pattern
        already used for Running/Loitering)."""
        duration = (now - self._confirmed_since) if self._confirmed_since else 0.0
        return {
            "state": self.state,
            "idle": self.state == "NORMAL" and self._accum_since is None,
            "duration_seconds": round(duration, 1),
            "peak_count": self._peak_count,
            "members": self._last_members,
        }


class LoiteringReIDMemory:
    """
    Cross-session "is this the SAME person coming back" memory for
    Loitering (see check_loitering() and ai_service/reid.py's module
    docstring for the visual-descriptor tradeoffs). Standard tracking
    assigns a brand-new track_id every time a person leaves frame and
    re-enters — this keeps a short-term, per-camera memory of recently-
    departed people's visual descriptor + accumulated dwell time, so a
    newly-appeared track can be linked back to them instead of starting
    its loitering clock from zero.

    Deliberately per-camera (never matches across different cameras) and
    deliberately a FIFO-pruned list rather than a dict keyed by identity —
    there is no stable identity key before a match is found; entries are
    just "recently seen" candidates, most of which will simply age out
    unmatched.
    """

    def __init__(self):
        # Each entry: {"camera_id", "signature", "exit_time", "cumulative_dwell"}
        self._entries: List[Dict[str, Any]] = []

    def register_exit(
        self, camera_id: int, signature: Dict[str, Any], cumulative_dwell: float, now: float,
        time_window_seconds: float,
    ) -> None:
        """Record a departing track's descriptor + accumulated dwell time,
        and prune anything older than time_window_seconds (from ANY
        camera — a uniform housekeeping pass, not gated on this specific
        camera_id, so memory doesn't grow unbounded across a long-running
        multi-camera session)."""
        self._entries.append({
            "camera_id": camera_id,
            "signature": signature,
            "exit_time": now,
            "cumulative_dwell": cumulative_dwell,
        })
        cutoff = now - time_window_seconds
        self._entries = [e for e in self._entries if e["exit_time"] >= cutoff]

    def try_match(
        self, camera_id: int, signature: Dict[str, Any], now: float,
        similarity_threshold: float, time_window_seconds: float,
    ) -> float:
        """
        Looks for a same-camera departed entry whose visual descriptor
        matches `signature` (cosine/correlation similarity >=
        similarity_threshold) within the last time_window_seconds. Returns
        that entry's cumulative_dwell (consuming/removing the entry, so it
        can't be matched again by a second, different new track) or 0.0 if
        no match. Picks the BEST match among candidates, not just the
        first, so a crowded scene with several recently-departed people
        doesn't link to whichever one happened to be inserted first.
        """
        cutoff = now - time_window_seconds
        best_idx: Optional[int] = None
        best_sim = 0.0
        for idx, entry in enumerate(self._entries):
            if entry["camera_id"] != camera_id or entry["exit_time"] < cutoff:
                continue
            sim = reid.descriptor_similarity(signature, entry["signature"])
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx is None:
            return 0.0
        matched = self._entries.pop(best_idx)
        return matched["cumulative_dwell"]


class BehaviourEngine:
    """
    Analyses detected activities and applies rule-based logic.

    Pipeline:
    1. Receive tracked-person positions/speed/dwell-time for the frame
    2. Apply rule-based detections (loitering, zones, crowd)
    3. Route to alert engine or review queue based on confidence
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Loitering — see person_is_loitering() below for the full
        # three-rule design (demo/dwell-in-place/confined-pacing) and
        # LoiteringReIDMemory for the cross-session "same person
        # returning" layer.
        self.loitering_demo_dwell_seconds = self.config.get("loitering_demo_dwell_seconds", 10.0)
        self.loitering_demo_radius_px = self.config.get("loitering_demo_radius_px", 120)
        self.loitering_dwell_seconds = self.config.get("loitering_dwell_seconds", 45)
        self.loitering_pacing_window_seconds = self.config.get("loitering_pacing_window_seconds", 120)
        self.loitering_pacing_radius_px = self.config.get("loitering_pacing_radius_px", 150)
        self.reid_similarity_threshold = self.config.get("reid_similarity_threshold", 0.85)
        self.reid_time_window_seconds = self.config.get("reid_time_window_seconds", 900)  # 15 minutes
        self._reid_memory = LoiteringReIDMemory()
        # (camera_id, track_id) -> dwell seconds inherited from a matched
        # prior identity that left and came back — see reid_try_link().
        self._reid_inherited_dwell: Dict[Tuple[int, int], float] = {}
        self.crowd_threshold = self.config.get("crowd_threshold", 10)  # person count
        # Person count must stay >= crowd_threshold continuously for this
        # long before Crowd Formation fires — see _check_rules.
        self.crowd_persist_seconds = self.config.get("crowd_persist_seconds", 60)
        # Restricted zones: a person must stay inside this long (continuously,
        # same tracked identity) before it becomes an "Unauthorized Entry".
        self.zone_unauthorized_seconds = self.config.get("zone_unauthorized_seconds", 20.0)
        # (camera_id, zone_name, track_id) -> time this person entered the zone
        self._zone_entry: Dict[tuple, float] = {}
        # ...AND their average pairwise centroid distance must stay under
        # this many pixels — a real "crowd" implies proximity, not just a
        # headcount scattered across the whole frame.
        self.crowd_max_avg_distance_px = self.config.get("crowd_max_avg_distance_px", 150)

        # Overcrowding — a SEPARATE, more detailed subsystem from Crowd
        # Formation above (deliberately left untouched): clustering by
        # connected components (find_largest_overcrowding_cluster()) plus
        # dedicated per-camera hysteresis (OvercrowdingState), instead of
        # Crowd Formation's whole-frame average-pairwise-distance check.
        # See config.py's OVERCROWDING_* block for the full rationale.
        self.overcrowding_min_count = self.config.get("overcrowding_min_count", 5)
        self.overcrowding_max_radius_px = self.config.get("overcrowding_max_radius_px", 150)
        self.overcrowding_dwell_seconds = self.config.get("overcrowding_dwell_seconds", 3.0)
        self.overcrowding_pretrigger_gap_seconds = self.config.get("overcrowding_pretrigger_gap_seconds", 1.0)
        # Per-camera Overcrowding hysteresis state (one active incident at
        # a time per camera, not per specific set of track_ids — see
        # OvercrowdingState's docstring).
        self._overcrowding_states: Dict[int, OvercrowdingState] = {}

        # A person is "stationary" if they haven't wandered further than this
        # many pixels from their own footprint over the loitering window.
        self.loiter_radius_px = self.config.get("loiter_radius_px", 150)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.80)
        # Scale-invariant running definition — see is_running_speed() above
        # for the full reasoning. Speed must qualify continuously for
        # running_min_seconds before Running fires — see _check_rules.
        self.running_body_heights_per_sec = self.config.get("running_body_heights_per_sec", 1.5)
        self.running_min_abs_speed_px_s = self.config.get("running_min_abs_speed_px_s", 60)
        self.running_speed_px_s = self.config.get("running_speed_px_s", 220)  # fallback when no bbox available
        self.running_min_seconds = self.config.get("running_min_seconds", 1.0)
        self.running_min_distance_body_heights = self.config.get("running_min_distance_body_heights", 0.5)
        self.running_min_distance_px = self.config.get("running_min_distance_px", 20)
        # Person-in-asset-zone AND CNN Kicking/Fighting must both hold
        # continuously for this long before Vandalism fires — see
        # check_vandalism().
        self.vandalism_min_seconds = self.config.get("vandalism_min_seconds", 3.0)
        self.vandalism_aggression_threshold = self.config.get("vandalism_aggression_threshold", 0.65)

        # Falling: aspect ratio (h/w) must drop below this to count as a
        # "horizontal posture" candidate.
        self.falling_aspect_ratio_max = self.config.get("falling_aspect_ratio_max", 0.8)
        # ...and centroid drift since that drop must stay under this many
        # pixels, continuously, for this many seconds, before the CNN's
        # Falling probability is allowed through.
        self.falling_motionless_px = self.config.get("falling_motionless_px", 5)
        self.falling_motionless_seconds = self.config.get("falling_motionless_seconds", 5.0)

        # Snatching: Person A must first be within this many pixels of
        # Person B (contact), then depart at at least this speed (px/s) —
        # measured, like tracker.py's Track.speed, over a short trailing
        # window, not a single-frame delta — within this many seconds of
        # that contact for the divergence to count.
        self.snatching_contact_px = self.config.get("snatching_contact_px", 15)
        self.snatching_departure_px_s = self.config.get("snatching_departure_px_s", 72)
        self.snatching_departure_window_seconds = self.config.get("snatching_departure_window_seconds", 1.5)

        # Fighting's 4-Gate config (Gates 1+2 geometry + Gate 4 hysteresis —
        # Gate 3's crop/model params live in inference_engine.py, which owns
        # frame/model access). See ai_service/config.py's FIGHTING_* block
        # for the full rationale behind each value.
        self.fighting_proximity_multiplier = self.config.get("fighting_proximity_multiplier", 1.5)
        self.fighting_iou_min = self.config.get("fighting_iou_min", 0.02)
        self.fighting_motion_min_px = self.config.get("fighting_motion_min_px", 2.0)
        self.fighting_confirm_seconds = self.config.get("fighting_confirm_seconds", 2.0)
        self.fighting_neural_confirm_threshold = self.config.get("fighting_neural_confirm_threshold", 0.70)
        self.fighting_recovery_prob_threshold = self.config.get("fighting_recovery_prob_threshold", 0.40)
        self.fighting_recovery_prob_seconds = self.config.get("fighting_recovery_prob_seconds", 3.0)
        self.fighting_proximity_fail_seconds = self.config.get("fighting_proximity_fail_seconds", 2.0)
        self.fighting_pretrigger_gap_seconds = self.config.get("fighting_pretrigger_gap_seconds", 1.0)
        # Per-camera Gate 4 hysteresis state (one active fight incident at
        # a time per camera, not per track_id pair) — see
        # update_fighting_incident_state().
        self._fighting_pair_states: Dict[int, FightingPairState] = {}

        # Per-camera continuous-crowd start time — keyed by camera_id since
        # this engine is a SINGLE shared instance across all cameras (see
        # InferenceEngine.__init__); without keying by camera_id, one
        # camera's crowd forming would leak its start time into another
        # camera's unrelated person count.
        self._crowd_start_time: Dict[int, float] = {}
        # Per-(camera_id, track_id) continuous-high-speed start time — see
        # the Running rule in _check_rules.
        # Per-(camera_id, track_id): {"start_time", "anchor"} — anchor is
        # that track's own position the moment it started qualifying as a
        # running candidate, forming its "running vector" (anchor -> current
        # position) used for the distance-covered check — see _check_rules.
        self._running_start_time: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Last `now` seen per camera — get_running_track_ids() needs this to
        # compute elapsed duration outside of _check_rules's own call.
        self._last_now: Dict[int, float] = {}
        # Per-(camera_id, track_id) most recent center/bbox-height, updated
        # every _check_rules call for every tracked person (not just
        # runners) — get_running_track_ids() needs a track's CURRENT
        # position to check its running distance, since it's called
        # separately from _check_rules and can't rely on that call's local
        # `tracked_persons` list.
        self._last_position: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Per-camera continuous (asset-zone-presence AND CNN-aggression)
        # start time — see check_vandalism().
        self._vandalism_start_time: Dict[int, float] = {}
        # Per-(camera_id, track_id) Falling candidate state: when this
        # track's aspect ratio first dropped below falling_aspect_ratio_max,
        # and its centroid at that moment — see check_falling_candidate().
        self._fall_candidates: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Per-(camera_id, low_track_id, high_track_id) last recorded contact
        # between two tracks — see check_snatch_divergence().
        self._snatch_contacts: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def analyze_activity(
        self,
        persons_detected: int,
        tracked_persons: List[Dict] = None,
        zones: List[Dict] = None,
        camera_id: int = 0,
        now: Optional[float] = None,
    ) -> Optional[DetectionResult]:
        """
        Apply rule-based detections to the current frame's tracked persons.
        Returns a DetectionResult if an event should be generated, or None.
        `now` should be the same clock the caller uses elsewhere (wall-clock
        for live streaming, a synthesized video-relative epoch for offline
        batch processing) — defaults to wall-clock if not supplied.
        """
        return self._check_rules(persons_detected, tracked_persons or [], zones, camera_id, now if now is not None else time.time())

    def get_zone_dwell(self, camera_id: int) -> Dict[int, float]:
        """track_id -> seconds that person has currently spent inside a
        restricted zone (longest across zones). Used by the overlay timer."""
        now = self._last_now.get(camera_id)
        out: Dict[int, float] = {}
        if now is None:
            return out
        for (cid, _zone, tid), entered in self._zone_entry.items():
            if cid == camera_id:
                out[tid] = max(out.get(tid, 0.0), now - entered)
        return out

    def _check_rules(
        self,
        persons_detected: int,
        tracked_persons: List[Dict],
        zones: List[Dict],
        camera_id: int,
        now: float,
    ) -> Optional[DetectionResult]:
        """Apply rule-based detections."""
        self._last_now[camera_id] = now
        current_track_ids = {p["track_id"] for p in tracked_persons if p.get("track_id") is not None}
        for key in [k for k in self._last_position if k[0] == camera_id and k[1] not in current_track_ids]:
            del self._last_position[key]
        for p in tracked_persons:
            track_id = p.get("track_id")
            bbox = p.get("bbox")
            if track_id is None:
                continue
            self._last_position[(camera_id, track_id)] = {
                "center": p.get("center", (0, 0)),
                "bbox_height": (bbox[3] - bbox[1]) if bbox else 0.0,
            }

        # Overcrowding's hysteresis timer must advance every single window
        # regardless of what (if anything) this method ends up returning
        # below — held here and only turned into this window's actual
        # return value at the very end (lowest priority among all the
        # rules in this method), so an unrelated higher-priority rule
        # firing first this window never causes Overcrowding's dwell
        # accumulation to silently stall. Does not touch persons_detected/
        # tracked_persons or any state Crowd Formation below reads/writes.
        overcrowding_state = self.update_overcrowding_state(camera_id, now, tracked_persons)
        overcrowding_result: Optional[DetectionResult] = None
        if overcrowding_state["state"] == "CONFIRMED":
            overcrowding_result = DetectionResult(
                camera_id=camera_id,
                activity="Overcrowding",
                confidence=0.92,
                severity="warning",
                persons_detected=len(overcrowding_state["members"]),
                duration_seconds=overcrowding_state["duration_seconds"],
                timestamp=datetime.utcnow(),
                metadata={
                    "rule": "overcrowding",
                    "peak_count": overcrowding_state["peak_count"],
                    "track_ids": [m.get("track_id") for m in overcrowding_state["members"]],
                },
            )

        # Crowd formation — requires BOTH a person-count threshold AND those
        # people actually being close together (average pairwise centroid
        # distance < crowd_max_avg_distance_px), sustained continuously for
        # crowd_persist_seconds. The count-only version used to also fire
        # for 5 people scattered across a wide frame who have nothing to do
        # with each other — a real "crowd" implies proximity, not just a
        # headcount.
        avg_distance = self._average_pairwise_distance(tracked_persons)
        crowd_condition = (
            persons_detected >= self.crowd_threshold
            and avg_distance is not None
            and avg_distance < self.crowd_max_avg_distance_px
        )
        if crowd_condition:
            start = self._crowd_start_time.get(camera_id)
            if start is None:
                self._crowd_start_time[camera_id] = now
            else:
                elapsed = now - start
                if elapsed >= self.crowd_persist_seconds:
                    return DetectionResult(
                        camera_id=camera_id,
                        activity="Crowd Formation",
                        confidence=0.95,
                        severity="warning",
                        persons_detected=persons_detected,
                        duration_seconds=elapsed,
                        timestamp=datetime.utcnow(),
                        metadata={
                            "rule": "crowd_threshold",
                            "count": persons_detected,
                            "avg_distance_px": round(avg_distance, 1),
                        },
                    )
        else:
            self._crowd_start_time.pop(camera_id, None)

        # Restricted zones — a timer starts when a tracked person enters
        # (body centre OR feet inside the polygon); only after they have
        # stayed continuously for zone_unauthorized_seconds (default 20s) are
        # they flagged as an Unauthorized Entry. Leaving (or losing the track)
        # resets that person's timer.
        inside_now = set()
        if zones and tracked_persons:
            for zone in zones:
                if zone.get("zone_type") != "restricted":
                    continue
                zone_pts = zone.get("polygon_points", [])
                for p in tracked_persons:
                    tid = p.get("track_id")
                    if tid is None:
                        continue
                    cx, cy = p.get("center", (0, 0))
                    bbox = p.get("bbox")
                    foot_inside = False
                    if bbox is not None and len(bbox) >= 4:
                        foot_inside = self._point_in_polygon(
                            {"x": (bbox[0] + bbox[2]) / 2.0, "y": bbox[3]}, zone_pts
                        )
                    if foot_inside or self._point_in_polygon({"x": cx, "y": cy}, zone_pts):
                        inside_now.add((camera_id, zone.get("name"), tid))
        for key in [k for k in self._zone_entry if k[0] == camera_id and k not in inside_now]:
            del self._zone_entry[key]
        violator = None
        for key in inside_now:
            entered = self._zone_entry.setdefault(key, now)
            dwell = now - entered
            if dwell >= self.zone_unauthorized_seconds and (violator is None or dwell > violator[1]):
                violator = (key, dwell)
        if violator is not None:
            (_, zone_name, tid), dwell = violator
            return DetectionResult(
                camera_id=camera_id,
                activity="Unauthorized Entry",
                confidence=0.98,
                severity="critical",
                persons_detected=1,
                duration_seconds=dwell,
                timestamp=datetime.utcnow(),
                metadata={"rule": "restricted_zone", "zone": zone_name, "track_id": tid,
                          "dwell_seconds": round(dwell, 1)},
            )

        # Vandalism is now checked separately via check_vandalism() — it
        # needs the CNN's Kicking/Fighting probability, which isn't
        # available yet at this point in the pipeline (the neural
        # classifier runs AFTER this rule-based pass — see
        # inference_engine.py's analyze()).

        # Loitering: dwelling AND staying within a small area — a person who
        # has simply been walking around the whole time does not count.
        # person_is_loitering() applies three OR'd sub-rules (see its
        # docstring / config.py's LOITERING_* block for the full
        # rationale): a fast DEMO rule (10s/120px, for live demos), the
        # production DWELL rule (45s/150px), and a longer-window PACING
        # rule (120s/150px) that catches confined pacing a shorter window
        # might miss. "Effective dwell time" = this track's own fresh
        # dwell_time PLUS any inherited dwell from a matched prior
        # identity that left and came back (see reid_try_link()/
        # LoiteringReIDMemory) — this is what makes a returning person's
        # loitering clock resume instead of restarting from zero, and is
        # also what can make a REPEAT LOITERING alert fire the instant
        # they're re-identified, rather than needing another full dwell
        # window of fresh time.
        loiterers = [p for p in tracked_persons if self.person_is_loitering(camera_id, p)]
        if loiterers:
            longest = max(loiterers, key=lambda p: self.get_effective_dwell(camera_id, p))
            is_repeat = any(
                self._reid_inherited_dwell.get((camera_id, p.get("track_id")), 0.0) > 0.0
                for p in loiterers
            )
            return DetectionResult(
                camera_id=camera_id,
                activity="Repeat Loitering" if is_repeat else "Loitering",
                confidence=0.90,
                severity="warning",
                persons_detected=len(loiterers),
                duration_seconds=self.get_effective_dwell(camera_id, longest),
                timestamp=datetime.utcnow(),
                metadata={"rule": "loitering", "track_ids": [p.get("track_id") for p in loiterers]},
            )

        # Running: a tracked person's instantaneous speed sustained
        # CONTINUOUSLY above the threshold for >= running_min_seconds, AND
        # covering real ground while doing so (>= running_min_distance from
        # that SAME track's own start-of-run anchor — its "running vector":
        # one continuous track's start->current displacement, not a generic
        # distance check). The docstring always claimed "sustained", but the
        # actual check here used to fire on a single qualifying frame with
        # no timer at all, and had no distance requirement either — a
        # speed reading alone can't tell a real sustained run from a
        # detection-box jump/teleport that happens to compute a high
        # instantaneous speed without the track actually having gone
        # anywhere. Real impact of the missing timer specifically: brief
        # fast limb/body movement during an actual fight (jostling,
        # swinging, staggering) routinely crosses RUNNING_SPEED_PX_S for one
        # frame, and because this rule-based result took display priority
        # over a properly 8-window-confirmed neural Fighting detection (see
        # inference_engine.py's overlay priority fix), a real fight would
        # render as "RUNNING" instead of "FIGHTING CONFIRMED". Both the
        # persistence timer and the distance/vector anchor are keyed per
        # (camera_id, track_id), same pattern as Falling's motionless timer.
        current_runners: set = set()
        runner_by_key: Dict[Tuple[int, int], Dict] = {}
        for p in tracked_persons:
            track_id = p.get("track_id")
            if track_id is None:
                continue
            if is_running_speed(
                p.get("speed", 0.0), p.get("bbox"),
                self.running_body_heights_per_sec, self.running_min_abs_speed_px_s, self.running_speed_px_s,
            ):
                current_runners.add((camera_id, track_id))
                runner_by_key[(camera_id, track_id)] = p
        for key in [k for k in self._running_start_time if k[0] == camera_id and k not in current_runners]:
            del self._running_start_time[key]
        for key in current_runners:
            if key not in self._running_start_time:
                self._running_start_time[key] = {
                    "start_time": now,
                    "anchor": runner_by_key[key].get("center", (0, 0)),
                }

        sustained_runners = [
            p for p in tracked_persons
            if p.get("track_id") is not None
            and self._running_qualifies(camera_id, p, now)
        ]
        if sustained_runners:
            fastest = max(sustained_runners, key=lambda p: p.get("speed", 0.0))
            key = (camera_id, fastest["track_id"])
            state = self._running_start_time[key]
            elapsed = now - state["start_time"]
            distance, direction = self._running_vector(fastest, state)
            return DetectionResult(
                camera_id=camera_id,
                activity="Running",
                confidence=0.85,
                severity="warning",
                persons_detected=len(sustained_runners),
                duration_seconds=elapsed,
                timestamp=datetime.utcnow(),
                metadata={
                    "rule": "running",
                    "track_ids": [p.get("track_id") for p in sustained_runners],
                    "speed_px_s": fastest.get("speed", 0.0),
                    "distance_px": round(distance, 1),
                    "direction": direction,
                },
            )

        # Overcrowding — lowest priority of all the rules in this method
        # (computed at the top of this method, unconditionally, so its
        # dwell timer never stalls on a window some other rule wins; see
        # the comment there). Only actually returned if nothing above
        # already claimed this window's single DetectionResult slot.
        if overcrowding_result is not None:
            return overcrowding_result

    def _running_vector(self, p: Dict, state: Dict) -> Tuple[float, Tuple[float, float]]:
        """Distance and unit direction from a running track's recorded
        start-of-run anchor to its current position — its actual "running
        vector", not a generic speed number."""
        ax, ay = state["anchor"]
        cx, cy = p.get("center", (0, 0))
        dx, dy = cx - ax, cy - ay
        distance = (dx ** 2 + dy ** 2) ** 0.5
        direction = (dx / distance, dy / distance) if distance > 0 else (0.0, 0.0)
        return distance, direction

    def _running_qualifies(self, camera_id: int, p: Dict, now: float) -> bool:
        """True if this SAME track has been a running candidate for
        >= running_min_seconds AND has covered >= the minimum distance
        along its own running vector since then — both conditions on the
        one continuous track, not independently on different frames."""
        track_id = p.get("track_id")
        if track_id is None:
            return False
        key = (camera_id, track_id)
        state = self._running_start_time.get(key)
        if state is None:
            return False
        if (now - state["start_time"]) < self.running_min_seconds:
            return False

        distance, _ = self._running_vector(p, state)
        bbox = p.get("bbox")
        if bbox:
            bbox_height = bbox[3] - bbox[1]
            if bbox_height > 0:
                return (distance / bbox_height) >= self.running_min_distance_body_heights
        return distance >= self.running_min_distance_px

    def get_running_track_ids(self, camera_id: int) -> Set[int]:
        """
        Track IDs CURRENTLY qualifying as sustained-running (duration AND
        distance, via _running_qualifies) for this camera — used by the
        overlay renderer (inference_engine.py) so the per-person box's
        "running" display and the actual Running alert rule share one
        definition and can never disagree, instead of the renderer running
        its own separate instantaneous, unguarded speed check.

        Needs each track's CURRENT position (not just its start-of-run
        anchor) to check distance, which _running_start_time alone doesn't
        carry — reads it from _last_position, refreshed every _check_rules
        call.
        """
        now = self._last_now.get(camera_id)
        if now is None:
            return set()
        result: Set[int] = set()
        for (cam, track_id), state in self._running_start_time.items():
            if cam != camera_id:
                continue
            if (now - state["start_time"]) < self.running_min_seconds:
                continue
            last_pos = self._last_position.get((cam, track_id))
            if last_pos is None:
                continue
            cx, cy = last_pos["center"]
            ax, ay = state["anchor"]
            distance = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
            bbox_height = last_pos.get("bbox_height", 0.0)
            ok = (
                (distance / bbox_height) >= self.running_min_distance_body_heights
                if bbox_height > 0
                else distance >= self.running_min_distance_px
            )
            if ok:
                result.add(track_id)
        return result

    # ------------------------------------------------------------------
    # Loitering Re-ID — called from inference_engine.py, which owns frame
    # access needed to build visual descriptors (see ai_service/reid.py).
    # This engine only ever sees already-extracted descriptor dicts.
    # ------------------------------------------------------------------

    def person_is_loitering(self, camera_id: int, person: Dict[str, Any]) -> bool:
        """
        The SAME three-rule (demo / dwell-in-place / confined-pacing)
        check the loitering DetectionResult logic applies — exposed
        standalone so the overlay renderer (inference_engine.py) can ask
        "is THIS specific person currently loitering" for a single
        already-tracked person dict without re-deriving its own,
        potentially-diverging copy of the rule (the same "overlay and
        alert rule share one definition" principle already applied to
        Running's get_running_track_ids()). See config.py's LOITERING_*
        block for the full rationale behind each of the three sub-rules.
        """
        effective_dwell = self.get_effective_dwell(camera_id, person)
        return (
            effective_dwell >= self.loitering_demo_dwell_seconds
            and person.get("roaming_radius_10s", float("inf")) <= self.loitering_demo_radius_px
        ) or (
            effective_dwell >= self.loitering_dwell_seconds
            and person.get("roaming_radius_45s", float("inf")) <= self.loiter_radius_px
        ) or (
            effective_dwell >= self.loitering_pacing_window_seconds
            and person.get("roaming_radius_120s", float("inf")) <= self.loitering_pacing_radius_px
        )

    def get_effective_dwell(self, camera_id: int, person: Dict[str, Any]) -> float:
        """A tracked person's own fresh dwell_time PLUS any dwell inherited
        from a matched prior identity that left and came back (see
        reid_try_link()) — the single source of truth check_loitering()
        and inference_engine.py's exit-registration both use, so a
        multi-return chain (left and came back more than once) always
        carries the FULL accumulated total forward, not just the most
        recent session's fresh time."""
        tid = person.get("track_id")
        inherited = self._reid_inherited_dwell.get((camera_id, tid), 0.0) if tid is not None else 0.0
        return person.get("dwell_time", 0.0) + inherited

    def reid_register_exit(
        self, camera_id: int, signature: Dict[str, Any], cumulative_dwell: float, now: float,
    ) -> None:
        """A tracked person has just left frame (evicted from the active
        tracker) — record their visual descriptor + accumulated dwell
        time so a later re-entry can be linked back to them. Only
        meaningful for someone who actually accrued some dwell (a person
        who was on screen for one frame and left has nothing worth
        remembering); callers should still call this unconditionally and
        let cumulative_dwell simply be small/zero for such tracks — the
        cost of one extra small memory entry is negligible and there is
        no reliable, false-positive-free way to pre-filter "worth
        remembering" here without just duplicating the loitering
        threshold logic."""
        self._reid_memory.register_exit(
            camera_id, signature, cumulative_dwell, now, self.reid_time_window_seconds
        )

    def reid_try_link(
        self, camera_id: int, track_id: int, signature: Dict[str, Any], now: float,
    ) -> float:
        """A brand-new track_id has just appeared — check whether its
        visual descriptor matches a recently-departed identity on this
        SAME camera. If matched, links track_id to that identity's
        accumulated dwell time (stored so check_loitering()'s effective-
        dwell calculation picks it up immediately) and returns it;
        returns 0.0 if no match. Idempotent to call more than once for
        the same track_id in the rare case a caller does — the memory
        entry is consumed on first match, so a second call simply finds
        nothing and returns 0.0 without double-counting."""
        inherited = self._reid_memory.try_match(
            camera_id, signature, now, self.reid_similarity_threshold, self.reid_time_window_seconds
        )
        if inherited > 0.0:
            self._reid_inherited_dwell[(camera_id, track_id)] = inherited
        return inherited

    def reid_release_track(self, camera_id: int, track_id: int) -> None:
        """Discards any inherited-dwell linkage for a track_id once it's
        no longer relevant — called after reid_register_exit() has
        already captured that track's TOTAL accumulated dwell (own fresh
        dwell_time + whatever it inherited) into the new memory entry, so
        the stale per-track linkage doesn't leak indefinitely."""
        self._reid_inherited_dwell.pop((camera_id, track_id), None)

    def update_overcrowding_state(
        self, camera_id: int, now: float, tracked_persons: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Overcrowding orchestration for one camera — finds this window's
        largest qualifying cluster (find_largest_overcrowding_cluster())
        and advances that camera's OvercrowdingState with it, using this
        engine's configured thresholds. Called unconditionally every
        window from _check_rules (BEFORE that method decides what, if
        anything, to actually return as this window's alert) so the
        accumulation/hysteresis timer keeps advancing in real time even on
        windows where some OTHER rule's DetectionResult ends up taking
        priority — exactly the same reason Fighting's Gate 3/4 state is
        always updated regardless of what inference_engine.py's analyze()
        ultimately returns.

        Prunes the state back out once it returns to idle, so a
        de-escalated crowd doesn't leak memory over a long-running camera
        session.
        """
        cluster = find_largest_overcrowding_cluster(
            tracked_persons, self.overcrowding_min_count, self.overcrowding_max_radius_px
        )
        state = self._overcrowding_states.get(camera_id)
        if state is None:
            state = OvercrowdingState()
            self._overcrowding_states[camera_id] = state

        result = state.update(
            now, cluster, self.overcrowding_dwell_seconds, self.overcrowding_pretrigger_gap_seconds
        )
        if result["idle"]:
            self._overcrowding_states.pop(camera_id, None)
        return result

    def get_overcrowding_state(self, camera_id: int, now: float) -> Dict[str, Any]:
        """Read-only snapshot of this camera's CURRENT Overcrowding state,
        without advancing anything — for the overlay renderer
        (inference_engine.py), which needs to know "is there a confirmed
        crowd right now" and who its members are, independently of
        whether Overcrowding happened to be the rule that won this
        window's single DetectionResult slot (see update_overcrowding_
        state()'s docstring for why it's always advanced regardless)."""
        state = self._overcrowding_states.get(camera_id)
        if state is None:
            return {"state": "NORMAL", "idle": True, "duration_seconds": 0.0, "peak_count": 0, "members": []}
        return state.snapshot(now)

    # ------------------------------------------------------------------
    # Hybrid CNN-gating helpers — called from inference_engine.py AFTER the
    # neural classifier has run (unlike _check_rules above, which runs
    # BEFORE it and has no CNN probabilities available yet).
    # ------------------------------------------------------------------

    def find_fighting_pair_configured(self, tracked_persons: List[Dict[str, Any]]) -> Optional[Tuple[Dict, Dict]]:
        """find_fighting_pair() (Gates 1+2) using this engine's own
        configured thresholds — convenience wrapper so inference_engine.py
        doesn't have to thread separate config values through the call
        site by hand."""
        return find_fighting_pair(
            tracked_persons,
            self.fighting_proximity_multiplier,
            self.fighting_iou_min,
            self.fighting_motion_min_px,
        )

    def fighting_pair_still_qualifies_configured(
        self, person_a: Dict[str, Any], person_b: Dict[str, Any]
    ) -> bool:
        """fighting_pair_still_qualifies() (Gates 1+2 for one specific
        pair) using this engine's own configured thresholds — see
        fighting_pair_still_qualifies()'s docstring for why the caller
        needs this "is my current pair still valid" check in addition to
        find_fighting_pair_configured()'s "find any valid pair" search."""
        return fighting_pair_still_qualifies(
            person_a, person_b,
            self.fighting_proximity_multiplier,
            self.fighting_iou_min,
            self.fighting_motion_min_px,
        )

    def update_fighting_incident_state(
        self,
        camera_id: int,
        now: float,
        gates_1_2_pass: bool,
        gate3_prob: float,
    ) -> Dict[str, Any]:
        """
        GATE 4 orchestration for a camera's currently-suspected fight
        incident — advances (or creates) that camera's FightingPairState
        with this window's Gate 1+2 result and Gate 3 (cropped-region
        neural) probability, using this engine's configured hysteresis
        thresholds.

        Deliberately keyed by CAMERA, not by a specific track_id pair —
        see _fighting_gate3_prob's docstring in inference_engine.py for
        the full rationale: on real chaotic multi-person footage, the
        exact pair of IDs Gates 1+2 flag as "who's fighting" was observed
        to change identity every few hundred milliseconds, so no single
        pair_key ever accumulated the required continuous confirm_seconds
        even though the same physical fight was visibly, continuously
        present the whole time. Tracking "is a fight happening in this
        camera right now" instead of "is this EXACT pair of numeric IDs
        still fighting" follows the real event instead of the tracker's
        unstable identity labels for it — one active incident per camera
        also matches the practical goal of raising one alert per fight,
        not one per relabeled ID pair.

        Prunes the state back out once it returns to idle, so a
        de-escalated incident doesn't leak memory over a long-running
        camera session.
        """
        state = self._fighting_pair_states.get(camera_id)
        if state is None:
            state = FightingPairState()
            self._fighting_pair_states[camera_id] = state

        result = state.update(
            now, gates_1_2_pass, gate3_prob,
            self.fighting_confirm_seconds,
            self.fighting_neural_confirm_threshold,
            self.fighting_recovery_prob_threshold,
            self.fighting_recovery_prob_seconds,
            self.fighting_proximity_fail_seconds,
            self.fighting_pretrigger_gap_seconds,
        )
        if result["idle"]:
            self._fighting_pair_states.pop(camera_id, None)
        return result

    def check_vandalism(
        self,
        camera_id: int,
        now: float,
        tracked_persons: List[Dict],
        zones: List[Dict],
        kicking_prob: float,
        fighting_prob: float,
    ) -> Optional[DetectionResult]:
        """
        Vandalism = a tracked person's centroid inside an "asset"-type zone
        AND the CNN's Kicking/Fighting probability (the closest available
        "aggressive motion" signal — there is no trained Vandalism class)
        both holding true, continuously, for >= vandalism_min_seconds.
        Dwell time inside the zone ALONE is deliberately no longer
        sufficient (see module docstring) — someone standing near a shelf
        looking for their keys must not alert.

        On confirmation, the returned DetectionResult.metadata carries the
        specific evidence a guard needs to see WHY this fired (not just
        that it fired): which zone ("zone_polygon"), which tracked person
        ("track_id"/"bbox"), and which CNN signal was reading aggressive
        ("aggression_label" — "Kicking" or "Fighting", whichever was
        higher) — consumed by inference_engine.py's overlay renderer.
        """
        aggression_prob = max(kicking_prob, fighting_prob)
        aggression_label = "Kicking" if kicking_prob >= fighting_prob else "Fighting"

        matched_zone: Optional[Dict] = None
        matched_person: Optional[Dict] = None
        if zones and tracked_persons and aggression_prob > self.vandalism_aggression_threshold:
            asset_zones = [z for z in zones if z.get("zone_type") == "asset"]
            for p in tracked_persons:
                cx, cy = p.get("center", (0, 0))
                for z in asset_zones:
                    if self._point_in_polygon({"x": cx, "y": cy}, z.get("polygon_points", [])):
                        matched_zone = z
                        matched_person = p
                        break
                if matched_person:
                    break

        condition = matched_person is not None
        if not condition:
            self._vandalism_start_time.pop(camera_id, None)
            return None

        start = self._vandalism_start_time.get(camera_id)
        if start is None:
            self._vandalism_start_time[camera_id] = now
            return None

        elapsed = now - start
        if elapsed < self.vandalism_min_seconds:
            return None

        return DetectionResult(
            camera_id=camera_id,
            activity="Vandalism",
            confidence=round(aggression_prob, 3),
            severity="critical",
            persons_detected=1,
            duration_seconds=elapsed,
            timestamp=datetime.utcnow(),
            metadata={
                "rule": "asset_zone_aggression",
                "aggression_prob": round(aggression_prob, 3),
                "aggression_label": aggression_label,
                "zone_polygon": matched_zone.get("polygon_points", []) if matched_zone else [],
                "zone_name": matched_zone.get("name") if matched_zone else None,
                "track_id": matched_person.get("track_id"),
                "bbox": matched_person.get("bbox"),
            },
        )

    def check_falling_candidate(
        self,
        camera_id: int,
        now: float,
        tracked_persons: List[Dict],
    ) -> bool:
        """
        True if any tracked person currently qualifies as a "fallen"
        candidate: bounding-box aspect ratio (h/w) below
        falling_aspect_ratio_max (a horizontal posture) AND centroid drift
        since that posture first appeared has stayed under
        falling_motionless_px, continuously, for >= falling_motionless_seconds.
        This is a GATE — the caller (inference_engine.py) still requires the
        CNN's own Falling probability on top of this before confirming.
        """
        current_ids = set()
        qualifies = False
        for p in tracked_persons:
            track_id = p.get("track_id")
            bbox = p.get("bbox")
            if track_id is None or not bbox:
                continue
            x1, y1, x2, y2 = bbox
            bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
            aspect = bh / bw
            key = (camera_id, track_id)

            if aspect >= self.falling_aspect_ratio_max:
                self._fall_candidates.pop(key, None)
                continue

            current_ids.add(key)
            cx, cy = p.get("center", (0, 0))
            state = self._fall_candidates.get(key)
            if state is None:
                self._fall_candidates[key] = {"anchor": (cx, cy), "start_time": now}
                continue

            ax, ay = state["anchor"]
            drift = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
            if drift > self.falling_motionless_px:
                # Moved too much since the posture dropped — not motionless;
                # restart the clock from here rather than giving up
                # entirely, in case they settle right after.
                state["anchor"] = (cx, cy)
                state["start_time"] = now
                continue

            if now - state["start_time"] >= self.falling_motionless_seconds:
                qualifies = True

        # Prune candidates for tracks that are no longer in a horizontal
        # posture this frame (handled above) or no longer tracked at all.
        for key in [k for k in self._fall_candidates if k[0] == camera_id and k not in current_ids]:
            del self._fall_candidates[key]

        return qualifies

    def find_snatch_divergence(
        self,
        camera_id: int,
        now: float,
        tracked_persons: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """
        Returns proof data ({"track_id", "bbox", "center", "direction": (dx,
        dy) unit vector}) for a tracked person who was recently in contact
        with another (within snatching_contact_px) and has since departed
        at >= snatching_departure_px_s within snatching_departure_window_
        seconds of that contact — the "grab and run" signature a static hug
        or handshake doesn't produce — or None if no such divergence is
        currently happening. This is a GATE — the caller (inference_engine.
        py) still requires the CNN's own Snatching probability on top of
        this before confirming.
        """
        boxes = {p["track_id"]: p for p in tracked_persons if p.get("track_id") is not None and p.get("bbox")}
        ids = sorted(boxes.keys())
        proof: Optional[Dict[str, Any]] = None

        current_pairs = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                pair_key = (camera_id, id_a, id_b)
                pa, pb = boxes[id_a], boxes[id_b]
                if _bbox_proximate(pa["bbox"], pb["bbox"], self.snatching_contact_px):
                    current_pairs.add(pair_key)
                    self._snatch_contacts[pair_key] = {
                        "pos_a": pa.get("center", (0, 0)),
                        "pos_b": pb.get("center", (0, 0)),
                        "contact_time": now,
                    }

        for pair_key, contact in list(self._snatch_contacts.items()):
            if pair_key[0] != camera_id:
                continue
            if now - contact["contact_time"] > self.snatching_departure_window_seconds:
                del self._snatch_contacts[pair_key]
                continue
            if pair_key in current_pairs:
                # Still in contact — no departure to measure yet.
                continue
            _, id_a, id_b = pair_key
            for track_id, other_pos in ((id_a, contact["pos_b"]), (id_b, contact["pos_a"])):
                p = boxes.get(track_id)
                if p is None:
                    continue
                speed = p.get("speed", 0.0)
                if speed < self.snatching_departure_px_s:
                    continue
                cx, cy = p.get("center", (0, 0))
                ox, oy = other_pos
                dx, dy = cx - ox, cy - oy
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > self.snatching_contact_px:
                    proof = {
                        "track_id": track_id,
                        "bbox": p["bbox"],
                        "center": (cx, cy),
                        "direction": (dx / dist, dy / dist) if dist > 0 else (0.0, 0.0),
                    }

        return proof

    def check_snatch_divergence(
        self,
        camera_id: int,
        now: float,
        tracked_persons: List[Dict],
    ) -> bool:
        """Boolean convenience wrapper around find_snatch_divergence() — see
        there for the gate's actual logic."""
        return self.find_snatch_divergence(camera_id, now, tracked_persons) is not None

    @staticmethod
    def _average_pairwise_distance(tracked_persons: List[Dict]) -> Optional[float]:
        """Mean centroid-to-centroid distance across every pair of tracked
        persons this frame — None if fewer than 2 (no pair exists). Used to
        require a "crowd" to actually be a group standing/moving close
        together, not just a headcount scattered across the whole frame."""
        centers = [p.get("center") for p in tracked_persons if p.get("center") is not None]
        if len(centers) < 2:
            return None
        total = 0.0
        count = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                (x1, y1), (x2, y2) = centers[i], centers[j]
                total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                count += 1
        return total / count if count else None

    @staticmethod
    def _point_in_polygon(point: Dict, polygon: List[Dict]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        if not polygon or len(polygon) < 3:
            return False
        x, y = point.get("x", 0), point.get("y", 0)
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].get("x", 0), polygon[i].get("y", 0)
            xj, yj = polygon[j].get("x", 0), polygon[j].get("y", 0)
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside
