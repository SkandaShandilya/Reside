"""
Fire/Smoke Hysteresis State Machine — a SEPARATE subsystem from
ai_service/activity_state.py (the UCF-Crime-style behaviour classifier's
state machine). Deliberately self-contained rather than importing/sharing
state with the behaviour system: fire/smoke evidence, thresholds, and
persistence requirements have nothing to do with person/activity
recognition, and this module must be tunable independently without
touching behaviour code at all.

Feeds from ai_service.detector.FireSmokeDetector's HSV + optical-flow
output (raw per-frame confidence), never from the neural activity
classifier — see the module docstring in inference_engine.py for why
Fire/Smoke are excluded from that classifier's display path.

State machine (one instance per camera per class — fire and smoke each
get their own, since they can be independently true). Internal state
names are kept as NORMAL/POSSIBLE/CONFIRMED/RECOVERY for backward
compatibility with existing overlay/telemetry consumers (frontend
FireSmoke.tsx, render_fire_smoke_only()) — conceptually these map to the
more descriptive NORMAL/CONFIRMING/ACTIVE/GRACE vocabulary used in this
module's docstrings and in inference_engine.py's debug logging:

    NORMAL --(any raw candidate)--> POSSIBLE ("CONFIRMING")
    POSSIBLE --(smoothed >= START threshold for N windows)--> CONFIRMED ("ACTIVE")
    POSSIBLE --(smoothed < END threshold)--> NORMAL
    CONFIRMED --(smoothed < END threshold)--> RECOVERY ("GRACE", timer starts)
    RECOVERY --(smoothed >= END threshold)--> CONFIRMED (recovered, timer cancelled)
    RECOVERY --(END_GRACE_SECONDS elapsed with no recovery)--> NORMAL (event closes)

START > END is the actual hysteresis mechanism: confidence oscillating
around a single value can only flap the event if it crosses BOTH bars
repeatedly, which genuine noise around one threshold cannot do. The grace
period is TIME-based (seconds elapsed via the caller's `now`, which is
wall-clock for live streaming or a synthesized video-relative epoch for
offline batch processing — see ai_service/video_timeline_processor.py) so
it means the same thing regardless of how fast frames are actually being
processed.
"""
from dataclasses import dataclass, field
from typing import Optional

from ai_service.config import AIConfig

NORMAL = "NORMAL"
POSSIBLE = "POSSIBLE"
CONFIRMED = "CONFIRMED"
RECOVERY = "RECOVERY"

# Friendlier vocabulary for logging (see inference_engine.py) — the
# internal state constants above stay unchanged for backward compatibility.
DISPLAY_NAME = {
    NORMAL: "NORMAL",
    POSSIBLE: "CONFIRMING",
    CONFIRMED: "ACTIVE",
    RECOVERY: "GRACE",
}


@dataclass
class FireSmokeStateResult:
    state: str
    raw_confidence: float
    smoothed_confidence: float
    is_alertable: bool
    duration_seconds: float = 0.0  # Time spent CONFIRMED so far, for the incident record
    peak_confidence: float = 0.0
    average_confidence: float = 0.0
    # Absolute "clock" this event was confirmed at (same unit as the `now`
    # passed into update() — real epoch seconds for live streaming, or a
    # synthesized epoch for offline batch processing, see
    # ai_service/video_timeline_processor.py). None while NORMAL.
    start_time: Optional[float] = None
    # Edge signals — True on ONLY the exact window a transition happens,
    # never on every window the state merely continues. A caller doing
    # event persistence (see ai_service/inference_engine.py) uses these to
    # create/finalize a DB row exactly once per real event, instead of
    # relying on a wall-clock cooldown that can both duplicate a long
    # event and suppress a genuinely new one.
    event_started: bool = False  # POSSIBLE -> CONFIRMED this window (not a RECOVERY re-confirm)
    event_ended: bool = False    # CONFIRMED/RECOVERY -> NORMAL this window (event fully over)


class FireSmokeStateMachine:
    """One instance tracks ONE class (fire OR smoke) for ONE camera."""

    def __init__(
        self,
        start_threshold: float,
        end_threshold: Optional[float] = None,
        alpha: float = AIConfig.FIRE_SMOKE_EMA_ALPHA,
        confirm_windows: int = AIConfig.FIRE_SMOKE_MIN_CONFIRM_WINDOWS,
        end_grace_seconds: float = AIConfig.END_GRACE_SECONDS,
    ):
        self.start_threshold = start_threshold
        # Falls back to the configured default rather than reusing
        # start_threshold — an explicit END_THRESHOLD must always be LOWER
        # than start_threshold (see module docstring); passing None is
        # only for callers that genuinely want the global default.
        self.end_threshold = end_threshold if end_threshold is not None else AIConfig.FIRE_END_THRESHOLD
        self.alpha = alpha
        self.confirm_windows = confirm_windows
        self.end_grace_seconds = end_grace_seconds

        self.state = NORMAL
        self._smoothed = 0.0
        self._ever_updated = False
        self._confirm_streak = 0
        self._confirmed_since: Optional[float] = None
        # Wall-clock (or video-relative) time the SAME open event most
        # recently dropped below end_threshold — None while solidly
        # CONFIRMED or while NORMAL. Distinguishes "just started fading"
        # from "been fading for a while", which the streak-count design
        # this replaced could not do in a time-meaningful way.
        self._grace_started_at: Optional[float] = None
        self._peak = 0.0
        self._conf_sum = 0.0
        self._conf_count = 0

    def _accumulate(self) -> None:
        self._peak = max(self._peak, self._smoothed)
        self._conf_sum += self._smoothed
        self._conf_count += 1

    def update(self, raw_confidence: float, now: float) -> FireSmokeStateResult:
        prev_state = self.state
        # Captured before a close can null these out — see the duration
        # calculation at the bottom of this method.
        confirmed_since_before = self._confirmed_since

        if not self._ever_updated:
            # First observation of this class's evidence stream ever —
            # start the EMA AT the raw value instead of damping it toward
            # the initial 0.0. Without this, a genuinely sustained sequence
            # like 0.82, 0.77, 0.84 (this module's own worked example for
            # "3 confirm windows") would need extra windows just to climb
            # out of zero-anchored damping before ever reaching
            # start_threshold — see ai_service/activity_state.py's
            # identical fix for the same reasoning.
            self._smoothed = raw_confidence
            self._ever_updated = True
        else:
            self._smoothed = self.alpha * raw_confidence + (1 - self.alpha) * self._smoothed

        if self.state == NORMAL:
            if raw_confidence > 0.0:
                self.state = POSSIBLE
                self._confirm_streak = 0
            # Deliberately NOT `elif` below — a NORMAL -> POSSIBLE
            # transition above must be evaluated against the confirm
            # condition in this SAME window, not just from the next call
            # onward. Without this fallthrough, a genuinely sustained
            # sequence (e.g. 0.82, 0.77, 0.84 with confirm_windows=3) would
            # need 4 windows to confirm instead of 3, one more than this
            # module's own worked example (see inference_engine.py spec
            # doc) specifies.

        if self.state == POSSIBLE:
            if self._smoothed >= self.start_threshold:
                self._confirm_streak += 1
                if self._confirm_streak >= self.confirm_windows:
                    self.state = CONFIRMED
                    self._confirmed_since = now
                    self._grace_started_at = None
                    self._peak = self._smoothed
                    self._conf_sum = self._smoothed
                    self._conf_count = 1
            else:
                self._confirm_streak = 0
                if self._smoothed < self.end_threshold:
                    # Never confirmed and evidence has genuinely faded —
                    # give up. Nothing was ever alerted, matching "a single
                    # false-positive spike must not create an event".
                    self.state = NORMAL
                    self._smoothed = 0.0
                    self._ever_updated = False

        elif self.state == CONFIRMED:
            self._accumulate()
            if self._smoothed < self.end_threshold:
                # Confidence has genuinely dropped below the (lower) END
                # bar — start the grace countdown, but do NOT close yet.
                self._grace_started_at = now
                self.state = RECOVERY
            # else: still solidly active, no grace timer running.

        elif self.state == RECOVERY:
            self._accumulate()
            if self._smoothed >= self.end_threshold:
                # Recovered before the grace period elapsed — cancel the
                # timer and resume being fully active. Same open event,
                # not a new one.
                self.state = CONFIRMED
                self._grace_started_at = None
            elif self._grace_started_at is not None and (now - self._grace_started_at) >= self.end_grace_seconds:
                # Absent for the entire grace period — the event is over.
                self.state = NORMAL
                self._confirm_streak = 0
                self._confirmed_since = None
                self._grace_started_at = None
                self._smoothed = 0.0
                self._ever_updated = False
            # else: still within the grace window, remain in RECOVERY.

        event_started = prev_state == POSSIBLE and self.state == CONFIRMED
        event_ended = prev_state in (CONFIRMED, RECOVERY) and self.state == NORMAL

        # On the exact window an event ends, self._confirmed_since has
        # already been reset to None (state is NORMAL) — use the
        # pre-transition value so THIS window still reports the event's
        # real final duration instead of 0.
        effective_confirmed_since = confirmed_since_before if event_ended else self._confirmed_since
        duration = (
            (now - effective_confirmed_since)
            if effective_confirmed_since and (self.state in (CONFIRMED, RECOVERY) or event_ended)
            else 0.0
        )
        is_alertable = self.state == CONFIRMED and self._smoothed >= self.start_threshold
        average = round(self._conf_sum / self._conf_count, 3) if self._conf_count else 0.0
        return FireSmokeStateResult(
            state=self.state,
            raw_confidence=round(raw_confidence, 3),
            smoothed_confidence=round(self._smoothed, 3),
            is_alertable=is_alertable,
            duration_seconds=round(duration, 1),
            peak_confidence=round(self._peak, 3),
            average_confidence=average,
            start_time=effective_confirmed_since,
            event_started=event_started,
            event_ended=event_ended,
        )


@dataclass
class CameraFireSmokeState:
    """Bundles a camera's independent fire and smoke state machines."""
    fire: FireSmokeStateMachine = field(
        default_factory=lambda: FireSmokeStateMachine(
            start_threshold=AIConfig.FIRE_ALERT_THRESHOLD,
            end_threshold=AIConfig.FIRE_END_THRESHOLD,
            confirm_windows=AIConfig.FIRE_CONFIRM_WINDOWS,
        )
    )
    smoke: FireSmokeStateMachine = field(
        default_factory=lambda: FireSmokeStateMachine(
            start_threshold=AIConfig.SMOKE_ALERT_THRESHOLD,
            end_threshold=AIConfig.SMOKE_END_THRESHOLD,
            confirm_windows=AIConfig.SMOKE_CONFIRM_WINDOWS,
        )
    )
