"""
Activity Hysteresis State Machine.

Turns raw, noisy, per-window classifier probabilities into a temporally
stable overlay state. Without this, a probability that oscillates around a
single hard threshold (e.g. 0.71, 0.79, 0.68, 0.82 across consecutive
windows) makes the overlay flicker between "Normal" and "FIGHTING" several
times a second — visually useless and impossible to trust.

State machine (per class, independently):

    NORMAL --(smoothed >= START)--> ANALYZING
    ANALYZING --(smoothed >= CONFIRM for N windows)--> CONFIRMED
    ANALYZING --(smoothed < END)--> NORMAL
    CONFIRMED --(smoothed < END)--> RECOVERY        (immediately, on window 1)
    RECOVERY --(smoothed >= CONFIRM)--> CONFIRMED   (re-confirm)
    RECOVERY --(smoothed < CONFIRM for N windows)--> NORMAL
    RECOVERY --(else, budget not yet exhausted)--> RECOVERY (bridges the dip)

    The N-window persistence budget lives entirely in RECOVERY, not split
    between CONFIRMED and RECOVERY — CONFIRMED hands off the instant the
    signal dips, so RECOVERY always gets its full N windows to bridge a
    brief drop before actually giving up. (An earlier version spent the
    budget in CONFIRMED and left RECOVERY to reset to NORMAL on a single
    sub-CONFIRM window with no counter of its own — found, from a real CCTV
    video, to wipe out a genuine high-confidence spike whenever the EMA
    hadn't yet caught back up to confirm_threshold in one step.)

Each activity class (Fighting, Running, Loitering, ...) gets its OWN
independent instance of this machine, fed that class's own probability every
window — never just the current top-1 class. The previous design shared a
single machine across all classes and reset its entire smoothing/streak
state whenever the top-1 predicted class changed frame-to-frame, which meant
a sequence like:

    Fighting 0.84, Fighting 0.82, Normal 0.45, Fighting 0.81, Fighting 0.86

would throw away all accumulated "Fighting" evidence the moment "Normal"
briefly out-scored it, and be seen as several separate short spikes instead
of one sustained event — exactly the kind of noisy real-world classifier
output this hysteresis layer exists to survive. Feeding each class's own
probability into its own persistent machine every window (0.0 on a window
where that class isn't evidenced at all, same as genuinely low confidence)
means a momentary rank swap between two classes no longer erases either
one's progress.
"""
import time as _time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from ai_service.config import AIConfig


NORMAL = "NORMAL"
ANALYZING = "ANALYZING"
CONFIRMED = "CONFIRMED"
RECOVERY = "RECOVERY"


@dataclass
class ActivityStateResult:
    state: str
    label: Optional[str]
    raw_confidence: float
    smoothed_confidence: float
    is_alertable: bool  # CONFIRMED *and* smoothed >= ALERT_THRESHOLD
    # Event-lifecycle summary (Part 7/17/18) — 0 while NORMAL, then tracked
    # from the moment this class first entered ANALYZING so a caller can
    # persist ONE incident (start/peak/average) instead of one alert per
    # window. Mirrors ai_service/fire_smoke_state.py's duration_seconds.
    duration_seconds: float = 0.0
    peak_confidence: float = 0.0
    average_confidence: float = 0.0
    # Absolute "clock" this class entered ANALYZING (same unit as the `now`
    # passed into update() — real epoch seconds for live streaming, or a
    # synthesized epoch for offline/batch video-timeline processing, see
    # ai_service/video_timeline_processor.py). None while NORMAL.
    start_time: Optional[float] = None
    # Edge signals — True on ONLY the exact window a transition happens,
    # never on every window the class merely continues to be active. A
    # caller doing event persistence (see ai_service/inference_engine.py's
    # neural-alert block) uses these to create/finalize a DB row exactly
    # once per real event, instead of relying on wall-clock cooldowns that
    # can both duplicate a long event and suppress a genuinely new one.
    event_started: bool = False  # ANALYZING -> CONFIRMED this window (not a RECOVERY re-confirm)
    event_ended: bool = False    # CONFIRMED/RECOVERY -> NORMAL this window (event fully over)


class _ClassActivityMachine:
    """One instance tracks ONE activity class's evidence stream, with its
    own (optionally class-specific — see AIConfig.CLASS_SPECIFIC_THRESHOLDS)
    thresholds and confirm/end persistence requirements."""

    def __init__(self, label: str):
        self.label = label
        overrides = AIConfig.CLASS_SPECIFIC_THRESHOLDS.get(label, {})
        self.start_threshold = overrides.get("start_threshold", AIConfig.ACTIVITY_START_THRESHOLD)
        self.confirm_threshold = overrides.get("confirm_threshold", AIConfig.ACTIVITY_CONFIRM_THRESHOLD)
        self.end_threshold = overrides.get("end_threshold", AIConfig.ACTIVITY_END_THRESHOLD)
        self.alpha = overrides.get("alpha", AIConfig.ACTIVITY_EMA_ALPHA)
        self.min_confirm_windows = overrides.get(
            "min_consecutive_confirm_windows", AIConfig.MIN_CONSECUTIVE_CONFIRM_WINDOWS
        )
        self.min_end_windows = overrides.get(
            "min_consecutive_end_windows", AIConfig.MIN_CONSECUTIVE_END_WINDOWS
        )
        self.alert_threshold = overrides.get("alert_threshold", AIConfig.ALERT_THRESHOLD)

        self.state = NORMAL
        self._smoothed = 0.0
        self._confirm_streak = 0
        self._end_streak = 0
        self._start_time: Optional[float] = None
        self._peak = 0.0
        self._conf_sum = 0.0
        self._conf_count = 0

    def _accumulate(self) -> None:
        self._peak = max(self._peak, self._smoothed)
        self._conf_sum += self._smoothed
        self._conf_count += 1

    def update(self, raw_confidence: float, now: float) -> ActivityStateResult:
        prev_state = self.state
        # Captured BEFORE any transition below can null `self._start_time`
        # out — needed so the exact window an event ends on can still
        # report its real final duration (see the duration calculation at
        # the bottom of this method for why using self._start_time directly
        # there would always read back 0 on that window).
        start_time_before_this_call = self._start_time
        if self.state == NORMAL:
            # While NORMAL, there is no sustained-event memory worth
            # smoothing against — read each window's raw value directly
            # rather than damping it through an EMA, otherwise a fresh
            # spike (a brand-new class's first-ever window, or a genuinely
            # new event starting right after a previous one fully reset)
            # would need several consecutive high-confidence windows just
            # to climb out of the damping before ever reaching the start
            # threshold at all.
            #
            # Deliberately keyed on STATE, not a one-shot "_ever_updated"
            # flag consumed by the very next call regardless of its value
            # (the previous design) — that had a real edge case: if the
            # single call immediately after a reset happened to carry a
            # low/irrelevant residual value (e.g. the tail of a just-ended
            # event's decaying evidence), it "used up" the fresh-start
            # behavior on a value that didn't matter, and every window
            # after that — INCLUDING the actual start of a real new event
            # — went back to being damped from that near-zero residual,
            # delaying real detection by several windows. Reading raw
            # values directly for as long as we stay NORMAL means the
            # fresh-start behavior is available whenever a real signal
            # actually arrives, not just on whichever window happens to
            # follow a reset.
            self._smoothed = raw_confidence
        else:
            self._smoothed = self.alpha * raw_confidence + (1 - self.alpha) * self._smoothed

        if self.state == NORMAL:
            if self._smoothed >= self.start_threshold:
                self.state = ANALYZING
                self._start_time = now
                self._peak = self._smoothed
                self._conf_sum = self._smoothed
                self._conf_count = 1

        elif self.state == ANALYZING:
            self._accumulate()
            if self._smoothed >= self.confirm_threshold:
                self._confirm_streak += 1
                self._end_streak = 0
                if self._confirm_streak >= self.min_confirm_windows:
                    self.state = CONFIRMED
            elif self._smoothed >= self.end_threshold:
                # Still plausible evidence (between end and confirm
                # threshold) — do NOT reset confirm_streak to 0 here, only
                # pause it. A single mid-range dip inside an otherwise
                # strong burst (e.g. smoothed briefly reads 0.68 between two
                # 0.85+ windows) is not the same as evidence actually
                # weakening toward "this isn't happening" — only a drop
                # below end_threshold represents that, and already resets
                # confirm_streak in the branch below. Without this, two
                # non-consecutive-but-close-together confirm-worthy windows
                # separated by one plausible-but-not-quite-confirm window
                # could never accumulate toward min_consecutive_confirm_
                # windows, even though nothing about the underlying evidence
                # actually failed.
                self._end_streak = 0
            else:
                # smoothed < end_threshold here (the elif above already
                # captured end_threshold <= smoothed < confirm_threshold).
                self._confirm_streak = 0
                # Gap tolerance, matching CONFIRMED's below (see that
                # branch's comment) — a REAL production bug found on an
                # actual CCTV fighting video: raw confidence legitimately
                # oscillates within a genuine event (a several-second
                # burst of high-confidence windows with occasional single-
                # window dips from motion blur/occlusion/brief separation),
                # and a single dip below end_threshold used to wipe out
                # all accumulated evidence INSTANTLY here — unlike
                # CONFIRMED, which already tolerated MIN_CONSECUTIVE_END_
                # WINDOWS consecutive low windows before giving up. That
                # asymmetry meant genuinely real, sustained-but-noisy
                # evidence could be killed during ANALYZING before it
                # ever had a chance to accumulate enough consecutive
                # confirm windows to reach CONFIRMED at all — while a
                # competing, more statistically STABLE (not necessarily
                # more CORRECT) prediction for a different class sailed
                # through untouched. Same persistence requirement as
                # leaving CONFIRMED, not a stricter one.
                self._end_streak += 1
                if self._end_streak >= self.min_end_windows:
                    self.state = NORMAL
                    self._start_time = None
                    self._end_streak = 0
                    # Fully resolved back to NORMAL — clear the EMA memory
                    # too (not just start_time), so a LATER, genuinely
                    # separate occurrence of this class starts clean via
                    # the NORMAL-state direct-set path above instead of
                    # having to fight through this event's leftover
                    # smoothing residue first. Without this, "Fight ->
                    # Normal -> Fight" could take several extra windows
                    # to re-confirm the second fight purely because of
                    # the first one's fading memory.
                    self._smoothed = 0.0

        elif self.state == CONFIRMED:
            self._accumulate()
            if self._smoothed < self.end_threshold:
                # Hand off to RECOVERY immediately on the FIRST sub-threshold
                # window, rather than spending the min_end_windows patience
                # budget here and leaving RECOVERY with none of its own (see
                # the bug this replaces, documented on the RECOVERY branch
                # below) — CONFIRMED's only job is to notice the drop has
                # started; RECOVERY is what actually survives it.
                self.state = RECOVERY
                self._end_streak = 1
            else:
                self._end_streak = 0

        elif self.state == RECOVERY:
            self._accumulate()
            if self._smoothed >= self.confirm_threshold:
                # Genuine re-confirmation — same bar ANALYZING needed the
                # first time, not just "still above end_threshold" (that
                # would make RECOVERY too eager to cancel a real fade).
                self.state = CONFIRMED
                self._end_streak = 0
            else:
                # REAL PRODUCTION BUG this replaces (found on an actual CCTV
                # fighting video, see the module's Fighting-specific
                # threshold comment in config.py): the old code reset
                # straight to NORMAL here on a SINGLE window below
                # confirm_threshold — with no streak counter of its own —
                # even though CONFIRMED had already spent min_end_windows
                # getting here, leaving RECOVERY with zero actual patience
                # despite the docstring's stated intent of it being a grace
                # period. Concretely: state=RECOVERY, smoothed=0.441, then a
                # real raw=0.984 fight-confidence spike arrived — with
                # alpha=0.30 the EMA only reached ~0.604, still under
                # confirm_threshold (0.80), so this branch fired and
                # instantly zeroed out a live, genuine, high-confidence
                # fight signal instead of just noting the dip and waiting.
                #
                # Fix: give RECOVERY its OWN min_end_windows-window budget
                # (matching CONFIRMED's, including any CLASS_SPECIFIC_
                # THRESHOLDS override) before actually giving up — a brief
                # 1-2 window dip below confirm_threshold no longer kills a
                # sustained event; only min_end_windows CONSECUTIVE such
                # windows does.
                self._end_streak += 1
                if self._end_streak >= self.min_end_windows:
                    self.state = NORMAL
                    self._confirm_streak = 0
                    self._end_streak = 0
                    self._start_time = None
                    # See the matching reset in the ANALYZING branch above.
                    self._smoothed = 0.0
                # Else: stays in RECOVERY — the event/alert stays open,
                # bridging over the dip instead of being torn down.

        # A genuinely NEW confirmation (ANALYZING -> CONFIRMED), not a
        # RECOVERY -> CONFIRMED re-confirm (that's a continuation of the
        # SAME still-open event, not a new one).
        event_started = prev_state == ANALYZING and self.state == CONFIRMED
        # The event is only truly over once it fully resolves back to
        # NORMAL — RECOVERY is still "the same open event, currently fading".
        event_ended = prev_state in (CONFIRMED, RECOVERY) and self.state == NORMAL

        # On the exact window an event ends, self._start_time has already
        # been reset to None (state is NORMAL) — use the pre-transition
        # value so THIS window's result still reports the event's real
        # final duration instead of 0. Every other window uses the current
        # self._start_time as before.
        effective_start_time = start_time_before_this_call if event_ended else self._start_time
        duration = (now - effective_start_time) if (effective_start_time and (self.state != NORMAL or event_ended)) else 0.0
        is_alertable = self.state == CONFIRMED and self._smoothed >= self.alert_threshold
        average = round(self._conf_sum / self._conf_count, 3) if self._conf_count else 0.0
        return ActivityStateResult(
            state=self.state,
            label=self.label if self.state != NORMAL else None,
            raw_confidence=round(raw_confidence, 3),
            smoothed_confidence=round(self._smoothed, 3),
            is_alertable=is_alertable,
            duration_seconds=round(duration, 1),
            peak_confidence=round(self._peak, 3),
            average_confidence=average,
            start_time=effective_start_time,
            event_started=event_started,
            event_ended=event_ended,
        )


class ActivityStateMachine:
    """
    Per-camera manager owning one independent `_ClassActivityMachine` per
    activity class ever observed for this camera. Call `update(probs)` once
    per analysis window with the FULL per-class probability dict for that
    window — "Normal"/"Fire"/"Smoke" should already be excluded by the
    caller (Fire/Smoke are a separate subsystem, see fire_smoke_state.py).

    Every tracked class is stepped every window (fed 0.0 if absent from
    `probs`), so classes accumulate or decay evidence independently instead
    of one shared state resetting whenever the top-1 class changes. The
    reported result is whichever class is currently "most active": a
    CONFIRMED class wins over anything else; among equally-staged classes,
    the higher smoothed confidence wins.
    """

    def __init__(self):
        self._machines: Dict[str, _ClassActivityMachine] = {}

    def update(
        self, probs: Optional[Dict[str, float]], now: Optional[float] = None
    ) -> "Tuple[ActivityStateResult, List[ActivityStateResult], List[ActivityStateResult]]":
        """
        Returns (best, ended, active):
        - `best`: whichever class is currently "most active" this window (a
          CONFIRMED class wins over anything else; among equally-staged
          classes, the higher smoothed confidence wins) — NORMAL/no-label if
          nothing is active. Display-only (e.g. the single-line overlay HUD
          banner) — see `active` below for why alert persistence must NOT
          key off this alone.
        - `ended`: every class (usually 0 or 1) whose `event_ended` fired
          THIS window — a caller doing event persistence needs this even for
          a class that isn't `best` anymore (it just became NORMAL, so it
          never would be), so an open DB row for it can still be closed out
          exactly once.
        - `active`: EVERY class currently NOT NORMAL (ANALYZING/CONFIRMED/
          RECOVERY), independent of which one happens to be `best`. Real
          CCTV footage can have two classes both genuinely confirmable at
          once (e.g. a crowd loitering while two people within it are
          fighting) — picking only the single highest-confidence class for
          alerting would silently DROP a second, independently-confirmed
          class's event forever, since it can never outrank an
          already-very-high-confidence `best` on wall-clock terms alone.
          This was a real bug (see backend/app/services/alert_dispatcher.py's
          fix history): a sustained, dominant "Loitering" confirmation
          masked a simultaneous, high-confidence "Fighting" signal for an
          entire 40-second clip because Fighting's own smoothed confidence
          never had to just clear its OWN threshold — it also had to beat
          Loitering's, which had nothing to do with whether Fighting was
          real. Callers doing alert creation/refresh/finalize MUST iterate
          `active` (and `ended`), not just `best`.
        """
        now = now if now is not None else _time.time()
        probs = probs or {}

        classes_to_step = set(self._machines.keys()) | {
            c for c in probs.keys() if c and c.lower() not in ("normal", "fire", "smoke")
        }

        best: Optional[ActivityStateResult] = None
        ended: List[ActivityStateResult] = []
        active: List[ActivityStateResult] = []
        for cls in classes_to_step:
            machine = self._machines.get(cls)
            if machine is None:
                machine = _ClassActivityMachine(cls)
                self._machines[cls] = machine
            result = machine.update(probs.get(cls, 0.0), now)
            if result.event_ended:
                # `result.label` is None here (the class just went NORMAL,
                # and label is intentionally blanked on NORMAL so `best`
                # never shows a stale label) — but a caller closing out a
                # DB row needs to know WHICH class ended, so re-attach it
                # on this ended-list copy only.
                ended.append(replace(result, label=cls))
            if result.state == NORMAL:
                continue
            active.append(result)
            if best is None or (result.state == CONFIRMED and best.state != CONFIRMED):
                best = result
            elif result.state == best.state and result.smoothed_confidence > best.smoothed_confidence:
                best = result

        if best is None:
            best = ActivityStateResult(
                state=NORMAL, label=None, raw_confidence=0.0, smoothed_confidence=0.0,
                is_alertable=False, duration_seconds=0.0, peak_confidence=0.0, average_confidence=0.0,
            )
        return best, ended, active

    def update_single(
        self, label: Optional[str], raw_confidence: float, now: Optional[float] = None
    ) -> "Tuple[ActivityStateResult, List[ActivityStateResult], List[ActivityStateResult]]":
        """Backward-compatible convenience wrapper for a single (label,
        confidence) pair — builds the one-entry probs dict `update()` now
        expects. Prefer passing the full per-class probability dict when
        available (see inference_engine.py's `_run_neural_classifier`)."""
        probs = {label: raw_confidence} if label else {}
        return self.update(probs, now)
