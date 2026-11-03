# AI Service Configuration
import os

class AIConfig:
    # Model paths
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    
    # Detection settings
    # YOLO model for person detection. Deliberately kept at "n" (nano)
    # rather than switching to yolov8s/yolov8m for the RTX 5070 GPU
    # optimization pass — measured on this exact GPU with real frames
    # (see bench_yolo_models.py in the scratchpad used for that task):
    # yolov8n and yolov8s landed within ~0.5ms of each other (both
    # ~8-9.5ms median at imgsz=640, half on or off), so switching up would
    # have cost more GPU memory for zero measured latency benefit at
    # batch size 1. Revisit only if a future accuracy problem (missed
    # small/distant/partially-occluded persons) is traced specifically to
    # detector recall, not for raw speed.
    YOLO_MODEL = "yolov8n.pt"  # YOLO model for person detection
    # Kept as an alias of PERSON_DETECTION_CONFIDENCE below for any code that
    # still reads the old name — the two must never drift apart, since this
    # value is what decides whether a person box is drawn at all, never
    # whether a person's behaviour is suspicious (that's ACTIVITY_* below).
    # Lowered 0.40 -> 0.35 on request, to catch more small/distant/partial
    # detections — the direct tradeoff is more low-confidence boxes (some
    # of which will be false positives); not validated against real footage
    # at this specific value, same "starting point" status as before.
    YOLO_CONFIDENCE = 0.35
    PERSON_DETECTION_CONFIDENCE = 0.35  # Not validated as optimal — tune against real footage.
    PERSON_CLASS_ID = 0  # COCO class ID for person
    # NMS IoU threshold — how much two candidate person boxes must overlap
    # before YOLO's own non-max-suppression collapses them into one
    # detection. Previously only ever the PersonDetector class's own
    # hardcoded default (0.45) — never actually driven by this file, since
    # InferenceEngine's construction call didn't pass it. Now explicit and
    # configurable; value itself unchanged (0.45 was already reasonable —
    # low enough that two close-but-distinct people don't get merged into a
    # single box, which is the "prevents merging two close individuals"
    # requirement this exists for).
    DETECTION_IOU_THRESHOLD = 0.45
    # YOLO inference resolution (imgsz). 640 on GPU is the existing,
    # already-matching default — made explicit/configurable here instead of
    # a hardcoded literal in detector.py. Raise to 1280 only if a camera
    # stream is genuinely high-resolution and small/distant persons are
    # still being missed at 640 — roughly doubles GPU inference cost per
    # frame, so it's a deliberate per-deployment tradeoff, not a default.
    YOLO_IMGSZ = 640
    # Alias of YOLO_IMGSZ for callers/config audits that look for the more
    # generic "DETECTION_IMGSZ" name — derived from it so the two can
    # never drift apart.
    DETECTION_IMGSZ = YOLO_IMGSZ

    # Activity recognition
    SEQUENCE_LENGTH = 16  # Number of frames per clip
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 1
    
    # Processing
    PROCESSING_FPS = 60
    FRAME_BUFFER_SIZE = 600  # ~10s at 60 FPS

    # Max rate (per camera) at which YOLO/behaviour analysis actually runs.
    # The video itself still streams at native FPS (rendered every frame from
    # whatever analysis is most recent) — this only caps how often the GPU
    # is asked to do a fresh detection pass. Person/vehicle motion doesn't
    # need re-detecting 30x/sec, and with several cameras open at once this
    # is the single biggest GPU-load knob.
    AI_ANALYSIS_FPS = 6

    
    # Alert generation
    CONFIDENCE_THRESHOLD = 0.80
    ALERT_COOLDOWN_SECONDS = 30
    LOW_CONFIDENCE_THRESHOLD = 0.50  # Below this, don't generate any event
    REVIEW_THRESHOLD = 0.70  # Between low and this -> review queue

    # Rule-based behaviour thresholds (used by BehaviourEngine — deterministic,
    # position-based detections that stay the only source of "confirmed"
    # alerts, independent of how good the neural classifier below is)

    # ── Loitering (behaviour_engine.py's check_loitering()/person_is_
    # loitering(), stateful per track_id) — see LoiteringReIDMemory for the
    # cross-session "same person returning" layer. THREE OR'd sub-rules
    # (any one alone is sufficient):
    #  (a) DEMO: effective dwell time >= LOITERING_DWELL_TIME_SECONDS (10s
    #      — deliberately fast, for live demos where waiting a realistic
    #      45s+ isn't practical to show), AND confinement over that short
    #      trailing window (tracker.py's Track.roaming_radius(10)) stays
    #      within LOITERING_MAX_RADIUS_PX. This is intentionally the
    #      loosest/fastest-firing rule; rules (b)/(c) below are the more
    #      conservative, production-oriented thresholds and are kept
    #      un-replaced — a genuinely confined 10s dwell will almost always
    #      also satisfy them a little later anyway, so nothing is lost by
    #      leaving all three active together, and (b)/(c) remain available
    #      for a deployment that wants a stricter bar than the demo rule.
    #  (b) DWELL: effective dwell time (own dwell_time + any inherited
    #      Re-ID dwell) >= LOITERING_DWELL_SECONDS, AND their confinement
    #      over just that trailing window (roaming_radius(45)) stays
    #      within LOITER_RADIUS_PX.
    #  (c) PACING: effective dwell time >= LOITERING_PACING_WINDOW_SECONDS
    #      (a longer window), AND confinement over THAT window
    #      (roaming_radius(120)) stays within LOITERING_PACING_RADIUS_PX —
    #      catches someone visibly pacing/lingering in a compact area
    #      whose shorter-window confinement alone wouldn't have tripped
    #      rule (b), e.g. a wider pacing loop that only reveals its real
    #      boundedness over a longer sample.
    # Values here are explicit request defaults, not independently
    # validated against real footage at these exact numbers — same
    # "starting point, tune against real footage" status as this project's
    # other rule thresholds.
    LOITERING_DWELL_TIME_SECONDS = 10.0   # Demo rule: dwell time before a confined-in-place person counts as loitering
    LOITERING_MAX_RADIUS_PX = 120         # Demo rule: max roaming radius (px), over that short window, to still count as "confined"
    LOITERING_DWELL_SECONDS = 45     # Production dwell rule (b) — see above
    LOITER_RADIUS_PX = 150           # Max roaming radius (px), over the dwell window, to still count as "confined"
    LOITERING_PACING_WINDOW_SECONDS = 120   # Longer trailing window for the "confined pacing" sub-rule (c)
    LOITERING_PACING_RADIUS_PX = 150        # Max roaming radius (px) over that longer window

    # ── Loitering Re-ID (ai_service/reid.py + behaviour_engine.py's
    # LoiteringReIDMemory): lightweight color-histogram visual descriptor,
    # NOT a deep embedding model — see reid.py's module docstring for why,
    # and its real, disclosed limitations (fooled by near-identical
    # outfits, doesn't survive a clothing change). RE_ID_SIMILARITY_
    # THRESHOLD is a correlation score in [0,1]; 0.85 (the requested
    # default) is a genuinely high bar for this technique — unvalidated
    # against real "same person returns" footage at this exact value, same
    # "starting point" disclosure as above.
    RE_ID_SIMILARITY_THRESHOLD = 0.85
    RE_ID_TIME_WINDOW_MINUTES = 15
    CROWD_THRESHOLD = 5          # Person count that triggers a crowd formation event
    # Person count must stay >= CROWD_THRESHOLD continuously for this long
    # before a Crowd Formation event fires — a single instantaneous frame
    # (e.g. two groups briefly crossing paths) must not alert.
    CROWD_PERSIST_SECONDS = 60
    # ...AND average pairwise centroid distance among them must stay under
    # this many pixels — a real "crowd" implies proximity, not just a
    # headcount scattered across the whole frame (see behaviour_engine.py's
    # _average_pairwise_distance()).
    CROWD_MAX_AVG_DISTANCE_PX = 150

    # ── Overcrowding (behaviour_engine.py's find_largest_overcrowding_
    # cluster()/OvercrowdingState/update_overcrowding_state()) — a
    # SEPARATE, more detailed subsystem from Crowd Formation above
    # (deliberately left untouched): connected-components clustering by
    # centroid proximity (so a group doesn't need every pair mutually
    # close, just transitively linked — see find_overcrowding_clusters())
    # instead of Crowd Formation's single whole-frame average-pairwise-
    # distance number, plus dedicated per-camera hysteresis (same
    # "gap tolerance doesn't reset a real state" shape as Fighting's Gate
    # 4 — see OvercrowdingState's docstring) instead of Crowd Formation's
    # simpler continuous-persistence timer. OVERCROWDING_MIN_COUNT is the
    # minimum cluster size; OVERCROWDING_MAX_RADIUS_PX is the max centroid
    # distance between two DIRECTLY-linked members (not a group-wide
    # radius); OVERCROWDING_DWELL_SECONDS is how long a qualifying cluster
    # must persist (continuously, with gap tolerance) before CONFIRMED;
    # OVERCROWDING_PRETRIGGER_GAP_SECONDS is that gap tolerance, reused
    # symmetrically for both the pre-confirm accumulation phase and
    # post-confirm recovery. Values here are explicit request defaults,
    # not independently validated against real footage at these exact
    # numbers — same "starting point, tune against real footage" status as
    # this project's other rule thresholds.
    OVERCROWDING_MIN_COUNT = 5
    OVERCROWDING_MAX_RADIUS_PX = 150
    OVERCROWDING_DWELL_SECONDS = 3.0
    OVERCROWDING_PRETRIGGER_GAP_SECONDS = 1.0

    # Running is defined SCALE-INVARIANTLY, not by a raw pixel speed alone
    # (see behaviour_engine.py's is_running_speed()) — a fixed px/s number
    # means something completely different for a person filling a third of
    # the frame (close to camera) vs. a distant speck (wide/zoomed-out
    # view): the same real-world running speed produces very different
    # pixel speeds depending on camera distance/zoom. Instead:
    #
    #   1. PRIMARY: speed / bbox_height >= RUNNING_SPEED_BODY_HEIGHTS_PER_SEC
    #      ("how many of their own body-heights do they cover per second"),
    #      which stays roughly constant across camera distance/zoom for the
    #      same real gait. 1.5 is an engineering estimate from typical human
    #      gait speeds (adult walking ~1.2-1.4 m/s vs. a ~1.7m height ->
    #      ~0.7-0.8 heights/s; running ~3-4 m/s -> ~1.8-2.4 heights/s) — a
    #      STARTING POINT like every other threshold in this file, to be
    #      recalibrated against real footage once available, not a measured
    #      constant.
    #   2. FLOOR: raw speed must also clear RUNNING_MIN_ABS_SPEED_PX_S —
    #      guards against a tiny/distant/noisy bounding box trivially
    #      satisfying the body-relative ratio from a few pixels of jitter
    #      (a 10px-tall far-away box only needs ~15px/s to "qualify"
    #      relatively, which is well within ordinary detection jitter).
    #   3. Both are read from tracker.py's Track.speed, which is already
    #      NET displacement over a trailing ~1s window (not a sum of
    #      frame-to-frame deltas) — a person jittering/oscillating in place
    #      already reads as slow under this formula, so no separate
    #      directional-consistency check is needed on top.
    #
    # Lowered on request (2026-09-22, twice) to make Running trigger more
    # easily — 1.5 heights/s sat closer to a genuine sprint; 1.0 was closer
    # to the walk/jog boundary; 0.7 sits below that, closer to a brisk
    # walk, so even a purposeful fast walk now qualifies, not just a jog.
    # A deliberate sensitivity change, not a measurement — same "STARTING
    # POINT, recalibrate against real footage" status as before. Pushing
    # this much lower starts to risk flagging ordinary brisk walking as
    # "Running" — worth real-footage validation before going further.
    RUNNING_SPEED_BODY_HEIGHTS_PER_SEC = 0.7
    RUNNING_MIN_ABS_SPEED_PX_S = 25
    # Kept as a bare absolute fallback ONLY for a caller with no bbox height
    # available at all (e.g. legacy metadata) — see is_running_speed().
    RUNNING_SPEED_PX_S = 100
    # Speed must stay >= threshold CONTINUOUSLY for this long before Running
    # fires — previously unenforced (single qualifying frame fired
    # instantly), which let brief fast limb/body movement during an actual
    # fight masquerade as "Running" and, via the overlay's display priority,
    # visually hide a properly-confirmed Fighting detection.
    RUNNING_MIN_SECONDS = 1.0
    # Running "distance covered" — a genuine runner covers real ground from
    # wherever they were when they started qualifying, not just a momentary
    # speed reading sustained on paper. Measured as the SAME track's net
    # displacement (current position vs. that track's OWN recorded
    # start-of-run anchor point — its actual "running vector": one
    # continuous track's start->current motion, not a generic distance
    # check) — see BehaviourEngine._check_rules. Scale-invariant like the
    # speed check above, with an absolute-pixel floor for when no bbox is
    # available.
    RUNNING_MIN_DISTANCE_BODY_HEIGHTS = 0.5
    RUNNING_MIN_DISTANCE_PX = 20

    # Vandalism is a HYBRID gate (see ai_service/behaviour_engine.py's
    # check_vandalism()) — dwell time inside an "asset"-type CameraZone
    # ALONE was found to false-positive on ordinary use (e.g. someone
    # searching for their keys near a shelf), so it now ALSO requires the
    # CNN's own Kicking/Fighting probability (there is no trained
    # "Vandalism" class — see DEFAULT_ACTIVITY_CLASSES's docstring below)
    # to exceed VANDALISM_AGGRESSION_THRESHOLD, with BOTH conditions true
    # continuously for VANDALISM_MIN_SECONDS.
    VANDALISM_MIN_SECONDS = 3.0
    VANDALISM_AGGRESSION_THRESHOLD = 0.65

    # ── Fighting: 4-Gate scale-invariant detection ──────────────────────────
    # SUPERSEDED by the constants below (kept, not deleted, for historical
    # context — same "dormant, not destroyed" convention used elsewhere in
    # this file): fixed-pixel proximity/displacement thresholds don't work
    # across different camera distances/zoom levels the same real gap in
    # pixels means something completely different up close vs. far away.
    FIGHTING_PROXIMITY_TOLERANCE_PX = 30          # unused — see FIGHTING_PROXIMITY_MULTIPLIER
    FIGHTING_VELOCITY_TRAILING_WINDOWS = 4         # unused — see FIGHTING_MOTION_WINDOW_SECONDS
    FIGHTING_MIN_COMBINED_DISPLACEMENT_PX = 15     # unused — see FIGHTING_MOTION_HEIGHT_FRACTION

    # GATE 1 — Scale-invariant proximity (behaviour_engine.py's
    # find_fighting_pair()): passes if the pair's centroid distance is
    # <= FIGHTING_PROXIMITY_MULTIPLIER * their average bbox WIDTH, OR their
    # boxes overlap at all beyond a trivial amount (IoU > FIGHTING_IOU_MIN).
    # Both scale naturally with how large the pair appears in frame, unlike
    # a fixed pixel distance.
    FIGHTING_PROXIMITY_MULTIPLIER = 1.5
    FIGHTING_IOU_MIN = 0.02

    # GATE 2 — Kinetic motion (behaviour_engine.py's
    # _fighting_gate2_motion_ok()): ultra-permissive by design. Passes
    # whenever the pair's COMBINED centroid displacement over the last
    # FIGHTING_MOTION_WINDOW_SECONDS is above a near-zero ABSOLUTE pixel
    # floor (FIGHTING_MOTION_MIN_PX) — i.e. ANY non-zero localized
    # movement while in proximity, deliberately NOT scaled to bbox size
    # and deliberately NOT requiring large displacement. Rules out only
    # two people who are GENUINELY motionless close together (a
    # handshake/conversation, or pixel-rounding noise on a static pair);
    # everything else close-quarters (slow wrestling, grappling, a shove
    # with naturally small physical displacement) is left entirely for
    # Gate 3's neural re-inference (P(Fighting)>=0.70) to actually judge —
    # this gate is intentionally no longer where real filtering happens.
    # See _fighting_gate2_motion_ok's docstring for the full history of
    # what this gate used to check and why each stricter attempt was
    # walked back (bbox aspect-ratio jitter, then a constant-velocity
    # rejection, then a height-relative displacement fraction — each
    # measurably cost more real fight recall than it fixed for THIS gate
    # specifically, on real video evidence).
    FIGHTING_MOTION_WINDOW_SECONDS = 1.0
    FIGHTING_MOTION_MIN_PX = 2.0
    # unused — see _fighting_gate2_motion_ok's docstring point 3 (walked
    # back: a height-relative displacement fraction was still too strict
    # for slow, close-quarters struggles with naturally small combined
    # movement — replaced by the near-zero absolute floor above).
    FIGHTING_MOTION_HEIGHT_FRACTION = 0.5
    # unused — see _fighting_gate2_motion_ok's docstring point 1 (removed:
    # real chaotic footage showed YOLO box-merging during close proximity
    # creates fake aspect-ratio jitter on its own, indistinguishable from
    # genuine grappling).
    FIGHTING_ASPECT_JITTER_THRESHOLD = 0.3
    # unused — see _fighting_gate2_motion_ok's docstring point 2 and
    # behaviour_engine.py's _velocity_is_steady() (kept, not deleted): a
    # "reject constant-velocity/steady-heading motion" rejection rule was
    # tried in place of the displacement check above, to filter out two
    # people walking closely past each other without raising Gate 3's
    # confirm threshold. Walked back after real-video testing showed it
    # also suppressed genuine close-quarters/slow-developing fights, which
    # can look "steady" to a per-segment velocity check for the first
    # second or more — see this session's report for the real before/after
    # numbers. Kept here, documented, in case a future attempt wants to
    # revisit the exact thresholds rather than re-deriving them from
    # scratch.
    FIGHTING_VELOCITY_ANGLE_CHANGE_THRESHOLD_DEG = 35.0
    FIGHTING_VELOCITY_SPEED_CHANGE_RATIO_THRESHOLD = 0.5

    # GATE 3 — Neural confirmation on the SPECIFIC pair, not the whole
    # frame (see inference_engine.py's fighting-pair crop pipeline): once
    # Gates 1+2 pass, the union of both persons' boxes (padded by
    # FIGHTING_CROP_PADDING_FRACTION on each side for context) is cropped
    # from the raw frame, built into its own rolling SEQUENCE_LENGTH-frame
    # clip (same preprocessing as the whole-frame classifier), and run
    # through the SAME deployed checkpoint independently — a scene-level
    # Fighting probability can be dominated by activity elsewhere in frame;
    # this asks specifically "are THESE two people fighting". Passes at
    # >= FIGHTING_NEURAL_CONFIRM_THRESHOLD.
    FIGHTING_CROP_PADDING_FRACTION = 0.15
    FIGHTING_NEURAL_CONFIRM_THRESHOLD = 0.70

    # GATE 4 — Temporal persistence & hysteresis (behaviour_engine.py's
    # FightingPairState): a dedicated, self-contained state machine per
    # tracked pair (NORMAL -> CONFIRMED -> NORMAL), deliberately separate
    # from the shared ActivityStateMachine/CLASS_SPECIFIC_THRESHOLDS path
    # every other neural class still uses — same "own dedicated subsystem"
    # pattern as Fire/Smoke, Vandalism, Falling, Snatching. Gates 1+2+3 must
    # all pass CONTINUOUSLY for FIGHTING_CONFIRM_SECONDS before CONFIRMED.
    # Once CONFIRMED, recovery back to NORMAL requires EITHER the neural
    # probability staying below FIGHTING_RECOVERY_PROB_THRESHOLD for
    # FIGHTING_RECOVERY_PROB_SECONDS, OR Gate 1 (proximity) failing
    # continuously for FIGHTING_PROXIMITY_FAIL_SECONDS — whichever happens
    # first. Because the recovery probability floor (0.40) is deliberately
    # lower than the confirm/Gate-3 floor (0.70), a brief dip anywhere
    # between the two (or even briefly below 0.40, as long as it's under
    # FIGHTING_RECOVERY_PROB_SECONDS) never resets the confirmation on its
    # own — this is what "a one-second shove doesn't end the fight" means
    # in practice, without a separate special-cased timer for it.
    FIGHTING_CONFIRM_SECONDS = 2.0
    FIGHTING_RECOVERY_PROB_THRESHOLD = 0.40
    FIGHTING_RECOVERY_PROB_SECONDS = 3.0
    FIGHTING_PROXIMITY_FAIL_SECONDS = 2.0
    # Real per-frame tracker jitter (a single ~0.1s sample where Gates
    # 1+2 or Gate 3 briefly drop, e.g. one occluded window mid-scuffle)
    # was observed, on real chaotic multi-person footage, to reset the
    # PRE-confirm accumulation timer to zero every time it happened —
    # even with a genuine, visibly continuous fight and a sustained
    # >=0.85 Gate-3 probability the whole time, no single window's worth
    # of jitter ever let the full 2.0s continuous requirement be met.
    # This mirrors the exact same "a brief interruption doesn't undo
    # what's already been observed" principle the 4-gate spec already
    # applies to the POST-confirm recovery phase above (a <1.0s dropout
    # below 0.70 doesn't reset a CONFIRMED fight) — applied here to the
    # pre-confirm phase with the same 1.0s tolerance, rather than
    # inventing a new threshold. The elapsed time during the gap still
    # counts toward FIGHTING_CONFIRM_SECONDS; only a gap of >=1.0s
    # actually restarts the timer from scratch.
    FIGHTING_PRETRIGGER_GAP_SECONDS = 1.0

    # Falling: geometric + motionlessness gate (see behaviour_engine.py's
    # check_falling_candidate()) — a tracked person's bounding-box aspect
    # ratio (h/w) must drop below FALLING_ASPECT_RATIO_MAX (a horizontal
    # posture) and then stay within FALLING_MOTIONLESS_PX of where it was
    # when that posture first appeared, continuously, for
    # FALLING_MOTIONLESS_SECONDS — this is what distinguishes an actual
    # collapse from someone bending down briefly (tying a shoe, picking
    # something up), who straightens back up or keeps moving.
    FALLING_ASPECT_RATIO_MAX = 0.8
    FALLING_MOTIONLESS_PX = 5
    FALLING_MOTIONLESS_SECONDS = 5.0
    # Explicit floor on the CNN's own Falling probability, applied in
    # inference_engine.py BEFORE the state machine ever sees it. Matches
    # Falling's own confirm_threshold override in CLASS_SPECIFIC_THRESHOLDS
    # below (also 0.75, down from the global 0.80) — that override is what
    # makes this value the one actually governing confirmation, not just a
    # pre-filter sitting in front of a stricter global bar.
    FALLING_CNN_THRESHOLD = 0.75

    # Snatching: contact + departure gate (see behaviour_engine.py's
    # check_snatch_divergence()) — Person A and B must first be within
    # SNATCHING_CONTACT_PX of each other (contact), then one of them must
    # depart at >= SNATCHING_DEPARTURE_PX_S (measured over a short trailing
    # window, like tracker.py's Track.speed, not a single-frame delta)
    # within SNATCHING_DEPARTURE_WINDOW_SECONDS of that contact — the
    # "grab and run" signature a handshake or a passing brush doesn't
    # produce.
    SNATCHING_CONTACT_PX = 15
    # The spec expresses this as "12px per analysis window" — converted to
    # px/s (the unit tracker.py's Track.speed already uses) at this
    # pipeline's live AI_ANALYSIS_FPS (6): 12 * 6 = 72.
    SNATCHING_DEPARTURE_PX_S = 72
    # "Immediately following contact" is not itself numerically specified —
    # this is a deliberate, disclosed interpretation, not a measured value.
    SNATCHING_DEPARTURE_WINDOW_SECONDS = 1.5
    # Explicit floor on the CNN's own Snatching probability, applied in
    # inference_engine.py BEFORE the state machine ever sees it. Matches
    # Snatching's own confirm_threshold override in CLASS_SPECIFIC_THRESHOLDS
    # below (also 0.65) — NOTE this sits BELOW the global ACTIVITY_START_
    # THRESHOLD (0.70): since NORMAL->ANALYZING still requires clearing 0.70
    # first (that transition is not class-overridable — see
    # _ClassActivityMachine), by the time Snatching ever reaches ANALYZING
    # its smoothed value has already cleared 0.65 too. The practical effect
    # is that ANALYZING->CONFIRMED for Snatching is governed almost entirely
    # by min_consecutive_confirm_windows (persistence), not by this
    # confirm_threshold independently — disclosed here rather than silently
    # implemented as if the two bars were operating independently.
    SNATCHING_CNN_THRESHOLD = 0.65

    # Minimum neural-classifier confidence worth showing on the overlay at
    # all, as a non-committal "AI:" hint. 10-way random guessing averages
    # 10%, so 30%+ is a meaningfully above-chance signal — still shown with
    # its exact confidence, never phrased as confirmed. Escalating to an
    # actual alert requires clearing REVIEW_THRESHOLD instead (see
    # inference_engine.py), a much higher bar.
    NEURAL_DISPLAY_THRESHOLD = 0.30

    # ── Activity hysteresis state machine (per ActivityStateMachine) ────────
    # Every value below is a STARTING POINT to be evaluated against real
    # validation footage (see ai_service/training/evaluate_thresholds.py),
    # never claimed as universally correct. They exist so a single noisy
    # frame can't flip the overlay between "Normal" and "CONFIRMED FIGHTING"
    # — see the NORMAL -> ANALYZING -> CONFIRMED -> RECOVERY -> NORMAL state
    # machine in ai_service/activity_state.py.
    ACTIVITY_START_THRESHOLD   = 0.70  # Smoothed prob crossing this: NORMAL -> ANALYZING
    ACTIVITY_CONFIRM_THRESHOLD = 0.80  # Smoothed prob sustained above this: ANALYZING -> CONFIRMED
    ACTIVITY_END_THRESHOLD     = 0.55  # Smoothed prob sustained below this: CONFIRMED -> RECOVERY -> NORMAL
    ACTIVITY_EMA_ALPHA         = 0.30  # smoothed = alpha*raw + (1-alpha)*prev_smoothed
    MIN_CONSECUTIVE_CONFIRM_WINDOWS = 3  # Windows >= CONFIRM required before state becomes CONFIRMED
    MIN_CONSECUTIVE_END_WINDOWS     = 3  # Windows < END required before state leaves CONFIRMED

    # A visual "CONFIRMED" overlay is NOT a security alert — this is the
    # separate, higher bar an activity must clear (smoothed confidence, with
    # persistence already satisfied by the state machine) before an Alert
    # row / notification is actually generated. See section 22-23 of the
    # alerting spec this implements.
    ALERT_THRESHOLD = 0.80

    TRACK_LOST_TIMEOUT = 2.0  # Seconds a track can go unseen before its activity state is discarded

    # ── Class-specific temporal calibration overrides (see activity_state.py)
    # Every activity class has different temporal characteristics — a brief
    # violent burst (Fighting/Kicking/Snatching) should confirm fast so a
    # 3-4 second event isn't missed waiting out the slower global default; a
    # collapse (Falling) is nearly instantaneous and should confirm on the
    # very next window once seen, but stays visibly "fallen" for longer
    # (slower to end) since the person remains down. Any field omitted for a
    # class falls back to the matching global ACTIVITY_* constant above.
    # Loitering/Running/Crowd Formation are handled entirely by the
    # deterministic rules in behaviour_engine.py / tracker.py (dwell-time,
    # speed, person-count) — not by this neural classifier at all — and
    # Fire/Smoke are a fully separate detection subsystem (see below); none
    # of the five are trained classes here, so none need an entry.
    # STARTING POINTS, same as every other threshold in this file — recalibrate
    # against ai_service/training/evaluate_thresholds.py once enough
    # per-class validation footage exists.
    #
    # NOTE on Fighting/Kicking/Snatching specifically: a faster `alpha`
    # override (0.55, vs. the global 0.30) was tried after diagnosing a real
    # 40s CCTV fighting video where the classifier's genuine, frequent
    # high-confidence (0.85-0.99) Fighting spikes were being damped below
    # CONFIRM_THRESHOLD by the slower global EMA. It DID let that video's
    # fight confirm — but re-tested against a 49s all-normal video
    # (10 concatenated Normal training clips), the same faster alpha also
    # let the classifier's own spurious high-confidence Fighting spikes on
    # completely normal footage (raw values up to 0.998 were observed) reach
    # CONFIRMED 3 separate times. That is a real classifier reliability
    # problem on out-of-distribution footage (see PART 8/13 of the training
    # pipeline audit), not something a hysteresis/EMA tuning knob can fix
    # without directly trading missed fights for false "fighting" alerts on
    # normal video — an unacceptable trade explicitly ruled out. `alpha` is
    # therefore left at the global default for these classes; only the
    # window-count values below are overridden.
    #
    # Fighting's entry below is now DORMANT, not deleted: Fighting has its
    # own dedicated 4-Gate subsystem (see FIGHTING_CONFIRM_SECONDS etc.
    # above and behaviour_engine.py's FightingPairState) that fully bypasses
    # the shared ActivityStateMachine — inference_engine.py unconditionally
    # zeroes behaviour_probs["Fighting"] before it ever reaches
    # session.activity_state.update(), so this override can never actually
    # confirm anything anymore. Kept for historical reference/in case the
    # dedicated subsystem is ever disabled.
    #
    # Falling's confirm_threshold is explicitly overridden to 0.75 (down
    # from the global 0.80) and Snatching's to 0.65, matching
    # FALLING_CNN_THRESHOLD / SNATCHING_CNN_THRESHOLD above exactly — these
    # were previously just pre-filters sitting in front of a stricter,
    # unmodified global confirm bar (0.80), which made them redundant in
    # practice. With the override, the state machine's own confirmation
    # actually runs at these values, not the global default.
    CLASS_SPECIFIC_THRESHOLDS = {
        "Fighting":        {"confirm_threshold": 0.70, "min_consecutive_confirm_windows": 8, "min_consecutive_end_windows": 3},  # dormant, see above
        "Kicking":         {"min_consecutive_confirm_windows": 2, "min_consecutive_end_windows": 3},
        "Snatching":       {"confirm_threshold": 0.65, "min_consecutive_confirm_windows": 2, "min_consecutive_end_windows": 3},
        "Falling":         {"confirm_threshold": 0.75, "min_consecutive_confirm_windows": 1, "min_consecutive_end_windows": 5},
    }

    # ── Training-time temporal windowing (see ai_service/training/dataset.py)
    # A raw training video longer than TRAIN_SHORT_CLIP_MAX_SECONDS is treated
    # as an untrimmed/long recording: instead of one 16-frame sample spread
    # across its ENTIRE duration (which can dilute or completely miss a brief
    # event), it is split into multiple overlapping windows, each becoming
    # its own training sample carrying the same video-level label. Videos at
    # or under this length are assumed to already BE the event (e.g. a
    # pre-trimmed dataset clip) and keep the original whole-clip sampling.
    TRAIN_SHORT_CLIP_MAX_SECONDS = 6.0
    TRAIN_WINDOW_SECONDS = 4.0
    TRAIN_WINDOW_STRIDE_SECONDS = 2.0
    TRAIN_MAX_WINDOWS_PER_VIDEO = 6

    # ── Behaviour event lifecycle persistence (Phase 2) ─────────────────────
    # While a neural/behaviour event stays open (CONFIRMED/RECOVERY), its DB
    # row's peak/average/duration is refreshed at most this often — not
    # every analysis window — so a long-running event doesn't hammer the
    # database, while still keeping the persisted numbers reasonably live
    # for anyone viewing it before it ends. See ai_service/inference_engine.py
    # (the neural/behaviour event lifecycle block) and
    # backend/app/services/alert_dispatcher.py (dispatch_event_update_sync).
    BEHAVIOUR_EVENT_REFRESH_SECONDS = 3.0

    # ── Uploaded-video full-timeline analysis (Phase 2) ─────────────────────
    # A dedicated mode (see ai_service/video_timeline_processor.py) that
    # decodes an uploaded video file start-to-finish exactly once, running
    # the SAME analyze() pipeline live cameras use at a fixed, bounded
    # sampling rate — never paced to real-time, never looping — so the
    # entire file gets covered regardless of how slow inference is on CPU.
    # Independent of AI_ANALYSIS_FPS (which paces LIVE streaming): this
    # governs offline batch throughput/density instead.
    UPLOAD_ANALYSIS_FPS = 8

    # Not every class can be reliably pinned to one individual — Fighting/
    # Kicking/Snatching inherently involve >=2 people and the classifier
    # judges the whole clip, not a crop of one person (see module docstring
    # in inference_engine.py: UCF-Crime-style temporal supervision, not a
    # per-person-box dataset). Scene-level activities get a "SCENE ACTIVITY"
    # banner + highlighted group instead of a single attributed box.
    # "kicking"/"falling" are DORMANT as of v16.0.0 (see DEFAULT_ACTIVITY_
    # CLASSES's docstring) — no checkpoint currently emits those class
    # names, so they never match, but the entries are kept rather than
    # deleted.
    PERSON_LEVEL_ACTIVITIES = {"falling"}
    SCENE_LEVEL_ACTIVITIES  = {"fighting", "kicking", "snatching"}

    # Verbose per-frame diagnostics (FPS, track IDs, raw vs smoothed
    # confidence, state, thresholds, device) — off by default, never meant
    # for the resident/guard-facing UI.
    DEBUG_OVERLAY = False

    # Event recording
    PRE_EVENT_SECONDS = 5
    POST_EVENT_SECONDS = 10

    # ── Fire/Smoke — a SEPARATE detection subsystem from the UCF-Crime-style
    # behaviour classifier above. Never share thresholds/state with
    # ACTIVITY_*/PERSON_DETECTION_CONFIDENCE — a fire alarm's evidence
    # requirements have nothing to do with a person-detection confidence or
    # a behaviour-classifier calibration, and conflating them would mean
    # tuning one silently changes the other. See ai_service/detector.py
    # (FireSmokeDetector — HSV segmentation + optical-flow turbulence gate,
    # NOT the neural activity classifier) and ai_service/fire_smoke_state.py.
    #
    # NOTE: FireSmokeDetector's own internal fire/smoke confidence formula
    # (fill-ratio + solidity based) already only produces values roughly in
    # 0.6-0.98 for fire and 0.5-0.92 for smoke once its shape/motion gates
    # pass — DETECTION_THRESHOLD below is applied on top of that, so it is
    # deliberately not the same numeric scale as a raw sigmoid/softmax
    # output from a trained classifier would be. Passed into
    # FireSmokeDetector's constructor (ai_service/inference_engine.py's
    # CameraInferenceSession) as fire_threshold/smoke_threshold.
    FIRE_DETECTION_THRESHOLD  = 0.82  # Minimum single-frame confidence to even count as a raw candidate
    # Lowered from 0.68 after diagnosing a real production false negative:
    # a visually-confirmed, actively-venting smoke plume (320x240 source
    # video) produced a correctly-shaped, correctly-colored, correctly-
    # sized contour whose confidence formula evaluated to 0.655
    # (fill_ratio=0.28, solidity=0.55) — just under the old 0.68 gate. Real
    # diffuse smoke naturally has a lower fill_ratio than a tightly-shaped
    # object (it doesn't fill its own bounding box as compactly), so the
    # formula structurally under-rewards genuine large plumes; 0.60 gives
    # real smoke of this shape room to pass while still requiring the
    # shape/motion gates above to have passed first.
    SMOKE_DETECTION_THRESHOLD = 0.60
    # START threshold — smoothed confidence required, sustained for
    # FIRE_SMOKE_MIN_CONFIRM_WINDOWS consecutive windows, to open a new
    # event. Doubles as the "worth alerting" bar (is_alertable).
    #
    # Each of these must be checked against its own detector's internal
    # per-frame gate (FIRE_DETECTION_THRESHOLD / SMOKE_DETECTION_THRESHOLD
    # above) — the detector only ever EMITS a confirmed detection whose raw
    # confidence already cleared that internal gate, so this START
    # threshold is comparing smoothed values against a floor that's already
    # been filtered once. If this is set HIGHER than the detector's own
    # gate, a genuine dead zone opens up: real, correctly-identified
    # smoke/fire that the detector confirms (because it passed the color/
    # shape/motion checks) can still never make the state machine start
    # climbing toward CONFIRMED, because its own confidence never happens
    # to clear this second, stricter bar.
    #
    # FIRE_ALERT_THRESHOLD=0.70 vs FIRE_DETECTION_THRESHOLD=0.82: no gap —
    # any detector-confirmed fire already exceeds this trivially.
    #
    # SMOKE_ALERT_THRESHOLD was 0.70 vs SMOKE_DETECTION_THRESHOLD=0.60 — a
    # real 0.10 dead zone. Measured directly against a real, visually-
    # confirmed smoke video (backend/storage/videos/uploaded_cameras/
    # cam_e4b66c3b.avi): every detector-confirmed smoke frame across the
    # whole clip fell in the 0.63-0.81 range (mean 0.71) — i.e. a
    # meaningfully weaker/more-distant real plume that still correctly
    # passes the detector's own validation could plausibly plateau in the
    # 0.60-0.70 band and never once reach the old 0.70 start bar, showing
    # as "detected" in the raw signal but never reaching a CONFIRMED alert.
    # Lowered to close that gap, matching (not undercutting) the
    # detector's own already-validated confidence floor — not a blind
    # "lower it until something fires" change.
    # OPERATOR CHOICE: alert as soon as fire/smoke is detected at all, rather
    # than after sustained confirmation — thresholds lowered to ~detector
    # floor and confirm windows set to 1 (see FIRE_SMOKE_MIN_CONFIRM_WINDOWS).
    # Trade-off: more sensitive, so brief false detections can now alert.
    FIRE_ALERT_THRESHOLD  = 0.30
    SMOKE_ALERT_THRESHOLD = 0.30
    # END threshold — deliberately LOWER than the START threshold above.
    # An open event only starts its closing grace period once smoothed
    # confidence drops below THIS (lower) bar, not merely below the START
    # bar again. Without this gap, confidence oscillating right around one
    # single threshold (e.g. 0.68, 0.71, 0.69, 0.72) would flap the event
    # open/closed every window. START > END is the actual mechanism that
    # prevents that — see FireSmokeStateMachine.
    FIRE_END_THRESHOLD  = 0.20
    SMOKE_END_THRESHOLD = 0.20
    # Higher than the behaviour system's ACTIVITY_EMA_ALPHA (0.30) — the
    # fire/smoke DETECTOR (see detector.py) already applies its own internal
    # 6-of-8-frame vote before ever emitting a nonzero confidence, so what
    # arrives here is already once-filtered evidence, not a raw per-frame
    # value. A slow-converging EMA on top of an already-conservative signal
    # meant real, sustained smoke plateauing at ~0.70-0.79 raw for only a
    # handful of frames never reached a smoothed value high enough to
    # confirm — verified against real smoke footage (see test_fire_smoke.py
    # output). 0.50 tracks genuinely sustained evidence within a few
    # windows while still damping single-frame noise.
    FIRE_SMOKE_EMA_ALPHA = 0.50
    # Consecutive windows >= the START threshold required to open a new
    # event — matches the behaviour pipeline's naming convention
    # (min_consecutive_confirm_windows) applied to fire/smoke specifically.
    # Raised 3 -> 12 (at AI_ANALYSIS_FPS=6 this is ~2s of sustained evidence
    # instead of ~0.5s): 3 windows let a 1-second lighter flare / headlight
    # reflection / momentary color-and-shape coincidence confirm as a full
    # event. Requiring ~2s of sustained evidence before CONFIRMED is a
    # deliberate anti-spam tightening, not a detector accuracy claim —
    # real, ongoing fire/smoke stays lit for far longer than 2s, so this
    # only costs a small amount of confirmation latency on genuine events.
    FIRE_SMOKE_MIN_CONFIRM_WINDOWS = 1
    FIRE_CONFIRM_WINDOWS  = FIRE_SMOKE_MIN_CONFIRM_WINDOWS
    SMOKE_CONFIRM_WINDOWS = FIRE_SMOKE_MIN_CONFIRM_WINDOWS
    # TIME-based grace period (NOT a window/frame count) between confidence
    # dropping below the END threshold and the event actually closing.
    # Critical for uploaded-video processing (ai_service/
    # video_timeline_processor.py), which can run much faster or slower
    # than real-time: a frame-count grace corresponds to a wildly different
    # amount of actual video time depending on processing speed, while a
    # SECONDS-based grace means "genuinely absent for 5 seconds of video
    # time" regardless of how fast the batch job is crunching frames. This
    # replaces the previous FIRE_SMOKE_MIN_END_WINDOWS (a window count,
    # ~1s at AI_ANALYSIS_FPS=6 for live streaming but a meaningless amount
    # of real/video time for a fast-running batch job).
    # Lowered 5.0 -> 4.0: a shorter grace window closes a genuinely-ended
    # event (and lets a real re-ignition open a fresh one, subject to
    # RETRIGGER_COOLDOWN_SECONDS below) slightly sooner. Does not affect
    # false-positive rate — grace only extends an ALREADY-CONFIRMED event's
    # tail, it cannot cause a new one to start.
    END_GRACE_SECONDS = 4.0
    # Applied strictly AFTER an event fully closes (event_ended) — a
    # SEPARATE concept from the active-event dedup (which is structural:
    # no new alert is ever created while state stays CONFIRMED/RECOVERY,
    # regardless of any cooldown). If evidence reappears within this window
    # of a close, it's treated as a continuation of the SAME incident
    # (reopens the same DB row) rather than a brand-new alert — this is
    # what actually stops a fast confirm/close/confirm flicker cycle from
    # creating a new row every cycle, which a bare "don't alert more than
    # once every N seconds" cooldown (the old FIRE_SMOKE_ALERT_COOLDOWN_
    # SECONDS role) could not correctly express since it was keyed off
    # alert CREATION time, not the event's actual CLOSE time.
    RETRIGGER_COOLDOWN_SECONDS = 30
    # Pre/post-event clip buffer specific to fire/smoke incidents (spec
    # section 17) — independently configurable from PRE/POST_EVENT_SECONDS
    # above, which the behaviour pipeline uses.
    FIRE_SMOKE_PRE_EVENT_SECONDS = 5
    FIRE_SMOKE_POST_EVENT_SECONDS = 5

    # Activity classes — must match the DEPLOYED checkpoint's own
    # `activity_classes` field exactly (inference_engine.py reads classes
    # FROM the checkpoint at load time, never from this constant directly —
    # see InferenceEngine._apply_checkpoint). This is only the fallback
    # default used before any checkpoint has loaded, and the class list new
    # training runs are built against.
    #
    # DELIBERATELY 2 CLASSES ONLY as of v16.0.0 (2026-09-22): Normal,
    # Fighting. Kicking/Snatching/Falling were DROPPED, not merged or
    # relabeled — a direct audit of the only two datasets this project is
    # permitted to train the behaviour classifier from (backend/storage/
    # datasets/RWF-2000-dataset: Fight/NonFight only; backend/storage/
    # datasets/ucf-dataset: Fighting/NormalVideos/RoadAccidents/Shooting/
    # Vandalism only) found ZERO legitimate examples of Kicking, Snatching,
    # or Falling in either — no folder, no label, nothing to honestly train
    # those classes from. Fabricating them (e.g. assuming "Fighting" clips
    # contain kicks) was explicitly ruled out. See ai_service/training/
    # build_behaviour_manifest.py and train_behaviour_v2.py for the full
    # from-scratch training pipeline and dataset audit.
    #
    # The OLD 5-class checkpoint (suspicious_activity_v15.0.0.pt) is backed
    # up at backend/storage/behaviour_backup_old/, not deleted, and remains
    # loadable on its own terms (a checkpoint's own activity_classes always
    # governs, regardless of this constant) if ever redeployed.
    #
    # CLASS_SPECIFIC_THRESHOLDS / PERSON_LEVEL_ACTIVITIES /
    # SCENE_LEVEL_ACTIVITIES below still list Kicking/Snatching/Falling —
    # left in place deliberately, not deleted, since removing them would
    # mean also ripping out the (separately validated, still-correct)
    # Falling/Snatching gating code in behaviour_engine.py/inference_
    # engine.py. With no checkpoint ever emitting those class names again,
    # `behaviour_probs.get("Falling"/"Snatching", 0.0)` is always 0.0, so
    # those gates are simply dormant, not deleted — ready to reactivate the
    # moment real training data exists for them, without rewriting any gate
    # logic.
    #
    # Loitering/Running/Crowd Formation/Vandalism are handled by
    # deterministic rule-based systems instead (dwell-time, tracker speed,
    # person-count, and asset-zone-dwell+aggression respectively — see
    # ai_service/behaviour_engine.py), never trained here. Fire/Smoke are a
    # fully separate detection subsystem (ai_service/detector.py's
    # FireSmokeDetector + ai_service/fire_smoke_state.py) that must never be
    # trained as part of this behaviour classifier. A checkpoint trained
    # under one class list is NOT index-compatible with one trained under a
    # different list — every training script here always trains from
    # scratch rather than fine-tuning across a class-list change.
    DEFAULT_ACTIVITY_CLASSES = [
        "Normal",
        "Fighting",
    ]

    # Severity mapping. Kicking/Snatching/Falling entries kept (dormant, not
    # deleted — see DEFAULT_ACTIVITY_CLASSES above) so they're ready to use
    # again without rewriting this dict once real training data exists.
    SEVERITY_MAP = {
        "Fighting":        "critical",
        "Kicking":         "critical",
        "Snatching":       "critical",
        "Falling":         "critical",
        "Normal":          "info",
        # Rule-based (BehaviourEngine sets DetectionResult.severity directly,
        # so this entry isn't actually consulted for it) — kept here so
        # Vandalism's severity is documented in the same place as every
        # other activity's.
        "Vandalism":       "critical",
    }
