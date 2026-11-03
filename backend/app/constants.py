"""
Shared constants for categorizing events/alerts without a schema migration.

Fire/Smoke alerts and behaviour alerts already use unambiguous, disjoint
`activity` string values (see ai_service/alert_generator.py and
app/services/alert_dispatcher.py) vs. behaviour names like "Fighting" /
"Restricted Area Entry". Rather than add a new `event_type` column (Alert
and SecurityEvent already carry model_version, confidence, camera,
duration, clip, thumbnail, severity, timestamp, status — see spec
section 22), this set lets every endpoint filter reliably off the field
that already exists.

Fire and smoke are combined into ONE application-level activity —
"Fire/Smoke Detected" — fire OR smoke OR both all map to it (see
ai_service/inference_engine.py's combined-event construction); the
underlying detector still tracks fire/smoke independently, and which
one(s) triggered a given alert is preserved in its
SecurityEvent.metadata_json (fire_detected/smoke_detected), not lost.
The three older, pre-combination labels are kept in this set (not
removed) purely so historical rows already in the database — created
before this change — still correctly categorize as "fire_smoke" in
existing alert/incident history views; no new alert will ever use them
again.
"""
FIRE_SMOKE_ACTIVITIES = {
    "Fire/Smoke Detected",
    "Fire Detected", "Smoke Detected", "Fire + Smoke Detected",  # legacy labels — historical rows only
}


def is_fire_smoke_activity(activity: str) -> bool:
    return activity in FIRE_SMOKE_ACTIVITIES


def event_category(activity: str) -> str:
    """'fire_smoke' or 'behaviour' — used as the `category` field on
    AlertResponse/IncidentResponse so the frontend never has to guess by
    matching activity name substrings itself."""
    return "fire_smoke" if is_fire_smoke_activity(activity) else "behaviour"


# The detection classes this project actually produces today — the deployed
# behaviour model (Normal/Fighting) plus the rule-based detectors and the
# fire/smoke subsystem. Analytics reports only these: classes the project
# dropped (Kicking/Snatching/Falling), and non-detection rows such as a
# resident's manual report, are deliberately left out.
PROJECT_ACTIVITY_CLASSES = (
    "Fighting",
    "Loitering",
    "Repeat Loitering",
    "Running",
    "Crowd Formation",
    "Overcrowding",
    "Vandalism",
    "Unauthorized Entry",
    "Fire/Smoke Detected",
)
