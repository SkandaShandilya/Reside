"""
Alert Generator — Creates alerts with deduplication and cooldown.

Prevents alert flooding by implementing:
1. Cooldown periods per camera per activity
2. Incident continuation tracking
3. Confidence-based routing (high → alert, low → review queue)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time


@dataclass
class AlertEvent:
    """An alert ready to be sent to the backend."""
    camera_id: int
    camera_name: str
    camera_location: str
    activity: str
    confidence: float
    severity: str
    persons_detected: int
    duration_seconds: float
    send_to_review: bool = False
    timestamp: datetime = None
    # Event-lifecycle fields (Phase 2 — timeline-aware events). Optional and
    # None by default so callers that genuinely have no per-window evidence
    # history (rule-based behaviour_engine detections, which are single-shot
    # position checks with no smoothed confidence to peak/average) don't
    # have to fabricate a value — only the ActivityStateMachine-driven path
    # (ai_service/inference_engine.py's neural/behaviour event block)
    # currently populates these.
    peak_confidence: Optional[float] = None
    average_confidence: Optional[float] = None
    start_time: Optional[float] = None  # epoch seconds — see ActivityStateResult.start_time
    # Optional internal detail that would otherwise be lost once several
    # underlying signals are combined into one application-level activity
    # string (e.g. Fire/Smoke Detected's fire_detected/smoke_detected —
    # see ai_service/inference_engine.py's combined_metadata). None for
    # every other alert type, which has nothing to preserve here. Threaded
    # through to SecurityEvent.metadata_json by alert_dispatcher.py — no
    # schema migration needed, that column already exists and was unused.
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_location": self.camera_location,
            "activity": self.activity,
            "confidence": self.confidence,
            "severity": self.severity,
            "persons_detected": self.persons_detected,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "peak_confidence": self.peak_confidence,
            "average_confidence": self.average_confidence,
            "metadata": self.metadata,
        }


class AlertGenerator:
    """
    Generates alerts with deduplication.
    
    Cooldown logic:
    - After an alert is generated for camera X + activity Y,
      subsequent detections within the cooldown period are suppressed
    - The incident continues to be tracked for duration measurement
    """

    def __init__(self, cooldown_seconds: int = 30, confidence_threshold: float = 0.80):
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold
        self.review_threshold = 0.50
        
        # Tracks: (camera_id, activity) -> last_alert_timestamp
        self._cooldown_map: Dict[Tuple[int, str], float] = {}

    def should_generate_alert(
        self,
        camera_id: int,
        activity: str,
        confidence: float,
    ) -> bool:
        """Check if an alert should be generated (respecting cooldowns)."""
        if confidence < self.review_threshold:
            return False
        
        key = (camera_id, activity)
        now = time.time()
        
        last_alert_time = self._cooldown_map.get(key, 0)
        if now - last_alert_time < self.cooldown_seconds:
            return False
        
        return True

    def generate_alert(
        self,
        camera_id: int,
        camera_name: str,
        camera_location: str,
        activity: str,
        confidence: float,
        severity: str,
        persons_detected: int = 0,
        duration_seconds: float = 0,
        peak_confidence: Optional[float] = None,
        average_confidence: Optional[float] = None,
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AlertEvent]:
        """
        Generate an alert if not in cooldown.
        Returns AlertEvent or None if suppressed.
        """
        if not self.should_generate_alert(camera_id, activity, confidence):
            return None

        # Record the alert time
        key = (camera_id, activity)
        self._cooldown_map[key] = time.time()

        # Determine if this goes to review queue or direct alert
        send_to_review = confidence < self.confidence_threshold

        return AlertEvent(
            camera_id=camera_id,
            camera_name=camera_name,
            camera_location=camera_location,
            activity=activity,
            confidence=confidence,
            severity=severity,
            persons_detected=persons_detected,
            duration_seconds=duration_seconds,
            send_to_review=send_to_review,
            peak_confidence=peak_confidence,
            average_confidence=average_confidence,
            start_time=start_time,
            metadata=metadata,
        )

    def clear_cooldown(self, camera_id: int, activity: str = None):
        """Clear cooldown for a specific camera/activity."""
        if activity:
            self._cooldown_map.pop((camera_id, activity), None)
        else:
            # Clear all for camera
            keys_to_remove = [k for k in self._cooldown_map if k[0] == camera_id]
            for k in keys_to_remove:
                del self._cooldown_map[k]
