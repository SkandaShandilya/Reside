"""
Lightweight person Re-Identification (Re-ID) — Loitering's cross-session
"is this the same person coming back" memory layer.

Deliberately NOT a deep embedding network: a real Re-ID model (OSNet, etc.)
would be far more discriminative, but is a heavier dependency and a bigger
inference cost per tracked person per frame than this project's real-time
budget (see the RTX 5070 latency work elsewhere in this codebase) can
justify for a feature that only needs "probably the same person in the
same clothes within a ~15 minute window", not courtroom-grade identity
matching. A color-histogram descriptor of the upper/lower clothing crop is
cheap (a few cv2 calls, no GPU), and good enough for that bar — same
"lightweight, disclosed limitation" tradeoff as elsewhere in this project's
detection logic. It WILL be fooled by two different people wearing near-
identical outfits, and WON'T survive someone changing clothes; both are
accepted, disclosed limitations, not bugs.
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2

# HSV hue+saturation only (skip value/brightness) — brightness is the
# channel most sensitive to lighting changes between when someone left
# frame and when they might return, and clothing color/pattern is carried
# almost entirely by hue+saturation anyway.
HIST_BINS = (16, 16)
HIST_RANGES = [0, 180, 0, 256]


def _region_histogram(crop: np.ndarray) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), HIST_RANGES)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def extract_descriptor(frame: np.ndarray, bbox: List[float]) -> Optional[Dict[str, np.ndarray]]:
    """
    Builds a lightweight visual descriptor for the person in `bbox` — a
    normalized HSV hue+saturation histogram of their upper-body crop
    (roughly shirt/torso) and lower-body crop (roughly pants/legs),
    computed separately so a person in a light top + dark bottoms doesn't
    average out to a meaningless middle color. Returns None if the crop is
    degenerate (out of frame, zero-area) rather than raising — a Re-ID
    descriptor is optional evidence, never something that should crash the
    pipeline over a bad box on one frame.
    """
    if not bbox:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    mid_y = y1 + (y2 - y1) // 2
    upper_crop = frame[y1:mid_y, x1:x2]
    lower_crop = frame[mid_y:y2, x1:x2]

    upper_hist = _region_histogram(upper_crop)
    lower_hist = _region_histogram(lower_crop)
    if upper_hist is None and lower_hist is None:
        return None
    return {"upper_hist": upper_hist, "lower_hist": lower_hist}


def _hist_similarity(hist_a: Optional[np.ndarray], hist_b: Optional[np.ndarray]) -> Optional[float]:
    if hist_a is None or hist_b is None:
        return None
    # Correlation is in [-1, 1]; clip to [0, 1] since a real photographic
    # pair is never meaningfully anti-correlated and negative values would
    # otherwise just read as "very different", same as 0.
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, corr))


def descriptor_similarity(sig_a: Dict[str, np.ndarray], sig_b: Dict[str, np.ndarray]) -> float:
    """
    Combined similarity in [0, 1] between two descriptors from
    extract_descriptor() — the mean of the upper and lower region
    correlations, using whichever region(s) are available in BOTH
    descriptors (a degenerate crop on one side, e.g. someone half out of
    frame, doesn't have to invalidate the whole comparison). Returns 0.0
    if neither region is comparable.
    """
    upper_sim = _hist_similarity(sig_a.get("upper_hist"), sig_b.get("upper_hist"))
    lower_sim = _hist_similarity(sig_a.get("lower_hist"), sig_b.get("lower_hist"))
    sims = [s for s in (upper_sim, lower_sim) if s is not None]
    if not sims:
        return 0.0
    return sum(sims) / len(sims)
