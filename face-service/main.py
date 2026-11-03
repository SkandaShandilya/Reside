"""
Smart Apartment Security — Face & Plate Recognition Service
===========================================================
FastAPI service providing:
  POST /detect-and-embed   — InsightFace face embedding (UNCHANGED)
  POST /verify-plate       — Robust Indian number-plate OCR (redesigned)
  GET  /health             — Liveness check

OCR pipeline (redesigned — no merge-conflict markers remain):
  • Multi-signal plate-crop classifier (aspect ratio, edge density, border)
  • Multi-method plate candidate detection (Canny, threshold, morphological)
  • Perspective correction with configurable white-border padding
  • 9 preprocessing variants per crop (colour, grey, CLAHE, Otsu, sharpened,
    2× / 3× upscales)
  • PaddleOCR singleton (CPU, English)
  • Spatial line grouping of OCR tokens (bounding-box based)
  • Position-aware character ambiguity correction (NOT a global replace)
  • Candidate generation + multi-signal scoring + consensus selection
"""

import logging
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from paddleocr import PaddleOCR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("face-service")

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

face_app: Optional[FaceAnalysis] = None
paddle_ocr: Optional[PaddleOCR] = None


def get_face_app() -> FaceAnalysis:
    global face_app
    if face_app is None:
        logger.info("Initializing InsightFace model (buffalo_l) on CPU...")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        face_app = app
        logger.info("InsightFace model loaded successfully.")
    return face_app


def get_paddle_ocr() -> PaddleOCR:
    global paddle_ocr
    if paddle_ocr is None:
        logger.info("Initializing PaddleOCR (English, CPU)...")
        paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
        logger.info("PaddleOCR loaded successfully.")
    return paddle_ocr


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load both models on startup."""
    try:
        get_face_app()
    except Exception as e:
        logger.error(f"Failed to pre-load InsightFace during startup: {e}")
    try:
        get_paddle_ocr()
    except Exception as e:
        logger.error(f"Failed to pre-load PaddleOCR during startup: {e}")
    yield


app = FastAPI(
    title="Smart Apartment Security — Face & Plate Recognition Service",
    description=(
        "FastAPI service for face detection/embedding (InsightFace) "
        "and number-plate OCR (PaddleOCR)"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "face-recognition"}


# ===========================================================================
# Face Verification Endpoint — UNCHANGED
# ===========================================================================

@app.post("/detect-and-embed")
async def detect_and_embed(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided.",
        )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or unreadable image file. Please upload a valid image (JPEG, PNG, etc.).",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read or decode the uploaded image file.",
        )

    try:
        model = get_face_app()
        faces = model.get(img)
    except Exception as e:
        logger.error(f"Error during InsightFace detection/embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the face recognition model.",
        )

    face_count = len(faces)
    if face_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please upload a clear photo of a face.",
        )
    if face_count > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Multiple faces detected ({face_count}). Please upload an image containing exactly one face.",
        )

    face = faces[0]
    embedding = (
        face.embedding.tolist()
        if (hasattr(face, "embedding") and face.embedding is not None)
        else []
    )
    bbox = (
        [float(x) for x in face.bbox]
        if (hasattr(face, "bbox") and face.bbox is not None)
        else []
    )
    det_score = (
        float(face.det_score)
        if (hasattr(face, "det_score") and face.det_score is not None)
        else 0.0
    )

    return {
        "success": True,
        "face_detected": True,
        "face_count": 1,
        "embedding": embedding,
        "bbox": bbox,
        "det_score": det_score,
    }


# ===========================================================================
# Number-Plate OCR Pipeline — Fully Redesigned
# ===========================================================================

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OcrToken:
    text: str
    conf: float
    bbox: list          # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0

    def __post_init__(self):
        xs = [pt[0] for pt in self.bbox]
        ys = [pt[1] for pt in self.bbox]
        self.cx = (min(xs) + max(xs)) / 2.0
        self.cy = (min(ys) + max(ys)) / 2.0
        self.w = max(xs) - min(xs)
        self.h = max(ys) - min(ys)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "conf": round(self.conf, 4),
            "cx": round(self.cx),
            "cy": round(self.cy),
            "w": round(self.w),
            "h": round(self.h),
        }


@dataclass
class OcrVariantResult:
    variant_name: str
    tokens: List[OcrToken] = field(default_factory=list)

    @property
    def raw_texts(self) -> List[str]:
        return [t.text for t in self.tokens]


@dataclass
class PlateCandidate:
    crop: np.ndarray
    crop_type: str   # 'perspective_padN', 'bbox', 'direct'
    score: float


@dataclass
class ScoredCandidate:
    text: str
    score: float
    variant_count: int
    mean_conf: float
    sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Indian plate regex — used as ONE scoring signal, NOT an absolute gate
# ---------------------------------------------------------------------------

_INDIAN_PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$')

# Known Indian state codes for bonus scoring
_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TG", "TN", "TR", "TS", "UK", "UP", "UT", "WB",
}

# ---------------------------------------------------------------------------
# Character ambiguity map (OCR confusion pairs)
# ---------------------------------------------------------------------------

# Maps an OCR-output character to possible real alternatives.
# IMPORTANT: Substitutions are gated by positional class — only applied when
# the alternative matches the expected character class (alpha/digit/either).
_AMBIGUITY_MAP: Dict[str, List[str]] = {
    "O": ["0"],
    "0": ["O"],
    "I": ["1", "L"],
    "1": ["I", "L"],
    "L": ["1", "I"],
    "S": ["5"],
    "5": ["S"],
    "B": ["8"],
    "8": ["B"],
    "Z": ["2"],
    "2": ["Z"],
    "G": ["6"],
    "6": ["G"],
    "T": ["7"],
    "7": ["T"],
    "Q": ["0", "O"],
    "D": ["0"],
    # V and W are often confused with U by OCR at alpha positions.
    # Direction: if OCR reads V/W, consider U. NOT the reverse — if OCR
    # reads U, trust it (U is visually distinctive on plates).
    "V": ["U"],
    "W": ["U"],
}


def _normalize_plate(text: str) -> str:
    """Uppercase and strip all non-alphanumeric characters."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# ---------------------------------------------------------------------------
# Position-aware character class (for ambiguity correction)
# ---------------------------------------------------------------------------

def _get_expected_char_class(pos: int, total_len: int) -> str:
    """
    Given a character's 0-based position in a normalized plate string,
    return the expected character class: 'alpha', 'digit', or 'either'.

    Indian plate structure (approximate):
      [0-1]  : state code  → alpha
      [2-3]  : district    → digit  (may be 1 digit for old plates)
      [4-6]  : series      → alpha  (0-3 letters)
      [7-10] : number      → digit  (1-4 digits)

    For total lengths < 8 we fall back to 'either' for flexibility.
    """
    if total_len < 6:
        return "either"

    # Positions 0-1 are always alphabetic (state code)
    if pos <= 1:
        return "alpha"

    # Positions 2-3 are numeric (district code)
    if 2 <= pos <= 3:
        return "digit"

    # For plates with a 1-digit district (total length ~8-9)
    # pos 2 is digit, pos 3 onward could be alpha (series)
    # We use total_len to infer structure
    if total_len <= 9:
        # Typical: KA-01-AB-1234 (10 chars) or KA-1-A-1234 (8 chars)
        # Positions 4+ : could be alpha series then digits
        if pos <= 5:
            return "either"   # series letters or start of number
        return "digit"
    else:
        # Longer plate: more confident about structure
        if 4 <= pos <= 6:
            return "alpha"    # series letters
        return "digit"


def _generate_ambiguity_candidates(text: str) -> List[str]:
    """
    Given a normalized OCR text, generate alternative plate strings by
    substituting ambiguous characters based on their positional class.

    Rules:
    - Only substitute if the substitution produces a character that is
      appropriate for the expected class at that position.
    - Never substitute blindly (avoids turning valid digits into letters).
    - Cap at 32 alternatives to avoid combinatorial explosion.

    Returns a list including the original text.
    """
    candidates = {text}
    n = len(text)

    for i, ch in enumerate(text):
        if ch not in _AMBIGUITY_MAP:
            continue

        expected_class = _get_expected_char_class(i, n)
        alternatives = _AMBIGUITY_MAP[ch]

        new_candidates = set()
        for existing in list(candidates):
            for alt in alternatives:
                # Only substitute if alt matches the expected class
                if expected_class == "alpha" and not alt.isalpha():
                    continue
                if expected_class == "digit" and not alt.isdigit():
                    continue
                # 'either' → always allow

                new_text = existing[:i] + alt + existing[i + 1:]
                new_candidates.add(new_text)

                if len(candidates) + len(new_candidates) >= 32:
                    candidates.update(new_candidates)
                    return list(candidates)

        candidates.update(new_candidates)

    return list(candidates)


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four 2-D points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """Perspective-warp quadrilateral pts into a flat rectangle."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    if maxW < 10 or maxH < 10:
        return None

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def _perspective_crop_padded(
    img: np.ndarray,
    pts: np.ndarray,
    pad: int = 15,
) -> Optional[np.ndarray]:
    """
    Perspective-warp then add white border padding so OCR doesn't see
    characters touching the image edge (critical for K at left boundary).
    """
    warped = _four_point_transform(img, pts)
    if warped is None:
        return None
    return cv2.copyMakeBorder(
        warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


# ---------------------------------------------------------------------------
# Plate-type classifier
# ---------------------------------------------------------------------------

def _is_likely_plate_crop(img: np.ndarray) -> Tuple[bool, float]:
    """
    Determine whether the uploaded image itself is likely a plate crop,
    using multiple signals:
      1. Aspect ratio (plate-like: 1.2 – 7.5)
      2. Edge density (plates have many horizontal edges)
      3. Large outer rectangular border
      4. Colour uniformity (plates are mostly white/yellow)
      5. NOT using pixel-area as primary signal

    Returns (is_plate_crop: bool, confidence: float).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return False, 0.0

    ar = w / h
    score = 0.0

    # --- Signal 1: Aspect ratio ---
    if 1.2 <= ar <= 7.5:
        # Single-line Indian plates ~3-5, two-line ~1.5-3
        ar_score = 1.0 - min(abs(ar - 3.5) / 4.0, 1.0)
        score += ar_score * 0.30
    else:
        # Clearly not plate-like AR → strongly not a plate crop
        score -= 0.30

    # --- Signal 2: Edge density ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w)
    # Plates have moderate edge density (characters create edges)
    if 0.03 <= edge_density <= 0.35:
        score += 0.20
    elif edge_density > 0.35:
        # Too many edges → complex scene (full vehicle image)
        score -= 0.10

    # --- Signal 3: Large rectangular border ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_found = False
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            cx, cy, cw, ch_cnt = cv2.boundingRect(approx)
            cnt_ar = cw / (ch_cnt + 1e-6)
            fill_w = cw / (w + 1e-6)
            fill_h = ch_cnt / (h + 1e-6)
            # A plate border should fill most of the image and have plate-like AR
            if fill_w > 0.70 and fill_h > 0.70 and 1.1 <= cnt_ar <= 8.0:
                border_found = True
                break
    if border_found:
        score += 0.25

    # --- Signal 4: Relative colour uniformity (plate backgrounds are mostly
    #               one colour — white or yellow) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_mean = float(np.mean(hsv[:, :, 2]))
    # White plates: low saturation, high value
    if sat_mean < 60 and val_mean > 160:
        score += 0.15
    # Yellow plates: moderate-high saturation, high hue ~25-35
    hue_mean = float(np.mean(hsv[:, :, 0]))
    if 10 <= hue_mean <= 35 and sat_mean > 80 and val_mean > 150:
        score += 0.10

    # --- Signal 5: Minimum dimension sanity ---
    # Even large zoomed plates are typically taller than 40px
    if h >= 30 and w >= 60:
        score += 0.10

    is_crop = score >= 0.35
    confidence = min(max(score, 0.0), 1.0)
    logger.info(
        f"Plate-crop classifier: AR={ar:.2f} edge_density={edge_density:.3f} "
        f"border={border_found} score={score:.3f} → is_crop={is_crop}"
    )
    return is_crop, confidence


# ---------------------------------------------------------------------------
# Plate candidate detection (multi-method)
# ---------------------------------------------------------------------------

def _detect_plate_candidates(img: np.ndarray) -> List[PlateCandidate]:
    """
    Detect plate candidate crops from a full vehicle image.

    Uses three parallel detection methods:
      A. Bilateral filter + Canny (original method, good for clean images)
      B. Morphological close + Canny (better for faint plate borders)
      C. Simple adaptive threshold edges (robust to lighting variation)

    For each good contour, produces up to 4 geometric variants:
      1. Perspective crop + 15 px white padding
      2. Perspective crop + 30 px white padding
      3. Axis-aligned bbox crop (+10 % margins)
      4. Wider bbox crop (+20 % margins)

    Returns a deduplicated, score-sorted list of PlateCandidate objects.
    Aspect ratio acceptance: 1.0–7.5 (broad to catch tilted plates).
    """
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Method A: bilateral + Canny --
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges_a = cv2.Canny(filtered, 30, 200)

    # -- Method B: morphological close + Canny --
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
    edges_b = cv2.Canny(closed, 30, 200)

    # -- Method C: adaptive threshold --
    thresh_c = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_c = cv2.dilate(thresh_c, kernel_d, iterations=1)

    # Dilate A and B to connect nearby fragments
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_a = cv2.dilate(edges_a, kernel_dil, iterations=1)
    edges_b = cv2.dilate(edges_b, kernel_dil, iterations=1)

    all_contour_sets = [
        cv2.findContours(edges_a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0],
        cv2.findContours(edges_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0],
        cv2.findContours(edges_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
    ]

    scored_regions = []

    for contours in all_contour_sets:
        for contour in contours:
            area = cv2.contourArea(contour)
            # Plate: 0.3% – 35% of image area (broad range for zoomed vehicle shots)
            if area < img_area * 0.003 or area > img_area * 0.35:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            x, y, rw, rh = cv2.boundingRect(approx)
            if rh == 0 or rw == 0:
                continue

            ar = rw / rh
            # Accept broad AR range to handle angled / two-line plates
            if ar < 1.0 or ar > 7.5:
                continue

            # Score: prefer plate-like AR and reasonable area
            ar_single = 1.0 / (1.0 + abs(ar - 3.5))
            ar_double = 1.0 / (1.0 + abs(ar - 2.0))
            ar_score = max(ar_single, ar_double)
            area_score = min(area / img_area * 10.0, 1.0)
            score = ar_score * 0.65 + area_score * 0.35

            # Approximate polygon to 4 points for perspective transform
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                rect_min = cv2.minAreaRect(contour)
                pts = cv2.boxPoints(rect_min).astype(np.float32)

            scored_regions.append((score, x, y, rw, rh, pts, contour))

    if not scored_regions:
        return []

    # Sort by score and keep top candidates (limit OCR calls)
    scored_regions.sort(key=lambda c: c[0], reverse=True)
    top_regions = scored_regions[:5]

    candidates: List[PlateCandidate] = []
    seen_boxes = []  # for deduplication

    for score, x, y, rw, rh, pts, contour in top_regions:
        # Deduplication: skip if very similar bounding box already added
        duplicate = False
        for sx, sy, srw, srh in seen_boxes:
            overlap_x = min(x + rw, sx + srw) - max(x, sx)
            overlap_y = min(y + rh, sy + srh) - max(y, sy)
            if overlap_x > 0 and overlap_y > 0:
                iou = (overlap_x * overlap_y) / (
                    rw * rh + srw * srh - overlap_x * overlap_y + 1e-6
                )
                if iou > 0.60:
                    duplicate = True
                    break
        if duplicate:
            continue
        seen_boxes.append((x, y, rw, rh))

        # --- Geometric variant 1 & 2: perspective crops with different padding ---
        for pad in (15, 30):
            ordered = _order_points(pts)
            margin_px = max(6, int(min(rw, rh) * 0.04))
            exp = ordered.copy()
            exp[0] += [-margin_px, -margin_px]
            exp[1] += [margin_px, -margin_px]
            exp[2] += [margin_px, margin_px]
            exp[3] += [-margin_px, margin_px]
            exp[:, 0] = np.clip(exp[:, 0], 0, w - 1)
            exp[:, 1] = np.clip(exp[:, 1], 0, h - 1)
            try:
                crop = _perspective_crop_padded(img, exp, pad=pad)
                if crop is not None and crop.size > 0:
                    candidates.append(PlateCandidate(
                        crop=crop,
                        crop_type=f"perspective_pad{pad}",
                        score=score,
                    ))
            except Exception as e:
                logger.debug(f"Perspective crop failed (pad={pad}): {e}")

        # --- Geometric variants 3 & 4: bbox crops with different margins ---
        for margin_factor_x, margin_factor_y in ((0.10, 0.15), (0.20, 0.25)):
            mx = int(rw * margin_factor_x)
            my = int(rh * margin_factor_y)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + rw + mx)
            y2 = min(h, y + rh + my)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                mf_label = f"bbox_{int(margin_factor_x*100)}pct"
                candidates.append(PlateCandidate(
                    crop=crop,
                    crop_type=mf_label,
                    score=score * 0.9,   # slight penalty vs perspective
                ))

    logger.info(
        f"Plate detection: {len(scored_regions)} contour candidates → "
        f"{len(candidates)} crops generated"
    )
    return candidates


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------

def _preprocess_variants(
    crop: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate named preprocessing variants for a plate crop.

    Variant selection is adaptive:
    - Small crops (height < 80): apply 2× and 3× upscaling
    - Larger crops: 1× and 2× only (avoid unnecessary compute)

    Returns list of (variant_name, image) tuples.
    The original colour image is always first.
    """
    h, w = crop.shape[:2]
    variants: List[Tuple[str, np.ndarray]] = []

    # Ensure minimum height for PaddleOCR
    if h < 32:
        scale = max(2, 64 // h)
        base = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    else:
        base = crop

    bh, bw = base.shape[:2]

    # Colour at native resolution — always include, often best
    variants.append(("color_1x", base))

    # Grayscale (3-channel for PaddleOCR)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    variants.append(("gray_1x", gray3))

    # 2× upscale (cubic) — helps when crop is small
    if bh < 200 or bw < 400:
        up2 = cv2.resize(base, (bw * 2, bh * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(("color_2x_cubic", up2))

        gray_up2 = cv2.resize(gray3, (bw * 2, bh * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(("gray_2x_cubic", gray_up2))

    # 3× upscale — only for very small crops
    if bh < 80:
        up3 = cv2.resize(base, (bw * 3, bh * 3), interpolation=cv2.INTER_CUBIC)
        variants.append(("color_3x_cubic", up3))

    # CLAHE on grayscale 2×
    if bh < 200:
        gray2x = cv2.cvtColor(
            cv2.resize(base, (bw * 2, bh * 2), interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2GRAY,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray2x)
        variants.append(("clahe_2x", cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)))

    # Otsu threshold 2× — good for clean white plates
    if bh < 200:
        gray2x_s = cv2.resize(gray, (bw * 2, bh * 2), interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(gray2x_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu_2x", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))

    # Sharpened colour at 1× — helps when characters look blurry
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(base, -1, kernel_sharpen)
    variants.append(("sharpen_1x", sharpened))

    logger.debug(
        f"Preprocessing: crop {bw}x{bh} → {len(variants)} variants: "
        f"{[n for n, _ in variants]}"
    )
    return variants


# ---------------------------------------------------------------------------
# OCR token parsing + filtering + line grouping
# ---------------------------------------------------------------------------

def _parse_ocr_result(ocr_result) -> List[OcrToken]:
    """Parse raw PaddleOCR output into OcrToken list."""
    if not ocr_result or not ocr_result[0]:
        return []
    tokens = []
    for item in ocr_result[0]:
        try:
            bbox = item[0]
            text = item[1][0].strip()
            conf = float(item[1][1])
            if text:
                tokens.append(OcrToken(text=text, conf=conf, bbox=bbox))
        except (IndexError, TypeError, ValueError):
            continue
    return tokens


def _filter_plate_tokens(tokens: List[OcrToken]) -> List[OcrToken]:
    """
    Remove OCR tokens unlikely to be part of the registration number:
    - Tokens with very low confidence (< 0.15)
    - Tokens whose height is < 35% of the tallest token
      (catches IND country sticker, decorative text, etc.)
    - Pure numeric tokens with < 2 characters (single-digit noise)
    """
    if not tokens:
        return []

    # Confidence filter
    tokens = [t for t in tokens if t.conf >= 0.15]
    if not tokens:
        return []

    # Height filter
    max_h = max(t.h for t in tokens)
    height_thresh = max_h * 0.35
    filtered = []
    for t in tokens:
        if t.h >= height_thresh:
            filtered.append(t)
        else:
            logger.debug(
                f"  Height-filtered token '{t.text}' "
                f"(h={t.h:.0f} < thresh={height_thresh:.0f})"
            )

    # Remove pure-digit single characters (stray noise)
    filtered = [
        t for t in filtered
        if not (t.text.isdigit() and len(t.text) == 1 and t.conf < 0.50)
    ]

    return filtered


def _group_ocr_into_lines(
    tokens: List[OcrToken],
) -> List[List[OcrToken]]:
    """
    Group tokens into horizontal lines using centre-y spatial clustering.
    Each token's own height is used as the clustering threshold to be robust
    against variable font sizes.  Lines are sorted top-to-bottom; tokens
    within each line are sorted left-to-right.
    """
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: t.cy)
    lines: List[List[OcrToken]] = [[sorted_tokens[0]]]

    for token in sorted_tokens[1:]:
        prev_line = lines[-1]
        prev_cy = float(np.mean([t.cy for t in prev_line]))
        # Threshold = half of this token's own height (not average)
        thresh = max(token.h * 0.55, 8.0)
        if abs(token.cy - prev_cy) <= thresh:
            prev_line.append(token)
        else:
            lines.append([token])

    for line in lines:
        line.sort(key=lambda t: t.cx)

    return lines


def _run_ocr_on_variant(
    ocr: PaddleOCR,
    image: np.ndarray,
    variant_name: str,
) -> OcrVariantResult:
    """
    Run PaddleOCR on a single image variant. Log tokens. Return OcrVariantResult.
    """
    result = OcrVariantResult(variant_name=variant_name)
    try:
        raw = ocr.ocr(image, cls=True)
        result.tokens = _parse_ocr_result(raw)
        logger.info(
            f"  [OCR:{variant_name}] {image.shape[1]}x{image.shape[0]} → "
            f"tokens: {[(t.text, round(t.conf, 3)) for t in result.tokens]}"
        )
    except Exception as e:
        logger.warning(f"  [OCR:{variant_name}] Exception: {e}")
    return result


# ---------------------------------------------------------------------------
# Candidate building from line groups
# ---------------------------------------------------------------------------

def _build_raw_candidates_from_lines(
    lines: List[List[OcrToken]],
    variant_name: str,
) -> List[Tuple[str, List[float], str]]:
    """
    Build raw text candidates from spatially grouped lines.

    Generates:
      - Each individual line as a candidate
      - Each pair of adjacent lines concatenated (top + bottom)
      - All tokens spatially ordered (left-to-right, top-to-bottom)

    Returns list of (raw_text, [confidence_list], source_label).
    """
    candidates = []

    # Individual lines
    for i, line in enumerate(lines):
        text = "".join(t.text for t in line)
        confs = [t.conf for t in line]
        candidates.append((text, confs, f"{variant_name}:line{i}"))

    # Adjacent line pairs
    for i in range(len(lines) - 1):
        text = (
            "".join(t.text for t in lines[i])
            + "".join(t.text for t in lines[i + 1])
        )
        confs = [t.conf for t in lines[i]] + [t.conf for t in lines[i + 1]]
        candidates.append((text, confs, f"{variant_name}:lines{i}+{i+1}"))

    # All tokens in spatial order
    all_tokens = [t for line in lines for t in line]
    if len(all_tokens) > 1:
        spatial_text = "".join(t.text for t in all_tokens)
        confs = [t.conf for t in all_tokens]
        candidates.append((spatial_text, confs, f"{variant_name}:spatial"))

    return candidates


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _score_candidate(
    text: str,
    mean_conf: float,
    variant_count: int,
    total_variants: int,
    is_perspective: bool,
) -> float:
    """
    Multi-signal scoring for a plate candidate string.

    Signals:
      1. Indian plate format match (regex) — bonus, NOT a gate
      2. Recognized state code
      3. Length (prefer 6–10 chars)
      4. Structural digit/alpha at expected positions
      5. Number-section length (prefer 4-digit tails)
      6. OCR confidence
      7. Consensus across variants
      8. Perspective-crop bonus
      9. Compound penalty when neither regex nor state code matches
    """
    n = len(text)
    score = 0.0

    regex_match = bool(_INDIAN_PLATE_RE.match(text))
    state_match = n >= 2 and text[:2] in _STATE_CODES

    # --- Signal 1: Indian plate format regex ---
    if regex_match:
        score += 0.35

    # --- Signal 2: Recognized state code ---
    if state_match:
        score += 0.20

    # --- Signal 9 (early): Compound penalty ---
    # If NEITHER the regex matches NOR a valid state code is present,
    # this is very unlikely to be a real Indian plate. Apply a strong
    # penalty that makes it nearly impossible for random text to win.
    if not regex_match and not state_match:
        score -= 0.35

    # --- Signal 3: Length ---
    if 8 <= n <= 10:
        score += 0.20
    elif 6 <= n < 8:
        score += 0.08
    elif n > 10:
        score -= 0.05
    elif n < 5:
        score -= 0.20

    # --- Signal 4: Structural checks ---
    if n >= 2:
        if text[0].isalpha() and text[1].isalpha():
            score += 0.10
        else:
            score -= 0.10

    if n >= 4:
        if text[2].isdigit() and text[3].isdigit():
            score += 0.10
        elif text[2].isdigit():
            score += 0.05
        else:
            score -= 0.08

    # --- Signal 5: Number-section analysis ---
    # Identify series end and number section, then score by:
    # (a) whether number section is purely digits (no alpha contamination)
    # (b) length of number section (4 digits is standard, 3 is acceptable)
    if n >= 6:
        series_end = 4
        for j in range(4, min(n, 7)):
            if text[j].isalpha():
                series_end = j + 1
            else:
                break
        number_section = text[series_end:]
        if number_section:
            alpha_in_number = sum(1 for c in number_section if c.isalpha())
            if alpha_in_number == 0:
                score += 0.10   # clean digit-only number section
                # Bonus for 4-digit number section (standard Indian format)
                num_len = len(number_section)
                if num_len == 4:
                    score += 0.12   # strong: 4-digit is standard
                elif num_len == 3:
                    score += 0.04   # acceptable but less common
                elif num_len >= 1:
                    score += 0.01
            else:
                score -= 0.12 * alpha_in_number

    # Final char should be digit
    if n >= 4 and text[-1].isdigit():
        score += 0.05
    elif n >= 4:
        score -= 0.08

    # --- Signal 6: OCR confidence ---
    score += mean_conf * 0.10

    # --- Signal 7: Consensus across variants ---
    if total_variants > 0:
        consensus_ratio = variant_count / total_variants
        score += consensus_ratio * 0.10

    # --- Signal 8: Perspective bonus ---
    if is_perspective:
        score += 0.04

    return score


# ---------------------------------------------------------------------------
# Consensus selection
# ---------------------------------------------------------------------------

# Minimum score for a candidate to be returned as a detected plate.
# With the compound penalty (-0.35) for non-matching text, unrelated
# text will typically score below this threshold.
_MIN_PLATE_SCORE = 0.35


def _select_best_candidate(
    scored: List[ScoredCandidate],
) -> Optional[ScoredCandidate]:
    """
    Select the best candidate from a scored list.

    Rules (in priority order):
    1. Remove candidates below minimum score threshold.
    2. Remove candidates shorter than 5 chars (fragments).
    3. If any candidate with length ≥ 8 has a score ≥ 0.40, it is preferred
       over shorter candidates with higher raw score.
    4. Among top candidates (within 0.10 score gap), prefer structurally
       complete ones (longer, ending in digit).
    """
    if not scored:
        return None

    # Sort by score descending
    scored = sorted(scored, key=lambda c: c.score, reverse=True)

    # Apply minimum score gate
    scored = [c for c in scored if c.score >= _MIN_PLATE_SCORE]
    if not scored:
        return None

    # Filter out very short fragments if longer ones exist
    has_long = any(len(c.text) >= 7 for c in scored)
    if has_long:
        long_cands = [c for c in scored if len(c.text) >= 6]
        if long_cands:
            scored = long_cands

    # Among top candidates (within 0.10 score gap), prefer structurally
    # complete ones: longer AND ending with a digit
    if len(scored) >= 2:
        top_score = scored[0].score
        near_top = [c for c in scored if top_score - c.score <= 0.10]
        if near_top:
            # Sort by: ends_with_digit (bool desc), length (desc), score (desc)
            near_top.sort(
                key=lambda c: (c.text[-1].isdigit(), len(c.text), c.score),
                reverse=True,
            )
            scored = near_top + [c for c in scored if c not in near_top]

    return scored[0] if scored else None


# ---------------------------------------------------------------------------
# Master OCR orchestrator
# ---------------------------------------------------------------------------

def _run_plate_ocr_pipeline(
    crops: List[PlateCandidate],
) -> Tuple[Optional[str], List[str], float, List[dict], List[str], Dict[str, float]]:
    """
    Run the full OCR pipeline on a list of plate crop candidates.

    For each crop:
      1. Generate preprocessing variants
      2. Run PaddleOCR on each variant
      3. Filter tokens, group into lines
      4. Build raw text candidates
      5. Normalize and generate ambiguity alternatives
      6. Score each candidate

    Finally: consensus-select the best candidate.

    Returns:
      (detected_number, raw_texts, confidence,
       debug_tokens, candidate_numbers, candidate_scores)
    """
    ocr = get_paddle_ocr()

    all_raw_texts: List[str] = []
    all_debug_tokens: List[dict] = []

    # Map: normalized_text → ScoredCandidate (aggregated across variants)
    candidate_map: Dict[str, ScoredCandidate] = {}
    # Track which normalized texts came directly from OCR (before ambiguity)
    raw_normalized_texts: set = set()

    total_variants_run = 0

    for crop_idx, plate_cand in enumerate(crops):
        is_persp = "perspective" in plate_cand.crop_type
        variants = _preprocess_variants(plate_cand.crop)

        for var_name, var_img in variants:
            full_name = f"crop{crop_idx}_{plate_cand.crop_type}_{var_name}"
            variant_result = _run_ocr_on_variant(ocr, var_img, full_name)
            total_variants_run += 1

            if not variant_result.tokens:
                continue

            # Collect raw texts and debug tokens (from first informative variant)
            all_raw_texts.extend(variant_result.raw_texts)
            if not all_debug_tokens and variant_result.tokens:
                all_debug_tokens = [t.to_dict() for t in variant_result.tokens]

            # Filter + group
            filtered = _filter_plate_tokens(variant_result.tokens)
            if not filtered:
                continue

            lines = _group_ocr_into_lines(filtered)
            logger.info(
                f"  [Lines:{full_name}] {len(lines)} line(s): "
                f"{[[t.text for t in ln] for ln in lines]}"
            )

            # Build raw candidates from line structure
            raw_cands = _build_raw_candidates_from_lines(lines, full_name)

            for raw_text, conf_list, source in raw_cands:
                normalized = _normalize_plate(raw_text)
                if len(normalized) < 4:
                    continue

                # Track raw OCR texts (before ambiguity correction)
                raw_normalized_texts.add(normalized)

                mean_conf = float(np.mean(conf_list)) if conf_list else 0.0

                # Generate position-aware ambiguity alternatives
                alternatives = _generate_ambiguity_candidates(normalized)

                for alt_text in alternatives:
                    if len(alt_text) < 4:
                        continue

                    # Score this candidate
                    existing = candidate_map.get(alt_text)
                    vc = (existing.variant_count + 1) if existing else 1
                    sc = _score_candidate(
                        text=alt_text,
                        mean_conf=mean_conf,
                        variant_count=vc,
                        total_variants=max(total_variants_run, 1),
                        is_perspective=is_persp,
                    )

                    if existing:
                        # Update if this variant gives a better score
                        existing.variant_count = vc
                        existing.mean_conf = max(existing.mean_conf, mean_conf)
                        existing.score = max(existing.score, sc)
                        existing.sources.append(source)
                    else:
                        candidate_map[alt_text] = ScoredCandidate(
                            text=alt_text,
                            score=sc,
                            variant_count=vc,
                            mean_conf=mean_conf,
                            sources=[source],
                        )

    # Re-score with final total_variants count + originality bonus
    for cand in candidate_map.values():
        cand.score = _score_candidate(
            text=cand.text,
            mean_conf=cand.mean_conf,
            variant_count=cand.variant_count,
            total_variants=max(total_variants_run, 1),
            is_perspective=any("perspective" in s for s in cand.sources),
        )
        # Originality bonus: candidates that exactly match a raw OCR text
        # (no ambiguity substitution needed) get a small boost. This prevents
        # unnecessary corrections like B→8 from winning when OCR reads B.
        if cand.text in raw_normalized_texts:
            cand.score += 0.06

    if not candidate_map:
        return None, all_raw_texts, 0.0, all_debug_tokens, [], {}

    scored_list = sorted(
        candidate_map.values(), key=lambda c: c.score, reverse=True
    )

    # Log top candidates for debugging
    logger.info(f"Top candidates (total variants run: {total_variants_run}):")
    for c in scored_list[:10]:
        logger.info(
            f"  '{c.text}' score={c.score:.4f} conf={c.mean_conf:.4f} "
            f"variants={c.variant_count} sources={c.sources[:3]}"
        )

    best = _select_best_candidate(scored_list)
    if best is None:
        return None, all_raw_texts, 0.0, all_debug_tokens, [], {}

    candidate_numbers = [c.text for c in scored_list[:10]]
    candidate_scores = {c.text: round(c.score, 4) for c in scored_list[:10]}

    logger.info(
        f"FINAL: '{best.text}' score={best.score:.4f} "
        f"conf={best.mean_conf:.4f} variants={best.variant_count}"
    )

    return (
        best.text,
        all_raw_texts,
        round(best.mean_conf, 4),
        all_debug_tokens,
        candidate_numbers,
        candidate_scores,
    )


# ===========================================================================
# /verify-plate endpoint
# ===========================================================================

@app.post("/verify-plate")
async def verify_plate(file: UploadFile = File(...)):
    """
    Accept an uploaded vehicle or plate-crop image, detect the plate region
    with OpenCV, run PaddleOCR through a multi-variant pipeline, and return
    the normalised Indian registration number.

    Supports:
    - Full vehicle images (plate detected with OpenCV)
    - Pre-cropped / zoomed plate images (detected by multi-signal classifier)
    - High-resolution zoomed plates (NOT rejected by pixel-area threshold)
    - Single-line and two-line plates
    - Angled / perspective plates
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided.",
        )

    # --- Decode image ---
    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or unreadable image file.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading uploaded plate image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read or decode the uploaded image file.",
        )

    h, w = img.shape[:2]
    logger.info(f"verify-plate: received image {w}x{h}")

    # --- Classify image type and build crop list ---
    all_crops: List[PlateCandidate] = []

    try:
        is_crop, crop_conf = _is_likely_plate_crop(img)
        detected_cands = _detect_plate_candidates(img)

        if is_crop:
            # Treat whole image as a direct plate crop (with white padding)
            logger.info(
                f"Image classified as plate crop "
                f"(confidence={crop_conf:.3f}) — OCRing directly"
            )
            direct = cv2.copyMakeBorder(
                img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            all_crops.append(PlateCandidate(
                crop=direct,
                crop_type="direct",
                score=crop_conf,
            ))
            # Also add without padding (some plates benefit from original bounds)
            all_crops.append(PlateCandidate(
                crop=img,
                crop_type="direct_noPad",
                score=crop_conf * 0.9,
            ))

        if detected_cands:
            logger.info(f"OpenCV detection found {len(detected_cands)} candidate(s)")
            # Limit to top 4 to keep OCR calls bounded
            all_crops.extend(detected_cands[:4])

        # If ambiguous (both possible), always test both paths
        # (handled above — we already add direct + detected)

        if not all_crops:
            logger.info("No crops produced — returning no-plate response")
            return {
                "success": True,
                "plateDetected": False,
                "detectedNumber": None,
                "rawTexts": [],
                "confidence": 0,
                "debugTokens": [],
                "message": "No plate region detected in the image.",
                "candidateNumbers": [],
                "candidateScores": {},
                "ocrVariants": [],
            }

    except Exception as e:
        logger.error(f"Plate detection error: {e}", exc_info=True)
        all_crops = [PlateCandidate(crop=img, crop_type="direct_fallback", score=0.5)]

    # --- Run OCR pipeline ---
    try:
        (
            detected_plate,
            raw_texts,
            confidence,
            debug_tokens,
            candidate_numbers,
            candidate_scores,
        ) = _run_plate_ocr_pipeline(all_crops)
    except Exception as e:
        logger.error(f"OCR pipeline error: {e}", exc_info=True)
        return {
            "success": True,
            "plateDetected": False,
            "detectedNumber": None,
            "rawTexts": [],
            "confidence": 0,
            "debugTokens": [],
            "message": "OCR engine encountered an error processing the image.",
            "candidateNumbers": [],
            "candidateScores": {},
            "ocrVariants": [],
        }

    if detected_plate is None:
        return {
            "success": True,
            "plateDetected": False,
            "detectedNumber": None,
            "rawTexts": list(dict.fromkeys(raw_texts))[:10],
            "confidence": 0,
            "debugTokens": debug_tokens,
            "message": "No reliable number plate detected in the image.",
            "candidateNumbers": candidate_numbers,
            "candidateScores": candidate_scores,
            "ocrVariants": [],
        }

    logger.info(f"Final plate: '{detected_plate}' (confidence: {confidence:.4f})")

    return {
        "success": True,
        "plateDetected": True,
        "detectedNumber": detected_plate,
        "rawTexts": list(dict.fromkeys(raw_texts))[:10],
        "confidence": confidence,
        "debugTokens": debug_tokens,
        "message": "Number plate detected successfully.",
        "candidateNumbers": candidate_numbers,
        "candidateScores": candidate_scores,
        "ocrVariants": [f"crop{i}_{c.crop_type}" for i, c in enumerate(all_crops)],
    }
