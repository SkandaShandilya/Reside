"""
HTTP client for the separate face-service process (InsightFace +
PaddleOCR — see face-service/main.py, run in its own venv/process; see
app/config.py's FACE_SERVICE_URL docstring for why it's not in-process).

Ported error-handling semantics 1:1 from the migration's
`backend/src/services/face.service.js` / `vehicle.service.js`:
connection failures/timeouts -> 503 "service unavailable", a 4xx from
face-service -> forwarded as 400 with its detail message, a 5xx from
face-service -> 503. This is the ONE place that talks to face-service,
so a face-service outage degrades gracefully here and never raises
uncaught into a request — and, since this is a plain HTTP call, an
outage here cannot affect ai_service (Behaviour/Fire-Smoke), which
never imports from or calls into this module.
"""
import httpx
from fastapi import HTTPException, status
from app.config import settings

_TIMEOUT = httpx.Timeout(30.0)  # CPU inference (InsightFace/PaddleOCR) can be slow


class FaceServiceError(HTTPException):
    """Raised for any face-service call failure — always caught and
    turned into a clean HTTP response by the calling endpoint, never
    left to propagate as a raw 500."""
    pass


async def _post_image(endpoint: str, filename: str, content: bytes, content_type: str) -> dict:
    url = f"{settings.FACE_SERVICE_URL.rstrip('/')}{endpoint}"
    files = {"file": (filename or "image.jpg", content, content_type or "image/jpeg")}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(url, files=files)
    except httpx.ConnectError:
        raise FaceServiceError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition service is unavailable. Please try again later.",
        )
    except httpx.TimeoutException:
        raise FaceServiceError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition service timed out. Please try again.",
        )
    except httpx.HTTPError:
        raise FaceServiceError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to communicate with face recognition service.",
        )

    if response.status_code >= 500:
        raise FaceServiceError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition service encountered an internal error.",
        )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Face processing failed.")
        except Exception:
            detail = "Face processing failed."
        raise FaceServiceError(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    data = response.json()
    if not data or not data.get("success"):
        raise FaceServiceError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face recognition service returned an unexpected response.",
        )
    return data


async def detect_and_embed(filename: str, content: bytes, content_type: str) -> dict:
    """Calls face-service's /detect-and-embed. Returns {embedding, bbox,
    det_score} on success; raises FaceServiceError otherwise (unchanged
    endpoint contract — see face-service/main.py)."""
    data = await _post_image("/detect-and-embed", filename, content, content_type)
    return {
        "embedding": data.get("embedding") or [],
        "bbox": data.get("bbox") or [],
        "det_score": data.get("det_score") or 0.0,
    }


async def verify_plate(filename: str, content: bytes, content_type: str) -> dict:
    """Calls face-service's /verify-plate. Returns the full OCR result
    dict (plateDetected, detectedNumber, rawTexts, confidence, message —
    unchanged endpoint contract)."""
    data = await _post_image("/verify-plate", filename, content, content_type)
    return {
        "plate_detected": data.get("plateDetected", False),
        "detected_number": data.get("detectedNumber"),
        "raw_texts": data.get("rawTexts") or [],
        "confidence": data.get("confidence") or 0.0,
        "message": data.get("message") or "",
    }
