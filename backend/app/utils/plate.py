"""
Plate-number normalization + Indian plate format check — ported 1:1
from the migration's `backend/src/utils/plate.utils.js`. The regex is a
SCORING signal for the OCR pipeline (see face-service/main.py), not a
hard gate here either — callers use is_valid_indian_plate() only to
annotate/display, never to reject an OCR result outright.
"""
import re

_PLATE_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize_plate_number(raw: str | None) -> str:
    """Uppercase, strip everything but letters/digits — mirrors
    normalizePlateNumber()."""
    if not raw or not isinstance(raw, str):
        return ""
    return _NON_ALNUM.sub("", raw.upper())


def is_valid_indian_plate(plate: str | None) -> bool:
    """2 state letters + 1-2 district digits + 0-3 series letters + 1-4
    number digits — mirrors isValidIndianPlate()."""
    if not plate or not isinstance(plate, str):
        return False
    return bool(_PLATE_PATTERN.match(plate))
