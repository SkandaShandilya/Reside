import re

# Accepts an optional leading "+" followed by 7-15 digits (E.164 range),
# after stripping spaces/hyphens/parentheses — permissive enough for
# whatever country the resident is in, since apartment_number/phone are
# already free-text with no locale assumption baked in elsewhere in this
# schema, while still rejecting empty/junk input.
_CLEAN = re.compile(r"[\s\-()]")
_PHONE_PATTERN = re.compile(r"^\+?\d{7,15}$")


def normalize_phone(raw: str | None) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    return _CLEAN.sub("", raw.strip())


def is_valid_phone(phone: str | None) -> bool:
    if not phone or not isinstance(phone, str):
        return False
    return bool(_PHONE_PATTERN.match(phone))
