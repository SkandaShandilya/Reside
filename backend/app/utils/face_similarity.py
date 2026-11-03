"""
Face embedding validation + cosine similarity — ported 1:1 (algorithm
and thresholds unchanged) from the migration's
`backend/src/utils/embedding.utils.js` + `similarity.utils.js`, adapted
from hand-rolled loops to numpy since this project already depends on
it throughout (ai_service). Behavior is identical: same validation
rules (exactly 512 dimensions, all-finite, not a zero vector), same
cosine similarity formula, same rejection reasons.
"""
from typing import Optional, Sequence
import numpy as np

EMBEDDING_DIMENSION = 512


def is_valid_embedding(embedding: Optional[Sequence[float]]) -> tuple[bool, Optional[str]]:
    """Mirrors embedding.utils.js's isValidEmbedding(): must be a
    512-length sequence of finite numbers, not all-zero."""
    if embedding is None or not isinstance(embedding, (list, tuple)):
        return False, "Embedding must be an array"
    if len(embedding) != EMBEDDING_DIMENSION:
        return False, f"Embedding must have exactly {EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
    arr = np.asarray(embedding, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False, "Embedding contains invalid value"
    if np.all(arr == 0):
        return False, "Embedding is a zero vector"
    return True, None


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> tuple[float, bool, Optional[str]]:
    """Mirrors similarity.utils.js's cosineSimilarity(): dot(A,B) /
    (||A|| * ||B||), with the same validation (both must be exactly
    512-d, finite, non-zero-magnitude) before computing. Returns
    (similarity, valid, reason)."""
    if len(vec_a) != EMBEDDING_DIMENSION or len(vec_b) != EMBEDDING_DIMENSION:
        return 0.0, False, f"Both vectors must be {EMBEDDING_DIMENSION}-dimensional"

    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0, False, "Vectors contain non-finite values"

    magnitude_a = float(np.sqrt(np.sum(a * a)))
    magnitude_b = float(np.sqrt(np.sum(b * b)))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0, False, "Cannot compute similarity for zero-magnitude vector"

    similarity = float(np.dot(a, b) / (magnitude_a * magnitude_b))
    return similarity, True, None
