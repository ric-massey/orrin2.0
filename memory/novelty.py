# memory/novelty.py
# Novelty scoring for memory ingest: fast cosine-to-novelty with vector cache support and batch helpers (no external deps).

from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import numpy as np


# ---------- small utils ----------
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def _as2d_norm(recent_vecs: Iterable[np.ndarray]) -> np.ndarray:
    """
    Stack recent vectors into a (N, D) float32 array and L2-normalize rows.
    Returns empty (0, 0) if none.
    """
    mats: List[np.ndarray] = []
    for rv in recent_vecs:
        try:
            mats.append(_normalize(rv))
        except Exception:
            continue
    if not mats:
        return np.zeros((0, 0), dtype=np.float32)
    M = np.vstack(mats).astype(np.float32, copy=False)
    # Rows are already normalized by _normalize
    return M


# ---------- public API ----------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [−1, 1]; safe for zero vectors."""
    va, vb = _normalize(a), _normalize(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)


def max_cosine(vec: np.ndarray, recent_vecs: Iterable[np.ndarray]) -> float:
    """
    Max cosine similarity between vec and a set of recent vectors.
    Returns 0.0 if recent set is empty or dims mismatch (after normalization).
    """
    v = _normalize(vec)
    M = _as2d_norm(recent_vecs)
    if M.size == 0:
        return 0.0
    # Cosine == dot because both sides are L2-normalized
    try:
        sims = M @ v
        return float(np.max(sims))
    except Exception:
        # Fallback to loop if dims misaligned or matmul fails
        best = 0.0
        for r in M:
            s = float(np.dot(r, v))
            if s > best:
                best = s
        return best


def novelty(
    vec: np.ndarray,
    recent_vecs: Iterable[np.ndarray],
    *,
    floor: float = 0.05,
    temperature: float = 1.0,
) -> float:
    """
    Convert max cosine similarity to a novelty score in [floor, 1].
      - novelty = (1 - max_cosine) ** temperature
      - temperature < 1.0 makes the function more forgiving (higher novelty)
      - temperature > 1.0 makes it stricter (lower novelty)
      - floor ensures we never fully suppress low-sim items in early life

    If there are no recent vectors, returns 1.0 (max novelty).
    """
    if temperature <= 0:
        temperature = 1.0
    m = max_cosine(vec, recent_vecs)
    n = (1.0 - float(max(0.0, min(1.0, m)))) ** float(temperature)
    n = float(max(float(floor), min(1.0, n)))
    return n


def novelty_many(
    vecs: Iterable[np.ndarray],
    recent_vecs: Iterable[np.ndarray],
    *,
    floor: float = 0.05,
    temperature: float = 1.0,
) -> List[float]:
    """Batch novelty for a list of vectors against the same recent set."""
    M = _as2d_norm(recent_vecs)
    out: List[float] = []
    if M.size == 0:
        return [1.0 for _ in vecs]
    for v in vecs:
        vn = _normalize(v)
        try:
            sims = M @ vn
            m = float(np.max(sims))
        except Exception:
            # tiny fallback
            m = 0.0
            for r in M:
                s = float(np.dot(r, vn))
                if s > m:
                    m = s
        n = (1.0 - float(max(0.0, min(1.0, m)))) ** float(max(temperature, 1e-6))
        out.append(float(max(floor, min(1.0, n))))
    return out


# ---------- quick self-test ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    base = _normalize(rng.normal(size=384))
    near = _normalize(base + 0.05 * rng.normal(size=384))
    far  = _normalize(rng.normal(size=384))

    recent = [base, _normalize(rng.normal(size=384))]

    print("max_cosine(base, recent):", max_cosine(base, recent))
    print("novelty(base, recent):   ", novelty(base, recent))
    print("novelty(near, recent):   ", novelty(near, recent))
    print("novelty(far, recent):    ", novelty(far, recent))
