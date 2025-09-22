# memory/strength.py
# Frequency + time-decay → strength utilities. Provides clamp01(), normalize_log_freq(), strength_from(), and decay_strength().

from __future__ import annotations
import math

# ---------------------------
# Small helpers
# ---------------------------
def clamp01(x: float) -> float:
    """Clamp to [0, 1]."""
    try:
        return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))
    except Exception:
        return 0.0

def normalize_log_freq(freq: int, *, sat: float = 50.0) -> float:
    """
    Map usage frequency to [0,1] with diminishing returns (log curve).
    `sat` is the soft saturation point (≈ value where the curve nears 1).
    """
    f = max(0, int(freq))
    s = max(1e-6, float(sat))
    return clamp01(math.log1p(f) / math.log1p(s))

def decay_strength(prev: float, hours_since_last: float, tau_hours: float) -> float:
    """
    Exponential decay of an existing strength over elapsed hours.
    `tau_hours` is the e-folding time constant (not half-life).
    """
    h = max(0.0, float(hours_since_last))
    tau = max(1e-6, float(tau_hours))
    return clamp01(float(prev) * math.exp(-h / tau))

# ---------------------------
# Main model
# ---------------------------
def strength_from(
    freq: int,
    hours_since_last: float,
    goal_rel: float,
    tau_hours: float,
    *,
    sat: float = 50.0,
    w_freq: float = 0.7,
    w_goal: float = 0.3,
) -> float:
    """
    Compute retrieval strength from:
      - freq: how many times the item has been accessed/used (diminishing via log)
      - hours_since_last: recency (exponential decay with time constant tau_hours)
      - goal_rel: 0..1 relevance to current goals (slower-varying prior)
      - tau_hours: e-folding time constant (e.g., working≈72h, long≈168h, summary≈240h)

    Returns a value in [0,1] used in blending with cosine similarity.
    """
    # Frequency term with diminishing returns (then decayed by recency)
    f_norm = normalize_log_freq(freq, sat=sat)                    # 0..1
    recency = math.exp(-max(0.0, float(hours_since_last)) / max(1e-6, float(tau_hours)))  # 0..1
    freq_decayed = f_norm * recency

    # Goal relevance prior (already 0..1)
    g = clamp01(goal_rel)

    # Blend
    wf = clamp01(w_freq)
    wg = clamp01(w_goal)
    # Normalize weights in case caller passes odd values
    Z = max(1e-6, wf + wg)
    wf /= Z; wg /= Z

    strength = wf * freq_decayed + wg * g
    return clamp01(strength)

# ---------------------------
# Quick self-test
# ---------------------------
if __name__ == "__main__":
    print("freq=0, hrs=24 →", strength_from(0, 24, 0.0, 72))
    print("freq=5, hrs=0  →", strength_from(5, 0, 0.2, 72))
    print("freq=20,hrs=48 →", strength_from(20, 48, 0.1, 72))
    print("decay 0.8 over 72h @tau=72 →", decay_strength(0.8, 72, 72))
