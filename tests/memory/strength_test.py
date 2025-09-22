# tests/memory_tests/strength_test.py
import math
import numpy as np
import pytest

from memory.strength import (
    clamp01,
    normalize_log_freq,
    decay_strength,
    strength_from,
)


# -------------- clamp01 ----------------

def test_clamp01_bounds_and_passthrough():
    assert clamp01(-0.5) == 0.0
    assert clamp01(0.0) == 0.0
    assert clamp01(0.25) == 0.25
    assert clamp01(1.0) == 1.0
    assert clamp01(1.5) == 1.0

def test_clamp01_non_numeric_safe():
    # Invalid types should not crash and should return 0.0 via exception path.
    assert clamp01("nope") == 0.0  # type: ignore[arg-type]


# -------------- normalize_log_freq ----------------

def test_normalize_log_freq_basic_monotonic_and_range():
    vals = [normalize_log_freq(f) for f in range(0, 10)]
    # Monotonic non-decreasing
    assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    # Range
    assert vals[0] == 0.0
    assert all(0.0 <= v <= 1.0 for v in vals)

def test_normalize_log_freq_saturation_point_equals_one():
    # At freq == sat, formula is exactly 1
    assert normalize_log_freq(50, sat=50.0) == 1.0
    # Above saturation still clamps to 1
    assert normalize_log_freq(1000, sat=50.0) == 1.0

def test_normalize_log_freq_handles_negative_freq_and_bad_sat():
    # Negative freq is treated as 0
    assert normalize_log_freq(-10) == 0.0
    # sat <= 0 is coerced to tiny positive → divides by ~log(1+1e-6), clamps to 1
    assert normalize_log_freq(1, sat=0.0) == 1.0
    assert normalize_log_freq(0, sat=0.0) == 0.0


# -------------- decay_strength ----------------

def test_decay_strength_identity_at_zero_hours():
    assert decay_strength(0.8, 0.0, 72.0) == pytest.approx(0.8, rel=1e-6)

def test_decay_strength_monotonic_with_hours():
    s0 = decay_strength(0.8, 0.0, 72.0)
    s1 = decay_strength(0.8, 24.0, 72.0)
    s2 = decay_strength(0.8, 72.0, 72.0)
    assert s0 >= s1 >= s2
    assert 0.0 <= s2 <= 0.8

def test_decay_strength_tau_effect():
    # Larger tau decays less for the same hours
    small_tau = decay_strength(0.8, 24.0, 24.0)
    large_tau = decay_strength(0.8, 24.0, 240.0)
    assert large_tau > small_tau

def test_decay_strength_clamped_and_safe_inputs():
    # prev outside [0,1] gets clamped by return path
    assert 0.0 <= decay_strength(2.0, 24.0, 72.0) <= 1.0
    # Negative hours coerced to 0
    assert decay_strength(0.5, -10.0, 72.0) == pytest.approx(0.5, rel=1e-6)
    # Tiny tau prevented from divide-by-zero
    val = decay_strength(0.5, 10.0, 0.0)
    assert 0.0 <= val <= 0.5


# -------------- strength_from ----------------

def test_strength_from_zero_everything_is_zero():
    s = strength_from(freq=0, hours_since_last=100.0, goal_rel=0.0, tau_hours=72.0)
    assert s == 0.0

def test_strength_from_goal_only_when_w_freq_zero():
    s = strength_from(
        freq=0,
        hours_since_last=100.0,
        goal_rel=1.0,
        tau_hours=72.0,
        w_freq=0.0,
        w_goal=1.0,
    )
    assert s == 1.0

def test_strength_from_weights_are_normalized_and_clamped():
    # Odd weights (sum > 1) get normalized; negative gets clamped to 0 before normalization
    s1 = strength_from(freq=10, hours_since_last=0.0, goal_rel=1.0, tau_hours=72.0, w_freq=2.0, w_goal=2.0)
    s2 = strength_from(freq=10, hours_since_last=0.0, goal_rel=1.0, tau_hours=72.0, w_freq=-5.0, w_goal=3.0)
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0
    # With goal_rel=1, a nonzero w_goal should push strength up significantly
    assert s1 > 0.4
    assert s2 > 0.4

def test_strength_from_recency_decay_effect():
    # Same freq/goal, larger hours_since_last -> lower strength
    s0 = strength_from(freq=8, hours_since_last=0.0, goal_rel=0.2, tau_hours=72.0)
    s1 = strength_from(freq=8, hours_since_last=24.0, goal_rel=0.2, tau_hours=72.0)
    assert s0 > s1

def test_strength_from_tau_effect_on_recency_component():
    # With fixed hours, larger tau keeps more strength (slower decay)
    s_small_tau = strength_from(freq=8, hours_since_last=24.0, goal_rel=0.2, tau_hours=24.0)
    s_large_tau = strength_from(freq=8, hours_since_last=24.0, goal_rel=0.2, tau_hours=240.0)
    assert s_large_tau > s_small_tau

def test_strength_from_freq_saturation_diminishing_returns():
    # Ensure diminishing returns: delta from 1->5 > delta from 50->100 when sat=50
    low = strength_from(freq=1, hours_since_last=0.0, goal_rel=0.0, tau_hours=72.0, sat=50.0, w_freq=1.0, w_goal=0.0)
    mid = strength_from(freq=5, hours_since_last=0.0, goal_rel=0.0, tau_hours=72.0, sat=50.0, w_freq=1.0, w_goal=0.0)
    hi1 = strength_from(freq=50, hours_since_last=0.0, goal_rel=0.0, tau_hours=72.0, sat=50.0, w_freq=1.0, w_goal=0.0)
    hi2 = strength_from(freq=100, hours_since_last=0.0, goal_rel=0.0, tau_hours=72.0, sat=50.0, w_freq=1.0, w_goal=0.0)

    delta_low = mid - low
    delta_high = hi2 - hi1
    assert delta_low > 0
    assert delta_high >= 0
    assert delta_low > delta_high  # diminishing returns

def test_strength_from_input_robustness_and_bounds():
    # Negative freq treated as 0; goal_rel out of bounds clamped; negative hours -> 0
    s = strength_from(freq=-10, hours_since_last=-1.0, goal_rel=2.5, tau_hours=72.0)
    assert 0.0 <= s <= 1.0

def test_strength_from_always_within_bounds_randomized():
    rng = np.random.default_rng(123)
    for _ in range(100):
        freq = int(rng.integers(-10, 500))
        hrs = float(rng.uniform(-5.0, 500.0))
        goal = float(rng.uniform(-2.0, 2.0))
        tau = float(rng.uniform(0.0, 500.0))
        s = strength_from(freq=freq, hours_since_last=hrs, goal_rel=goal, tau_hours=tau)
        assert 0.0 <= s <= 1.0
