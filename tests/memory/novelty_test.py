# tests/memory_tests/novelty_test.py
import math
import numpy as np
import pytest

from memory.novelty import cosine, max_cosine, novelty, novelty_many


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def test_cosine_identical_is_one_and_opposite_is_minus_one():
    rng = np.random.default_rng(7)
    v = _norm(rng.normal(size=128))
    assert pytest.approx(cosine(v, v), abs=1e-6) == 1.0

    opp = -v
    assert pytest.approx(cosine(v, opp), abs=1e-6) == -1.0


def test_cosine_with_zero_vector_is_zero():
    v = _norm(np.ones(16, dtype=np.float32))
    z = np.zeros_like(v)
    # safe for zero vectors; should yield ~0
    assert pytest.approx(cosine(v, z), abs=1e-9) == 0.0
    assert pytest.approx(cosine(z, z), abs=1e-9) == 0.0


def test_max_cosine_empty_recent_returns_zero():
    rng = np.random.default_rng(11)
    v = _norm(rng.normal(size=64))
    assert max_cosine(v, []) == 0.0


def test_max_cosine_picks_best_of_many():
    rng = np.random.default_rng(21)
    base = _norm(rng.normal(size=96))
    near = _norm(base + 0.05 * rng.normal(size=96))
    far = _norm(rng.normal(size=96))

    recent = [far, near, _norm(rng.normal(size=96))]
    m = max_cosine(base, recent)
    # must be at least as similar as with 'near'
    assert m >= cosine(base, near) - 1e-6
    # and definitely higher than similarity to 'far' on average
    assert m >= cosine(base, far) - 1e-6


def test_novelty_bounds_and_floor_behavior():
    rng = np.random.default_rng(33)
    base = _norm(rng.normal(size=48))
    recent = [base, _norm(rng.normal(size=48))]
    # identical -> max_cosine=1 -> novelty 0 -> clamped to floor
    n = novelty(base, recent, floor=0.07)
    assert 0.07 - 1e-9 <= n <= 1.0
    assert pytest.approx(n, abs=1e-6) == 0.07

    # no recent -> novelty = 1.0
    assert pytest.approx(novelty(base, []), abs=1e-9) == 1.0


def test_novelty_monotonic_with_similarity():
    rng = np.random.default_rng(44)
    base = _norm(rng.normal(size=72))
    slightly_near = _norm(base + 0.05 * rng.normal(size=72))
    very_near = _norm(base + 0.005 * rng.normal(size=72))
    recent = [slightly_near]

    n1 = novelty(base, recent, floor=0.01)  # with slightly_near in recent
    # add a very_near → similarity increases → novelty should not increase
    n2 = novelty(base, [slightly_near, very_near], floor=0.01)
    assert n2 <= n1 + 1e-9


def test_novelty_temperature_effects():
    rng = np.random.default_rng(55)
    base = _norm(rng.normal(size=128))
    # craft a recent vector with moderate similarity
    recent = [_norm(base + 0.2 * rng.normal(size=128))]

    n_cold = novelty(base, recent, temperature=2.0)   # stricter (lower novelty)
    n_warm = novelty(base, recent, temperature=0.5)   # more forgiving (higher novelty)
    assert 0.0 < n_cold <= n_warm <= 1.0


def test_novelty_many_matches_scalar_calls_and_handles_empty_recent():
    rng = np.random.default_rng(66)
    vecs = [_norm(rng.normal(size=64)) for _ in range(5)]
    recent = [_norm(rng.normal(size=64)) for _ in range(3)]

    many = novelty_many(vecs, recent, floor=0.02, temperature=1.3)
    assert len(many) == len(vecs)

    for v, nm in zip(vecs, many):
        ns = novelty(v, recent, floor=0.02, temperature=1.3)
        assert pytest.approx(nm, rel=1e-6, abs=1e-6) == ns

    # empty recent -> all 1.0
    assert all(x == 1.0 for x in novelty_many(vecs, []))


def test_novelty_many_is_in_range_and_ordered_by_similarity():
    rng = np.random.default_rng(77)
    base = _norm(rng.normal(size=90))
    near = _norm(base + 0.05 * rng.normal(size=90))
    mid = _norm(base + 0.2 * rng.normal(size=90))
    far = _norm(rng.normal(size=90))
    recent = [near]

    scores = novelty_many([near, mid, far], recent, floor=0.01, temperature=1.0)
    # all in [floor, 1]
    assert all(0.01 <= s <= 1.0 for s in scores)
    # order: near (lowest novelty) <= mid <= far (highest novelty)
    assert scores[0] <= scores[1] <= scores[2] + 1e-9
