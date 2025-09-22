# tests/memory_tests/ingest_test.py
import numpy as np
import pytest

from memory.ingest import (
    sanitize_meta,
    social_relevance,
    compute_salience,
    kind_prior,
    decide_keep,
    build_item_from_event,
    SALIENCE_W_NOVELTY, SALIENCE_W_GOALREL, SALIENCE_W_SOCIAL, SALIENCE_W_IMPACT,
)
from memory.models import Event


# ---------------------------
# helpers
# ---------------------------

def _vec(v):
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a if n == 0 else a / n

def _mk_event(kind="chat:user", content="hello", **meta) -> Event:
    return Event.new(kind=kind, content=content, meta=meta or {})


# ---------------------------
# sanitize_meta
# ---------------------------

def test_sanitize_meta_strips_reserved_keys_and_keeps_custom():
    meta = {
        "id": "X",
        "layer": "long",
        "content": "nope",
        "embedding_id": "v",
        "_vec": [1, 2, 3],
        "custom": 123,
        "another": "ok",
    }
    out = sanitize_meta(meta)
    assert "id" not in out
    assert "layer" not in out
    assert "content" not in out
    assert "embedding_id" not in out
    assert "_vec" not in out
    assert out == {"custom": 123, "another": "ok"}

def test_sanitize_meta_handles_none_and_non_dict():
    assert sanitize_meta(None) == {}
    assert sanitize_meta("notadict") == {}


# ---------------------------
# social_relevance
# ---------------------------

def test_social_relevance_values():
    assert social_relevance("chat:user") == 1.0
    assert social_relevance("goal:update") == 0.3
    assert social_relevance("other") == 0.0
    assert social_relevance(123) == 0.0


# ---------------------------
# compute_salience + clamps
# ---------------------------

def test_compute_salience_linear_combo_and_clamp():
    # Pick values to compute exact linear combo
    n, g, s, i = 0.2, 0.4, 0.6, 0.8
    expected = (
        SALIENCE_W_NOVELTY * n +
        SALIENCE_W_GOALREL * g +
        SALIENCE_W_SOCIAL  * s +
        SALIENCE_W_IMPACT  * i
    )
    got = compute_salience(novelty=n, goal_rel=g, social_rel=s, impact=i)
    assert pytest.approx(got, rel=1e-6) == expected

    # Clamp >1
    got_hi = compute_salience(novelty=10, goal_rel=10, social_rel=10, impact=10)
    assert got_hi == 1.0
    # Clamp <0 (not expected from our inputs but guard anyway)
    got_lo = compute_salience(novelty=-10, goal_rel=-10, social_rel=-10, impact=-10)
    assert got_lo == 0.0


# ---------------------------
# kind_prior
# ---------------------------

def test_kind_prior_uses_config_and_clamps(monkeypatch):
    import memory.ingest as mod
    # Ensure priors exist and exercise both existing and fallback paths
    priors = dict(mod.MEMCFG.STRENGTH_PRIORS)
    priors["definition"] = 0.9
    monkeypatch.setattr(mod.MEMCFG, "STRENGTH_PRIORS", priors, raising=False)

    assert kind_prior("definition") == 0.9
    # fallback to "fact" prior for unknown kinds
    assert kind_prior("unknown_kind") == pytest.approx(mod.MEMCFG.STRENGTH_PRIORS["fact"], rel=1e-6)


# ---------------------------
# decide_keep
# ---------------------------

def test_decide_keep_respects_capture_all_and_explicit_and_threshold(monkeypatch):
    import memory.ingest as mod

    # 1) capture-all → always True
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", True, raising=False)
    assert decide_keep(salience=0.0, meta={}, capture_all=None, salience_keep=0.5) is True

    # 2) explicit remember → True
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", False, raising=False)
    assert decide_keep(salience=0.0, meta={"explicit_remember": True}, capture_all=None, salience_keep=0.9) is True

    # 3) threshold
    assert decide_keep(salience=0.6, meta={}, capture_all=None, salience_keep=0.5) is True
    assert decide_keep(salience=0.4, meta={}, capture_all=None, salience_keep=0.5) is False

    # 4) explicit capture_all param overrides config
    assert decide_keep(salience=0.0, meta={}, capture_all=True, salience_keep=0.99) is True


# ---------------------------
# build_item_from_event
# ---------------------------

def test_build_item_uses_precomputed_vector_and_sanitized_meta(monkeypatch):
    import memory.ingest as mod

    # Force capture (so keep=True regardless of salience)
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", True, raising=False)

    # Ensure deterministic model hint
    monkeypatch.setattr(mod, "model_hint", lambda: "hintX", raising=True)

    # Provide a precomputed vector via meta._vec
    pre = [0.0, 2.0, 0.0]
    ev = _mk_event(
        kind="other:evt",
        content="ignored for vec",
        _vec=pre,
        layer="should_be_stripped",
        custom="kept",
        goal_rel=0.0, impact=0.0,
    )

    res = build_item_from_event(ev, recent_vecs=[])
    assert res.kept is True
    assert res.item is not None and res.vector is not None

    # vector equals normalized precomputed vector
    assert np.allclose(res.vector, _vec(pre))
    # embedding_dim matches vector
    assert res.item.embedding_dim == len(res.vector)
    # meta was sanitized: "layer" and "_vec" removed, "custom" kept
    assert res.item.meta.get("custom") == "kept"
    assert "layer" not in res.item.meta and "_vec" not in res.item.meta
    # model hint set
    assert res.item.model_hint == "hintX"
    # freq and strength initialized
    assert res.item.freq == 0
    assert 0.0 <= float(res.item.strength) <= 1.0


def test_build_item_salience_formula_and_keep_threshold(monkeypatch):
    import memory.ingest as mod

    # turn off capture-all to exercise threshold
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", False, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "SALIENCE_KEEP", 0.6, raising=False)

    # Patch novelty_score to a fixed value so we can predict salience
    monkeypatch.setattr(mod, "novelty_score", lambda v, recent: 0.5, raising=True)
    # Patch get_embedding to a fixed vec (so len known)
    monkeypatch.setattr(mod, "get_embedding", lambda s: _vec([1.0, 0.0, 0.0]), raising=True)
    # Hint stable
    monkeypatch.setattr(mod, "model_hint", lambda: "hintY", raising=True)

    # Build event with goal_rel, social (from kind), impact
    ev = _mk_event(kind="goal:update", content="abc", goal_rel=0.4, impact=0.2)
    res = build_item_from_event(ev, recent_vecs=[_vec([0, 1, 0])])

    # Compute expected salience
    expected_sal = (
        SALIENCE_W_NOVELTY * 0.5 +
        SALIENCE_W_GOALREL * 0.4 +
        SALIENCE_W_SOCIAL  * 0.3 +   # social for "goal:"
        SALIENCE_W_IMPACT  * 0.2
    )
    assert pytest.approx(res.salience, rel=1e-6) == expected_sal

    # With threshold 0.6, ensure decision matches
    if expected_sal >= 0.6:
        assert res.kept is True
        assert res.item is not None
    else:
        assert res.kept is False
        assert res.item is None


def test_build_item_explicit_remember_overrides_low_salience(monkeypatch):
    import memory.ingest as mod
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", False, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "SALIENCE_KEEP", 0.99, raising=False)
    # Force novelty low -> salience low
    monkeypatch.setattr(mod, "novelty_score", lambda v, recent: 0.0, raising=True)
    monkeypatch.setattr(mod, "get_embedding", lambda s: _vec([0, 0, 1]), raising=True)
    monkeypatch.setattr(mod, "model_hint", lambda: "h", raising=True)

    ev = _mk_event(kind="other", content="zzz", explicit_remember=True)
    res = build_item_from_event(ev, recent_vecs=[_vec([1, 0, 0])])
    assert res.kept is True
    assert res.item is not None


def test_build_item_uses_novelty_1_when_no_recent_vecs(monkeypatch):
    import memory.ingest as mod
    # If there are no recent vectors, code uses novelty = 1.0 (and should NOT call novelty_score)
    called = {"nov": 0}
    def _nov(v, recent):
        called["nov"] += 1
        return 0.123
    monkeypatch.setattr(mod, "novelty_score", _nov, raising=True)
    monkeypatch.setattr(mod, "get_embedding", lambda s: _vec([1, 0, 0]), raising=True)
    monkeypatch.setattr(mod, "model_hint", lambda: "h", raising=True)
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", True, raising=False)

    ev = _mk_event(content="x")
    res = build_item_from_event(ev, recent_vecs=[])
    # novelty_score must not be called
    assert called["nov"] == 0
    assert res.kept and res.item is not None
    assert res.item.novelty == 1.0
