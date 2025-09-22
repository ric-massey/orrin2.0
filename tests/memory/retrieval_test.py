# tests/memory_tests/retrieval_test.py
import time
import numpy as np
import pytest

from memory.retrieval import retrieve, score_only
from memory.store.inmem import InMemoryStore
from memory.models import MemoryItem


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def _mk_item(kind: str, content: str, layer: str = "working", **meta) -> MemoryItem:
    it = MemoryItem.new(kind=kind, source="test:gen", content=content, layer=layer, **meta)
    it.embedding_id = f"vec_{it.id}"
    return it


def _seed_items_with_vecs(store: InMemoryStore, items, vecs):
    """Upsert items and their vectors into the store (normalized)."""
    store.upsert_items(items)
    store.upsert_vectors({it.embedding_id: _norm(v) for it, v in zip(items, vecs)})


def test_retrieve_orders_by_similarity_when_alpha_1():
    store = InMemoryStore()
    D = 32
    q = _norm(np.r_[1.0, np.zeros(D-1)])

    # Three items with descending similarity to q
    a = _mk_item("fact", "A")
    b = _mk_item("fact", "B")
    c = _mk_item("fact", "C")

    va = q
    vb = _norm(q + 0.2 * np.eye(1, D, 1).ravel())  # slightly off
    vc = _norm(np.random.default_rng(0).normal(size=D))  # random

    _seed_items_with_vecs(store, [a, b, c], [va, vb, vc])

    out = retrieve(store, query_vec=q, alpha=1.0, top_k=3, reinforce=False)
    assert [it.content for it in out] == ["A", "B", "C"]


def test_retrieve_orders_by_strength_when_alpha_0():
    store = InMemoryStore()
    D = 16
    q = _norm(np.r_[1.0, np.zeros(D-1)])

    a = _mk_item("fact", "A")
    b = _mk_item("fact", "B")
    c = _mk_item("fact", "C")

    # Make sims different, but strength will dominate (alpha=0)
    va, vb, vc = q, _norm(q + 0.2 * np.eye(1, D, 1).ravel()), _norm(np.random.default_rng(1).normal(size=D))
    a.strength = 0.2
    b.strength = 0.9  # highest strength, should rank first
    c.strength = 0.1

    _seed_items_with_vecs(store, [a, b, c], [va, vb, vc])

    out = retrieve(store, query_vec=q, alpha=0.0, top_k=3, reinforce=False)
    assert [it.content for it in out][0] == "B"


def test_reinforcement_updates_freq_and_last_access_and_strength():
    store = InMemoryStore()
    D = 24
    q = _norm(np.r_[1.0, np.zeros(D-1)])

    it = _mk_item("fact", "hello")
    it.strength = 0.0
    it.freq = 0
    v = q
    _seed_items_with_vecs(store, [it], [v])

    out = retrieve(store, query_vec=q, alpha=1.0, top_k=1, reinforce=True)
    assert len(out) == 1
    got = out[0]
    assert got.freq >= 1
    assert got.last_access is not None
    # strength should be nonzero after reinforcement
    assert got.strength > 0.0

    # Ensure the store actually persisted the change
    got2 = store.get_items([got.id])[0]
    assert got2.freq == got.freq
    assert got2.last_access == got.last_access
    assert got2.strength == got.strength


def test_kind_filter_and_meta_filter_work_together():
    store = InMemoryStore()
    D = 20
    q = _norm(np.r_[1.0, np.zeros(D-1)])

    a = _mk_item("fact", "A", topic="cars")                  # scalar meta
    b = _mk_item("rule", "B", topic=["cars", "racing"])      # list meta
    c = _mk_item("fact", "C", topic="music")

    va, vb, vc = q, q, _norm(np.random.default_rng(2).normal(size=D))
    _seed_items_with_vecs(store, [a, b, c], [va, vb, vc])

    # Filter to kind 'rule' and topic 'cars' (b should match via list overlap)
    out = retrieve(
        store,
        query_vec=q,
        kinds=["rule"],
        meta_filter={"topic": "cars"},
        top_k=5,
        reinforce=False,
    )
    assert [it.content for it in out] == ["B"]


def test_mmr_promotes_diversity_over_near_duplicates():
    store = InMemoryStore()
    D = 32
    rng = np.random.default_rng(123)
    q = _norm(rng.normal(size=D))

    # s1 and s2 are very similar to each other and to q
    s1 = _mk_item("fact", "s1")
    s2 = _mk_item("fact", "s2")
    diverse = _mk_item("fact", "diverse")

    vs1 = _norm(q + 0.01 * rng.normal(size=D))
    vs2 = _norm(q + 0.011 * rng.normal(size=D))  # almost same direction as s1
    vdiv = _norm(rng.normal(size=D))            # different direction

    _seed_items_with_vecs(store, [s1, s2, diverse], [vs1, vs2, vdiv])

    # Without MMR, blend by sim: s1, s2, diverse
    out_no_mmr = retrieve(store, query_vec=q, alpha=1.0, top_k=3, use_mmr=False, reinforce=False)
    assert [it.content for it in out_no_mmr][:2] == ["s1", "s2"]

    # With MMR and k=2, expect ["s1", "diverse"] rather than the redundant "s2"
    out_mmr = retrieve(store, query_vec=q, alpha=1.0, top_k=2, use_mmr=True, mmr_lambda=0.5, reinforce=False)
    assert [it.content for it in out_mmr] == ["s1", "diverse"]


def test_overfetch_and_top_k_do_not_exceed_limits():
    store = InMemoryStore()
    D = 16
    q = _norm(np.r_[1.0, np.zeros(D-1)])

    items = [_mk_item("fact", f"i{i}") for i in range(10)]
    vecs = [q] + [_norm(np.random.default_rng(i+10).normal(size=D)) for i in range(9)]
    _seed_items_with_vecs(store, items, vecs)

    out = retrieve(store, query_vec=q, top_k=5, overfetch=10, reinforce=False)
    assert len(out) == 5  # never exceeds top_k


def test_score_only_returns_scores_and_does_not_reinforce():
    store = InMemoryStore()
    D = 12
    q = _norm(np.r_[1.0, np.zeros(D-1)])

    a = _mk_item("fact", "A"); a.freq = 0; a.strength = 0.0
    _seed_items_with_vecs(store, [a], [q])

    res = score_only(store, query_vec=q, top_k=1)
    assert len(res) == 1
    item, score = res[0]
    assert isinstance(item, MemoryItem)
    assert isinstance(score, float)

    # Ensure no reinforcement happened
    fresh = store.get_items([a.id])[0]
    assert fresh.freq == 0
    assert fresh.last_access is None
    assert fresh.strength == 0.0


def test_query_text_path_is_accepted_and_returns_results():
    # This test exercises the query_text->embedding path without asserting exact ranking.
    store = InMemoryStore()
    D = 24
    rng = np.random.default_rng(99)

    # Create two items with distinct content
    a = _mk_item("fact", "alpha content about cars")
    b = _mk_item("fact", "beta content about cooking")

    # Give them different random vectors; we just want a stable, non-empty result.
    _seed_items_with_vecs(store, [a, b], [rng.normal(size=D), rng.normal(size=D)])

    out = retrieve(store, query_text="cars and engines", top_k=2, reinforce=False)
    assert len(out) >= 1  # at least something comes back; path works


def test_no_hits_returns_empty_list():
    store = InMemoryStore()
    q = _norm(np.ones(8, dtype=np.float32))
    # store is empty -> ann_search returns []
    assert retrieve(store, query_vec=q, top_k=3, reinforce=False) == []
