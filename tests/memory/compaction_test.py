# tests/memory_tests/compaction_test.py
import numpy as np
import pytest
import types

from memory.compaction import (
    should_compact,
    compact_and_promote,
    CompactionStats,
)
from memory.models import MemoryItem


# ------------------------
# Deterministic embed stub
# ------------------------
def _embed_stub(text: str) -> np.ndarray:
    """
    Map keywords to fixed directions so cluster/dup logic is predictable:
      - 'apple'       -> e1
      - 'apple mix'   -> e1+e2
      - 'banana'      -> e2
      - otherwise     -> zero
    """
    t = (text or "").lower()
    if "apple mix" in t:
        return np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    if "apple" in t:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if "banana" in t:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    # Patch compaction.get_embedding to deterministic stub
    import memory.compaction as mod
    monkeypatch.setattr(mod, "get_embedding", _embed_stub, raising=True)
    yield


# ------------------------
# Tiny fake store + WAL
# ------------------------
class FakeStore:
    """
    Minimal store used by compaction:
      - _items: id -> MemoryItem
      - _vecs:  embedding_id -> np.ndarray (normalized expected)
    """
    def __init__(self):
        self._items = {}
        self._vecs = {}
        self.upsert_items_calls = 0
        self.upsert_vectors_calls = 0

    def upsert_items(self, items):
        self.upsert_items_calls += 1
        for it in items:
            self._items[it.id] = it

    def upsert_vectors(self, vec_map):
        self.upsert_vectors_calls += 1
        for eid, v in vec_map.items():
            self._vecs[eid] = np.asarray(v, dtype=np.float32).reshape(-1)

class FakeWal:
    def __init__(self, raise_on=False):
        self.appended = []
        self.raise_on = raise_on
    def append_items(self, items):
        if self.raise_on:
            raise RuntimeError("WAL write failure (test)")
        self.appended.extend(items)


# ------------------------
# Helpers
# ------------------------
def _mk_item(text: str, *, layer="working", kind="fact") -> MemoryItem:
    it = MemoryItem.new(kind=kind, source="test", content=text, layer=layer)
    return it


# ------------------------
# should_compact tests
# ------------------------
def test_should_compact_by_cap_or_first_run(monkeypatch):
    # cap reached
    assert should_compact(working_cache_size=10, last_ts=123.0, cap=10, interval_minutes=60) is True
    # first run (last_ts <= 0)
    assert should_compact(working_cache_size=0, last_ts=0.0, cap=999, interval_minutes=60) is True

def test_should_compact_by_interval(monkeypatch):
    import memory.compaction as mod
    # make "now" be 600.0
    monkeypatch.setattr(mod.time, "time", lambda: 600.0, raising=False)
    # last compaction at t=540 -> 60s -> 1 min, equals interval → True
    assert should_compact(working_cache_size=1, last_ts=540.0, cap=999, interval_minutes=1) is True
    # last compaction at t=570 -> 30s < 60s → False
    assert should_compact(working_cache_size=1, last_ts=570.0, cap=999, interval_minutes=1) is False


# ------------------------
# compaction core behavior
# ------------------------
def test_compact_promotes_clusters_creates_summaries_and_drops_near_dups():
    st = FakeStore()
    wal = FakeWal()

    # Apple cluster: pivot, near-dup (cos=1.0), and related (apple mix cos≈0.707)
    i1 = _mk_item("apple base")
    i2 = _mk_item("apple dup")    # same vector as apple → near dup vs pivot
    i3 = _mk_item("apple mix")    # related but not a near-dup

    # Banana cluster: two related
    j1 = _mk_item("banana one")
    j2 = _mk_item("banana two")

    working = [i1, i2, i3, j1, j2]

    stats = compact_and_promote(
        st,
        working,
        sim_threshold=0.60,         # apple mix joins apple cluster
        duplicate_sim=0.96,         # i2 dropped as near duplicate of i1
        min_cluster_size=2,
        max_bullets=5,
        bullet_chars=40,
        promote_layer="long",
        wal=wal,
    )

    # --- stats ---
    assert isinstance(stats, CompactionStats)
    assert stats.processed == len(working)
    assert stats.clusters_formed == 2
    assert stats.near_duplicates_dropped == 1  # only i2 dropped
    # promoted: kept apples = {i1, i3} (2) + bananas {j1, j2} (2) = 4
    assert stats.promoted == 4
    assert stats.summary_items_created == 2

    # --- items promoted to 'long' ---
    for it in (i1, i3, j1, j2):
        assert it.layer == "long"

    # --- summaries written, vectors upserted ---
    summaries = [x for x in st._items.values() if getattr(x, "kind", "") == "summary"]
    assert len(summaries) == 2
    for sm in summaries:
        assert sm.layer == "summary"
        assert sm.strength == pytest.approx(0.30)
        assert sm.summary_of and all(isinstance(x, str) for x in sm.summary_of)
        assert sm.embedding_id and sm.embedding_id in st._vecs
        assert isinstance(sm.embedding_dim, int) and sm.embedding_dim > 0
        assert sm.content.startswith("Summary of related items:\n")
        # bullets should be <= max_bullets and each line begins with '• '
        bullet_lines = [ln for ln in sm.content.splitlines() if ln.startswith("• ")]
        assert 1 <= len(bullet_lines) <= 5

    # --- WAL observed both promotions and summaries ---
    wal_ids = {it.id for it in wal.appended}
    promoted_ids = {i1.id, i3.id, j1.id, j2.id}
    assert promoted_ids.issubset(wal_ids)
    assert any(it.kind == "summary" for it in wal.appended)


def test_small_cluster_promotes_without_summary():
    st = FakeStore()
    wal = FakeWal()

    # Single item → cluster size 1 < min_cluster_size (2) → no summary
    it = _mk_item("apple solo")
    working = [it]

    stats = compact_and_promote(
        st,
        working,
        sim_threshold=0.7,
        duplicate_sim=0.95,
        min_cluster_size=2,
        max_bullets=3,
        bullet_chars=30,
        promote_layer="long",
        wal=wal,
    )

    assert stats.processed == 1
    assert stats.promoted == 1
    assert stats.summary_items_created == 0
    assert it.layer == "long"
    assert not [x for x in st._items.values() if getattr(x, "kind", "") == "summary"]


def test_store_vecs_are_preferred_over_content_embeddings_for_clustering(monkeypatch):
    """
    If an item has embedding_id and store._vecs contains that id, compaction should
    reuse that vector (and NOT recompute from content). We verify by forcing two
    'apple' items to be *far* apart via store._vecs so they do NOT cluster.
    """
    st = FakeStore()
    wal = FakeWal()

    a1 = _mk_item("apple base")
    a2 = _mk_item("apple base")
    # Give them embedding ids and custom vectors that are orthogonal
    a1.embedding_id = f"vec_{a1.id}"
    a2.embedding_id = f"vec_{a2.id}"
    st._vecs[a1.embedding_id] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)  # e1
    st._vecs[a2.embedding_id] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)  # e2

    stats = compact_and_promote(
        st,
        [a1, a2],
        sim_threshold=0.90,        # would cluster if using content (both 'apple')
        duplicate_sim=0.99,
        min_cluster_size=2,
        max_bullets=5,
        bullet_chars=40,
        promote_layer="long",
        wal=wal,
    )

    # With store vectors orthogonal, we should get two separate clusters
    assert stats.clusters_formed == 2
    # Neither cluster reaches size 2 → both promoted individually, no summaries
    assert stats.promoted == 2
    assert stats.summary_items_created == 0


def test_wal_failures_are_swallowed():
    st = FakeStore()
    wal = FakeWal(raise_on=True)

    i1 = _mk_item("apple base")
    i2 = _mk_item("banana base")

    # Should not raise even though WAL raises in append_items
    stats = compact_and_promote(
        st,
        [i1, i2],
        sim_threshold=0.5,
        duplicate_sim=0.95,
        min_cluster_size=2,
        max_bullets=3,
        bullet_chars=30,
        promote_layer="long",
        wal=wal,
    )
    assert stats.processed == 2
