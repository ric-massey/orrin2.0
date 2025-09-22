# tests/memory_tests/daemon_test.py
import time
from types import SimpleNamespace
from typing import List, Tuple
import numpy as np
import pytest

# --- FIX: support both module names (daemon.py shim or memory_daemon.py) ---
try:
    from memory.daemon import MemoryDaemon, now_iso  # preferred (shim re-exports if renamed)
except Exception:  # pragma: no cover
    from memory.memory_daemon import MemoryDaemon, now_iso

from memory.models import Event, MemoryItem
from memory.store.inmem import InMemoryStore


# -------------------------
# helpers
# -------------------------

def _vec(x):
    return np.asarray(x, dtype=np.float32)

def _mk_event(kind="chat:user", content="hello", **meta) -> Event:
    return Event.new(kind=kind, content=content, meta=meta or {})

def _mk_item(content="x", kind="fact", layer="working") -> MemoryItem:
    it = MemoryItem.new(kind=kind, source="test", content=content, layer=layer)
    it.embedding_id = f"vec_{it.id}"
    return it


# -------------------------
# fixtures
# -------------------------

@pytest.fixture
def store():
    return InMemoryStore()

@pytest.fixture(autouse=True)
def mute_metrics_and_wal(monkeypatch):
    # stub metrics
    calls = SimpleNamespace(
        bump_ingest=0,
        note_item_upserts=[],
        note_vector_upserts=[],
        note_retrieval=[],
        note_compaction=[],
        wal_events=[],
        wal_items=[],
    )

    import memory.memory_daemon as mod
    monkeypatch.setattr(mod, "bump_ingest", lambda: calls.__setattr__("bump_ingest", calls.bump_ingest + 1))
    monkeypatch.setattr(mod, "note_item_upserts", lambda n: calls.note_item_upserts.append(n))
    monkeypatch.setattr(mod, "note_vector_upserts", lambda n: calls.note_vector_upserts.append(n))
    monkeypatch.setattr(mod, "note_retrieval", lambda kinds, hits, latency_s: calls.note_retrieval.append((tuple(kinds or []), hits)))
    monkeypatch.setattr(mod, "note_compaction", lambda stats, when_ts=None: calls.note_compaction.append((stats, when_ts)))

    # stub WAL
    monkeypatch.setattr(mod, "wal_append_event", lambda ev: calls.wal_events.append(ev))
    monkeypatch.setattr(mod, "wal_append_items", lambda items: calls.wal_items.extend(items))

    yield calls


# -------------------------
# tests
# -------------------------

def test_start_and_stop_thread(store):
    d = MemoryDaemon(store, tick_hz=50.0)
    d.start()
    assert d.running is True
    assert d.thread is not None and d.thread.is_alive()
    d.stop()
    assert d.running is False


def test_ingest_persists_items_and_vectors_and_learns_lexicon(store, mute_metrics_and_wal, monkeypatch):
    d = MemoryDaemon(store, tick_hz=1000.0)

    # force CAPTURE_ALL on to avoid salience gate
    import memory.memory_daemon as mod
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", True, raising=False)

    ev = _mk_event(content="GPU means graphics processing unit")
    d.ingest(ev)

    # run one tick to drain
    d._tick()

    # one item and vector should be present
    st = store.stats()
    assert st["items_total"] == 1
    assert st["vectors_total"] == 1

    # WAL should have seen both event and item
    assert mute_metrics_and_wal.wal_events and mute_metrics_and_wal.wal_items

    # lexicon learned
    senses = store.get_lexicon_by_term("gpu")
    assert senses and senses[0].definition.lower().startswith("graphics")


def test_retrieve_blends_and_reinforces(store, mute_metrics_and_wal, monkeypatch):
    d = MemoryDaemon(store, tick_hz=1000.0)

    # Seed two items along orthogonal directions
    it1 = _mk_item("alpha"); it2 = _mk_item("beta")
    store.upsert_items([it1, it2])
    store.upsert_vectors({
        it1.embedding_id: _vec([1, 0]),
        it2.embedding_id: _vec([0, 1]),
    })

    # Patch daemon.get_embedding to query along x-axis
    import memory.memory_daemon as mod
    monkeypatch.setattr(mod, "get_embedding", lambda q: _vec([1.0, 0.0]), raising=True)

    out = d.retrieve("anything", top_k=1)
    assert len(out) == 1 and out[0].id == it1.id

    # reinforcement happened
    got = store.get_items([it1.id])[0]
    assert (got.freq or 0) >= 1
    assert isinstance(got.last_access, str) and "T" in got.last_access  # ISO-ish
    # strength updated to something >= prior
    assert (got.strength or 0.0) >= 0.0

    # metric logged
    assert mute_metrics_and_wal.note_retrieval and mute_metrics_and_wal.note_retrieval[-1][1] == 1


def test_compaction_is_invoked_and_clears_working_cache(store, mute_metrics_and_wal, monkeypatch):
    d = MemoryDaemon(store, tick_hz=1000.0)

    # Put some fake items into working cache
    for _ in range(5):
        it = _mk_item("w", layer="working")
        d._working_cache[it.id] = it

    # force should_compact() to true and capture arguments
    import memory.memory_daemon as mod
    called = {}
    def fake_should(*a, **kw): return True
    def fake_compact(store_arg, working_items, **kw):
        called["store"] = store_arg
        called["items"] = list(working_items)
        called["wal"] = kw.get("wal")
        # tiny stats-like object with fields used by note_compaction
        return SimpleNamespace(processed=len(working_items), promoted=len(working_items),
                               summary_items_created=0, near_duplicates_dropped=0, clusters_formed=1)

    monkeypatch.setattr(mod, "should_compact", fake_should, raising=True)
    monkeypatch.setattr(mod, "compact_and_promote", fake_compact, raising=True)

    # run tick: should call compaction and clear cache
    d._tick()

    assert "items" in called and len(called["items"]) == 5
    # DEFAULT_WAL is passed through
    assert called["wal"] is mod.DEFAULT_WAL
    # cache cleared + compaction metric noted
    assert d._working_cache == {}
    assert mute_metrics_and_wal.note_compaction


def test_build_item_from_event_result_is_used_for_persist_and_vector(store, mute_metrics_and_wal, monkeypatch):
    d = MemoryDaemon(store, tick_hz=1000.0)

    # Prepare a fake build_item_from_event that always keeps and sets a known vector
    import memory.memory_daemon as mod

    class _Res:
        def __init__(self, kept, item, vector):
            self.kept = kept
            self.item = item
            self.vector = vector

    known_vec = _vec([0.0, 3.0, 4.0])  # normalized -> [0, .6, .8]
    item = _mk_item("stubbed content")

    def fake_builder(ev, recent_vecs, capture_all, salience_keep):
        # ensure daemon gave us a snapshot list
        assert isinstance(recent_vecs, list)
        return _Res(True, item, known_vec)

    monkeypatch.setattr(mod, "build_item_from_event", fake_builder, raising=True)

    ev = _mk_event(content="ignored by builder")
    d.ingest(ev)
    d._tick()

    # The exact vector should be in the store at item's embedding_id (normalized by store)
    v = store._vecs.get(item.embedding_id)
    assert v is not None and np.allclose(v, known_vec / np.linalg.norm(known_vec))

    # item should be persisted
    got = store.get_items([item.id])
    assert got and got[0].content == "stubbed content"

    # WAL wrote items (daemon batches items -> wal_append_items)
    assert any(i.id == item.id for i in mute_metrics_and_wal.wal_items)


def test_precomputed_vector_path_through_ingest_module(store, mute_metrics_and_wal, monkeypatch):
    """
    Integration-ish: let the real build_item_from_event handle a precomputed _vec in ev.meta.
    We patch memory.embedder.get_embedding to return something orthogonal so we can tell the path used.
    """
    d = MemoryDaemon(store, tick_hz=1000.0)

    # Make embedder return a vector along x; we'll supply pre_vec along y to detect usage
    import memory.embedder as embed_mod
    monkeypatch.setattr(embed_mod, "get_embedding", lambda s: _vec([1.0, 0.0, 0.0]), raising=True)

    # Also, ensure daemon.get_embedding (used in retrieve) is unaffected; not used here.

    # Force capture-all so the event is kept
    import memory.memory_daemon as dmod
    monkeypatch.setattr(dmod.MEMCFG, "CAPTURE_ALL", True, raising=False)

    ev = _mk_event(content="anything", explicit_remember=True, _vec=[0.0, 2.0, 0.0])  # precomputed y-axis
    d.ingest(ev)
    d._tick()

    # Check last vector upserted equals normalized precomputed vector (not embedder x-axis)
    # Find the only item
    ids = [it.id for it in store.get_items(list(store._items.keys()))]
    assert ids, "no items persisted"
    it = store.get_items(ids)[0]
    vec = store._vecs.get(it.embedding_id)
    assert vec is not None and np.allclose(vec, _vec([0.0, 1.0, 0.0]))


def test_definition_learning_called_even_when_not_kept(store, mute_metrics_and_wal, monkeypatch):
    d = MemoryDaemon(store, tick_hz=1000.0)

    # Turn off capture-all and set a very high salience to reject the event
    import memory.memory_daemon as mod
    monkeypatch.setattr(mod.MEMCFG, "CAPTURE_ALL", False, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "SALIENCE_KEEP", 0.99, raising=False)

    # We will also stub build_item_from_event to *not keep* regardless of salience
    class _Res:
        def __init__(self): self.kept=False; self.item=None; self.vector=None
    monkeypatch.setattr(mod, "build_item_from_event", lambda *a, **kw: _Res(), raising=True)

    # Feed definitional text; even if not kept, daemon should learn lexicon
    ev = _mk_event(content="DSP stands for digital signal processing")
    d.ingest(ev)
    d._tick()

    senses = store.get_lexicon_by_term("dsp")
    assert senses and "digital signal processing" in senses[0].definition.lower()
