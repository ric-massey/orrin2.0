# tests/memory_tests/health_test.py
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from memory.health import (
    StoreStats,
    collect_store_stats,
    assess_health,
    snapshot,
)
from memory.models import MemoryItem


# ---------------------------
# helpers
# ---------------------------

class DuckStore:
    """Duck-typed InMemory-like store exposing _items/_vecs only."""
    def __init__(self):
        self._items = {}
        self._vecs = {}

class StatsStore:
    """Store that exposes a .stats() dict the health module should prefer."""
    def __init__(self, d):
        self._d = dict(d)
    def stats(self):
        return dict(self._d)


def _mk_item(layer="working", with_emb=False) -> MemoryItem:
    it = MemoryItem.new(kind="fact", source="test", content="x", layer=layer)
    if with_emb:
        it.embedding_id = f"vec_{it.id}"
    return it


# ---------------------------
# collect_store_stats
# ---------------------------

def test_collect_store_stats_prefers_stats_method():
    expected = {
        "items_total": 5,
        "items_by_layer": {"working": 1, "long": 3, "summary": 1},
        "index_lag": 7,
        "vectors_total": 99,
        "vector_bytes_total": 123456,
        "gc_eligible": 4,
    }
    ss = collect_store_stats(StatsStore(expected))
    assert isinstance(ss, StoreStats)
    assert ss.items_total == expected["items_total"]
    assert ss.items_by_layer == expected["items_by_layer"]
    assert ss.index_lag == expected["index_lag"]
    assert ss.vectors_total == expected["vectors_total"]
    assert ss.vector_bytes_total == expected["vector_bytes_total"]
    assert ss.gc_eligible == expected["gc_eligible"]


def test_collect_store_stats_duck_typed_counts_layers_index_lag_and_bytes():
    st = DuckStore()

    # items: 2 working (1 with emb id), 1 long with emb id, 1 summary no emb id
    w1 = _mk_item("working", with_emb=True)
    w2 = _mk_item("working", with_emb=False)
    l1 = _mk_item("long", with_emb=True)
    s1 = _mk_item("summary", with_emb=False)

    # only l1’s vector is present; w1 is "lagging"
    st._items = {i.id: i for i in (w1, w2, l1, s1)}
    st._vecs = {
        l1.embedding_id: np.zeros(8, dtype=np.float32),
    }

    ss = collect_store_stats(st)
    assert ss.items_total == 4
    assert ss.items_by_layer == {"working": 2, "long": 1, "summary": 1}
    assert ss.vectors_total == 1
    assert ss.index_lag == 1  # w1 has emb id but missing vector
    # vector_bytes_total should reflect numpy nbytes
    assert ss.vector_bytes_total == st._vecs[l1.embedding_id].nbytes


def test_collect_store_stats_vec_bytes_fallback_without_nbytes():
    st = DuckStore()
    it = _mk_item("working", with_emb=True)
    st._items = {it.id: it}
    # Use a plain list so _approx_vec_nbytes falls back to len*4
    st._vecs = {it.embedding_id: [0] * 10}
    ss = collect_store_stats(st)
    assert ss.vectors_total == 1
    assert ss.vector_bytes_total == 40  # 10 * 4


# ---------------------------
# assess_health
# ---------------------------

def test_assess_health_ok_when_under_thresholds(monkeypatch):
    import memory.health as mod
    # Soften thresholds high so we don't trigger anything
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_INDEX_LAG_SOFT", 10_000, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_COMPACTION_STALLED_MIN", 9_999, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_FLUSH_FAILURES_SOFT", 99, raising=False)

    ss = StoreStats(items_total=1, items_by_layer={"working": 1, "long": 0, "summary": 0},
                    index_lag=0, vectors_total=0, vector_bytes_total=0)
    status, signals, notes = assess_health(
        store_stats=ss,
        working_cache_size=0,
        last_compaction_ts=time.time(),
        flush_failures=0,
        now=time.time(),
    )
    assert status == "ok"
    assert isinstance(signals, dict) and "memory.index_lag" in signals
    assert notes == []


def test_assess_health_warns_on_each_individual_issue(monkeypatch):
    import memory.health as mod
    # low thresholds so we trigger
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_INDEX_LAG_SOFT", 1, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_COMPACTION_STALLED_MIN", 10, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_FLUSH_FAILURES_SOFT", 2, raising=False)

    # 1) index lag only
    ss = StoreStats(index_lag=2, items_by_layer={"working":0,"long":0,"summary":0})
    status, _, notes = assess_health(ss, working_cache_size=0, last_compaction_ts=time.time(), flush_failures=0, now=time.time())
    assert status == "warn"
    assert any("index_lag" in n for n in notes)

    # 2) compaction stalled only (reset ss so only this issue triggers)
    ss = StoreStats(index_lag=0, items_by_layer={"working":0,"long":0,"summary":0})
    t_now = 2000.0
    status, _, notes = assess_health(ss, working_cache_size=0, last_compaction_ts=t_now - (11*60), flush_failures=0, now=t_now)
    assert status == "warn"
    assert any("compaction stalled" in n for n in notes)

    # 3) flush failures only (keep others under thresholds)
    ss = StoreStats(index_lag=0, items_by_layer={"working":0,"long":0,"summary":0})
    status, _, notes = assess_health(ss, working_cache_size=0, last_compaction_ts=t_now, flush_failures=3, now=t_now)
    assert status == "warn"
    assert any("flush failures" in n for n in notes)


def test_assess_health_escalates_to_error_when_multiple_issues(monkeypatch):
    import memory.health as mod
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_INDEX_LAG_SOFT", 1, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_COMPACTION_STALLED_MIN", 10, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_FLUSH_FAILURES_SOFT", 2, raising=False)

    ss = StoreStats(index_lag=5, items_by_layer={"working":0,"long":0,"summary":0})
    # Trigger index_lag and compaction stalled (2 issues -> error)
    t_now = 10_000.0
    status, signals, notes = assess_health(
        ss,
        working_cache_size=123,
        last_compaction_ts=t_now - (20 * 60),
        flush_failures=0,
        now=t_now,
    )
    assert status == "error"
    assert signals["memory.index_lag"] == 5
    assert signals["memory.compaction_stalled_min"] >= 20.0
    assert any("index_lag" in n for n in notes) and any("compaction stalled" in n for n in notes)


# ---------------------------
# snapshot wrapper
# ---------------------------

def test_snapshot_packages_status_signals_notes_and_store_stats(monkeypatch):
    # Build a duck store with one lagging item and one vector
    st = DuckStore()
    it = _mk_item("working", with_emb=True)
    st._items[it.id] = it
    st._vecs["other_vec"] = np.zeros(4, dtype=np.float32)

    import memory.health as mod
    # Thresholds such that index_lag alone causes 'warn'
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_INDEX_LAG_SOFT", 0, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_COMPACTION_STALLED_MIN", 9999, raising=False)
    monkeypatch.setattr(mod.MEMCFG, "HEALTH_FLUSH_FAILURES_SOFT", 9999, raising=False)

    now = 5000.0
    snap = snapshot(
        st,
        working_cache_size=42,
        last_compaction_ts=4900.0,
        flush_failures=1,
        now=now,
    )

    assert set(snap.keys()) == {"status", "signals", "notes", "store_stats"}
    assert snap["status"] in {"ok", "warn", "error"}
    assert isinstance(snap["signals"], dict)
    for k in [
        "memory.index_lag", "memory.compaction_stalled_min", "memory.working_cache",
        "memory.flush_failures", "memory.items.working", "memory.items.long",
        "memory.items.summary", "memory.vectors.total", "memory.vectors.bytes"
    ]:
        assert k in snap["signals"]
    assert isinstance(snap["notes"], list)
    assert isinstance(snap["store_stats"], StoreStats)
    # minutes since compaction should be ~ (5000-4900)/60
    assert snap["signals"]["memory.compaction_stalled_min"] == pytest.approx((now - 4900.0)/60.0, rel=1e-6)
