# tests/memory_tests/memory_metrics_test.py
import types
import time
import numpy as np  # just to ensure numpy interplay doesn't break anything
import pytest

import memory.metrics as m


# --------------------------
# Fakes
# --------------------------
class FakeCounter:
    def __init__(self):
        self.total = 0
        self.label_calls = []
        self.labelled = {}  # key -> FakeCounter
    def labels(self, *args):
        key = tuple(args)
        self.label_calls.append(key)
        c = self.labelled.get(key)
        if c is None:
            c = FakeCounter()
            self.labelled[key] = c
        return c
    def inc(self, n=1):
        self.total += n


class _ChildGauge:
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        self.last = None
        self.calls = 0
    def set(self, v):
        self.calls += 1
        self.last = v
        self.parent.children[self.key] = v


class FakeGauge:
    def __init__(self):
        self.last = None
        self.calls = 0
        self.children = {}  # (labels...) -> value
    def labels(self, *labels):
        return _ChildGauge(self, tuple(labels))
    def set(self, v):
        self.calls += 1
        self.last = v


class FakeHistogram:
    def __init__(self):
        self.values = []
    def observe(self, v):
        self.values.append(float(v))


# --------------------------
# Helpers
# --------------------------
def _reset_server(monkeypatch, has_prom=False):
    # Control Prometheus branch
    monkeypatch.setattr(m, "_HAS_PROM", has_prom, raising=True)
    # Reset idempotence
    monkeypatch.setattr(m, "_server_started", False, raising=True)
    # Make a dummy registry object for generate_latest calls
    monkeypatch.setattr(m, "_registry", object(), raising=True)


# --------------------------
# Tests: counters & basic helpers
# --------------------------
def test_bump_ingest_and_upserts_increment(monkeypatch):
    fc_ingest = FakeCounter()
    fc_items = FakeCounter()
    fc_vecs = FakeCounter()
    monkeypatch.setattr(m, "ingest_events_total", fc_ingest, raising=True)
    monkeypatch.setattr(m, "items_upserts_total", fc_items, raising=True)
    monkeypatch.setattr(m, "vectors_upserts_total", fc_vecs, raising=True)

    m.bump_ingest(3)
    m.note_item_upserts(2)
    m.note_vector_upserts(5)

    assert fc_ingest.total == 3
    assert fc_items.total == 2
    assert fc_vecs.total == 5


def test_note_retrieval_counts_and_latency_with_labels(monkeypatch):
    fq = FakeCounter()
    fh = FakeCounter()
    hist = FakeHistogram()
    monkeypatch.setattr(m, "retrieval_queries_total", fq, raising=True)
    monkeypatch.setattr(m, "retrieval_hits_total", fh, raising=True)
    monkeypatch.setattr(m, "retrieval_latency_seconds", hist, raising=True)

    # kinds sorted + joined
    m.note_retrieval(["goal", "fact"], hits=7, latency_s=0.123)
    # None kinds -> "any"
    m.note_retrieval(None, hits=0, latency_s=None)

    # label usage recorded
    assert ("fact", "goal") in fq.labelled or ("goal", "fact") in fq.labelled
    # "any" bucket used
    assert ("any",) in fq.labelled

    # hits aggregated
    assert fh.total == 7 + 0

    # latency recorded once
    assert len(hist.values) == 1 and abs(hist.values[0] - 0.123) < 1e-6


def test_note_compaction_updates_counters_and_ts(monkeypatch):
    comp = FakeCounter()
    sums = FakeCounter()
    dups = FakeCounter()
    ts_g = FakeGauge()
    monkeypatch.setattr(m, "compactions_total", comp, raising=True)
    monkeypatch.setattr(m, "summaries_created_total", sums, raising=True)
    monkeypatch.setattr(m, "duplicates_folded_total", dups, raising=True)
    monkeypatch.setattr(m, "last_compaction_ts", ts_g, raising=True)

    class Stats:
        summary_items_created = 4
        near_duplicates_dropped = 9

    m.note_compaction(Stats(), when_ts=123.456)

    assert comp.total == 1
    assert sums.total == 4
    assert dups.total == 9
    assert abs(ts_g.last - 123.456) < 1e-9


def test_set_health_gauges_sets_all(monkeypatch):
    g_idx = FakeGauge()
    g_vt = FakeGauge()
    g_vb = FakeGauge()
    g_wc = FakeGauge()
    g_items = FakeGauge()
    g_last = FakeGauge()
    monkeypatch.setattr(m, "index_lag", g_idx, raising=True)
    monkeypatch.setattr(m, "vectors_total", g_vt, raising=True)
    monkeypatch.setattr(m, "vectors_bytes", g_vb, raising=True)
    monkeypatch.setattr(m, "working_cache_size", g_wc, raising=True)
    monkeypatch.setattr(m, "items_by_layer", g_items, raising=True)
    monkeypatch.setattr(m, "last_compaction_ts", g_last, raising=True)

    m.set_health_gauges(
        idx_lag=10, vec_total=111, vec_bytes=2222,
        working_cache=33, items_working=44, items_long=55, items_summary=66,
        last_compact=777.0
    )

    assert g_idx.last == 10
    assert g_vt.last == 111
    assert g_vb.last == 2222
    assert g_wc.last == 33
    # labelled children
    assert g_items.children.get(("working",)) == 44
    assert g_items.children.get(("long",)) == 55
    assert g_items.children.get(("summary",)) == 66
    assert g_last.last == 777.0


def test_timer_context_manager_records_histogram():
    h = FakeHistogram()
    with m.timer(h):
        time.sleep(0.001)  # tiny
    assert len(h.values) == 1
    assert h.values[0] >= 0.0


# --------------------------
# Tests: metrics server & dump_text
# --------------------------
def test_start_metrics_server_without_prom(monkeypatch):
    _reset_server(monkeypatch, has_prom=False)
    assert m.start_metrics_server(9999) is False  # no-op branch


def test_start_metrics_server_with_prom_idempotent(monkeypatch):
    _reset_server(monkeypatch, has_prom=True)
    calls = {"n": 0}

    def fake_start_http_server(port, registry):
        calls["n"] += 1
        assert isinstance(port, int)
        assert registry is m._registry

    monkeypatch.setattr(m, "start_http_server", fake_start_http_server, raising=True)

    # first time starts
    assert m.start_metrics_server(8088) is True
    # second time returns True without calling again
    assert m.start_metrics_server(8088) is True
    assert calls["n"] == 1


def test_dump_text_without_prom_returns_none(monkeypatch):
    _reset_server(monkeypatch, has_prom=False)
    assert m.dump_text() is None


def test_dump_text_with_prom_returns_bytes(monkeypatch):
    _reset_server(monkeypatch, has_prom=True)
    monkeypatch.setattr(m, "generate_latest", lambda reg: b"PROM", raising=True)
    out = m.dump_text()
    assert out == b"PROM"


# --------------------------
# Tests: media-specific metric
# --------------------------
def test_note_media_save_increments_and_sets_bytes(monkeypatch):
    fc = FakeCounter()
    gg = FakeGauge()
    monkeypatch.setattr(m, "media_items_total", fc, raising=True)
    monkeypatch.setattr(m, "media_bytes_total", gg, raising=True)

    # This function uses internal state; just ensure no crash and calls happen
    m.note_media_save(1234)
    assert fc.total == 1
    assert gg.last is not None and gg.last >= 1234
