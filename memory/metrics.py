# memory/metrics.py
# Lightweight metrics for Orrin2.0 memory (Prometheus if available; safe no-op fallback). Counters/Gauges/Histograms + helpers and a timer() context.

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable
from contextlib import contextmanager
import time
import threading

# ---------- Optional Prometheus ----------
_HAS_PROM = False
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter as _PCounter,
        Gauge as _PGauge,
        Histogram as _PHistogram,
        generate_latest,
        start_http_server,
    )
    _HAS_PROM = True
except Exception:
    # We'll define compatible shims below
    _HAS_PROM = False


# ---------- Fallback shims ----------
class _NoopMetric:
    def __init__(self, *_args, **_kw): pass
    def labels(self, *_args, **_kw): return self
    def inc(self, *_args, **_kw): return None
    def set(self, *_args, **_kw): return None
    def observe(self, *_args, **_kw): return None

class _NoopRegistry:
    def __init__(self): self._m = {}
    def register(self, name: str, metric: _NoopMetric): self._m[name] = metric
    def get(self, name: str): return self._m.get(name)

# ---------- Public Registry ----------
_registry = CollectorRegistry() if _HAS_PROM else _NoopRegistry()
_server_guard = threading.Lock()
_server_started = False

# ---------- Metric factories ----------
def _counter(name: str, desc: str, labels: Iterable[str] = ()):
    if _HAS_PROM:
        return _PCounter(name, desc, list(labels), registry=_registry)
    return _NoopMetric()

def _gauge(name: str, desc: str, labels: Iterable[str] = ()):
    if _HAS_PROM:
        return _PGauge(name, desc, list(labels), registry=_registry)
    return _NoopMetric()

def _histogram(name: str, desc: str, labels: Iterable[str] = (), buckets: Optional[list[float]] = None):
    if _HAS_PROM:
        if buckets is None:
            # default buckets tuned for ms→s scale
            buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        return _PHistogram(name, desc, list(labels), buckets=buckets, registry=_registry)
    return _NoopMetric()

# ---------- Metrics (namespaced "orrin_memory_") ----------
ingest_events_total          = _counter("orrin_memory_ingest_events_total", "Number of events ingested into memory.")
items_upserts_total          = _counter("orrin_memory_items_upserts_total", "Number of MemoryItems upserted.")
vectors_upserts_total        = _counter("orrin_memory_vectors_upserts_total", "Number of vectors upserted into the index.")

retrieval_queries_total      = _counter("orrin_memory_retrieval_queries_total", "Number of retrieval queries issued.", labels=["kinds"])
retrieval_hits_total         = _counter("orrin_memory_retrieval_hits_total", "Number of items returned by retrieval.")
retrieval_latency_seconds    = _histogram("orrin_memory_retrieval_latency_seconds", "Latency of retrieval end-to-end in seconds.")

compactions_total            = _counter("orrin_memory_compactions_total", "Number of compaction/promote passes executed.")
summaries_created_total      = _counter("orrin_memory_summaries_created_total", "Summary MemoryItems created by compaction.")
duplicates_folded_total      = _counter("orrin_memory_duplicates_folded_total", "Logical near-duplicates folded into pivots.")

gc_purges_total              = _counter("orrin_memory_gc_purges_total", "Items purged by GC (post-summarization).")
lexicon_upserts_total        = _counter("orrin_memory_lexicon_upserts_total", "Lexicon senses upserted/updated.")

# Gauges for current state
index_lag                    = _gauge("orrin_memory_index_lag", "Count of items with missing vectors (index backlog).")
vectors_total                = _gauge("orrin_memory_vectors_total", "Total number of vectors in the index.")
vectors_bytes                = _gauge("orrin_memory_vectors_bytes", "Approximate bytes held by vectors.")

items_by_layer               = _gauge("orrin_memory_items_by_layer", "Number of items per layer.", labels=["layer"])
working_cache_size           = _gauge("orrin_memory_working_cache", "Size of daemon working cache (items).")
last_compaction_ts           = _gauge("orrin_memory_last_compaction_ts", "Epoch seconds of last compaction run.")

media_bytes_total            = _gauge("orrin_memory_media_bytes_total", "Raw bytes of media saved on disk.")
media_items_total            = _counter("orrin_memory_media_items_total", "Count of media items ingested.")

# ---------- Helpers ----------
def start_metrics_server(port: int = 8008) -> bool:
    """Start Prometheus HTTP server on given port (idempotent). Returns True if started."""
    global _server_started
    if not _HAS_PROM:
        return False
    with _server_guard:
        if _server_started:
            return True
        try:
            start_http_server(port, registry=_registry)
            _server_started = True
            return True
        except Exception:
            return False

@contextmanager
def timer(histogram_metric=retrieval_latency_seconds):
    """Context manager: with timer(retrieval_latency_seconds): ..."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        try:
            histogram_metric.observe(dt)
        except Exception:
            pass

def bump_ingest(n: int = 1) -> None:
    try:
        ingest_events_total.inc(n)
    except Exception:
        pass

def note_item_upserts(n: int = 1) -> None:
    try:
        items_upserts_total.inc(n)
    except Exception:
        pass

def note_vector_upserts(n: int = 1) -> None:
    try:
        vectors_upserts_total.inc(n)
    except Exception:
        pass

def note_retrieval(kinds: Optional[list[str]], hits: int, latency_s: Optional[float] = None) -> None:
    try:
        # Prefer multi-label call (what tests expect); fall back to single string.
        if kinds:
            try:
                retrieval_queries_total.labels(*sorted(kinds)).inc()
            except Exception:
                retrieval_queries_total.labels(",".join(sorted(kinds))).inc()
        else:
            retrieval_queries_total.labels("any").inc()

        retrieval_hits_total.inc(hits)
        if latency_s is not None:
            retrieval_latency_seconds.observe(float(latency_s))
    except Exception:
        pass


def note_compaction(stats: Any, when_ts: Optional[float] = None) -> None:
    """
    Accepts CompactionStats from memory/compaction.py and updates metrics.
    """
    try:
        compactions_total.inc()
        summaries_created_total.inc(int(getattr(stats, "summary_items_created", 0) or 0))
        duplicates_folded_total.inc(int(getattr(stats, "near_duplicates_dropped", 0) or 0))
        if when_ts:
            last_compaction_ts.set(float(when_ts))
    except Exception:
        pass

def set_health_gauges(*, idx_lag: int, vec_total: int, vec_bytes: int, working_cache: int, items_working: int, items_long: int, items_summary: int, last_compact: Optional[float] = None) -> None:
    """
    Fast path to push health snapshot numbers (from memory/health.snapshot()) into gauges.
    """
    try:
        index_lag.set(int(idx_lag))
        vectors_total.set(int(vec_total))
        vectors_bytes.set(int(vec_bytes))
        working_cache_size.set(int(working_cache))
        items_by_layer.labels("working").set(int(items_working))
        items_by_layer.labels("long").set(int(items_long))
        items_by_layer.labels("summary").set(int(items_summary))
        if last_compact is not None:
            last_compaction_ts.set(float(last_compact))
    except Exception:
        pass

def note_media_save(byte_count: int) -> None:
    try:
        media_items_total.inc()
        if byte_count > 0:
            media_bytes_total.set(float(byte_count) + float(getattr(note_media_save, "_acc", 0.0)))
            note_media_save._acc = float(media_bytes_total._value.get()) if _HAS_PROM else float(byte_count)  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------- Optional: export current metrics (Prometheus text format) ----------
def dump_text() -> Optional[bytes]:
    if not _HAS_PROM:
        return None
    try:
        return generate_latest(_registry)
    except Exception:
        return None
