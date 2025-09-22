# Health checks & Reaper-friendly signals for Orrin2.0 memory (index lag, compaction stalling, cache size, etc.).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time

from .config import MEMCFG


@dataclass
class StoreStats:
    # High-level counts
    items_total: int = 0
    items_by_layer: Dict[str, int] = field(
        default_factory=lambda: {"working": 0, "long": 0, "summary": 0}
    )
    # Embedding coverage / lag
    index_lag: int = 0                     # items with an embedding_id but no vector indexed
    vectors_total: int = 0                 # number of vectors in the store
    vector_bytes_total: int = 0            # approximate memory footprint (bytes)
    # Optional GC signals (if caller computes them elsewhere)
    gc_eligible: int = 0


def _approx_vec_nbytes(vec) -> int:
    try:
        return int(vec.nbytes)  # numpy arrays
    except Exception:
        try:
            return int(len(vec)) * 4  # conservative fallback assuming float32
        except Exception:
            return 0


def collect_store_stats(store) -> StoreStats:
    """
    Best-effort, backend-agnostic stats collection.
    Works with InMemoryStore by duck-typing private members (_items/_vecs).
    For other backends, you can add a 'stats()' method that returns a dict
    with compatible keys and we will consume it here.
    """
    # If the store provides a native stats() method, prefer it.
    if hasattr(store, "stats") and callable(getattr(store, "stats")):
        d = dict(store.stats())  # type: ignore
        ss = StoreStats()
        ss.items_total = int(d.get("items_total", 0))
        ss.items_by_layer = dict(d.get("items_by_layer", {"working": 0, "long": 0, "summary": 0}))
        ss.index_lag = int(d.get("index_lag", 0))
        ss.vectors_total = int(d.get("vectors_total", 0))
        ss.vector_bytes_total = int(d.get("vector_bytes_total", 0))
        ss.gc_eligible = int(d.get("gc_eligible", 0))
        return ss

    # Duck-type InMemoryStore (dev backend)
    items = getattr(store, "_items", None)
    vecs = getattr(store, "_vecs", None)
    ss = StoreStats()
    if isinstance(items, dict):
        ss.items_total = len(items)
        by_layer = {"working": 0, "long": 0, "summary": 0}
        lag = 0
        for it in items.values():
            layer = getattr(it, "layer", None) or "working"
            if layer in by_layer:
                by_layer[layer] += 1
            # index lag = items that declare an embedding_id but don't have a vector yet
            emb_id = getattr(it, "embedding_id", None)
            if emb_id and isinstance(vecs, dict) and emb_id not in vecs:
                lag += 1
        ss.items_by_layer = by_layer
        ss.index_lag = lag
    if isinstance(vecs, dict):
        ss.vectors_total = len(vecs)
        # estimate bytes
        total_b = 0
        for v in vecs.values():
            total_b += _approx_vec_nbytes(v)
        ss.vector_bytes_total = total_b
    return ss


def assess_health(
    store_stats: StoreStats,
    working_cache_size: int,
    last_compaction_ts: float,
    flush_failures: int = 0,
    now: Optional[float] = None,
) -> Tuple[str, Dict[str, int | float], list[str]]:
    """
    Returns (status, signals, notes)
      - status: "ok" | "warn" | "error"
      - signals: dict suitable for Reaper consumption
      - notes: brief human-readable explanations
    """
    now = time.time() if now is None else float(now)
    minutes_since_compaction = (now - (last_compaction_ts or 0.0)) / 60.0

    # Build signals
    signals: Dict[str, int | float] = {
        "memory.index_lag": int(store_stats.index_lag),
        "memory.compaction_stalled_min": float(minutes_since_compaction),
        "memory.working_cache": int(working_cache_size),
        "memory.flush_failures": int(flush_failures),
        "memory.items.working": int(store_stats.items_by_layer.get("working", 0)),
        "memory.items.long": int(store_stats.items_by_layer.get("long", 0)),
        "memory.items.summary": int(store_stats.items_by_layer.get("summary", 0)),
        "memory.vectors.total": int(store_stats.vectors_total),
        "memory.vectors.bytes": int(store_stats.vector_bytes_total),
    }

    # Threshold-based evaluation (soft limits)
    issues: list[str] = []

    if store_stats.index_lag > MEMCFG.HEALTH_INDEX_LAG_SOFT:
        issues.append(f"index_lag {store_stats.index_lag} > {MEMCFG.HEALTH_INDEX_LAG_SOFT}")

    if minutes_since_compaction > MEMCFG.HEALTH_COMPACTION_STALLED_MIN:
        issues.append(
            f"compaction stalled for {minutes_since_compaction:.1f} min (> {MEMCFG.HEALTH_COMPACTION_STALLED_MIN})"
        )

    if flush_failures >= MEMCFG.HEALTH_FLUSH_FAILURES_SOFT:
        issues.append(f"flush failures {flush_failures} ≥ {MEMCFG.HEALTH_FLUSH_FAILURES_SOFT}")

    # Severity: 0 -> ok, 1 -> warn, 2+ -> error (as tests expect)
    if len(issues) >= 2:
        status = "error"
    elif len(issues) == 1:
        status = "warn"
    else:
        status = "ok"

    return status, signals, issues


def snapshot(
    store,
    *,
    working_cache_size: int,
    last_compaction_ts: float,
    flush_failures: int = 0,
    now: Optional[float] = None,
) -> Dict[str, object]:
    """
    Convenience wrapper: collect store stats, assess, and return a packaged snapshot.
    """
    ss = collect_store_stats(store)
    status, signals, notes = assess_health(
        store_stats=ss,
        working_cache_size=working_cache_size,
        last_compaction_ts=last_compaction_ts,
        flush_failures=flush_failures,
        now=now,
    )
    return {
        "status": status,           # "ok" | "warn" | "error"
        "signals": signals,         # flat dict for Reaper
        "notes": notes,             # list[str] explanations
        "store_stats": ss,          # the raw counts (dataclass)
    }
