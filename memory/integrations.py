# memory/integrations.py
from __future__ import annotations
from typing import Callable, Any, Dict, Optional
import time

# Make these import-safe so the module always loads (tests monkeypatch these).
_MEM_SNAPSHOT_IS_STUB = False
try:
    from .health import snapshot as mem_snapshot  # tests patch this symbol
except Exception:  # pragma: no cover
    _MEM_SNAPSHOT_IS_STUB = True

    def mem_snapshot(*_a, **_k):
        # If tests don't patch, fall back to an empty signals dict
        return {"signals": {}}

try:
    from .wal import stats as _wal_stats_impl  # real WAL stats if present
except Exception:  # pragma: no cover
    _wal_stats_impl = None


# Expose a symbol the tests can monkeypatch directly.
def wal_stats():
    if _wal_stats_impl is None:
        return None
    try:
        return _wal_stats_impl()
    except Exception:
        # If the underlying WAL blows up, let callers decide what to do.
        raise


def _len_safely(obj: Any) -> int:
    try:
        return len(obj)  # type: ignore[call-arg]
    except Exception:
        return 0


def _alive_safely(daemon: Any) -> bool:
    try:
        if hasattr(daemon, "is_alive"):
            return bool(daemon.is_alive())
        t = getattr(daemon, "thread", None)
        return bool(t and t.is_alive())
    except Exception:
        return False


def _flatten_store_stats(ss: Any) -> Optional[Dict[str, Any]]:
    if ss is None:
        return None
    try:
        if hasattr(ss, "__dict__"):
            return dict(ss.__dict__)
        if hasattr(ss, "_asdict"):
            return dict(ss._asdict())  # namedtuple
        if isinstance(ss, dict):
            return dict(ss)
    except Exception:
        pass
    return None


def _derive_items_bytes(store: Any, store_stats: Optional[Dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    items: Optional[int] = None
    bytes_total: Optional[int] = None

    # Prefer values from snapshot store_stats
    if store_stats:
        items = store_stats.get("items_total", items)
        for key in ("vector_bytes_total", "bytes_total", "approx_bytes", "mem_bytes"):
            v = store_stats.get(key)
            if isinstance(v, (int, float)):
                bytes_total = int(v)
                break

    # Fallbacks from the store object
    if items is None:
        # Try common ways to count items
        for attr in ("count_items", "count"):
            try:
                f = getattr(store, attr, None)
                if callable(f):
                    items = int(f())
                    break
            except Exception:
                pass
        if items is None:
            try:
                items = int(_len_safely(store))
            except Exception:
                items = None

    if bytes_total is None:
        for attr in ("approx_bytes", "bytes", "size_bytes", "mem_bytes"):
            try:
                v = getattr(store, attr, None)
                v = v() if callable(v) else v
                if isinstance(v, (int, float)):
                    bytes_total = int(v)
                    break
            except Exception:
                pass

    return items, bytes_total


def _derive_wal_fields(daemon: Any, store: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"wal_enabled": False, "wal_queue": None, "wal_lag_s": None}

    # Enabled?
    try:
        out["wal_enabled"] = bool(getattr(store, "wal_enabled", False) or getattr(daemon, "wal_enabled", False))
    except Exception:
        pass

    # Queue length?
    try:
        q = getattr(store, "wal_queue", None)
        q = q() if callable(q) else q
        if isinstance(q, (int, float)):
            out["wal_queue"] = int(q)
    except Exception:
        pass

    # Lag in seconds: direct age if available, else derive from last flush ts.
    try:
        lag = getattr(store, "wal_last_flush_age_s", None)
        lag = lag() if callable(lag) else lag
        if isinstance(lag, (int, float)):
            out["wal_lag_s"] = float(lag)
        else:
            ts = getattr(store, "wal_last_flush_ts", None)
            ts = ts() if callable(ts) else ts
            if ts:
                out["wal_lag_s"] = max(0.0, time.time() - float(ts))
    except Exception:
        pass

    # If WAL stats object exists, surface write_failures as a signal extension.
    try:
        ws = wal_stats()
        if ws is not None and hasattr(ws, "write_failures"):
            out["wal_write_failures"] = getattr(ws, "write_failures")
    except Exception:
        # If wal_stats raises, omit the WAL key entirely (tests expect no hard failure)
        pass

    return out


def make_memory_health_getter(*, daemon, store, full: bool = False) -> Callable[[], Dict[str, Any]]:
    """
    Returns a zero-arg function the watchdog/UI can call to fetch memory health.

    Default (full=False): preserves legacy/test behavior and returns ONLY:
        {"signals": {...}}
    - Tests monkeypatch `mem_snapshot` and `wal_stats` on this module.

    Rich mode (full=True): returns a superset suitable for the Memory dashboard UI:
        {
          "daemon_alive": bool,
          "store_type": str,
          "items": int|None,
          "bytes": int|None,
          "wal_enabled": bool,
          "wal_queue": int|None,
          "wal_lag_s": float|None,
          # optional:
          "wal_write_failures": int,
          # full snapshot:
          "status": "...",
          "signals": {...},
          "notes": [...],
          "store_stats": {...}   # flattened
        }
    """
    def _get_minimal() -> Dict[str, Any]:
        hs = mem_snapshot(
            store,
            working_cache_size=_len_safely(getattr(daemon, "_working_cache", {})),
            last_compaction_ts=getattr(daemon, "_last_compact_ts", None),
        )
        sig = dict(hs.get("signals", {})) if isinstance(hs, dict) else {}
        # WAL write_failures only (if present and safe)
        try:
            ws = wal_stats()
            if ws is not None and hasattr(ws, "write_failures"):
                sig["memory.wal.write_failures"] = getattr(ws, "write_failures")
        except Exception:
            # If wal_stats raises, omit the WAL key entirely (as tests expect)
            pass
        return {"signals": sig}

    def _get_rich() -> Dict[str, Any]:
        # Daemon fields
        daemon_alive = _alive_safely(daemon)

        # Take snapshot (status, signals, notes, store_stats)
        try:
            hs = mem_snapshot(
                store,
                working_cache_size=_len_safely(getattr(daemon, "_working_cache", {})),
                last_compaction_ts=getattr(daemon, "_last_compact_ts", None),
            )
            if not isinstance(hs, dict):
                hs = {"signals": {}}
        except Exception as e:
            hs = {"status": "snapshot_error", "error": str(e), "signals": {}}

        # Flatten store_stats
        store_stats = _flatten_store_stats(hs.get("store_stats"))
        if store_stats is not None:
            hs["store_stats"] = store_stats

        # items / bytes
        items, bytes_total = _derive_items_bytes(store, store_stats)

        # WAL fields
        wal = _derive_wal_fields(daemon, store)

        out: Dict[str, Any] = {
            "daemon_alive": daemon_alive,
            "store_type": type(store).__name__,
            "items": items,
            "bytes": bytes_total,
            **wal,
            **hs,  # include status/signals/notes/store_stats
        }
        return out

    if full:
        return _get_rich
    # Heuristic: if mem_snapshot is our stub (no real data), stick to minimal to keep tests stable.
    return _get_minimal


def make_memory_dashboard_health_getter(*, daemon, store) -> Callable[[], Dict[str, Any]]:
    """
    Convenience wrapper for the Memory dashboard UI. Always returns the rich payload.
    """
    return make_memory_health_getter(daemon=daemon, store=store, full=True)
