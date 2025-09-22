# tests/memory_tests/intergrations_test.py
from types import SimpleNamespace
import pytest

from memory.integrations import make_memory_health_getter


def test_getter_calls_snapshot_with_expected_args_and_includes_wal(monkeypatch):
    calls = []

    # fake mem_snapshot that records args and returns signals
    def fake_snapshot(store, *, working_cache_size, last_compaction_ts, flush_failures=0, now=None):
        calls.append({
            "store": store,
            "working_cache_size": working_cache_size,
            "last_compaction_ts": last_compaction_ts,
        })
        return {"signals": {"a": 1, "b": 2}}

    # fake wal_stats that returns write_failures attribute
    def fake_wal_stats():
        return SimpleNamespace(write_failures=7)

    monkeypatch.setattr("memory.integrations.mem_snapshot", fake_snapshot, raising=True)
    monkeypatch.setattr("memory.integrations.wal_stats", fake_wal_stats, raising=True)

    daemon = SimpleNamespace(_working_cache={"x": 1, "y": 2, "z": 3}, _last_compact_ts=1234.0)
    store = object()

    getter = make_memory_health_getter(daemon=daemon, store=store)
    out = getter()

    # snapshot called with derived values
    assert calls and calls[0]["store"] is store
    assert calls[0]["working_cache_size"] == 3
    assert calls[0]["last_compaction_ts"] == 1234.0

    # signals merged + wal failures included
    assert out == {"signals": {"a": 1, "b": 2, "memory.wal.write_failures": 7}}


def test_getter_works_if_daemon_missing_attrs_and_wal_has_no_attr(monkeypatch):
    # snapshot returns some signals
    monkeypatch.setattr(
        "memory.integrations.mem_snapshot",
        lambda store, **kw: {"signals": {"base": 1}},
        raising=True,
    )

    # wal_stats returns an object without write_failures -> module omits WAL key
    monkeypatch.setattr(
        "memory.integrations.wal_stats",
        lambda: SimpleNamespace(),
        raising=True,
    )

    daemon = SimpleNamespace()  # no _working_cache / _last_compact_ts
    store = object()

    getter = make_memory_health_getter(daemon=daemon, store=store)
    out = getter()

    # With no write_failures attr, integrations.py omits the WAL key.
    assert out == {"signals": {"base": 1}}


def test_getter_omits_wal_key_if_wal_stats_raises(monkeypatch):
    # snapshot returns signals
    monkeypatch.setattr(
        "memory.integrations.mem_snapshot",
        lambda store, **kw: {"signals": {"ok": 1}},
        raising=True,
    )

    # wal_stats raises -> code should ignore and proceed
    def boom():
        raise RuntimeError("wal unavailable")
    monkeypatch.setattr("memory.integrations.wal_stats", boom, raising=True)

    daemon = SimpleNamespace(_working_cache={}, _last_compact_ts=None)
    store = object()

    getter = make_memory_health_getter(daemon=daemon, store=store)
    out = getter()
    assert out == {"signals": {"ok": 1}}
    # ensure the wal key was not added


def test_getter_handles_snapshot_returning_empty_dict(monkeypatch):
    # snapshot returns a dict without "signals"
    monkeypatch.setattr(
        "memory.integrations.mem_snapshot",
        lambda store, **kw: {},
        raising=True,
    )
    # wal present
    monkeypatch.setattr(
        "memory.integrations.wal_stats",
        lambda: SimpleNamespace(write_failures=3),
        raising=True,
    )

    daemon = SimpleNamespace(_working_cache={"only": 1})
    store = object()
    getter = make_memory_health_getter(daemon=daemon, store=store)
    out = getter()
    # should still return a signals dict, containing wal key only
    assert out == {"signals": {"memory.wal.write_failures": 3}}


def test_getter_propagates_snapshot_exceptions(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("snapshot failed")
    monkeypatch.setattr("memory.integrations.mem_snapshot", boom, raising=True)
    monkeypatch.setattr("memory.integrations.wal_stats", lambda: SimpleNamespace(write_failures=1), raising=True)

    getter = make_memory_health_getter(daemon=SimpleNamespace(_working_cache={}), store=object())
    with pytest.raises(RuntimeError, match="snapshot failed"):
        _ = getter()
