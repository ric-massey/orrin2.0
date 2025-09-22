# tests/memory_tests/wal_test.py
from __future__ import annotations
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

import memory.wal as wal_mod
from memory.wal import WAL
from memory.models import Event, MemoryItem


def _mk_item(content="note", kind="fact", layer="working") -> MemoryItem:
    it = MemoryItem.new(kind=kind, source="chat:user", content=content, layer=layer)
    it.embedding_id = f"vec_{it.id}"
    it.embedding_dim = 8
    it.model_hint = "test-hint"
    it.salience = 0.5
    it.novelty = 0.8
    it.goal_relevance = 0.1
    it.impact_signal = 0.0
    it.freq = 2
    it.strength = 0.3
    it.summary_of = []
    it.cross_refs = []
    it.pinned = None
    it.expiry_hint = None
    it.meta = {"foo": "bar"}
    return it


def test_append_event_writes_jsonl_and_sanitizes_vec(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    items_path = tmp_path / "items.jsonl"
    w = WAL(events_path, items_path, max_bytes=10_000_000, gzip_rotate=False)

    # meta includes a heavy _vec that must be dropped
    meta = {"x": 1, "_vec": np.arange(16).tolist()}
    ev = Event(kind="chat:user", content="hello world", meta=meta)

    w.append_event(ev)
    w.flush()

    data = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    rec = json.loads(data[0])
    assert rec["kind"] == "chat:user"
    assert rec["content"] == "hello world"
    # _vec is sanitized out
    assert "_vec" not in rec["meta"]
    assert rec["meta"]["x"] == 1
    # stats updated
    assert w.stats.events_written == 1
    assert w.stats.write_failures == 0


def test_append_items_and_roundtrip_to_dict(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    items_path = tmp_path / "items.jsonl"
    w = WAL(events_path, items_path, max_bytes=10_000_000, gzip_rotate=False)

    items = [_mk_item("a"), _mk_item("b")]
    n = w.append_items(items)
    w.flush()

    assert n == 2
    lines = items_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # Roundtrip via record_to_item
    rec0 = json.loads(lines[0])
    obj0 = WAL.record_to_item(rec0)
    assert isinstance(obj0, MemoryItem)
    assert obj0.content in {"a", "b"}
    assert obj0.embedding_id and obj0.embedding_dim == 8


def test_rotation_and_gzip_for_events(tmp_path: Path):
    # Force tiny max_bytes so we rotate quickly
    events_path = tmp_path / "events.jsonl"
    items_path = tmp_path / "items.jsonl"
    w = WAL(events_path, items_path, max_bytes=250, gzip_rotate=True)

    # Each event line ~100+ bytes; write several to trigger rotation
    for i in range(6):
        ev = Event(kind="chat:user", content=f"msg {i}", meta={"k": i})
        w.append_event(ev)
        # slight delay so timestamped rotate names differ if needed
        time.sleep(0.01)

    w.flush()

    # Should have at least one rotated .gz and a fresh events.jsonl
    rotated = list(tmp_path.glob("events.*.jsonl.gz"))
    assert rotated, "expected at least one gz-rotated events file"
    assert events_path.exists(), "fresh events.jsonl should exist after rotation"
    assert w.stats.rotate_events >= 1
    assert w.stats.events_written == 6

    # Replay from a rotated .gz
    seen = 0
    for rec in WAL.replay_events(rotated[0]):
        assert "kind" in rec and "content" in rec
        seen += 1
    assert seen >= 1


def test_rotation_and_gzip_for_items(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    items_path = tmp_path / "items.jsonl"
    w = WAL(events_path, items_path, max_bytes=300, gzip_rotate=True)

    # Write items until rotation occurs
    count = 0
    while w.stats.rotate_items == 0:
        w.append_items([_mk_item(f"c{count}"), _mk_item(f"d{count}")])
        count += 1
        if count > 20:  # safety guard
            break
    w.flush()

    rotated = list(tmp_path.glob("items.*.jsonl.gz"))
    assert rotated, "expected at least one gz-rotated items file"
    assert items_path.exists()
    assert w.stats.rotate_items >= 1
    assert w.stats.items_written > 0

    # Replay via replay_items (same as events reader)
    total = 0
    for rec in WAL.replay_items(rotated[0]):
        assert "id" in rec and "layer" in rec and "content" in rec
        total += 1
    assert total > 0


def test_record_to_event_and_item_helpers(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    items_path = tmp_path / "items.jsonl"
    w = WAL(events_path, items_path, max_bytes=10_000_000, gzip_rotate=False)

    ev = Event(kind="chat:user", content="roundtrip", meta={"a": 1})
    w.append_event(ev)

    it = _mk_item("roundtrip-item")
    w.append_items([it])
    w.flush()

    e_rec = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    i_rec = json.loads(items_path.read_text(encoding="utf-8").splitlines()[0])

    e2 = WAL.record_to_event(e_rec)
    i2 = WAL.record_to_item(i_rec)

    assert isinstance(e2, Event) and e2.kind == "chat:user" and e2.content == "roundtrip"
    assert isinstance(i2, MemoryItem) and i2.content == "roundtrip-item"


def test_flush_and_close_are_safe_and_idempotent(tmp_path: Path):
    w = WAL(tmp_path / "e.jsonl", tmp_path / "i.jsonl", max_bytes=1024)
    # No writes; flush/close should not error and may increment write_failures only on OS issues
    w.flush()
    w.close()
    # Idempotent close
    w.close()


def test_fsync_every_true_path(tmp_path: Path):
    w = WAL(tmp_path / "e.jsonl", tmp_path / "i.jsonl", fsync_every=True, max_bytes=10_000_000)
    w.append_event(Event(kind="chat:user", content="sync", meta={}))
    w.append_items([_mk_item("sync")])
    # If fsync path had problems, write_failures would likely be > 0; not strict, but we expect success
    w.flush()
    assert w.stats.events_written == 1
    assert w.stats.items_written == 1


def test_module_level_wrappers_can_be_monkeypatched(tmp_path: Path, monkeypatch):
    # Use a test WAL as the module-level DEFAULT_WAL
    test_default = WAL(tmp_path / "E.jsonl", tmp_path / "I.jsonl", max_bytes=10_000_000, gzip_rotate=False)
    monkeypatch.setattr(wal_mod, "DEFAULT_WAL", test_default, raising=True)

    wal_mod.append_event(Event(kind="chat:user", content="wrapped-ev", meta={"m": 1}))
    wal_mod.append_items([_mk_item("wrapped-item")])
    wal_mod.flush()
    st = wal_mod.stats()

    assert isinstance(st, wal_mod.WalStats)
    assert st.events_written == 1
    assert st.items_written == 1

    # Files actually contain the records
    assert (tmp_path / "E.jsonl").exists()
    assert (tmp_path / "I.jsonl").exists()
    assert "wrapped-ev" in (tmp_path / "E.jsonl").read_text(encoding="utf-8")
    assert "wrapped-item" in (tmp_path / "I.jsonl").read_text(encoding="utf-8")


def test_replay_plain_files(tmp_path: Path):
    w = WAL(tmp_path / "plain_events.jsonl", tmp_path / "plain_items.jsonl", max_bytes=10_000_000, gzip_rotate=False)
    for i in range(3):
        w.append_event(Event(kind="chat:assistant", content=f"e{i}", meta={"i": i}))
    w.append_items([_mk_item("x"), _mk_item("y")])
    w.flush()

    # Replay plain (non-gz) paths
    evs = list(WAL.replay_events(tmp_path / "plain_events.jsonl"))
    its = list(WAL.replay_items(tmp_path / "plain_items.jsonl"))
    assert len(evs) == 3
    assert len(its) == 2
    assert evs[0]["kind"] == "chat:assistant"
    assert its[0]["kind"] in {"fact", "media", "rule", "goal", "summary", "introspection"}
