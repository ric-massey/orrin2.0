# tests/goals/test_wal.py
# Pytest for goals.wal (append/iter/tail/follow, replay_to_store, rotate)

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Dict

import pytest

from goals import wal as W
from goals.store import FileGoalsStore


def _records_equal(a: Dict, b: Dict, keys=("id", "k")) -> bool:
    return all(a.get(x) == b.get(x) for x in keys)


def test_append_and_iter_lines_roundtrip(tmp_path: Path):
    wal = tmp_path / "wal.log"
    # single append
    one = {"type": "custom", "k": 1}
    W.append(wal, one)
    # many
    many = [{"type": "custom", "k": i} for i in range(2, 6)]
    W.append_many(wal, many)

    got = list(W.iter_lines(wal))
    assert len(got) == 1 + len(many)
    assert got[0]["type"] == "custom" and got[0]["k"] == 1
    assert got[-1]["k"] == 5
    # ts is injected
    assert "ts" in got[0]


def test_tail_returns_last_n(tmp_path: Path):
    wal = tmp_path / "wal.log"
    for i in range(10):
        W.append(wal, {"type": "custom", "k": i})
    last3 = W.tail(wal, n=3)
    assert [r["k"] for r in last3] == [7, 8, 9]


def test_follow_yields_new_appends(tmp_path: Path):
    wal = tmp_path / "wal.log"
    seen: List[Dict] = []
    stop = threading.Event()

    def reader():
        for rec in W.follow(wal, from_end=True, poll_seconds=0.05, stop=stop):
            seen.append(rec)
            if len(seen) >= 3:
                stop.set()
                break

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # Give the follower time to start and seek
    time.sleep(0.05)

    # Write three records; follow() should yield them in order
    for i in range(3):
        W.append(wal, {"type": "custom", "k": i})
        time.sleep(0.05)

    t.join(timeout=2.0)
    assert [r.get("k") for r in seen] == [0, 1, 2]


def test_replay_to_store_applies_wal(tmp_path: Path):
    wal = tmp_path / "wal.log"
    store = FileGoalsStore(data_dir=tmp_path / "store")

    # Craft minimal upserts (fields expected by wal._dict_to_goal/_dict_to_step)
    W.append(wal, {
        "type": "goal_upsert",
        "goal": {
            "id": "g1", "title": "T", "kind": "research", "spec": {}, "status": "NEW", "priority": "NORMAL"
        },
    })
    W.append(wal, {
        "type": "step_upsert",
        "step": {
            "id": "s1", "goal_id": "g1", "name": "N", "action": {"op": "noop"}, "status": "READY"
        },
    })

    report = W.replay_to_store(store, wal)
    assert report["goals"] == 1 and report["steps"] == 1 and report["skipped"] == 0

    g = store.get_goal("g1")
    assert g is not None and g.id == "g1" and g.kind == "research"
    s_ids = {s.id for s in store.steps_for("g1")}
    assert "s1" in s_ids


def test_rotate_compacts_wal(tmp_path: Path):
    wal = tmp_path / "wal.log"
    for i in range(20):
        W.append(wal, {"type": "custom", "k": i})

    rotated_dir = tmp_path / "rot"
    gz = W.rotate(wal, rotated_dir=rotated_dir, keep_tail_lines=5)
    assert gz is not None and gz.exists()

    # Remaining WAL should have 5 lines
    rem = list(W.iter_lines(wal))
    assert len(rem) == 5
    assert [r["k"] for r in rem] == [15, 16, 17, 18, 19]
