# tests/goals/test_store.py
# Pytest for FileGoalsStore CRUD, indexing, filtering, and WAL behavior

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from goals.store import FileGoalsStore
from goals.model import Goal, Step, Status, Priority


@pytest.fixture()
def store(tmp_path: Path) -> FileGoalsStore:
    return FileGoalsStore(data_dir=tmp_path / "goals-data")


def _read_wal_lines(path: Path) -> List[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            pass
    return out


def test_paths_and_files_exist(store: FileGoalsStore):
    p = store.paths()
    assert set(p.keys()) == {"state", "wal", "dir"}
    # state.jsonl and wal.log should exist (created on init)
    assert Path(p["state"]).exists()
    assert Path(p["wal"]).exists()
    assert Path(p["dir"]).exists()


def test_upsert_goal_persists_and_writes_wal(store: FileGoalsStore, tmp_path: Path):
    g = Goal(
        id="g1",
        title="First goal",
        kind="research",
        spec={"queries": ["foo"]},
        priority=Priority.HIGH,
        status=Status.NEW,
    )
    store.upsert_goal(g)

    # Read back
    g2 = store.get_goal("g1")
    assert g2 is not None and g2.id == "g1" and g2.kind == "research"

    # WAL should contain a goal_upsert with matching id
    wal = Path(store.paths()["wal"])
    recs = _read_wal_lines(wal)
    assert any(r.get("type") == "goal_upsert" and (r.get("goal") or {}).get("id") == "g1" for r in recs)


def test_upsert_step_and_indexing_and_wal(store: FileGoalsStore):
    # Add base goal
    store.upsert_goal(Goal(id="g2", title="G2", kind="coding", spec={}, status=Status.READY))

    s1 = Step(id="s1", goal_id="g2", name="one", action={"op": "noop"}, status=Status.READY)
    s2 = Step(id="s2", goal_id="g2", name="two", action={"op": "noop"}, status=Status.WAITING)
    s3 = Step(id="s3", goal_id="g3", name="other goal", action={"op": "noop"}, status=Status.READY)

    store.upsert_step(s1)
    store.upsert_step(s2)
    store.upsert_step(s3)

    # Indexing: steps_for("g2") returns s1 and s2 only
    g2_steps = {s.id for s in store.steps_for("g2")}
    assert g2_steps == {"s1", "s2"}

    # WAL entries exist for steps
    wal = Path(store.paths()["wal"])
    recs = _read_wal_lines(wal)
    ids_in_wal = { (r.get("step") or {}).get("id") for r in recs if r.get("type") == "step_upsert" }
    assert {"s1", "s2", "s3"}.issubset(ids_in_wal)


def test_list_steps_filters_by_status_and_goal(store: FileGoalsStore):
    gid = "gF"
    store.upsert_goal(Goal(id=gid, title="filtering", kind="housekeeping", spec={}, status=Status.READY))

    a = Step(id="sf_a", goal_id=gid, name="A", action={"op": "noop"}, status=Status.READY)
    b = Step(id="sf_b", goal_id=gid, name="B", action={"op": "noop"}, status=Status.WAITING)
    c = Step(id="sf_c", goal_id=gid, name="C", action={"op": "noop"}, status=Status.DONE)
    store.upsert_step(a); store.upsert_step(b); store.upsert_step(c)

    all_for_goal = {s.id for s in store.list_steps(goal_id=gid)}
    assert all_for_goal == {"sf_a", "sf_b", "sf_c"}

    only_ready = {s.id for s in store.list_steps(goal_id=gid, statuses=[Status.READY])}
    assert only_ready == {"sf_a"}

    non_terminal = {s.id for s in store.list_steps(goal_id=gid, statuses=[Status.READY, Status.WAITING])}
    assert non_terminal == {"sf_a", "sf_b"}

    # steps_for(None) returns all steps (with optional status filter)
    all_any_goal = {s.id for s in store.steps_for(None)}
    assert {"sf_a", "sf_b", "sf_c"}.issubset(all_any_goal)
    only_done_any = {s.id for s in store.steps_for(None, statuses=[Status.DONE])}
    assert "sf_c" in only_done_any and "sf_a" not in only_done_any


def test_ready_goals_returns_expected_statuses(store: FileGoalsStore):
    # Create a mix of statuses
    store.upsert_goal(Goal(id="gr_ready", title="r", kind="k", spec={}, status=Status.READY))
    store.upsert_goal(Goal(id="gr_run", title="r", kind="k", spec={}, status=Status.RUNNING))
    store.upsert_goal(Goal(id="gr_wait", title="r", kind="k", spec={}, status=Status.WAITING))
    store.upsert_goal(Goal(id="gr_new", title="r", kind="k", spec={}, status=Status.NEW))
    store.upsert_goal(Goal(id="gr_done", title="r", kind="k", spec={}, status=Status.DONE))

    ids = {g.id for g in store.ready_goals()}
    assert ids == {"gr_ready", "gr_run", "gr_wait"}


def test_iter_list_counts_and_overwrite(store: FileGoalsStore):
    g = Goal(id="gX", title="X", kind="coding", spec={}, status=Status.NEW)
    store.upsert_goal(g)
    store.upsert_goal(Goal(id="gY", title="Y", kind="research", spec={}, status=Status.READY))

    # counts
    cnt = store.counts()
    assert cnt["goals"] == 2

    # list & iter are consistent
    ids1 = {x.id for x in store.list_goals()}
    ids2 = {x.id for x in store.iter_goals()}
    assert ids1 == ids2 == {"gX", "gY"}

    # overwrite gX (update title/status), ensure stored value changes and WAL appends again
    before_wal_len = len(_read_wal_lines(Path(store.paths()["wal"])))
    g_updated = Goal(id="gX", title="X2", kind="coding", spec={}, status=Status.READY)
    store.upsert_goal(g_updated)
    after = store.get_goal("gX")
    assert after is not None and after.title == "X2" and after.status == Status.READY

    after_wal_len = len(_read_wal_lines(Path(store.paths()["wal"])))
    assert after_wal_len > before_wal_len, "Expected another goal_upsert record appended to WAL"
