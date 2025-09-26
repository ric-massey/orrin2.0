# tests/goals_test/test_runner_smoke.py
from __future__ import annotations

import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from goals.runner import StepRunner
from goals.registry import GoalRegistry
from goals.store import FileGoalsStore
from goals.model import Goal, Step, Status, Priority
from goals.handlers.base import BaseGoalHandler, HandlerContext

UTCNOW = lambda: datetime.now(timezone.utc)


# -----------------------------
# helpers
# -----------------------------

def wait_until(pred, *, timeout=3.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


class DummyHandler(BaseGoalHandler):
    """
    Minimal handler that:
      - sets started_at and RUNNING on first tick
      - raises if action["op"] == "boom"
      - otherwise marks DONE and sets finished_at
      - optional action["sleep_ms"] adds a small busy-wait
    """
    kind = "dummy"

    def plan(self, goal: Goal, ctx: HandlerContext):
        return []

    def is_blocked(self, goal: Goal, ctx: HandlerContext):
        return False, None

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        if step.started_at is None:
            step.started_at = UTCNOW()          # ← changed from self._utcnow()
            step.status = Status.RUNNING

        if (step.action or {}).get("op") == "boom":
            raise RuntimeError("boom")

        ms = int((step.action or {}).get("sleep_ms", 0))
        if ms > 0:
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) * 1000.0 < ms:
                pass

        step.status = Status.DONE
        step.finished_at = UTCNOW()             # ← changed from self._utcnow()
        return step


# -----------------------------
# fixtures
# -----------------------------

@pytest.fixture()
def store(tmp_path: Path) -> FileGoalsStore:
    return FileGoalsStore(data_dir=tmp_path / "goals-data")


@pytest.fixture()
def registry() -> GoalRegistry:
    reg = GoalRegistry()
    reg.register(DummyHandler(), replace=True)
    return reg


@pytest.fixture()
def reaper_sink():
    events: List[Dict[str, Any]] = []
    def sink(ev: Dict[str, Any]) -> None:
        events.append(dict(ev))
    return events, sink


@pytest.fixture()
def runner(store: FileGoalsStore, registry: GoalRegistry, reaper_sink):
    events, sink = reaper_sink
    q: "queue.Queue[Step]" = queue.Queue()
    r = StepRunner(store=store, registry=registry, step_queue=q, workers=2, ctx={}, reaper_sink=sink)
    r.start()
    yield r, q, events
    r.stop()
    r.join(timeout=2.0)


# -----------------------------
# tests
# -----------------------------

def test_executes_and_emits_events(store: FileGoalsStore, runner):
    r, q, events = runner

    g = Goal(id="g_ok", title="ok", kind="dummy", spec={}, status=Status.READY, priority=Priority.NORMAL)
    store.upsert_goal(g)

    # Optional sanity check: make sure the registry resolves 'dummy'
    assert r._get_handler(g) is not None, "No handler resolved for kind='dummy'"

    s = Step(id="s_ok", goal_id="g_ok", name="do", action={"op": "ok"}, status=Status.READY)
    store.upsert_step(s)

    # enqueue via public API
    r.submit(s)

    # goal should reach DONE (avoid falling back to stale 'g')
    def is_done():
        cur = store.get_goal("g_ok")
        return cur is not None and cur.status == Status.DONE

    assert wait_until(is_done, timeout=3.0)

    # step terminal
    s2 = store.list_steps(goal_id="g_ok")[0]
    assert s2.status == Status.DONE and s2.finished_at is not None

    # events present (StepStarted + StepFinished)
    kinds = [e.get("kind") for e in events]
    assert "StepStarted" in kinds and "StepFinished" in kinds
    fin = next(e for e in events if e.get("kind") == "StepFinished")
    assert fin.get("extra", {}).get("duration_sec") is not None


def test_marks_goal_failed_on_handler_error_no_retries(store: FileGoalsStore, runner):
    r, q, events = runner

    g = Goal(id="g_boom", title="boom", kind="dummy", spec={}, status=Status.READY)
    store.upsert_goal(g)
    s = Step(id="s_boom", goal_id="g_boom", name="boom", action={"op": "boom"}, status=Status.READY, max_attempts=1)
    store.upsert_step(s)

    r.submit(s)

    def is_failed():
        cur = store.get_goal("g_boom")
        return cur is not None and cur.status == Status.FAILED

    # goal should fail because handler raises and no retries remain
    assert wait_until(is_failed, timeout=3.0)
    s2 = store.list_steps(goal_id="g_boom")[0]
    assert s2.status == Status.FAILED and (s2.attempts or 0) >= (s2.max_attempts or 0)
    assert any(e.get("kind") == "StepFailed" for e in events)


def test_marks_step_failed_when_goal_missing(store: FileGoalsStore, runner):
    r, q, events = runner

    s = Step(id="s_orphan", goal_id="nope", name="orphan", action={"op": "ok"}, status=Status.READY)
    store.upsert_step(s)
    r.submit(s)

    # orphan step should be marked FAILED
    assert wait_until(lambda: next((ss for ss in store.list_steps() if ss.id == "s_orphan"), None).status == Status.FAILED, timeout=1.5)
    failed = next(e for e in events if e.get("kind") == "StepFailed" and e.get("step_id") == "s_orphan")
    assert failed.get("extra", {}).get("reason") == "goal_missing"


def test_active_workers_spike_then_goal_done(store: FileGoalsStore, runner):
    r, q, events = runner

    g = Goal(id="g_busy", title="busy", kind="dummy", spec={}, status=Status.READY)
    store.upsert_goal(g)

    # 4 short steps → observe active_workers > 0 during execution
    for i in range(4):
        s = Step(id=f"s_busy_{i}", goal_id="g_busy", name=f"busy {i}", action={"op": "ok", "sleep_ms": 60}, status=Status.READY)
        store.upsert_step(s)
        r.submit(s)

    saw_activity = wait_until(lambda: r.active_workers > 0, timeout=1.0)
    assert saw_activity

    def busy_done():
        cur = store.get_goal("g_busy")
        return cur is not None and cur.status == Status.DONE

    assert wait_until(busy_done, timeout=3.0)


def test_marks_goal_failed_when_handler_missing(store: FileGoalsStore, reaper_sink):
    events, sink = reaper_sink
    q: "queue.Queue[Step]" = queue.Queue()

    # Registry with no 'dummy2' handler
    empty_reg = GoalRegistry()

    r = StepRunner(store=store, registry=empty_reg, step_queue=q, workers=1, ctx={}, reaper_sink=sink)
    r.start()
    try:
        g = Goal(id="g_noh", title="no handler", kind="dummy2", spec={}, status=Status.READY)
        store.upsert_goal(g)
        s = Step(id="s_noh", goal_id="g_noh", name="x", action={"op": "ok"}, status=Status.READY)
        store.upsert_step(s)

        r.submit(s)

        def noh_failed():
            cur = store.get_goal("g_noh")
            return cur is not None and cur.status == Status.FAILED

        assert wait_until(noh_failed, timeout=3.0)
        kinds = [e.get("kind") for e in events]
        assert "StepFailed" in kinds and "GoalFailed" in kinds
    finally:
        r.stop()
        r.join(timeout=2.0)
