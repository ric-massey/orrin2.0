# tests/goals/test_state_machine.py
# End-to-end state-machine style tests for goals/steps across NEW→READY→RUNNING→{DONE,FAILED},
# plus BLOCKED↔READY and PAUSED gating.

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pytest

# --- FIX: support both module names (daemon.py shim or goals_daemon.py) ---
try:
    from goals.daemon import GoalsDaemon  # preferred (shim re-exports if you renamed)
except Exception:  # pragma: no cover
    from goals.goals_daemon import GoalsDaemon

from goals.registry import GoalRegistry
from goals.store import FileGoalsStore
from goals.model import Goal, Step, Status, Priority
from goals.handlers.base import BaseGoalHandler, HandlerContext


UTCNOW = lambda: datetime.now(timezone.utc)


# -----------------------------
# Test helpers
# -----------------------------

def wait_until(pred, *, timeout=3.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


# -----------------------------
# Dummy handlers for state-machine exercises
# -----------------------------

class SuccHandler(BaseGoalHandler):
    """Plans one step and immediately succeeds."""
    kind = "succ"

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        return [Step(id=f"{goal.id}_s1", goal_id=goal.id, name="do", action={"op": "ok"}, status=Status.READY)]

    def is_blocked(self, goal: Goal, ctx: HandlerContext):
        return False, None

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        if step.started_at is None:
            step.started_at = self._utcnow()
            step.status = Status.RUNNING
        step.status = Status.DONE
        step.finished_at = self._utcnow()
        return step


class BlockThenUnblockHandler(BaseGoalHandler):
    """Blocks for ~0.15s from goal.created_at, then unblocks and completes."""
    kind = "blocky"

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        return [Step(id=f"{goal.id}_s1", goal_id=goal.id, name="wait_then_do", action={"op": "ok"}, status=Status.READY)]

    def is_blocked(self, goal: Goal, ctx: HandlerContext):
        # Stay blocked until 150ms have elapsed since creation
        return (UTCNOW() - goal.created_at).total_seconds() < 0.15, "cooldown"

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        if step.started_at is None:
            step.started_at = self._utcnow()
            step.status = Status.RUNNING
        step.status = Status.DONE
        step.finished_at = self._utcnow()
        return step


class FlakyOnceHandler(BaseGoalHandler):
    """Fails the first attempt (raises), then succeeds on retry."""
    kind = "flaky"

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        return [Step(id=f"{goal.id}_s1", goal_id=goal.id, name="flaky_once", action={"op": "ok"}, status=Status.READY, max_attempts=3)]

    def is_blocked(self, goal: Goal, ctx: HandlerContext):
        return False, None

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        if step.started_at is None:
            step.started_at = self._utcnow()
            step.status = Status.RUNNING
        # Fail on first attempt, succeed afterwards
        if (step.attempts or 0) == 0:
            raise RuntimeError("transient")
        step.status = Status.DONE
        step.finished_at = self._utcnow()
        return step


class PauseHandler(BaseGoalHandler):
    """Simple success handler; used to test PAUSED gating when a READY step already exists."""
    kind = "pause"

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        # Not used in the PAUSED test (we inject a step manually)
        return [Step(id=f"{goal.id}_s1", goal_id=goal.id, name="noop", action={"op": "ok"}, status=Status.READY)]

    def is_blocked(self, goal: Goal, ctx: HandlerContext):
        return False, None

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        if step.started_at is None:
            step.started_at = self._utcnow()
            step.status = Status.RUNNING
        step.status = Status.DONE
        step.finished_at = self._utcnow()
        return step


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture()
def store(tmp_path: Path) -> FileGoalsStore:
    return FileGoalsStore(data_dir=tmp_path / "goals-data")


@pytest.fixture()
def registry() -> GoalRegistry:
    reg = GoalRegistry()
    reg.register(SuccHandler(), replace=True)
    reg.register(BlockThenUnblockHandler(), replace=True)
    reg.register(FlakyOnceHandler(), replace=True)
    reg.register(PauseHandler(), replace=True)
    return reg


@pytest.fixture()
def daemon(store: FileGoalsStore, registry: GoalRegistry):
    d = GoalsDaemon(store=store, registry=registry, workers=2, tick_seconds=0.05, ctx={})
    d.start()
    yield d
    d.stop()
    d.join(timeout=2.0)


# -----------------------------
# Tests
# -----------------------------

def test_new_ready_running_done_path(store: FileGoalsStore, daemon: GoalsDaemon):
    g = Goal(id="g_sm_ok", title="happy", kind="succ", spec={}, status=Status.NEW)
    store.upsert_goal(g)

    assert wait_until(lambda: (store.get_goal("g_sm_ok") or g).status == Status.DONE, timeout=2.0)

    steps = store.steps_for("g_sm_ok")
    assert steps and steps[0].status == Status.DONE
    assert steps[0].started_at is not None and steps[0].finished_at is not None


def test_blocked_then_unblocked_then_done(store: FileGoalsStore, daemon: GoalsDaemon):
    g = Goal(id="g_block", title="block cycle", kind="blocky", spec={}, status=Status.NEW)
    store.upsert_goal(g)

    # It should be BLOCKED shortly after planning
    saw_blocked = wait_until(lambda: (store.get_goal("g_block") or g).status == Status.BLOCKED, timeout=0.6)
    assert saw_blocked, "expected goal to enter BLOCKED first"

    # Then it should unblock and complete
    assert wait_until(lambda: (store.get_goal("g_block") or g).status == Status.DONE, timeout=2.0)


def test_retry_after_handler_error_then_done(store: FileGoalsStore, daemon: GoalsDaemon):
    g = Goal(id="g_retry", title="flaky once", kind="flaky", spec={}, status=Status.NEW, priority=Priority.HIGH)
    store.upsert_goal(g)

    # First attempt should fail → step READY again with attempts=1; eventually DONE
    assert wait_until(lambda: (store.get_goal("g_retry") or g).status == Status.DONE, timeout=2.5)

    s = store.steps_for("g_retry")[0]
    assert s.status == Status.DONE and (s.attempts or 0) >= 1


def test_paused_goal_with_ready_step_is_ignored_until_resumed(store: FileGoalsStore, daemon: GoalsDaemon):
    # Create a PAUSED goal (so daemon ignores it), but inject a READY step manually.
    g = Goal(id="g_paused", title="hold", kind="pause", spec={}, status=Status.PAUSED)
    store.upsert_goal(g)

    s = Step(id="g_paused_s1", goal_id="g_paused", name="noop", action={"op": "ok"}, status=Status.READY)
    store.upsert_step(s)

    # Ensure it does not run while PAUSED
    time.sleep(0.3)
    g_now = store.get_goal("g_paused") or g
    assert g_now.status == Status.PAUSED
    s_now = next(ss for ss in store.list_steps(goal_id="g_paused") if ss.id == "g_paused_s1")
    assert s_now.status == Status.READY, "step should not start while goal is PAUSED"

    # Resume → daemon should consider it and complete
    store.upsert_goal(replace(g_now, status=Status.READY, updated_at=UTCNOW()))
    assert wait_until(lambda: (store.get_goal("g_paused") or g).status == Status.DONE, timeout=2.0)
