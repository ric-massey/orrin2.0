# tests/goals/test_model.py
# Pytest tests for core dataclasses/enums in goals.model (Status, Priority, Progress, Step, Goal)

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from goals.model import Goal, Step, Progress, Status, Priority


def test_priority_enum_order_and_values():
    assert int(Priority.LOW) == 0
    assert int(Priority.NORMAL) == 1
    assert int(Priority.HIGH) == 2
    assert int(Priority.CRITICAL) == 3
    assert Priority.CRITICAL > Priority.HIGH > Priority.NORMAL > Priority.LOW


def test_status_members_are_strings():
    # simple smoke: names & values should match
    assert Status.NEW.name == "NEW" and Status.NEW.value == "NEW"
    assert Status.DONE.name == "DONE" and Status.DONE.value == "DONE"
    assert isinstance(Status.BLOCKED.value, str)


def test_progress_set_clamps_and_updates_note():
    p = Progress()
    assert p.percent == 0.0 and p.note == ""
    p.set(percent=150)  # clamp to 100
    assert p.percent == 100.0
    p.set(percent=-5)   # clamp to 0
    assert p.percent == 0.0
    p.set(note="working")
    assert p.note == "working"
    # evidence is a dict and unique per instance
    p.evidence["k"] = "v"
    q = Progress()
    assert "k" not in q.evidence


def test_step_defaults_and_mutability_lists():
    s = Step(id="s1", goal_id="g1", name="do thing", action={"op": "x"})
    assert s.status == Status.READY
    assert s.attempts == 0 and s.max_attempts == 3
    assert s.deps == [] and s.artifacts == []
    assert s.started_at is None and s.finished_at is None and s.last_error is None
    # ensure lists are not shared across instances
    s.deps.append("other")
    s2 = Step(id="s2", goal_id="g1", name="do 2", action={"op": "y"})
    assert s2.deps == []


def test_goal_terminal_and_touch_updates_timestamp():
    g = Goal(id="g1", title="t", kind="k", spec={}, status=Status.NEW)
    t0 = g.updated_at
    g.touch()
    assert g.updated_at >= t0
    # terminal states
    for st in (Status.DONE, Status.FAILED, Status.CANCELLED):
        gg = Goal(id=f"g_{st}", title="t", kind="k", spec={}, status=st)
        assert gg.is_terminal() is True
    # non-terminal
    gg2 = Goal(id="g2", title="t", kind="k", spec={}, status=Status.READY)
    assert gg2.is_terminal() is False


def test_goal_overdue_true_false_cases():
    now = datetime.now(timezone.utc)
    past = now - timedelta(hours=1)
    future = now + timedelta(hours=1)

    g_past = Goal(id="gp", title="t", kind="k", spec={}, status=Status.READY, deadline=past)
    assert g_past.overdue(now) is True

    g_future = Goal(id="gf", title="t", kind="k", spec={}, status=Status.READY, deadline=future)
    assert g_future.overdue(now) is False

    # terminal goals are never overdue
    g_done = Goal(id="gd", title="t", kind="k", spec={}, status=Status.DONE, deadline=past)
    assert g_done.overdue(now) is False


def test_goal_overdue_with_naive_deadline_is_safe():
    now = datetime.now(timezone.utc)
    # Naive (no tz) deadline; method should not raise and should return False due to safe-guard
    naive_past = (datetime.now() - timedelta(hours=1)).replace(tzinfo=None)  # explicit naive
    g = Goal(id="gn", title="t", kind="k", spec={}, status=Status.READY, deadline=naive_past)
    assert g.overdue(now) in (False,)  # specifically False in current implementation
