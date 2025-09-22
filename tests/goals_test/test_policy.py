# tests/goals/test_policy.py
# Pytest for the scheduler policy (choose_next_steps): priority, deadlines, fairness, locks, capacity

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pytest

from goals.policy import choose_next_steps
from goals.model import Goal, Step, Status, Priority
from goals.locks import LockManager


UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _mk_goal(
    gid: str,
    *,
    title: str = "g",
    kind: str = "dummy",
    priority: Priority = Priority.NORMAL,
    status: Status = Status.READY,
    deadline: datetime | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> Goal:
    g = Goal(
        id=gid,
        title=title,
        kind=kind,
        spec={},
        priority=priority,
        status=status,
        deadline=deadline,
    )
    if created_at:
        g.created_at = created_at
    if updated_at:
        g.updated_at = updated_at
    return g


def _mk_step(gid: str, sid: str) -> Step:
    return Step(id=sid, goal_id=gid, name=f"{sid}", action={"op": "noop"}, status=Status.READY)


def _pair(g: Goal, *steps: Step) -> List[Tuple[Goal, Step]]:
    return [(g, s) for s in steps]


def test_overdue_escalates_above_higher_priority():
    now = _now()
    # g_low is LOW but overdue → escalates to CRITICAL
    g_low_overdue = _mk_goal("g_low", priority=Priority.LOW, deadline=now - timedelta(hours=2))
    s_low = _mk_step(g_low_overdue.id, "s_low")

    # g_high is HIGH but not overdue
    g_high = _mk_goal("g_high", priority=Priority.HIGH, deadline=now + timedelta(hours=5))
    s_high = _mk_step(g_high.id, "s_high")

    candidates = _pair(g_low_overdue, s_low) + _pair(g_high, s_high)
    chosen = choose_next_steps(candidates=candidates, store=None, ctx={}, capacity=1)

    assert chosen and chosen[0].id == "s_low", "Overdue LOW should outrank non-overdue HIGH"


def test_deadline_urgency_prefers_sooner_when_otherwise_equal():
    now = _now()
    base_created = now - timedelta(days=1)
    # Same priority & status; only deadlines differ
    g_soon = _mk_goal("g_soon", priority=Priority.NORMAL, deadline=now + timedelta(hours=1), created_at=base_created)
    g_later = _mk_goal("g_later", priority=Priority.NORMAL, deadline=now + timedelta(hours=6), created_at=base_created)

    s1 = _mk_step(g_soon.id, "s1")
    s2 = _mk_step(g_later.id, "s2")

    chosen = choose_next_steps(candidates=_pair(g_soon, s1) + _pair(g_later, s2), store=None, ctx={}, capacity=1)
    assert chosen[0].id == "s1", "Closer deadline should be selected first"


def test_capacity_and_one_per_goal_limit():
    g1 = _mk_goal("g1", priority=Priority.HIGH)
    g2 = _mk_goal("g2", priority=Priority.HIGH)
    s1a, s1b = _mk_step("g1", "s1a"), _mk_step("g1", "s1b")
    s2a, s2b = _mk_step("g2", "s2a"), _mk_step("g2", "s2b")

    # With capacity 2 we should get at most one step per goal on the first pass
    chosen = choose_next_steps(
        candidates=_pair(g1, s1a, s1b) + _pair(g2, s2a, s2b),
        store=None,
        ctx={},
        capacity=2,
    )
    goal_ids = {s.goal_id for s in chosen}
    assert len(chosen) == 2
    assert goal_ids == {"g1", "g2"}, "Should pick at most one step per goal per pulse"


def test_fairness_picks_the_other_goal_next_pulse_when_equal():
    # Two identical goals (no deadlines, same priority). Fairness should alternate over pulses.
    created = _now() - timedelta(days=1)
    gA = _mk_goal("gA", priority=Priority.NORMAL, created_at=created, updated_at=created)
    gB = _mk_goal("gB", priority=Priority.NORMAL, created_at=created, updated_at=created)
    sA = _mk_step("gA", "sA")
    sB = _mk_step("gB", "sB")

    ctx = {}  # reused between pulses to keep fairness memory

    # First pulse: either can be chosen (tie-breakers). Record which.
    first = choose_next_steps(candidates=_pair(gA, sA) + _pair(gB, sB), store=None, ctx=ctx, capacity=1)
    assert first and first[0].id in {"sA", "sB"}
    first_id = first[0].id

    # Tiny wait to give some "waited" credit difference
    time.sleep(0.02)

    # Second pulse: the other one should now have higher fairness credit and be chosen
    second = choose_next_steps(candidates=_pair(gA, sA) + _pair(gB, sB), store=None, ctx=ctx, capacity=1)
    assert second and second[0].id in {"sA", "sB"}
    assert second[0].id != first_id, "Fairness should prefer the other goal on the next pulse when equal"


def test_locks_hint_prefers_available_lock():
    now = _now()
    # Two goals compete for the same lock "L"
    g1 = _mk_goal("g1", priority=Priority.NORMAL, deadline=now + timedelta(hours=2))
    g2 = _mk_goal("g2", priority=Priority.NORMAL, deadline=now + timedelta(hours=2))

    s1 = _mk_step("g1", "s1"); s1.action["locks"] = ["L"]
    s2 = _mk_step("g2", "s2"); s2.action["locks"] = ["L"]

    # Simulate that lock "L" is already held by g1 → policy should prefer s1
    locks = LockManager(ttl_seconds=60.0)
    assert locks.acquire("L", holder_id="g1")
    ctx = {"locks": locks}

    chosen = choose_next_steps(candidates=_pair(g1, s1) + _pair(g2, s2), store=None, ctx=ctx, capacity=1)
    assert chosen and chosen[0].id == "s1", "Lock availability hint should favor the goal that already holds the lock"


def test_second_pass_allows_extra_for_critical_overdue():
    now = _now()
    # One goal with two steps, overdue → effective priority escalates to CRITICAL.
    g = _mk_goal("g_overdue", priority=Priority.LOW, deadline=now - timedelta(minutes=30))
    s1 = _mk_step(g.id, "s1")
    s2 = _mk_step(g.id, "s2")

    # Capacity 2: first pass picks one step; second pass may add another for CRITICAL overdue goals.
    chosen = choose_next_steps(candidates=_pair(g, s1, s2), store=None, ctx={}, capacity=2)
    assert len(chosen) == 2, "Policy should allow a second step for overdue CRITICAL goals when capacity permits"
    assert {s.id for s in chosen} == {"s1", "s2"}
