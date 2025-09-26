# goals/policy.py
# Scheduler policy for selecting which READY steps to run next (priority, deadlines, fairness, optional locks)

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .model import Goal, Step, Status, Priority

UTCNOW = lambda: datetime.now(timezone.utc)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def choose_next_steps(
    *,
    candidates: List[Tuple[Goal, Step]],
    store: Any,
    ctx: Dict[str, Any],
    capacity: Optional[int] = None,
) -> List[Step]:
    """
    Select up to `capacity` Steps to enqueue now.

    Inputs
    ------
    candidates : list[(Goal, Step)]
        Pairs that are Status.READY and deps-satisfied (prepared by daemon).
    store : GoalsStore-like
        Unused here (duck-typed hook if you want to read extra store hints).
    ctx : dict
        HandlerContext; used for fair scheduling state and optional LockManager (ctx["locks"]).
        We persist a tiny fairness memory under ctx["_policy_fair"] between pulses.
    capacity : int | None
        Maximum number of steps to pick. If None, treat as unlimited.

    Policy
    ------
    1) Hard cap: at most one step per goal per pulse (prevents single-goal hogging).
    2) Priority tiers: CRITICAL > HIGH > NORMAL > LOW; overdue deadlines escalate to CRITICAL.
    3) Deadlines: overdue gets a large bump; near deadlines get a smaller bump.
    4) Fairness: goals not picked recently accrue "wait" credit (soft bump).
    5) Optional locks hint: if step.action has "locks": ["name", ...], prefer steps whose locks are available.
    6) Second pass: if capacity remains, allow additional steps from the SAME goal
       only when that goal is effectively CRITICAL because its deadline is overdue.

    Returns
    -------
    list[Step]: chosen steps (length ≤ capacity).
    """
    if not candidates:
        return []
    cap = len(candidates) if capacity is None else max(0, int(capacity))
    if cap == 0:
        return []

    # Build fairness state in ctx (survives across pulses)
    fair = ctx.setdefault("_policy_fair", {"last_pick_ts": {}})  # type: ignore[assignment]
    last_pick_ts: Dict[str, float] = fair["last_pick_ts"]  # type: ignore[index]

    now = UTCNOW()

    # Compute scores for each (goal, step)
    scored: List[Tuple[Tuple, Goal, Step]] = []

    for g, s in candidates:
        if s.status != Status.READY:
            continue

        prio_eff = _effective_priority(g, now)
        overdue = g.overdue(now)

        # Deadline urgency: seconds until deadline (negative if overdue)
        dl_urg = _deadline_urgency(g, now)  # higher is more urgent

        # Fairness bump: seconds since this goal was last picked (capped, scaled)
        waited = _wait_credit(last_pick_ts.get(g.id), now_ts=_now_ts())

        # Locks availability hint (1 if free/ok, 0 if appears busy) — soft factor
        locks_ok = _locks_available_hint(ctx, s, holder=g.id)

        # Composite sort key (descending by each term):
        #  - overdue (1/0)
        #  - effective priority
        #  - deadline urgency (bigger → sooner deadline / negative if far future)
        #  - locks_ok (prefer steps that won't block immediately)
        #  - waited (fairness)
        #  - tiebreaker: created_at/updated_at if available on goal (older first)
        key = (
            1 if overdue else 0,
            prio_eff,
            dl_urg,
            locks_ok,
            waited,
            _ts_safe(getattr(g, "created_at", None)),
            _ts_safe(getattr(g, "updated_at", None)),
        )
        scored.append((key, g, s))

    if not scored:
        return []

    # Sort by key descending
    scored.sort(key=lambda t: t[0], reverse=True)

    # Pick at most one step per goal per pulse
    chosen: List[Step] = []
    used_goals: set[str] = set()
    chosen_ids: set[str] = set()

    for _key, g, s in scored:
        if len(chosen) >= cap:
            break
        if g.id in used_goals:
            continue
        if s.id in chosen_ids:
            continue
        chosen.append(s)
        used_goals.add(g.id)
        chosen_ids.add(s.id)

    # Second pass: allow extra steps from goals that are overdue → effectively CRITICAL,
    # only if capacity remains, and never duplicate already-chosen steps.
    if len(chosen) < cap:
        for _key, g, s in scored:
            if len(chosen) >= cap:
                break
            if g.id not in used_goals:
                continue  # only top-up goals that already got one step
            if _effective_priority(g, now) >= Priority.CRITICAL and g.overdue(now):
                if s.id in chosen_ids:
                    continue
                chosen.append(s)
                chosen_ids.add(s.id)

    # Update fairness timestamps for goals we picked
    ts_now = _now_ts()
    for step in chosen:
        # Find the goal_id for this step from candidates
        gid = _goal_id_for_step(step, candidates)
        if gid:
            last_pick_ts[gid] = ts_now

    return chosen


# -----------------------------------------------------------------------------
# Helpers (scoring and hints)
# -----------------------------------------------------------------------------

def _effective_priority(goal: Goal, now: datetime) -> int:
    """Escalate to CRITICAL when overdue; otherwise map Enum → int for comparison."""
    try:
        if goal.overdue(now):
            return int(Priority.CRITICAL)
        return int(goal.priority)
    except Exception:
        # Fallback for non-enum custom priorities
        try:
            return int(getattr(goal, "priority", Priority.NORMAL))
        except Exception:
            return int(Priority.NORMAL)


def _deadline_urgency(goal: Goal, now: datetime) -> float:
    """
    Convert deadline proximity into a bounded score:
      - Overdue → large positive bump proportional to hours overdue (clamped).
      - Future → smaller positive bump as it approaches (inverse time).
    """
    dl = getattr(goal, "deadline", None)
    if not dl:
        return 0.0
    try:
        # seconds positive if now past deadline
        delta_sec = (now - dl).total_seconds()
    except Exception:
        return 0.0

    if delta_sec >= 0:
        # Overdue: growth ~ log(1 + hours_overdue)
        hours = delta_sec / 3600.0
        return 100.0 + math.log1p(min(hours, 24.0)) * 10.0  # capped influence
    else:
        # Future: inverse of time remaining (closer deadline → larger score)
        hours_left = (-delta_sec) / 3600.0
        return 10.0 / max(0.25, min(hours_left, 72.0))  # within 3 days gets meaningful bump


def _wait_credit(last_pick_ts: Optional[float], now_ts: float) -> float:
    """
    Fairness credit increases with time since last pick; capped to avoid runaway.
    """
    if not last_pick_ts:
        return 1.0
    waited = max(0.0, now_ts - last_pick_ts)  # seconds
    # Diminishing returns: log-scale, cap at ~30 sec worth
    return min(5.0, math.log1p(waited))


def _locks_available_hint(ctx: Dict[str, Any], step: Step, *, holder: str) -> float:
    """
    If step.action declares required locks (['locks']), prefer those whose locks appear available.
    This is a soft hint: returns 1.0 if likely available, 0.0 if likely busy, 0.5 unknown.
    """
    try:
        req = step.action.get("locks")
        if not req:
            return 0.5
        lm = ctx.get("locks")
        if not lm:
            return 0.5
        ok_all = True
        for name in req:
            # If lock is held by someone else, likely busy
            held_by = lm.held_by(name)
            if held_by is None:
                continue  # free
            if held_by != holder:
                ok_all = False
                break
        return 1.0 if ok_all else 0.0
    except Exception:
        return 0.5


def _goal_id_for_step(step: Step, pairs: List[Tuple[Goal, Step]]) -> Optional[str]:
    for g, s in pairs:
        if s is step:
            return g.id
    return None


def _ts_safe(dt: Optional[datetime]) -> float:
    if not dt:
        return 0.0
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _now_ts() -> float:
    return time.monotonic()


__all__ = ["choose_next_steps"]
