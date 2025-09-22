# goals/events.py
# Typed events for goals/steps and adapters to reaper & WAL (newline-delimited JSON)

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .model import Goal, Step

UTCNOW = lambda: datetime.now(timezone.utc)
EVENT_VERSION = 1


# ----------- Event kinds -----------

class EventKind(str, Enum):
    # Goal lifecycle
    GoalCreated   = "GoalCreated"
    GoalUpdated   = "GoalUpdated"
    GoalCancelled = "GoalCancelled"
    GoalPlanned   = "GoalPlanned"
    GoalBlocked   = "GoalBlocked"
    GoalUnblocked = "GoalUnblocked"
    GoalFinished  = "GoalFinished"
    GoalFailed    = "GoalFailed"

    # Step lifecycle
    StepPlanned   = "StepPlanned"
    StepStarted   = "StepStarted"
    StepFinished  = "StepFinished"
    StepFailed    = "StepFailed"


# ----------- Base + typed events -----------

@dataclass
class BaseEvent:
    ts: str = field(default_factory=lambda: UTCNOW().isoformat())
    kind: str = ""
    src: str = "goals"
    level: str = "info"
    v: int = EVENT_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalEvent(BaseEvent):
    goal_id: str = ""
    goal_kind: str = ""
    title: str = ""
    status: str = ""
    priority: str = ""
    deadline: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class StepEvent(BaseEvent):
    step_id: str = ""
    goal_id: str = ""
    name: str = ""
    status: str = ""
    attempts: int = 0
    max_attempts: int = 0


# ----------- Factories -----------

def make_goal_event(kind: str | EventKind, goal: Goal, *, level: str = "info", extra: Optional[Dict[str, Any]] = None) -> GoalEvent:
    return GoalEvent(
        kind=str(kind),
        level=level,
        goal_id=goal.id,
        goal_kind=goal.kind,
        title=goal.title or "",
        status=getattr(goal.status, "name", str(goal.status)),
        priority=getattr(goal.priority, "name", str(goal.priority)),
        deadline=goal.deadline.isoformat() if goal.deadline else None,
        tags=tuple(goal.tags or []),
        extra=dict(extra or {}),
    )


def make_step_event(kind: str | EventKind, step: Step, *, level: str = "info", extra: Optional[Dict[str, Any]] = None) -> StepEvent:
    return StepEvent(
        kind=str(kind),
        level=level,
        step_id=step.id,
        goal_id=step.goal_id,
        name=step.name or "",
        status=getattr(step.status, "name", str(step.status)),
        attempts=int(step.attempts or 0),
        max_attempts=int(step.max_attempts or 0),
        extra=dict(extra or {}),
    )


# ----------- Adapters -----------

def to_reaper_event(ev: BaseEvent) -> Dict[str, Any]:
    """
    Flatten to a dict suitable for your reaper/observability pipeline.
    (You can pass this to the daemon/api reaper_sink.)
    """
    d = asdict(ev)
    # Keep payload compact & consistent
    d["source"] = d.pop("src")
    return d


def to_memory_note(ev: BaseEvent) -> Tuple[str, str, Dict[str, Any]]:
    """
    Lightweight mapping to your memory writer hook:
      returns (kind, content, meta)
    """
    if isinstance(ev, GoalEvent):
        content = f"{ev.kind}: {ev.title} [{ev.goal_kind}/{ev.priority}]"
        meta = {"goal_id": ev.goal_id, "status": ev.status, "priority": ev.priority, "deadline": ev.deadline}
        return "goal_event", content, meta
    if isinstance(ev, StepEvent):
        content = f"{ev.kind}: {ev.name} ({ev.step_id}) [{ev.status}]"
        meta = {"goal_id": ev.goal_id, "step_id": ev.step_id, "attempts": ev.attempts, "max_attempts": ev.max_attempts}
        return "goal_step", content, meta
    # Fallback
    return "goal_event", f"{ev.kind}", asdict(ev)


# ----------- WAL (JSONL) helpers -----------

def event_to_wal_line(ev: BaseEvent) -> str:
    """
    Serialize to a single-line JSON string for append-only WAL.
    """
    d = asdict(ev)
    # Ensure no newlines to keep WAL one-event-per-line
    s = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
    return s.replace("\n", "\\n")


def event_from_wal_line(line: str) -> Dict[str, Any]:
    """
    Parse a WAL JSON line back to a dict (typed reconstruction optional upstream).
    """
    try:
        return json.loads(line)
    except Exception:
        # tolerate escaped newlines
        return json.loads(line.replace("\\n", "\n"))


__all__ = [
    "EventKind",
    "BaseEvent",
    "GoalEvent",
    "StepEvent",
    "make_goal_event",
    "make_step_event",
    "to_reaper_event",
    "to_memory_note",
    "event_to_wal_line",
    "event_from_wal_line",
]
