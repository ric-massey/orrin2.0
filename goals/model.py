# goals/model.py
# Core dataclasses and enums for the Goals subsystem (Goal, Step, Status, Priority, Progress)

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

UTCNOW = lambda: datetime.now(timezone.utc)


class Status(str, Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    WAITING = "WAITING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class Priority(int, Enum):
    LOW = 0          # ROUTINE
    NORMAL = 1       # IMPORTANT
    HIGH = 2
    CRITICAL = 3


@dataclass
class Progress:
    percent: float = 0.0
    note: str = ""
    evidence: Dict[str, str] = field(default_factory=dict)

    def set(self, *, percent: Optional[float] = None, note: Optional[str] = None) -> None:
        if percent is not None:
            # Clamp to [0, 100]
            self.percent = max(0.0, min(100.0, float(percent)))
        if note is not None:
            self.note = note


@dataclass
class Step:
    id: str
    goal_id: str
    name: str
    action: Dict
    status: Status = Status.READY
    attempts: int = 0
    max_attempts: int = 3
    deps: List[str] = field(default_factory=list)

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_error: Optional[str] = None

    artifacts: List[str] = field(default_factory=list)


@dataclass
class Goal:
    id: str
    title: str
    kind: str
    spec: Dict

    priority: Priority = Priority.NORMAL
    status: Status = Status.NEW

    created_at: datetime = field(default_factory=UTCNOW)
    updated_at: datetime = field(default_factory=UTCNOW)
    deadline: Optional[datetime] = None

    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    progress: Progress = field(default_factory=Progress)
    acceptance: Dict = field(default_factory=dict)

    last_error: Optional[str] = None
    step_order: List[str] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = UTCNOW()

    def is_terminal(self) -> bool:
        return self.status in {Status.DONE, Status.FAILED, Status.CANCELLED}

    def overdue(self, now: Optional[datetime] = None) -> bool:
        if not self.deadline or self.is_terminal():
            return False
        now = now or UTCNOW()
        try:
            return (now - self.deadline).total_seconds() > 0
        except Exception:
            return False


__all__ = ["Status", "Priority", "Progress", "Step", "Goal"]
