# goals/handlers/dummy.py
from __future__ import annotations
from typing import Optional
from datetime import datetime, timezone

from .dummy import DummyHandler
from ..model import Goal, Step, Status
from .base import BaseGoalHandler, HandlerContext

UTCNOW = lambda: datetime.now(timezone.utc)

class DummyHandler(BaseGoalHandler):
    """Tiny test handler for kind='dummy' that completes a step immediately."""
    kind = "dummy"

    def plan(self, goal: Goal, ctx: HandlerContext):
        return []

    def is_blocked(self, goal: Goal, ctx: HandlerContext):
        return False, None

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        if step.started_at is None:
            step.started_at = UTCNOW()
            step.status = Status.RUNNING
            return step
        step.status = Status.DONE
        step.finished_at = UTCNOW()
        return step

__all__ = ["DummyHandler"]