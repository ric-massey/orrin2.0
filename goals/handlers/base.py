# goals/handlers/base.py
# Base protocol and helpers for goal handlers in the Goals daemon (defines handler interface)

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Optional, Tuple, List, Dict, Any

from ..model import Goal, Step

# Public type alias for anything we pass around to handlers (services, paths, config, clients, etc.)
HandlerContext = Dict[str, Any]


@runtime_checkable
class GoalHandler(Protocol):
    """
    Minimal contract each goal handler must satisfy.

    Semantics:
      - accept(goal): True if this handler is appropriate for goal.kind/spec.
      - plan(goal, ctx): Return an initial list of Steps (may be empty). Called when a NEW goal is admitted.
      - is_blocked(goal, ctx): Return (True, reason) if the goal cannot currently progress (e.g., waiting on lock, IO).
      - tick(goal, step, ctx): Execute or advance one step; return the updated Step (or None if no change this tick).
      - on_event(goal, event, ctx): Optional hook for reacting to external system events (reaper/memory/watchdogs).
    """
    kind: str

    def accept(self, goal: Goal) -> bool: ...
    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]: ...
    def is_blocked(self, goal: Goal, ctx: HandlerContext) -> Tuple[bool, Optional[str]]: ...
    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]: ...
    def on_event(self, goal: Goal, event: Dict[str, Any], ctx: HandlerContext) -> None: ...


class BaseGoalHandler(ABC):
    """
    Convenience abstract base class that implements sensible defaults.
    Subclass and override as needed; set `kind` to your handler's name.
    """
    kind: str = "base"

    def accept(self, goal: Goal) -> bool:
        # Default: match by kind string
        return getattr(goal, "kind", None) == self.kind

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        # Default: no preplanned steps; daemon/other hooks may enqueue later
        return []

    def is_blocked(self, goal: Goal, ctx: HandlerContext) -> Tuple[bool, Optional[str]]:
        # Default: not blocked
        return False, None

    @abstractmethod
    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        """
        Required: advance work for a single step.
        Return the updated Step (mutated or replacement), or None if nothing changed this tick.
        Raise exceptions only for *unexpected* errors; expected transient issues should surface via is_blocked().
        """
        raise NotImplementedError

    def on_event(self, goal: Goal, event: Dict[str, Any], ctx: HandlerContext) -> None:
        # Default: ignore external events
        return None


__all__ = ["GoalHandler", "BaseGoalHandler", "HandlerContext"]
