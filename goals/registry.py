# goals/registry.py
# Handler registry mapping goal kinds to handlers (lookup + registration utilities)

from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional

from .handlers.base import GoalHandler


def _key(kind: str) -> str:
    return str(kind).strip().lower()


class GoalRegistry:
    """
    Minimal registry that maps goal.kind → GoalHandler instance.

    Usage:
      reg = GoalRegistry([CodingHandler(), ResearchHandler()])
      h = reg.get("coding")
      reg.register(HousekeepingHandler())
    """

    def __init__(self, handlers: Optional[Iterable[GoalHandler]] = None) -> None:
        self._mu = threading.Lock()
        self._by_kind: Dict[str, GoalHandler] = {}
        if handlers:
            self.register_many(list(handlers), replace=True)

    # ---------- CRUD ----------

    def register(self, handler: GoalHandler, *, replace: bool = False) -> None:
        k = _key(getattr(handler, "kind", ""))
        if not k:
            raise ValueError("handler.kind must be a non-empty string")
        with self._mu:
            if not replace and k in self._by_kind:
                raise ValueError(f"handler for kind '{k}' already registered")
            self._by_kind[k] = handler

    def register_many(self, handlers: Iterable[GoalHandler], *, replace: bool = False) -> None:
        for h in handlers:
            self.register(h, replace=replace)

    def get(self, kind: str) -> Optional[GoalHandler]:
        return self._by_kind.get(_key(kind))

    def kinds(self) -> List[str]:
        with self._mu:
            return list(self._by_kind.keys())

    def remove(self, kind: str) -> Optional[GoalHandler]:
        with self._mu:
            return self._by_kind.pop(_key(kind), None)

    def as_dict(self) -> Dict[str, GoalHandler]:
        with self._mu:
            return dict(self._by_kind)


# ---------- convenience factory ----------

def build_default_registry() -> GoalRegistry:
    """
    Best-effort registry with built-in handlers if importable.
    Safe to call even if some handlers are missing.
    """
    handlers: List[GoalHandler] = []
    try:
        from .handlers.coding import CodingHandler  # type: ignore
        handlers.append(CodingHandler())
    except Exception:
        pass
    try:
        from .handlers.research import ResearchHandler  # type: ignore
        handlers.append(ResearchHandler())
    except Exception:
        pass
    try:
        from .handlers.housekeeping import HousekeepingHandler  # type: ignore
        handlers.append(HousekeepingHandler())
    except Exception:
        pass
    return GoalRegistry(handlers)


__all__ = ["GoalRegistry", "build_default_registry"]
