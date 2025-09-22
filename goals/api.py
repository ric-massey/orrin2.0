# goals/api.py
# Public facade for creating/updating/cancelling/listing goals and notifying the Goals daemon & observers

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import Goal, Status, Priority, Progress

Subscriber = Callable[[Dict[str, Any]], None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _mk_id(prefix: str = "g_") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def _parse_deadline(deadline: Optional[Any]) -> Optional[datetime]:
    if deadline is None:
        return None
    if isinstance(deadline, datetime):
        return deadline if deadline.tzinfo else deadline.replace(tzinfo=timezone.utc)
    if isinstance(deadline, (int, float)):
        # treat as unix epoch seconds
        return datetime.fromtimestamp(float(deadline), tz=timezone.utc)
    if isinstance(deadline, str):
        s = deadline.strip()
        try:
            # tolerate Z suffix
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _to_priority(p: Any) -> Priority:
    if isinstance(p, Priority):
        return p
    if isinstance(p, str):
        s = p.strip().upper()
        mapping = {"ROUTINE": Priority.LOW, "IMPORTANT": Priority.NORMAL, "CRITICAL": Priority.CRITICAL, "LOW": Priority.LOW, "NORMAL": Priority.NORMAL, "HIGH": Priority.HIGH}
        if s in mapping:
            return mapping[s]
        try:
            # allow "0/1/2/3" strings
            return Priority(int(s))
        except Exception:
            return Priority.NORMAL
    if isinstance(p, (int, float)):
        try:
            return Priority(int(p))
        except Exception:
            pass
    return Priority.NORMAL


def _store_upsert(store: Any, goal: Goal) -> None:
    if hasattr(store, "upsert_goal"):
        store.upsert_goal(goal)
        return
    if hasattr(store, "save_goal"):
        store.save_goal(goal)
        return
    if hasattr(store, "add_goal") and hasattr(store, "get_goal"):
        if store.get_goal(goal.id) is None:
            store.add_goal(goal)
            return
    if hasattr(store, "update_goal"):
        store.update_goal(goal)
        return
    raise AttributeError("GoalsStore does not expose upsert/save/add APIs I recognize")


def _store_get(store: Any, goal_id: str) -> Optional[Goal]:
    if hasattr(store, "get_goal"):
        return store.get_goal(goal_id)
    if hasattr(store, "by_id"):
        return store.by_id(goal_id)
    if hasattr(store, "find_goal"):
        return store.find_goal(goal_id)
    raise AttributeError("GoalsStore does not expose get/by_id/find APIs I recognize")


def _store_iter(store: Any) -> Iterable[Goal]:
    if hasattr(store, "iter_goals"):
        return store.iter_goals()
    if hasattr(store, "list_goals"):
        return store.list_goals()
    if hasattr(store, "all"):
        return store.all()
    raise AttributeError("GoalsStore does not expose iter/list/all APIs I recognize")


class GoalsAPI:
    """
    Lightweight API surface so Orrin's main loop (and UIs/tests) can interact with the Goals subsystem
    without touching internals.

    Dependencies:
      - store: any object with upsert/get/list-like methods (duck-typed)
      - daemon (optional): object exposing submit(goal_id: str)
      - reaper_sink (optional): callable(event_dict) to send observability events
      - memory_writer (optional): callable(kind: str, content: str, meta: dict) → None
    """

    def __init__(
        self,
        store: Any,
        *,
        daemon: Optional[Any] = None,
        reaper_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        memory_writer: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        plan_on_create: bool = True,
    ) -> None:
        self.store = store
        self.daemon = daemon
        self.reaper_sink = reaper_sink
        self.memory_writer = memory_writer
        self.plan_on_create = plan_on_create

        self._subs: List[Subscriber] = []
        self._lock = threading.Lock()

    # ---------- Public API ----------

    def create_goal(
        self,
        *,
        title: str,
        kind: str,
        spec: Optional[Dict[str, Any]] = None,
        priority: Any = Priority.NORMAL,
        deadline: Optional[Any] = None,
        tags: Optional[Sequence[str]] = None,
        parent_id: Optional[str] = None,
        acceptance: Optional[Dict[str, Any]] = None,
        triggers: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Goal:
        gid = _mk_id()
        spec = dict(spec or {})
        if triggers:
            spec.setdefault("triggers", list(triggers))
        goal = Goal(
            id=gid,
            title=title,
            kind=kind,
            spec=spec,
            priority=_to_priority(priority),
            status=Status.NEW,
            deadline=_parse_deadline(deadline),
            parent_id=parent_id,
            tags=list(tags or []),
            progress=Progress(),
            acceptance=dict(acceptance or {}),
        )
        _store_upsert(self.store, goal)
        self._emit_event("GoalCreated", goal, extra={"reason": "api.create"})
        self._write_memory_event("goal_event", f"Created goal: {goal.title} [{goal.kind}/{goal.priority.name}]", goal)

        # Optionally kick the daemon immediately
        if self.plan_on_create:
            self.submit(goal.id)

        return goal

    def update_goal(self, goal_id: str, **fields: Any) -> Optional[Goal]:
        goal = _store_get(self.store, goal_id)
        if goal is None:
            return None

        # Normalize certain fields
        if "priority" in fields:
            fields["priority"] = _to_priority(fields["priority"])
        if "deadline" in fields:
            fields["deadline"] = _parse_deadline(fields["deadline"])
        if "tags" in fields and fields["tags"] is not None:
            fields["tags"] = list(fields["tags"])
        if "spec" in fields and fields["spec"] is not None:
            fields["spec"] = dict(fields["spec"])

        updated = replace(goal, **fields, updated_at=_utcnow())
        _store_upsert(self.store, updated)

        self._emit_event("GoalUpdated", updated, extra={"changed": list(fields.keys())})
        return updated

    def cancel_goal(self, goal_id: str, reason: str = "api.cancel") -> Optional[Goal]:
        goal = _store_get(self.store, goal_id)
        if goal is None:
            return None
        if goal.status in {Status.DONE, Status.FAILED, Status.CANCELLED}:
            return goal
        updated = replace(goal, status=Status.CANCELLED, updated_at=_utcnow(), last_error=reason)
        _store_upsert(self.store, updated)
        self._emit_event("GoalCancelled", updated, extra={"reason": reason})
        self._write_memory_event("goal_event", f"Cancelled goal: {updated.title}", updated)
        return updated

    def list_goals(
        self,
        *,
        kinds: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[Status]] = None,
        priorities: Optional[Sequence[Priority]] = None,
        tags: Optional[Sequence[str]] = None,
        text: Optional[str] = None,
        limit: Optional[int] = None,
        sort: str = "-updated_at",  # or "created_at", "-priority"
    ) -> List[Goal]:
        def _match(g: Goal) -> bool:
            if kinds and g.kind not in kinds:
                return False
            if statuses and g.status not in statuses:
                return False
            if priorities and g.priority not in priorities:
                return False
            if tags and not set(tags).issubset(set(g.tags or [])):
                return False
            if text:
                t = text.lower()
                if t not in (g.title or "").lower() and t not in (g.last_error or "").lower():
                    return False
            return True

        items = [g for g in _store_iter(self.store) if _match(g)]
        keyrev = False
        keyname = sort
        if sort.startswith("-"):
            keyrev = True
            keyname = sort[1:]

        def _key(g: Goal):
            return getattr(g, keyname, None)

        items.sort(key=_key, reverse=keyrev)
        return items[:limit] if limit else items

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        return _store_get(self.store, goal_id)

    def submit(self, goal_id: str) -> None:
        """Kick the daemon to (re)consider a goal now."""
        if self.daemon and hasattr(self.daemon, "submit"):
            try:
                self.daemon.submit(goal_id)
            except Exception:
                # Best-effort; don't raise to caller
                pass

    # ---------- Subscriptions ----------

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Subscribe to API events. Returns an unsubscribe function."""
        with self._lock:
            self._subs.append(callback)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subs.remove(callback)
                except ValueError:
                    pass

        return _unsub

    # ---------- Internals ----------

    def _emit_event(self, kind: str, goal: Goal, *, extra: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "ts": _utcnow().isoformat(),
            "kind": kind,
            "goal_id": goal.id,
            "goal_kind": goal.kind,
            "status": goal.status.value if hasattr(goal.status, "value") else str(goal.status),
            "priority": goal.priority.value if hasattr(goal.priority, "value") else int(goal.priority),
            "title": goal.title,
            "deadline": goal.deadline.isoformat() if goal.deadline else None,
            "tags": list(goal.tags or []),
            "extra": dict(extra or {}),
        }
        # Notify subscribers
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                continue
        # Send to reaper/observability if provided
        if callable(self.reaper_sink):
            try:
                self.reaper_sink(event)
            except Exception:
                pass

    def _write_memory_event(self, kind: str, content: str, goal: Goal) -> None:
        if not callable(self.memory_writer):
            return
        try:
            meta = {"goal_id": goal.id, "goal_kind": goal.kind, "priority": int(goal.priority), "deadline": goal.deadline.isoformat() if goal.deadline else None}
            self.memory_writer(kind, content, meta)
        except Exception:
            pass


__all__ = ["GoalsAPI"]
