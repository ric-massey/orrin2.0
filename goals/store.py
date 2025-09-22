# goals/store.py
# File-backed goals store: loads on start, persists to JSONL + WAL, provides filtering/indexing APIs.

from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .model import Goal, Step, Status, Priority, Progress
from . import wal as WAL
from .utils import ensure_dir, iso as _iso, parse_iso as _parse_iso, append_jsonl, iter_jsonl, write_text

UTCNOW = lambda: datetime.now(timezone.utc)


def _to_status(x: Any) -> Status:
    if isinstance(x, Status):
        return x
    try:
        return Status[str(x).upper()]
    except Exception:
        return Status.READY

def _to_priority(x: Any) -> Priority:
    if isinstance(x, Priority):
        return x
    try:
        return Priority[str(x).upper()]
    except Exception:
        try:
            return Priority(int(x))
        except Exception:
            return Priority.NORMAL

def _goal_from_dict(d: Dict[str, Any]) -> Goal:
    return Goal(
        id=str(d["id"]),
        title=str(d.get("title", "")),
        kind=str(d.get("kind", "")),
        spec=dict(d.get("spec") or {}),
        priority=_to_priority(d.get("priority", Priority.NORMAL)),
        status=_to_status(d.get("status", Status.NEW)),
        created_at=_parse_iso(d.get("created_at")) or UTCNOW(),
        updated_at=_parse_iso(d.get("updated_at")) or UTCNOW(),
        deadline=_parse_iso(d.get("deadline")),
        parent_id=d.get("parent_id"),
        tags=list(d.get("tags") or []),
        progress=Progress(**(d.get("progress") or {})),
        acceptance=dict(d.get("acceptance") or {}),
        last_error=d.get("last_error"),
        step_order=list(d.get("step_order") or []),
    )

def _step_from_dict(d: Dict[str, Any]) -> Step:
    return Step(
        id=str(d["id"]),
        goal_id=str(d.get("goal_id") or d.get("goalId") or ""),
        name=str(d.get("name", "")),
        action=dict(d.get("action") or {}),
        status=_to_status(d.get("status", Status.READY)),
        attempts=int(d.get("attempts", 0)),
        max_attempts=int(d.get("max_attempts", 3)),
        deps=list(d.get("deps") or []),
        started_at=_parse_iso(d.get("started_at")),
        finished_at=_parse_iso(d.get("finished_at")),
        last_error=d.get("last_error"),
        artifacts=list(d.get("artifacts") or []),
    )

def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, datetime):
                out[k] = _iso(v)
            else:
                out[k] = _jsonable(v)
        return out
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return _iso(obj)
    return obj


class FileGoalsStore:
    """
    Simple file-backed store.

    Directory layout:
      data_dir/
        state.jsonl   # append-only snapshots: {"goal":{...}} / {"step":{...}}
        wal.log       # append-only operational log (goal_upsert/step_upsert)

    On init, we load from state.jsonl if present; otherwise we replay WAL.
    """

    def __init__(self, data_dir: Path | str):
        self._dir = ensure_dir(Path(data_dir))
        self._state = self._dir / "state.jsonl"
        self._wal = self._dir / "wal.log"
        # Ensure files exist
        self._state.touch(exist_ok=True)
        self._wal.touch(exist_ok=True)

        self._goals: Dict[str, Goal] = {}
        self._steps: Dict[str, Step] = {}
        self._by_goal_steps: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

        self._load()

    # ---------------- paths / counts ----------------

    def paths(self) -> Dict[str, str]:
        return {"dir": str(self._dir), "state": str(self._state), "wal": str(self._wal)}

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {"goals": len(self._goals), "steps": len(self._steps)}

    # ---------------- loading ----------------

    def _load(self) -> None:
        """
        Load state from state.jsonl; if empty, try replaying wal.log.
        """
        loaded_any = False
        try:
            for rec in iter_jsonl(self._state):
                if "goal" in rec:
                    g = _goal_from_dict(rec["goal"])
                    self._goals[g.id] = g
                    loaded_any = True
                if "step" in rec:
                    s = _step_from_dict(rec["step"])
                    self._steps[s.id] = s
                    self._by_goal_steps.setdefault(s.goal_id, []).append(s.id)
                    loaded_any = True
        except Exception:
            # fall back to WAL replay below
            loaded_any = False

        if not loaded_any:
            # Try WAL replay
            try:
                WAL.replay_to_store(self, self._wal)
            except Exception:
                pass  # start empty if even WAL is unreadable

    # ---------------- CRUD: goals ----------------

    def upsert_goal(self, g: Goal) -> Goal:
        with self._lock:
            # normalize/update timestamps
            g.updated_at = UTCNOW() if getattr(g, "updated_at", None) is None else g.updated_at
            if getattr(g, "created_at", None) is None:
                g.created_at = UTCNOW()

            self._goals[g.id] = g

            # append state snapshot & wal
            append_jsonl(self._state, [{"goal": _jsonable(g)}])
            WAL.append(self._wal, {"type": "goal_upsert", "goal": _jsonable(g)})
            return g

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        with self._lock:
            return self._goals.get(goal_id)

    def list_goals(
        self,
        *,
        kinds: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[Status]] = None,
        priorities: Optional[Sequence[Priority]] = None,
        tags: Optional[Sequence[str]] = None,
        text: Optional[str] = None,
        limit: Optional[int] = None,
        sort: str = "-updated_at",
    ) -> List[Goal]:
        with self._lock:
            items = list(self._goals.values())

        if kinds:
            kset = {k.lower() for k in kinds}
            items = [g for g in items if (g.kind or "").lower() in kset]
        if statuses:
            sset = {s for s in statuses}
            items = [g for g in items if g.status in sset]
        if priorities:
            pset = {p for p in priorities}
            items = [g for g in items if g.priority in pset]
        if tags:
            tset = {t.lower() for t in tags}
            items = [g for g in items if tset.issubset({t.lower() for t in (g.tags or [])})]
        if text:
            t = text.lower()
            items = [g for g in items if t in (g.title or "").lower() or t in (g.last_error or "").lower()]

        # sort
        rev = sort.startswith("-")
        key = sort[1:] if rev else sort
        def _key(g: Goal):
            v = getattr(g, key, None)
            if isinstance(v, datetime):
                return v.timestamp()
            if isinstance(v, Priority):
                return int(v)
            if isinstance(v, Status):
                return v.name
            return v
        try:
            items.sort(key=_key, reverse=rev)
        except Exception:
            pass

        if limit:
            items = items[: int(limit)]
        return items

    def iter_goals(self) -> Iterator[Goal]:
    # Take a snapshot under the lock, then yield outside to avoid deadlocks
        with self._lock:
            items = list(self._goals.values())
        for g in items:
            yield g

    def ready_goals(self) -> List[Goal]:
        """Goals that should be considered by a scheduler (READY/RUNNING/WAITING)."""
        with self._lock:
            return [g for g in self._goals.values() if g.status in {Status.READY, Status.RUNNING, Status.WAITING}]

    # ---------------- CRUD: steps ----------------

    def upsert_step(self, s: Step) -> Step:
        with self._lock:
            self._steps[s.id] = s
            lst = self._by_goal_steps.setdefault(s.goal_id, [])
            if s.id not in lst:
                lst.append(s.id)

            append_jsonl(self._state, [{"step": _jsonable(s)}])
            WAL.append(self._wal, {"type": "step_upsert", "step": _jsonable(s)})
            return s

    def get_step(self, step_id: str) -> Optional[Step]:
        with self._lock:
            return self._steps.get(step_id)

    def list_steps(
        self,
        *,
        goal_id: Optional[str] = None,
        statuses: Optional[Sequence[Status]] = None,
    ) -> List[Step]:
        with self._lock:
            if goal_id:
                ids = list(self._by_goal_steps.get(goal_id, []))
                items = [self._steps[i] for i in ids if i in self._steps]
            else:
                items = list(self._steps.values())

        if statuses:
            sset = set(statuses)
            items = [s for s in items if s.status in sset]
        return items

    def iter_steps(self) -> Iterator[Step]:
    # Same pattern for steps
        with self._lock:
            items = list(self._steps.values())
        for s in items:
            yield s

    def steps_for(self, goal_id: Optional[str], *, statuses: Optional[Sequence[Status]] = None) -> List[Step]:
        return self.list_steps(goal_id=goal_id, statuses=statuses)
