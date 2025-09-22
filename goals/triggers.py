# goals/triggers.py
# Time- and event-based triggers for goals (cron-ish schedules, time-of-day, predicates, and event hooks)

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:  # optional: nicer cron scheduling if installed
    from croniter import croniter  # type: ignore
except Exception:  # pragma: no cover
    croniter = None  # type: ignore

UTCNOW = lambda: datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Trigger state (kept by caller; we don't persist anything internally)
# ---------------------------------------------------------------------------

@dataclass
class TriggerState:
    last_fired_at: Optional[datetime] = None
    next_due_at: Optional[datetime] = None
    count: int = 0  # times fired


StateMap = Dict[str, TriggerState]  # e.g., ctx.setdefault("trigger_state", {})[goal.id] = StateMap


# ---------------------------------------------------------------------------
# Trigger base
# ---------------------------------------------------------------------------

class Trigger:
    """
    Abstract trigger interface.

    Subclasses must implement:
      - next_due(now, state) -> datetime|None
        Compute the next due datetime *strictly in the future* (>= now allowed) given current state.

      - describe() -> str
        Human-readable summary for logs/UI.

    The helper method:
      - due(now, state, *, set_next=True) -> bool
        Returns True if the trigger is due *at* now (within epsilon). Optionally updates state's next_due_at.
    """

    name: str = "trigger"

    def next_due(self, now: datetime, state: TriggerState) -> Optional[datetime]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def due(self, now: Optional[datetime], state: TriggerState, *, set_next: bool = True) -> bool:
        now = now or UTCNOW()
        nd = state.next_due_at or self.next_due(now, state)
        # Consider due if next_due_at is not set or is now/past (with tiny epsilon)
        if nd is None:
            # One-shot "now" semantics
            if set_next:
                state.next_due_at = None
            return True
        due = (nd - now).total_seconds() <= 0.001
        if due and set_next:
            # advance next occurrence to avoid immediate re-fire
            state.last_fired_at = now
            state.count += 1
            state.next_due_at = self.next_due(now + timedelta(milliseconds=1), state)
        elif set_next and state.next_due_at is None:
            state.next_due_at = nd
        return due


# ---------------------------------------------------------------------------
# Concrete triggers
# ---------------------------------------------------------------------------

class Every(Trigger):
    """
    Fires at a fixed interval.

    Args:
      seconds|minutes|hours: interval length (you can combine; we sum them)
      jitter_seconds: optional uniform jitter added to each period (+/- jitter/2)
      align: if True, align to wall-clock boundaries (e.g., minute or hour) when feasible
    """
    def __init__(self, *, seconds: int = 0, minutes: int = 0, hours: int = 0, jitter_seconds: float = 0.0, align: bool = False, name: str = "every") -> None:
        self.interval = timedelta(seconds=seconds + minutes * 60 + hours * 3600)
        if self.interval.total_seconds() <= 0:
            raise ValueError("Every(): interval must be > 0")
        self.jitter = float(jitter_seconds)
        self.align = bool(align)
        self.name = name

    def next_due(self, now: datetime, state: TriggerState) -> Optional[datetime]:
        base = state.last_fired_at or now
        if state.last_fired_at is None and self.align:
            # Align first fire to the next multiple of the interval from epoch
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            elapsed = (now - epoch).total_seconds()
            step = self.interval.total_seconds()
            next_multiple = math.floor(elapsed / step) * step + step
            nd = epoch + timedelta(seconds=next_multiple)
        else:
            nd = base + self.interval
        if self.jitter > 0:
            nd = nd + timedelta(seconds=(random.random() - 0.5) * self.jitter)
        return nd


class At(Trigger):
    """
    Fires at a specific time of day (UTC by default) on selected weekdays.

    Args:
      hour, minute, second: time of day (UTC) to fire
      weekdays: optional list of allowed weekdays (0=Mon..6=Sun); if omitted, fires every day
      tz_offset_minutes: integer minutes offset for local time (e.g., -300 for America/New_York EST);
                         we convert local time -> UTC for scheduling
    """
    def __init__(
        self,
        *,
        hour: int,
        minute: int = 0,
        second: int = 0,
        weekdays: Optional[List[int]] = None,
        tz_offset_minutes: int = 0,
        name: str = "at",
    ) -> None:
        for v, n, hi in [(hour, "hour", 23), (minute, "minute", 59), (second, "second", 59)]:
            if not (0 <= int(v) <= hi):
                raise ValueError(f"At(): {n} out of range")
        self.local_time = dtime(hour=int(hour), minute=int(minute), second=int(second))
        self.weekdays = [int(w) for w in weekdays] if weekdays is not None else None
        self.tz_offset = int(tz_offset_minutes)
        self.name = name

    def next_due(self, now: datetime, state: TriggerState) -> Optional[datetime]:
        # Convert 'now' into local clock
        local_now = now + timedelta(minutes=self.tz_offset)
        candidate = datetime.combine(local_now.date(), self.local_time, tzinfo=timezone.utc) - timedelta(minutes=self.tz_offset)

        if candidate <= now:
            candidate = candidate + timedelta(days=1)

        # If weekdays are restricted, move forward until it matches
        if self.weekdays is not None:
            while candidate.weekday() not in self.weekdays:
                candidate = candidate + timedelta(days=1)

        return candidate


class Cron(Trigger):
    """
    Standard cron expression trigger (requires croniter). If croniter is not installed, we degrade to Every(5m).
    """
    def __init__(self, expr: str, *, name: str = "cron") -> None:
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("Cron(): expr must be a non-empty string")
        self.expr = expr.strip()
        self.name = name
        self._degraded = croniter is None

    def next_due(self, now: datetime, state: TriggerState) -> Optional[datetime]:
        if croniter is None:
            # degrade gracefully
            return Every(minutes=5).next_due(now, state)
        itr = croniter(self.expr, now)
        return itr.get_next(datetime)  # type: ignore[arg-type]

    def describe(self) -> str:
        if self._degraded:
            return f"{self.name} (degraded: every 5m) {self.expr!r}"
        return f"{self.name} {self.expr!r}"


class When(Trigger):
    """
    Predicate trigger that fires when a callable returns True.
    Use with care; always provide min_interval to avoid hot loops.

    Args:
      predicate(ctx) -> bool
      min_interval_seconds: minimum time between firings
    """
    def __init__(self, predicate: Callable[[Dict[str, Any]], bool], *, min_interval_seconds: float = 5.0, name: str = "when") -> None:
        if not callable(predicate):
            raise ValueError("When(): predicate must be callable")
        self.predicate = predicate
        self.min_interval = float(min_interval_seconds)
        self.name = name

    def next_due(self, now: datetime, state: TriggerState) -> Optional[datetime]:
        # If we fired recently, enforce min interval; otherwise "now" if predicate passes else unknown
        if state.last_fired_at:
            wait = (state.last_fired_at + timedelta(seconds=self.min_interval))
            if wait > now:
                return wait
        # Not enforceable until predicate is true; return now to allow due() to check
        return now

    def due(self, now: Optional[datetime], state: TriggerState, *, set_next: bool = True, ctx: Optional[Dict[str, Any]] = None) -> bool:  # type: ignore[override]
        now = now or UTCNOW()
        # Enforce spacing first
        if state.last_fired_at and (now - state.last_fired_at).total_seconds() < self.min_interval:
            return False
        if self.predicate(ctx or {}):
            if set_next:
                state.last_fired_at = now
                state.count += 1
                state.next_due_at = self.next_due(now, state)
            return True
        # Not due; schedule a short revisit
        if set_next and (state.next_due_at is None or state.next_due_at < now + timedelta(seconds=1)):
            state.next_due_at = now + timedelta(seconds=1)
        return False


class Event(Trigger):
    """
    Fires when a named event is received via notify(event_name, payload).

    Args:
      name_match: exact event kind/name to react to (string or list of strings)
      cooldown_seconds: minimum spacing between successive fires
    """
    def __init__(self, name_match: Union[str, List[str]], *, cooldown_seconds: float = 0.0, name: str = "event") -> None:
        names = [name_match] if isinstance(name_match, str) else list(name_match)
        self.matches = {str(x).strip() for x in names if str(x).strip()}
        if not self.matches:
            raise ValueError("Event(): name_match must contain at least one non-empty name")
        self.cooldown = float(cooldown_seconds)
        self.name = name
        self._pending: int = 0  # count of queued events

    def next_due(self, now: datetime, state: TriggerState) -> Optional[datetime]:
        if self._pending > 0:
            if state.last_fired_at and self.cooldown > 0:
                nd = state.last_fired_at + timedelta(seconds=self.cooldown)
                return max(now, nd)
            return now
        # no pending events; unknown future
        return None

    def notify(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if event_name in self.matches:
            self._pending += 1

    def due(self, now: Optional[datetime], state: TriggerState, *, set_next: bool = True) -> bool:  # type: ignore[override]
        now = now or UTCNOW()
        if self._pending <= 0:
            return False
        # Respect cooldown
        if state.last_fired_at and self.cooldown > 0:
            if (now - state.last_fired_at).total_seconds() < self.cooldown:
                return False
        # consume one
        self._pending = max(0, self._pending - 1)
        if set_next:
            state.last_fired_at = now
            state.count += 1
            state.next_due_at = self.next_due(now, state)
        return True

    def describe(self) -> str:
        return f"{self.name} on {sorted(self.matches)} (cooldown {self.cooldown}s)"


# ---------------------------------------------------------------------------
# Helpers for (de)serializing specs (Goal.spec['triggers'])
# ---------------------------------------------------------------------------

def from_spec(spec: Dict[str, Any]) -> Trigger:
    """
    Build a Trigger from a JSON-like spec. Examples:

      {"type":"every","minutes":5}
      {"type":"at","hour":9,"minute":0,"weekdays":[0,1,2,3,4],"tz_offset_minutes":-300}
      {"type":"cron","expr":"*/15 * * * *"}
      {"type":"when","min_interval_seconds":10}
      {"type":"event","names":["GoalCreated","GoalFinished"],"cooldown_seconds":0}

    For "when", you must wire the predicate in code; we create a placeholder that never fires.
    """
    t = (spec.get("type") or "").strip().lower()
    if t == "every":
        return Every(
            seconds=int(spec.get("seconds", 0)),
            minutes=int(spec.get("minutes", 0)),
            hours=int(spec.get("hours", 0)),
            jitter_seconds=float(spec.get("jitter_seconds", 0.0)),
            align=bool(spec.get("align", False)),
            name=spec.get("name", "every"),
        )
    if t == "at":
        wd = spec.get("weekdays")
        return At(
            hour=int(spec.get("hour", 0)),
            minute=int(spec.get("minute", 0)),
            second=int(spec.get("second", 0)),
            weekdays=[int(x) for x in wd] if isinstance(wd, (list, tuple)) else None,
            tz_offset_minutes=int(spec.get("tz_offset_minutes", 0)),
            name=spec.get("name", "at"),
        )
    if t == "cron":
        return Cron(str(spec.get("expr", "*/5 * * * *")), name=spec.get("name", "cron"))
    if t == "event":
        names = spec.get("names") or spec.get("name") or []
        if isinstance(names, str):
            names = [names]
        return Event(list(names), cooldown_seconds=float(spec.get("cooldown_seconds", 0.0)), name=spec.get("name", "event"))
    if t == "when":
        # Placeholder false-predicate; caller should replace with a real one via wire_predicates()
        return When(lambda _ctx: False, min_interval_seconds=float(spec.get("min_interval_seconds", 5.0)), name=spec.get("name", "when"))
    # default: every 5 minutes
    return Every(minutes=5, name=spec.get("name", "every"))


def compile_triggers(specs: Optional[Iterable[Dict[str, Any]]]) -> List[Trigger]:
    """Compile a list of trigger specs into Trigger objects."""
    if not specs:
        return []
    return [from_spec(dict(s or {})) for s in specs]


def wire_predicates(triggers: Iterable[Trigger], *, when: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
    """
    Replace placeholder When() predicates with a concrete callable, if provided.
    Useful if your Goal.spec uses {"type":"when"} but you want to inject logic at runtime.
    """
    if when is None:
        return
    for t in triggers:
        if isinstance(t, When) and t.predicate is not when:
            t.predicate = when  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Evaluation helpers (to use from your daemon/watchdogs)
# ---------------------------------------------------------------------------

def evaluate(
    *,
    triggers: Iterable[Trigger],
    state: StateMap,
    now: Optional[datetime] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> List[Trigger]:
    """
    Check which triggers are due *now* and update state in-place.

    Returns a list of Trigger objects that fired.
    """
    now = now or UTCNOW()
    fired: List[Trigger] = []
    for idx, t in enumerate(triggers):
        key = f"t{idx}_{t.describe()}"
        ts = state.setdefault(key, TriggerState())
        if isinstance(t, When):
            if t.due(now, ts, set_next=True, ctx=(ctx or {})):
                fired.append(t)
        else:
            if t.due(now, ts, set_next=True):
                fired.append(t)
    return fired


def notify_events(triggers: Iterable[Trigger], event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """
    Push an external event into any Event triggers (so they can fire on next evaluate()).
    """
    for t in triggers:
        if isinstance(t, Event):
            t.notify(event_name, payload)


__all__ = [
    "Trigger",
    "Every",
    "At",
    "Cron",
    "When",
    "Event",
    "TriggerState",
    "StateMap",
    "from_spec",
    "compile_triggers",
    "wire_predicates",
    "evaluate",
    "notify_events",
]
