# goals/metrics.py
# Prometheus metrics for the Goals subsystem: counters/gauges/histograms + helpers to refresh/set/observe

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

try:
    from prometheus_client import Counter as _Counter, Gauge as _Gauge, Histogram as _Histogram
except Exception as _e:  # pragma: no cover
    _Counter = _Gauge = _Histogram = None  # type: ignore

from .model import Goal, Step, Status, Priority

UTCNOW = lambda: datetime.now(timezone.utc)

# ------------------------------
# Metric objects (initialized once via init_metrics)
# ------------------------------
_METRICS_INITIALIZED = False

goals_events_total: _Counter  # type: ignore
steps_events_total: _Counter  # type: ignore
goals_latency_seconds: _Histogram  # type: ignore
steps_exec_seconds: _Histogram  # type: ignore

goals_status_total: _Gauge  # type: ignore
goals_priority_total: _Gauge  # type: ignore
goals_kind_total: _Gauge  # type: ignore
goals_overdue_total: _Gauge  # type: ignore

steps_status_total: _Gauge  # type: ignore

goals_queue_depth: _Gauge  # type: ignore
goals_workers_active: _Gauge  # type: ignore


def init_metrics() -> None:
    """
    Create Prometheus metric objects (idempotent). Call once during app bootstrap.
    """
    global _METRICS_INITIALIZED
    if _METRICS_INITIALIZED:  # already done
        return
    if _Counter is None or _Gauge is None or _Histogram is None:  # pragma: no cover
        raise RuntimeError("prometheus_client not available")

    global goals_events_total, steps_events_total
    global goals_latency_seconds, steps_exec_seconds
    global goals_status_total, goals_priority_total, goals_kind_total, goals_overdue_total
    global steps_status_total
    global goals_queue_depth, goals_workers_active

    # Event counters
    goals_events_total = _Counter(
        "goals_events_total",
        "Total goal lifecycle events",
        ["event", "kind"],
    )
    steps_events_total = _Counter(
        "goals_steps_events_total",
        "Total step lifecycle events",
        ["event"],
    )

    # Duration histograms
    goals_latency_seconds = _Histogram(
        "goals_latency_seconds",
        "Latency from goal creation to terminal state",
        ["kind", "outcome"],
        # buckets tuned for human/task scale
        buckets=(1, 2, 5, 10, 30, 60, 120, 300, 900, 1800, 3600, float("inf")),
    )
    steps_exec_seconds = _Histogram(
        "goals_step_exec_seconds",
        "Execution time for individual steps (when available)",
        ["kind", "result"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300, float("inf")),
    )

    # Snapshot gauges (set via refresh_from_store())
    goals_status_total = _Gauge(
        "goals_status_total",
        "Current count of goals by status",
        ["status"],
    )
    goals_priority_total = _Gauge(
        "goals_priority_total",
        "Current count of goals by priority",
        ["priority"],
    )
    goals_kind_total = _Gauge(
        "goals_kind_total",
        "Current count of goals by kind",
        ["kind"],
    )
    goals_overdue_total = _Gauge(
        "goals_overdue_total",
        "Count of non-terminal goals past their deadline",
    )

    steps_status_total = _Gauge(
        "goals_steps_status_total",
        "Current count of steps by status",
        ["status"],
    )

    # Queue/worker gauges (set by daemon/runner)
    goals_queue_depth = _Gauge(
        "goals_queue_depth",
        "Scheduler queue size (ready steps enqueued to runner)",
    )
    goals_workers_active = _Gauge(
        "goals_workers_active",
        "Number of active step worker threads",
    )

    _METRICS_INITIALIZED = True


# Initialize on import (safe & idempotent)
init_metrics()


# ------------------------------
# Public helpers (safe to call anywhere)
# ------------------------------

def observe_goal_event(event: Dict[str, Any]) -> None:
    """
    Increment event counters and, for terminal events with a Goal object, record latency.

    Accepts either:
      - dicts emitted by GoalsDaemon/API (keys like 'kind','goal_kind','goal_id','status',...)
      - or a richer payload including {"goal": Goal} for latency measurement
    """
    try:
        kind = str(event.get("kind") or "Unknown")
        gkind = str(event.get("goal_kind") or "unknown")
        goals_events_total.labels(event=kind, kind=gkind).inc()
        # Terminal outcomes latency (requires Goal object to read created_at)
        if kind in {"GoalFinished", "GoalFailed", "GoalCancelled"}:
            goal_obj = event.get("goal")
            outcome = (
                "finished" if kind == "GoalFinished"
                else "failed" if kind == "GoalFailed"
                else "cancelled"
            )
            if isinstance(goal_obj, Goal):
                try:
                    created = getattr(goal_obj, "created_at", None)
                    finished_at = getattr(goal_obj, "updated_at", None) or UTCNOW()
                    if created:
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        if finished_at.tzinfo is None:
                            finished_at = finished_at.replace(tzinfo=timezone.utc)
                        dur = max(0.0, (finished_at - created).total_seconds())
                        goals_latency_seconds.labels(kind=goal_obj.kind, outcome=outcome).observe(dur)
                except Exception:
                    pass
    except Exception:
        # Never throw from metrics
        pass


def observe_step_event(event: Dict[str, Any]) -> None:
    """
    Increment step event counters and (if provided) observe execution duration.
    Expected keys:
      - 'kind' (event name): StepStarted|StepFinished|StepFailed
      - 'extra': may contain 'duration_sec' (float)
      - 'goal_kind' (optional): handler kind for histogram labeling
    """
    try:
        ekind = str(event.get("kind") or "Unknown")
        steps_events_total.labels(event=ekind).inc()

        if ekind in {"StepFinished", "StepFailed"}:
            # Prefer explicit duration if provided by runner
            dur = None
            extra = event.get("extra") or {}
            try:
                dur = float(extra.get("duration_sec"))
            except Exception:
                dur = None
            gkind = str(event.get("goal_kind") or "unknown")
            result = "ok" if ekind == "StepFinished" else "error"
            if dur is not None and dur >= 0:
                steps_exec_seconds.labels(kind=gkind, result=result).observe(dur)
    except Exception:
        pass


def update_queue(depth: int, workers_active: int) -> None:
    """
    Set queue depth and worker count gauges.
    """
    try:
        goals_queue_depth.set(max(0, int(depth)))
        goals_workers_active.set(max(0, int(workers_active)))
    except Exception:
        pass


def refresh_from_store(store: Any, *, now: Optional[datetime] = None) -> None:
    """
    Recompute snapshot gauges from the store. Safe to call periodically.
    Duck-typed store:
      - iter_goals() | list_goals() | all()
      - iter_steps() | list_steps()
    """
    now = now or UTCNOW()

    goals = list(_iter_goals(store))
    steps = list(_iter_steps(store))

    # --- goals by status/priority/kind
    by_status = Counter(_status_name(g.status) for g in goals)
    by_priority = Counter(_priority_name(getattr(g, "priority", Priority.NORMAL)) for g in goals)
    by_kind = Counter((g.kind or "unknown") for g in goals)

    # set gauges
    _set_labeled_gauge(goals_status_total, "status", by_status)
    _set_labeled_gauge(goals_priority_total, "priority", by_priority)
    _set_labeled_gauge(goals_kind_total, "kind", by_kind)

    # overdue (non-terminal with past deadline)
    terminal = {"DONE", "FAILED", "CANCELLED"}
    overdue = 0
    for g in goals:
        dl = getattr(g, "deadline", None)
        if not dl:
            continue
        try:
            if _status_name(g.status) not in terminal and (now - dl).total_seconds() > 0:
                overdue += 1
        except Exception:
            continue
    try:
        goals_overdue_total.set(overdue)
    except Exception:
        pass

    # --- steps by status
    by_step_status = Counter(_status_name(s.status) for s in steps) if steps else Counter()
    _set_labeled_gauge(steps_status_total, "status", by_step_status)


# ------------------------------
# Internals (duck-typing helpers)
# ------------------------------

def _iter_goals(store: Any) -> Iterable[Goal]:
    if hasattr(store, "iter_goals"):
        return store.iter_goals()
    if hasattr(store, "list_goals"):
        return store.list_goals()
    if hasattr(store, "all"):
        return store.all()
    return []  # graceful fallback


def _iter_steps(store: Any) -> Iterable[Step]:
    if hasattr(store, "iter_steps"):
        return store.iter_steps()
    if hasattr(store, "list_steps"):
        return store.list_steps()
    return []


def _status_name(x: Any) -> str:
    try:
        return x.name  # Enum
    except Exception:
        return str(x)


def _priority_name(x: Any) -> str:
    try:
        return x.name  # Enum
    except Exception:
        return str(x)


def _set_labeled_gauge(gauge: _Gauge, label: str, counts: Counter) -> None:  # type: ignore
    try:
        seen = set()
        for k, v in counts.items():
            gauge.labels(**{label: str(k)}).set(float(v))
            seen.add(str(k))
        # Optionally, we could zero-out old labels, but Prometheus client doesn't list them directly.
        # Keeping it simple to avoid label cardinality churn.
    except Exception:
        pass


__all__ = [
    "init_metrics",
    "observe_goal_event",
    "observe_step_event",
    "update_queue",
    "refresh_from_store",
    # Gauges (useful for daemon)
    "goals_queue_depth",
    "goals_workers_active",
]
