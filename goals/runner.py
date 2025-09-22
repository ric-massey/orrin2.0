# goals/runner.py
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .model import Goal, Step, Status, Priority
from .handlers.base import GoalHandler, HandlerContext
from . import metrics as metrics_mod

UTCNOW = lambda: datetime.now(timezone.utc)

def _dbg_enabled() -> bool:
    try:
        return os.getenv("GOALS_DEBUG", "0") not in ("0", "", "false", "False")
    except Exception:
        return False

def _dbg(*a):
    if _dbg_enabled():
        try:
            print("[runner]", *a, flush=True)
        except Exception:
            pass

# ---------- minimal duck-typed store helpers ----------

def _iter_goals(store: Any) -> Iterable[Goal]:
    if hasattr(store, "iter_goals"):
        return store.iter_goals()
    if hasattr(store, "list_goals"):
        return store.list_goals()
    if hasattr(store, "all"):
        return store.all()
    return []  # graceful fallback


def _get_goal(store: Any, goal_id: str) -> Optional[Goal]:
    if hasattr(store, "get_goal"):
        return store.get_goal(goal_id)
    for g in _iter_goals(store):
        if g.id == goal_id:
            return g
    return None


def _upsert_goal(store: Any, goal: Goal) -> None:
    if hasattr(store, "upsert_goal"):
        store.upsert_goal(goal); return
    if hasattr(store, "save_goal"):
        store.save_goal(goal); return
    if hasattr(store, "update_goal"):
        store.update_goal(goal); return


def _upsert_step(store: Any, step: Step) -> None:
    if hasattr(store, "upsert_step"):
        store.upsert_step(step); return
    if hasattr(store, "save_step"):
        store.save_step(step); return
    if hasattr(store, "update_step"):
        store.update_step(step); return


def _list_steps(store: Any, goal_id: Optional[str] = None) -> List[Step]:
    if hasattr(store, "steps_for"):
        return store.steps_for(goal_id)
    if hasattr(store, "iter_steps"):
        out: List[Step] = []
        for s in store.iter_steps():
            if goal_id and s.goal_id != goal_id:
                continue
            out.append(s)
        return out
    if hasattr(store, "list_steps"):
        return store.list_steps(goal_id=goal_id)
    return []


# ---------- StepRunner ----------

class StepRunner:
    def __init__(
        self,
        *,
        store: Any,
        registry: Any,
        step_queue: "queue.Queue[Step]",
        workers: int = 3,
        ctx: Optional[HandlerContext] = None,
        reaper_sink: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.q: "queue.Queue[Step]" = step_queue
        self.workers = max(0, int(workers))
        self.ctx: HandlerContext = dict(ctx or {})
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._active_mu = threading.Lock()
        self._active_count = 0
        self._reaper_sink = reaper_sink

    # ----- lifecycle -----

    def start(self) -> None:
        _dbg("start workers:", self.workers)
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"GoalsStepWorker-{i+1}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        _dbg("stop called")
        self._stop.set()
        for _ in self._threads:
            try:
                self.q.put_nowait(None)  # type: ignore[arg-type]
            except Exception:
                break

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    # Public enqueue for schedulers/daemons
    def submit(self, step: Step) -> None:
        _dbg("submit step:", step.id, "goal:", step.goal_id)
        try:
            self.q.put_nowait(step)
        except Exception:
            self.q.put(step)

    # ----- metrics/introspection -----

    @property
    def active_workers(self) -> int:
        with self._active_mu:
            return self._active_count

    def _inc_active(self) -> None:
        with self._active_mu:
            self._active_count += 1
            self._set_worker_metrics()

    def _dec_active(self) -> None:
        with self._active_mu:
            self._active_count = max(0, self._active_count - 1)
            self._set_worker_metrics()

    def capacity_left(self) -> int:
        cap = max(0, self.workers - self.active_workers)
        _dbg("capacity_left:", cap)
        return cap

    def queue_size(self) -> int:
        try:
            return self.q.qsize()
        except Exception:
            return 0

    def _set_worker_metrics(self) -> None:
        try:
            metrics_mod.update_queue(self.queue_size(), self.active_workers)
        except Exception:
            pass

    # ----- core worker loop -----

    def _worker(self) -> None:
        _dbg("worker started")
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:  # poison
                _dbg("worker got poison pill")
                continue

            step: Step = item
            _dbg("dequeued step:", step.id, "goal:", step.goal_id)
            self._inc_active()
            try:
                self._execute_step(step)
            except Exception as e:
                self._emit({"kind": "StepRunnerError", "error": f"{type(e).__name__}: {e}", "ts": UTCNOW().isoformat()})
                _dbg("execute_step error:", e)
            finally:
                self._dec_active()
                self.q.task_done()
                self._set_worker_metrics()

    # ----- execution -----

    def _execute_step(self, step: Step) -> None:
        goal = _get_goal(self.store, step.goal_id)
        if goal is None:
            _dbg("goal missing for step:", step.id)
            step.status = Status.FAILED
            step.last_error = "goal missing"
            step.finished_at = UTCNOW()
            _upsert_step(self.store, step)
            self._emit_step_event("StepFailed", step, goal_kind="unknown", extra={"reason": "goal_missing"})
            return

        handler = self._get_handler(goal)
        if handler is None:
            _dbg("no handler for goal kind:", goal.kind)
            step.status = Status.FAILED
            step.last_error = f"no handler for kind '{goal.kind}'"
            step.finished_at = UTCNOW()
            _upsert_step(self.store, step)
            self._emit_step_event("StepFailed", step, goal_kind=goal.kind, extra={"reason": "no_handler"})
            ng = replace(goal, status=Status.FAILED, updated_at=UTCNOW(), last_error=step.last_error)
            _upsert_goal(self.store, ng)
            self._emit_goal_event("GoalFailed", ng, extra={"reason": "no_handler"})
            return

        started_emitted = False
        t0 = time.perf_counter()

        while not self._stop.is_set():
            try:
                new = handler.tick(goal, step, self._handler_ctx(goal))
                if new is not None:
                    step = new
                _dbg("tick ->", step.status.name, "step:", step.id)
            except Exception as e:
                step.last_error = f"{type(e).__name__}: {e}"
                step.attempts = int(step.attempts or 0) + 1
                _dbg("tick raised:", step.last_error, "attempts:", step.attempts, "/", step.max_attempts)
                if step.attempts >= step.max_attempts:
                    step.status = Status.FAILED
                    step.finished_at = UTCNOW()
                else:
                    step.status = Status.READY
                    step.started_at = None

            if not started_emitted and step.started_at is not None:
                self._emit_step_event("StepStarted", step, goal_kind=goal.kind)
                started_emitted = True

            _upsert_step(self.store, step)

            if step.status in {Status.DONE, Status.FAILED, Status.CANCELLED}:
                extra = {"duration_sec": max(0.0, time.perf_counter() - t0)}
                if step.status == Status.DONE:
                    self._emit_step_event("StepFinished", step, goal_kind=goal.kind, extra=extra)
                    _dbg("finished step:", step.id)
                elif step.status == Status.FAILED:
                    self._emit_step_event("StepFailed", step, goal_kind=goal.kind, extra=extra)
                    _dbg("failed step:", step.id)
                self._maybe_finalize_goal(goal, step)
                return

            if step.status in {Status.READY, Status.WAITING, Status.BLOCKED, Status.PAUSED}:
                _dbg("deferring step:", step.id, "status:", step.status.name)
                return

            time.sleep(0.02)

    # ----- helpers -----

    def _maybe_finalize_goal(self, goal: Goal, last_step: Step) -> None:
        steps = _list_steps(self.store, goal_id=goal.id)
        if last_step.status == Status.FAILED and (last_step.attempts or 0) >= (last_step.max_attempts or 0):
            ng = replace(goal, status=Status.FAILED, updated_at=UTCNOW(), last_error=last_step.last_error)
            _upsert_goal(self.store, ng)
            self._emit_goal_event("GoalFailed", ng, extra={"step_id": last_step.id})
            _dbg("finalized goal FAILED:", goal.id)
            return
        if steps and all(s.status in {Status.DONE, Status.CANCELLED} for s in steps):
            ng = replace(goal, status=Status.DONE, updated_at=UTCNOW())
            _upsert_goal(self.store, ng)
            self._emit_goal_event("GoalFinished", ng, extra={"steps_total": len(steps)})
            _dbg("finalized goal DONE:", goal.id)

    def _get_handler(self, goal: Goal) -> Optional[GoalHandler]:
        reg = self.registry
        if reg is None:
            return None
        for meth in ("get", "get_handler", "handler_for", "resolve", "lookup"):
            fn = getattr(reg, meth, None)
            if callable(fn):
                try:
                    h = fn(goal.kind)
                    if h:
                        return h
                except Exception:
                    pass
        for attr in ("by_kind", "handlers", "registry", "_by_kind", "_handlers"):
            m = getattr(reg, attr, None)
            if isinstance(m, dict):
                h = m.get(goal.kind)
                if h:
                    return h
        if isinstance(reg, dict):
            return reg.get(goal.kind)  # type: ignore[index]
        return None

    def _handler_ctx(self, goal: Goal) -> HandlerContext:
        ctx = dict(self.ctx)
        ctx["goal"] = goal
        return ctx

    # ----- event & metrics emitters -----

    def _emit(self, event: Dict[str, Any]) -> None:
        sink = self._reaper_sink
        if callable(sink):
            try:
                sink(event)
            except Exception:
                pass

    def _emit_step_event(self, kind: str, step: Step, *, goal_kind: str, extra: Optional[Dict[str, Any]] = None) -> None:
        evt = {
            "kind": kind,
            "ts": UTCNOW().isoformat(),
            "step_id": step.id,
            "goal_id": step.goal_id,
            "goal_kind": goal_kind,
            "name": step.name,
            "status": getattr(step.status, "name", str(step.status)),
            "attempts": int(step.attempts or 0),
            "max_attempts": int(step.max_attempts or 0),
            "extra": dict(extra or {}),
        }
        self._emit(evt)
        try:
            metrics_mod.observe_step_event(evt)
        except Exception:
            pass

    def _emit_goal_event(self, kind: str, goal: Goal, *, extra: Optional[Dict[str, Any]] = None) -> None:
        evt = {
            "kind": kind,
            "ts": UTCNOW().isoformat(),
            "goal_id": goal.id,
            "goal_kind": goal.kind,
            "status": getattr(goal.status, "name", str(goal.status)),
            "priority": getattr(goal.priority, "name", str(goal.priority)),
            "title": goal.title,
            "deadline": goal.deadline.isoformat() if goal.deadline else None,
            "extra": dict(extra or {}),
            "goal": goal,  # included for latency histogram when terminal
        }
        self._emit(evt)
        try:
            metrics_mod.observe_goal_event(evt)
        except Exception:
            pass


__all__ = ["StepRunner"]
