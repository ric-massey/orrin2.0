# goals/goals_daemon.py
# Orchestrates the Goals subsystem: planning NEW goals, scheduling READY steps, and running them via a worker pool

from __future__ import annotations

import threading
import queue
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import Goal, Step, Status, Priority
from .handlers.base import GoalHandler, HandlerContext
from . import policy as policy_mod  # expected to provide choose_next_steps(...)
from . import runner as runner_mod  # expected to provide StepRunner

UTCNOW = lambda: datetime.now(timezone.utc)


# ---------------- duck-typed store helpers ----------------

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _iter_goals(store: Any) -> Iterable[Goal]:
    if hasattr(store, "iter_goals"):
        return store.iter_goals()
    if hasattr(store, "list_goals"):
        return store.list_goals()
    if hasattr(store, "all"):
        return store.all()
    raise AttributeError("GoalsDaemon: store must expose iter_goals/list_goals/all()")


def _get_goal(store: Any, goal_id: str) -> Optional[Goal]:
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
    raise AttributeError("GoalsDaemon: store must expose upsert_goal/save_goal/update_goal()")


def _add_steps(store: Any, steps: List[Step]) -> None:
    if not steps:
        return
    if hasattr(store, "add_steps"):
        store.add_steps(steps); return
    if hasattr(store, "upsert_step"):
        for s in steps:
            store.upsert_step(s)
        return
    if hasattr(store, "save_step"):
        for s in steps:
            store.save_step(s)
        return
    raise AttributeError("GoalsDaemon: store must support add_steps/upsert_step/save_step")


def _list_steps(store: Any, goal_id: Optional[str] = None, statuses: Optional[Iterable[Status]] = None) -> List[Step]:
    # Duck-typed set of store APIs
    if hasattr(store, "steps_for"):
        return store.steps_for(goal_id, statuses=statuses)  # type: ignore[arg-type]
    if hasattr(store, "iter_steps"):
        out: List[Step] = []
        for s in store.iter_steps():
            if goal_id and s.goal_id != goal_id:
                continue
            if statuses and s.status not in set(statuses):
                continue
            out.append(s)
        return out
    if hasattr(store, "list_steps"):
        return store.list_steps(goal_id=goal_id, statuses=statuses)
    # If no step API is available yet, return empty to let you wire it later.
    return []


def _deps_satisfied(step: Step, store: Any) -> bool:
    if not step.deps:
        return True
    # If store exposes get_step, we can check real statuses; otherwise assume satisfied.
    if hasattr(store, "get_step"):
        for dep_id in step.deps:
            dep = store.get_step(dep_id)
            if not dep or dep.status != Status.DONE:
                return False
        return True
    return True


# ---------------- GoalsDaemon ----------------

class GoalsDaemon:
    """
    Runs in parallel to Orrin's main loop:
      - plans NEW goals by calling the appropriate handler.plan()
      - schedules READY steps using policy.choose_next_steps()
      - executes steps on a worker pool via runner.StepRunner
    """

    def __init__(
        self,
        store: Any,
        registry: Any,
        *,
        workers: int = 3,
        tick_seconds: float = 0.5,
        ctx: Optional[HandlerContext] = None,
        reaper_sink: Optional[Any] = None,
        memory_writer: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.tick_seconds = tick_seconds
        self.ctx: HandlerContext = dict(ctx or {})
        self.reaper_sink = reaper_sink
        self.memory_writer = memory_writer

        # Internal coordination
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Step queue + runner
        self._step_q: "queue.Queue[Step]" = queue.Queue()
        self._runner = runner_mod.StepRunner(
            store=self.store,
            registry=self.registry,
            step_queue=self._step_q,
            workers=workers,
            ctx=self.ctx,
            reaper_sink=self._emit_event,
        )

        # Health
        self._last_pulse_at: Optional[datetime] = None
        self._last_error: Optional[str] = None

    # ---------------- Lifecycle ----------------

    def start(self) -> None:
        """Start scheduler and worker pool."""
        self._runner.start()
        self._thread = threading.Thread(target=self._loop, name="GoalsDaemon", daemon=True)
        self._thread.start()
        self._emit_event({"kind": "GoalsDaemonStarted", "ts": UTCNOW().isoformat()})

    def stop(self) -> None:
        """Signal shutdown (graceful)."""
        self._stop.set()
        self._wake.set()
        try:
            self._runner.stop()
        except Exception:
            pass
        self._emit_event({"kind": "GoalsDaemonStopping", "ts": UTCNOW().isoformat()})

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for scheduler (and runner) to stop."""
        if self._thread:
            self._thread.join(timeout=timeout)
        try:
            self._runner.join(timeout=timeout)
        except Exception:
            pass

    def submit(self, goal_id: str) -> None:
        """
        External nudge: mark a goal for prompt consideration.
        We simply wake the scheduler; the next tick will re-evaluate.
        """
        self._wake.set()

    # ---------------- Introspection ----------------

    def health(self) -> Dict[str, Any]:
        return {
            "last_pulse_at": self._last_pulse_at.isoformat() if self._last_pulse_at else None,
            "last_error": self._last_error,
            "queue_size": self._step_q.qsize(),
            "workers_active": _safe_getattr(self._runner, "active_workers", 0),
        }

    # ---------------- Internal loop ----------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._pulse()
                self._last_error = None
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                self._emit_event({"kind": "GoalsDaemonError", "ts": UTCNOW().isoformat(), "error": self._last_error})
            finally:
                self._last_pulse_at = UTCNOW()

            # Sleep or wake early if submit() was called
            self._wake.wait(timeout=self.tick_seconds)
            self._wake.clear()

        # StepRunner.stop() handles worker shutdown.

    # One scheduling pulse
    def _pulse(self) -> None:
        # 1) Plan NEW goals
        _ = self._plan_new_goals()

        # 2) Gather candidate steps from READY goals
        candidates: List[Tuple[Goal, Step]] = self._gather_ready_steps()

        # 3) Choose which steps to run now (robust to empty/None policy output)
        steps_to_run: List[Step] = []
        cap: Optional[int] = None
        try:
            if hasattr(self._runner, "capacity_left"):
                cap = int(self._runner.capacity_left())
        except Exception:
            cap = None

        # Effective capacity: if unknown or 0/negative, allow at least one step when we have candidates
        if isinstance(cap, int):
            eff_cap = max(0, cap)
        else:
            eff_cap = len(candidates)
        if eff_cap == 0 and candidates:
            eff_cap = 1

        choose = getattr(policy_mod, "choose_next_steps", None)
        used_policy = False
        if callable(choose) and candidates and eff_cap > 0:
            try:
                chosen = choose(
                    candidates=candidates,
                    store=self.store,
                    ctx=self.ctx,
                    capacity=eff_cap,
                )
                # Accept only sane, *non-empty* outputs; else fallback
                if isinstance(chosen, list) and len(chosen) > 0:
                    if isinstance(chosen[0], tuple):
                        steps_to_run = [s for (_g, s) in chosen][:eff_cap]
                    else:
                        steps_to_run = [s for s in chosen if isinstance(s, Step)][:eff_cap]
                    if steps_to_run:
                        used_policy = True
            except Exception:
                used_policy = False  # fallback below

        if not used_policy:
            # Fallback: FIFO up to effective capacity
            steps_to_run = [s for (_g, s) in candidates[:eff_cap]]

        # 4) Enqueue selected steps
        for step in steps_to_run:
            self._enqueue_step(step)

        # 5) Finalize goals whose steps reached terminal states
        self._finalize_goals()

    def _plan_new_goals(self) -> bool:
        """Find NEW goals, call handler.plan(), persist steps, and mark them READY."""
        planned_any = False
        for g in list(_iter_goals(self.store)):
            if g.status != Status.NEW:
                continue
            handler = self._get_handler(g)
            if handler is None:
                # No handler; mark FAILED to avoid tight loop
                ng = replace(g, status=Status.FAILED, last_error=f"no handler for kind '{g.kind}'", updated_at=UTCNOW())
                _upsert_goal(self.store, ng)
                self._emit_goal_event("GoalFailed", ng, extra={"reason": "no_handler"})
                continue

            try:
                steps = handler.plan(g, self._handler_ctx(g))
            except Exception as e:
                ng = replace(g, status=Status.FAILED, last_error=f"plan error: {type(e).__name__}: {e}", updated_at=UTCNOW())
                _upsert_goal(self.store, ng)
                self._emit_goal_event("GoalFailed", ng, extra={"reason": "plan_error"})
                continue

            try:
                _add_steps(self.store, steps or [])
                ng = replace(g, status=Status.READY, updated_at=UTCNOW())
                _upsert_goal(self.store, ng)
                self._emit_goal_event("GoalPlanned", ng, extra={"steps": len(steps or [])})
                planned_any = True
            except Exception as e:
                ng = replace(g, status=Status.FAILED, last_error=f"persist steps error: {type(e).__name__}: {e}", updated_at=UTCNOW())
                _upsert_goal(self.store, ng)
                self._emit_goal_event("GoalFailed", ng, extra={"reason": "persist_steps_error"})
                continue
        return planned_any

    def _gather_ready_steps(self) -> List[Tuple[Goal, Step]]:
        """Collect (goal, step) pairs that are READY and dependency-satisfied."""
        out: List[Tuple[Goal, Step]] = []
        for g in _iter_goals(self.store):
            if g.status not in {Status.READY, Status.RUNNING, Status.WAITING, Status.BLOCKED}:
                continue

            # Goal-level block check
            handler = self._get_handler(g)
            if handler:
                blocked, reason = False, None
                try:
                    # Supply pending steps to handler via ctx hint
                    pending = _list_steps(self.store, goal_id=g.id, statuses=[Status.READY, Status.RUNNING, Status.WAITING])
                    self.ctx["pending_steps"] = pending
                    blocked, reason = handler.is_blocked(g, self._handler_ctx(g))
                except Exception as e:
                    blocked, reason = True, f"is_blocked error: {type(e).__name__}: {e}"
                if blocked:
                    if g.status != Status.BLOCKED:
                        ng = replace(g, status=Status.BLOCKED, updated_at=UTCNOW(), last_error=reason)
                        _upsert_goal(self.store, ng)
                        self._emit_goal_event("GoalBlocked", ng, extra={"reason": reason})
                    continue
                else:
                    if g.status == Status.BLOCKED:
                        ng = replace(g, status=Status.READY, updated_at=UTCNOW(), last_error=None)
                        _upsert_goal(self.store, ng)
                        self._emit_goal_event("GoalUnblocked", ng, extra={})

            # Step-level candidates
            ready_steps = _list_steps(self.store, goal_id=g.id, statuses=[Status.READY])
            for s in ready_steps:
                if _deps_satisfied(s, self.store):
                    out.append((g, s))

        return out

    def _enqueue_step(self, step: Step) -> None:
        """
        Submit a step to the runner. Prefer a direct submit() method if available;
        otherwise put into the shared queue for runners that poll a queue.
        """
        submit = _safe_getattr(self._runner, "submit", None)
        if callable(submit):
            try:
                submit(step)
                return
            except Exception:
                pass
        # Fallback: use the queue
        try:
            self._step_q.put_nowait(step)
        except queue.Full:
            self._step_q.put(step)

    def _finalize_goals(self) -> None:
        """
        Flip goals to DONE or FAILED when all their steps are terminal.
        This keeps progress moving even if the runner doesn't directly update goals.
        """
        for g in list(_iter_goals(self.store)):
            if g.status in {Status.DONE, Status.FAILED, Status.CANCELLED}:
                continue
            steps = _list_steps(self.store, goal_id=g.id)
            if not steps:
                # No steps → if READY/RUNNING/WATING and handler had nothing, mark DONE (noop)
                if g.status in {Status.READY, Status.RUNNING, Status.WAITING}:
                    ng = replace(g, status=Status.DONE, updated_at=UTCNOW())
                    _upsert_goal(self.store, ng)
                    self._emit_goal_event("GoalFinished", ng, extra={"reason": "no_steps"})
                continue

            any_running = any(s.status == Status.RUNNING for s in steps)
            any_ready = any(s.status == Status.READY for s in steps)
            any_waiting = any(s.status == Status.WAITING for s in steps)
            all_done = steps and all(s.status == Status.DONE for s in steps)
            any_failed = any(s.status == Status.FAILED for s in steps)

            if all_done:
                if g.status != Status.DONE:
                    ng = replace(g, status=Status.DONE, updated_at=UTCNOW())
                    _upsert_goal(self.store, ng)
                    self._emit_goal_event("GoalFinished", ng, extra={"reason": "all_steps_done"})
            elif any_failed and not (any_ready or any_running or any_waiting):
                # No more work pending and at least one failed → mark goal failed
                if g.status != Status.FAILED:
                    ng = replace(g, status=Status.FAILED, updated_at=UTCNOW(), last_error="step_failure")
                    _upsert_goal(self.store, ng)
                    self._emit_goal_event("GoalFailed", ng, extra={"reason": "step_failed"})

    # ---------------- Helpers ----------------

    def _get_handler(self, goal: Goal) -> Optional[GoalHandler]:
        """
        Be liberal in what we accept: try multiple common registry APIs,
        then peek at typical dict attributes, then dict-like indexing.
        """
        reg = self.registry
        if reg is None:
            return None

        # Try common method names
        for meth in ("get", "get_handler", "handler_for", "resolve", "lookup"):
            fn = getattr(reg, meth, None)
            if callable(fn):
                try:
                    h = fn(goal.kind)
                    if h:
                        return h
                except Exception:
                    pass

        # Peek at common mapping attributes
        for attr in ("by_kind", "handlers", "registry", "_by_kind", "_handlers"):
            m = getattr(reg, attr, None)
            if isinstance(m, dict) and goal.kind in m:
                try:
                    return m[goal.kind]
                except Exception:
                    pass

        # Dict-like registry
        if isinstance(reg, dict):
            return reg.get(goal.kind)  # type: ignore[index]

        return None

    def _handler_ctx(self, goal: Goal) -> HandlerContext:
        # Provide a shallow per-goal view that includes goal reference and common services
        ctx = dict(self.ctx)
        ctx["goal"] = goal
        return ctx

    def _emit_event(self, event: Dict[str, Any]) -> None:
        sink = self.reaper_sink
        if callable(sink):
            try:
                sink(event)
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
        }
        self._emit_event(evt)
        # Optionally mirror into memory
        mw = self.memory_writer
        if callable(mw):
            try:
                text = f"{kind}: {goal.title} [{goal.kind}/{getattr(goal.priority,'name',goal.priority)}]"
                meta = {"goal_id": goal.id, "kind": goal.kind, "status": getattr(goal.status, "name", str(goal.status))}
                mw("goal_event", text, meta)
            except Exception:
                pass


__all__ = ["GoalsDaemon"]
