# goals/handlers/housekeeping.py
# Concrete handler for recurring maintenance/cleanup tasks; snapshots, WAL pruning, logs, lint/tests

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..model import Goal, Step, Status
from .base import BaseGoalHandler, HandlerContext

UTCNOW = lambda: datetime.now(timezone.utc)


# ---------- small helpers ----------

def _new_step(goal_id: str, name: str, action: Dict[str, Any], *, max_attempts: int = 2, deps: Optional[List[str]] = None) -> Step:
    return Step(
        id=f"s_{uuid.uuid4().hex[:10]}",
        goal_id=goal_id,
        name=name,
        action=action,
        max_attempts=max_attempts,
        deps=list(deps or []),
        status=Status.READY,
    )


def _artifacts_dir(ctx: HandlerContext, goal: Goal) -> Path:
    base = Path(ctx.get("artifacts_dir") or "data/goals/artifacts").resolve()
    d = base / goal.id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_artifact(ctx: HandlerContext, goal: Goal, name: str, content: str) -> str:
    p = _artifacts_dir(ctx, goal) / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _ctx_path(ctx: HandlerContext, key: str, default: str) -> Path:
    # Try multiple common placements for paths
    paths = ctx.get("paths")
    if paths and hasattr(paths, key):
        return Path(getattr(paths, key)).resolve()
    mapping = ctx.get("PATHS") or {}
    if key in mapping:
        return Path(mapping[key]).resolve()
    if key in ctx:
        return Path(ctx[key]).resolve()
    return Path(default).resolve()


def _acquire(ctx: HandlerContext, name: str, goal_id: str) -> bool:
    locks = ctx.get("locks")
    if not locks:
        return True
    try:
        return locks.acquire(name, goal_id)
    except Exception:
        return False


def _release(ctx: HandlerContext, name: str, goal_id: str) -> None:
    locks = ctx.get("locks")
    if not locks:
        return
    try:
        locks.release(name, goal_id)
    except Exception:
        pass


def _which(bin_name: str) -> Optional[str]:
    return shutil.which(bin_name)


def _run(cmd: List[str], *, cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        timeout=timeout,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------- handler ----------

class HousekeepingHandler(BaseGoalHandler):
    """
    Recurring maintenance/cleanup tasks for Orrin:
      - snapshots (goals, memory)
      - WAL pruning (goals, memory)
      - log vacuum/rotation
      - environment checks (pip check), lint/format, smoke tests

    Expected goal.spec keys (all optional):
      tasks: list[str]  # any of: snapshot_goals, prune_goals_wal, snapshot_memory, prune_memory_wal,
                        #          vacuum_logs, pip_check, pytest_smoke, ruff_lint, ruff_format, clean_tmp, reindex_memory
      opts:  dict       # per-task options, e.g. {"vacuum_logs": {"older_than_days": 7, "delete_older_than_days": 90}}
      repo:  str        # working directory for test/lint/format (default ".")
    """
    kind: str = "housekeeping"

    # ---------- Planning ----------

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        spec = goal.spec or {}
        tasks: List[str] = list(spec.get("tasks") or [
            "snapshot_goals",
            "prune_goals_wal",
            "vacuum_logs",
            "pip_check",
        ])
        opts: Dict[str, Any] = dict(spec.get("opts") or {})

        steps: List[Step] = []
        prev_id: Optional[str] = None
        for t in tasks:
            action = {"op": t, "opts": opts.get(t, {})}
            s = _new_step(goal.id, t.replace("_", " "), action, deps=([prev_id] if prev_id else None))
            steps.append(s)
            prev_id = s.id
        return steps

    def is_blocked(self, goal: Goal, ctx: HandlerContext) -> Tuple[bool, Optional[str]]:
        # Most tasks are lightweight; we coordinate on a single fs-maintenance lock when needed inside tick().
        return False, None

    # ---------- Execution ----------

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        op = step.action.get("op", "")
        opts = step.action.get("opts") or {}
        repo = Path((goal.spec or {}).get("repo") or ctx.get("repo") or ".").resolve()
        started_now = False

        if step.started_at is None:
            step.started_at = UTCNOW()
            step.status = Status.RUNNING
            started_now = True

        try:
            # File-system mutating ops require a cooperative lock
            mut_ops = {
                "snapshot_goals", "prune_goals_wal",
                "snapshot_memory", "prune_memory_wal",
                "vacuum_logs", "clean_tmp",
            }
            if op in mut_ops and not _acquire(ctx, "fs-maintenance", goal.id):
                return self._defer(step, "fs-maintenance lock unavailable")

            if op == "snapshot_goals":
                note = self._op_snapshot_goals(goal, ctx, opts)
                _release(ctx, "fs-maintenance", goal.id)
                return self._finish_ok(goal, step, ctx, note)

            if op == "prune_goals_wal":
                note = self._op_prune_wal(goal, ctx, opts, target="goals")
                _release(ctx, "fs-maintenance", goal.id)
                return self._finish_ok(goal, step, ctx, note)

            if op == "snapshot_memory":
                note = self._op_snapshot_memory(goal, ctx, opts)
                _release(ctx, "fs-maintenance", goal.id)
                return self._finish_ok(goal, step, ctx, note)

            if op == "prune_memory_wal":
                note = self._op_prune_wal(goal, ctx, opts, target="memory")
                _release(ctx, "fs-maintenance", goal.id)
                return self._finish_ok(goal, step, ctx, note)

            if op == "vacuum_logs":
                note = self._op_vacuum_logs(goal, ctx, opts)
                _release(ctx, "fs-maintenance", goal.id)
                return self._finish_ok(goal, step, ctx, note)

            if op == "clean_tmp":
                note = self._op_clean_tmp(goal, ctx, opts)
                _release(ctx, "fs-maintenance", goal.id)
                return self._finish_ok(goal, step, ctx, note)

            if op == "pip_check":
                rc, out, err = _run([shutil.which("python") or "python", "-m", "pip", "check"])
                art = _write_artifact(ctx, goal, f"{step.id}_pip_check.txt", out + ("\n\n[stderr]\n" + err if err else ""))
                step.artifacts.append(art)
                if rc != 0:
                    raise RuntimeError("pip check reported issues")
                return self._finish_ok(goal, step, ctx, "pip check OK")

            if op == "pytest_smoke":
                py = shutil.which("pytest")
                if not py:
                    # Treat as soft success when pytest not installed
                    return self._finish_ok(goal, step, ctx, "pytest not installed; skipped")
                args = opts.get("args") or ["-q", "-k", "not slow and not integration"]
                rc, out, err = _run([py, *args], cwd=repo, timeout=opts.get("timeout", 900))
                art = _write_artifact(ctx, goal, f"{step.id}_pytest.txt", out + ("\n\n[stderr]\n" + err if err else ""))
                step.artifacts.append(art)
                if rc != 0:
                    raise RuntimeError("pytest smoke failed")
                return self._finish_ok(goal, step, ctx, "pytest smoke OK")

            if op == "ruff_lint":
                ruff = _which("ruff")
                if not ruff:
                    return self._finish_ok(goal, step, ctx, "ruff not installed; skipped")
                rc, out, err = _run([ruff, "check", "--quiet"], cwd=repo)
                art = _write_artifact(ctx, goal, f"{step.id}_ruff.txt", out + ("\n\n[stderr]\n" + err if err else ""))
                step.artifacts.append(art)
                if rc != 0:
                    raise RuntimeError("ruff lint errors")
                return self._finish_ok(goal, step, ctx, "ruff check OK")

            if op == "ruff_format":
                ruff = _which("ruff")
                if ruff:
                    rc, out, err = _run([ruff, "format"], cwd=repo)
                    art = _write_artifact(ctx, goal, f"{step.id}_format.txt", out + ("\n\n[stderr]\n" + err if err else ""))
                    step.artifacts.append(art)
                    if rc != 0:
                        raise RuntimeError("ruff format failed")
                    return self._finish_ok(goal, step, ctx, "ruff format OK")
                black = _which("black")
                if black:
                    rc, out, err = _run([black, "-q", "."], cwd=repo)
                    art = _write_artifact(ctx, goal, f"{step.id}_format.txt", (out or "") + ("\n\n[stderr]\n" + err if err else ""))
                    step.artifacts.append(art)
                    if rc != 0:
                        raise RuntimeError("black format failed")
                    return self._finish_ok(goal, step, ctx, "black format OK")
                return self._finish_ok(goal, step, ctx, "no formatter installed; skipped")

            if op == "reindex_memory":
                fn = ctx.get("memory_reindex")
                if callable(fn):
                    result = fn()
                    note = f"memory reindex invoked → {result!r}"
                else:
                    note = "no memory_reindex() provided; skipped"
                return self._finish_ok(goal, step, ctx, note)

            # Unknown op
            raise ValueError(f"unknown op: {op!r}")

        except Exception as e:
            step.last_error = f"{type(e).__name__}: {e}"
            step.attempts += 1
            if started_now:
                step.started_at = None
                step.status = Status.READY
            if step.attempts >= step.max_attempts:
                step.status = Status.FAILED
                step.finished_at = UTCNOW()
            # ensure any lock is released on failure
            if op in {"snapshot_goals", "prune_goals_wal", "snapshot_memory", "prune_memory_wal", "vacuum_logs", "clean_tmp"}:
                _release(ctx, "fs-maintenance", goal.id)
            return step

    # ---------- operations (pure python where possible) ----------

    def _op_snapshot_goals(self, goal: Goal, ctx: HandlerContext, opts: Dict[str, Any]) -> str:
        state = _ctx_path(ctx, "GOALS_SNAP", "data/goals/state.jsonl")
        dest_dir = _ctx_path(ctx, "GOALS_SNAP_DIR", "data/goals/snapshots")
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = UTCNOW().strftime("%Y%m%d-%H%M%S")
        dest = dest_dir / f"goals_state_{ts}.jsonl"
        if not state.exists():
            return "goals state.jsonl not found; skipped"
        # Write a small manifest with counts then copy file
        lines = state.read_text(encoding="utf-8").splitlines()
        meta = {"timestamp": UTCNOW().isoformat(), "lines": len(lines)}
        manifest = dest.with_suffix(".meta.json")
        manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"snapshot_goals → {dest.name} (lines={len(lines)})"

    def _op_prune_wal(self, goal: Goal, ctx: HandlerContext, opts: Dict[str, Any], *, target: str) -> str:
        keep = int(opts.get("keep_lines", 5000))
        if target == "goals":
            wal = _ctx_path(ctx, "GOALS_WAL", "data/goals/wal.log")
            rotated_dir = _ctx_path(ctx, "GOALS_WAL_DIR", "data/goals/wal-rotated")
        elif target == "memory":
            wal = _ctx_path(ctx, "MEMORY_WAL", "data/memory/wal.log")
            rotated_dir = _ctx_path(ctx, "MEMORY_WAL_DIR", "data/memory/wal-rotated")
        else:
            raise ValueError("unknown target for WAL prune")

        if not wal.exists():
            return f"{target} wal.log not found; skipped"

        rotated_dir.mkdir(parents=True, exist_ok=True)
        text = wal.read_text(encoding="utf-8").splitlines()
        before = len(text)
        if before <= keep:
            return f"{target} WAL within limit (lines={before} ≤ keep={keep}); no change"

        # move old to gz with timestamp
        ts = UTCNOW().strftime("%Y%m%d-%H%M%S")
        gz_path = rotated_dir / f"{target}_wal_{ts}.log.gz"
        with gzip.open(gz_path, "wb") as f:
            f.write(("\n".join(text[:-keep]) + "\n").encode("utf-8"))
        # rewrite wal with last `keep` lines
        wal.write_text("\n".join(text[-keep:]) + "\n", encoding="utf-8")
        return f"pruned {target} WAL: {before} → {keep} lines; rotated={gz_path.name}"

    def _op_snapshot_memory(self, goal: Goal, ctx: HandlerContext, opts: Dict[str, Any]) -> str:
        fn = ctx.get("memory_snapshot")
        if callable(fn):
            result = fn()
            return f"memory snapshot via hook → {result!r}"
        # fallback: copy known file if present
        state = _ctx_path(ctx, "MEMORY_SNAP", "data/memory/state.jsonl")
        dest_dir = _ctx_path(ctx, "MEMORY_SNAP_DIR", "data/memory/snapshots")
        if not state.exists():
            return "memory snapshot hook not provided and state.jsonl missing; skipped"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = UTCNOW().strftime("%Y%m%d-%H%M%S")
        dest = dest_dir / f"memory_state_{ts}.jsonl"
        dest.write_text(state.read_text(encoding="utf-8"), encoding="utf-8")
        return f"snapshot_memory → {dest.name}"

    def _op_vacuum_logs(self, goal: Goal, ctx: HandlerContext, opts: Dict[str, Any]) -> str:
        logs_dir = _ctx_path(ctx, "LOGS_DIR", "logs")
        if not logs_dir.exists():
            return "logs/ not found; skipped"
        older_than_days = int(opts.get("older_than_days", 7))
        delete_older_than_days = int(opts.get("delete_older_than_days", 90))
        now = UTCNOW()
        gzipped = 0
        deleted = 0

        for p in logs_dir.glob("**/*"):
            if p.is_dir():
                continue
            age_days = (now - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).days
            if p.suffix == ".gz":
                if age_days >= delete_older_than_days:
                    try:
                        p.unlink()
                        deleted += 1
                    except Exception:
                        pass
                continue
            # gzip .log and similar if old enough
            if age_days >= older_than_days:
                gz = p.with_suffix(p.suffix + ".gz")
                try:
                    with open(p, "rb") as fin, gzip.open(gz, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    p.unlink(missing_ok=True)
                    gzipped += 1
                except Exception:
                    # ignore individual file errors
                    continue
        return f"vacuum_logs: gzipped={gzipped}, deleted_old_gz={deleted}"

    def _op_clean_tmp(self, goal: Goal, ctx: HandlerContext, opts: Dict[str, Any]) -> str:
        tmp_dir = _ctx_path(ctx, "TMP_DIR", "tmp")
        if not tmp_dir.exists():
            return "tmp/ not found; skipped"
        removed = 0
        for p in tmp_dir.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                removed += 1
            except Exception:
                pass
        return f"clean_tmp: removed {removed} entries"

    # ---------- common step helpers ----------

    def _finish_ok(self, goal: Goal, step: Step, ctx: HandlerContext, note: str) -> Step:
        step.status = Status.DONE
        step.finished_at = UTCNOW()
        art = _write_artifact(ctx, goal, f"{step.id}_ok.txt", note)
        step.artifacts.append(art)
        step.last_error = None
        return step

    def _defer(self, step: Step, reason: str) -> Step:
        step.status = Status.READY
        step.last_error = f"DEFERRED: {reason}"
        step.started_at = None
        return step


__all__ = ["HousekeepingHandler"]
