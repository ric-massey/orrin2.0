# goals/handlers/coding.py
# Concrete handler for code/edit/test goals; plans git steps and executes them safely

from __future__ import annotations

import os
import json
import uuid
import shlex
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..model import Goal, Step, Status
from .base import BaseGoalHandler, HandlerContext

UTCNOW = lambda: datetime.now(timezone.utc)


def _new_step(goal_id: str, name: str, action: Dict[str, Any], *, max_attempts: int = 3, deps: Optional[List[str]] = None) -> Step:
    return Step(
        id=f"s_{uuid.uuid4().hex[:10]}",
        goal_id=goal_id,
        name=name,
        action=action,
        max_attempts=max_attempts,
        deps=list(deps or []),
        status=Status.READY,
    )


def _repo_path(ctx: HandlerContext, goal: Goal) -> Path:
    repo = (goal.spec or {}).get("repo") or ctx.get("repo") or "."
    return Path(repo).resolve()


def _artifacts_dir(ctx: HandlerContext, goal: Goal) -> Path:
    base = Path(ctx.get("artifacts_dir") or "data/goals/artifacts").resolve()
    d = base / goal.id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_artifact(ctx: HandlerContext, goal: Goal, name: str, content: str) -> str:
    p = _artifacts_dir(ctx, goal) / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _run(repo: Path, cmd: List[str], *, timeout: Optional[int] = None, input_text: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        input=(input_text.encode("utf-8") if input_text is not None else None),
        capture_output=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")


def _has_lock(ctx: HandlerContext, name: str, goal_id: str) -> bool:
    locks = ctx.get("locks")
    if not locks:
        return True  # no lock manager configured; proceed
    try:
        return locks.acquire(name, goal_id)
    except Exception:
        return False


def _release_lock(ctx: HandlerContext, name: str, goal_id: str) -> None:
    locks = ctx.get("locks")
    if not locks:
        return
    try:
        locks.release(name, goal_id)
    except Exception:
        pass


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in s.lower()).strip("-")


class CodingHandler(BaseGoalHandler):
    """
    Plans and executes repo-centric tasks:
      - create/switch branch
      - apply changes (files mapping or unified diff)
      - run tests/linters
      - summarize result

    Expected goal.spec keys (all optional unless noted):
      repo: str (path to repository; default '.')
      branch: str (if absent, auto-create 'orrin/<slug>-<yyyymmddHHMM>')
      allow_dirty: bool (default False) — skip clean working-tree check
      files: { "relative/path.py": "new file content", ... }  (mutually exclusive with 'diff')
      diff: str  (unified diff text to apply via 'git apply')
      commit_message: str (default derived from goal.title)
      tests: bool | str (True → run 'pytest -q'; or a shell command string)
      summary: bool (default True) — write a short summary artifact
    """
    kind: str = "coding"

    # ---------- Planning ----------

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        spec = goal.spec or {}
        steps: List[Step] = []

        # 1) optional: ensure working tree is clean (unless allow_dirty)
        if not bool(spec.get("allow_dirty", False)):
            steps.append(_new_step(goal.id, "ensure clean working tree", {"op": "git_status_clean"}))

        # 2) create/switch branch
        title = goal.title or "task"
        default_branch = f"orrin/{_slug(title)[:32]}-{UTCNOW().strftime('%Y%m%d%H%M')}"
        branch = spec.get("branch") or default_branch
        steps.append(_new_step(goal.id, f"switch to branch {branch}", {"op": "git_branch", "branch": branch}))

        # 3) apply edits (files or diff)
        if "files" in spec:
            steps.append(_new_step(goal.id, "apply file changes", {"op": "apply_files", "files": spec["files"], "commit_message": spec.get("commit_message")}))
        elif "diff" in spec:
            steps.append(_new_step(goal.id, "apply patch", {"op": "apply_patch", "diff": spec["diff"], "commit_message": spec.get("commit_message")}))

        # 4) run tests (if requested)
        if spec.get("tests"):
            test_cmd = "pytest -q" if spec.get("tests") is True else str(spec.get("tests"))
            steps.append(_new_step(goal.id, "run tests", {"op": "run_cmd", "cmd": test_cmd, "timeout": spec.get("tests_timeout", 900)}))

        # 5) summarize
        if spec.get("summary", True):
            steps.append(_new_step(goal.id, "summarize changes", {"op": "summarize"}))

        return steps

    def is_blocked(self, goal: Goal, ctx: HandlerContext) -> Tuple[bool, Optional[str]]:
        # Block if repo-write lock unavailable for write operations still pending
        # (cheap heuristic: if any remaining step has an op that mutates the repo).
        mut_ops = {"apply_files", "apply_patch", "git_branch"}
        if any(s for s in ctx.get("pending_steps", []) if s.action.get("op") in mut_ops):
            # We don't acquire here; acquisition happens inside tick right before the op.
            # This call is for informational blocking only; return False to let tick attempt acquisition.
            return False, None
        return False, None

    # ---------- Execution (one step per tick) ----------

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        repo = _repo_path(ctx, goal)
        op = step.action.get("op")
        started_now = False

        if step.started_at is None:
            step.started_at = UTCNOW()
            step.status = Status.RUNNING
            started_now = True

        try:
            if op == "git_status_clean":
                rc, out, _err = _run(repo, ["git", "status", "--porcelain"])
                if rc != 0:
                    raise RuntimeError("git status failed")
                if out.strip():
                    allow_dirty = bool((goal.spec or {}).get("allow_dirty", False))
                    if not allow_dirty:
                        raise RuntimeError("working tree not clean")
                step = self._finish_ok(goal, step, ctx, f"clean={not bool(out.strip())}")
                return step

            if op == "git_branch":
                branch = step.action.get("branch")
                if not branch:
                    raise ValueError("branch not specified")
                # Acquire repo-write for branch switch
                if not _has_lock(ctx, "repo-write", goal.id):
                    # Could not get lock; mark back to READY to retry later.
                    return self._defer(goal, step, "repo-write lock unavailable")
                # Try git switch -c (Git 2.23+) then fallback
                rc, out, err = _run(repo, ["git", "rev-parse", "--verify", branch])
                if rc == 0:
                    rc2, out2, err2 = _run(repo, ["git", "switch", branch])
                    if rc2 != 0:
                        rc2, out2, err2 = _run(repo, ["git", "checkout", branch])
                    if rc2 != 0:
                        raise RuntimeError(f"failed to switch branch: {err2 or err}")
                else:
                    rc2, out2, err2 = _run(repo, ["git", "switch", "-c", branch])
                    if rc2 != 0:
                        rc2, out2, err2 = _run(repo, ["git", "checkout", "-b", branch])
                    if rc2 != 0:
                        raise RuntimeError(f"failed to create/switch branch: {err2}")
                _release_lock(ctx, "repo-write", goal.id)
                step = self._finish_ok(goal, step, ctx, f"branch={branch}")
                return step

            if op == "apply_files":
                # Acquire repo-write
                if not _has_lock(ctx, "repo-write", goal.id):
                    return self._defer(goal, step, "repo-write lock unavailable")
                files: Dict[str, str] = step.action.get("files") or {}
                if not isinstance(files, dict) or not files:
                    raise ValueError("files mapping is empty")
                for rel, content in files.items():
                    p = repo / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8")
                _run(repo, ["git", "add", "-A"])
                msg = step.action.get("commit_message") or f"chore(orrin): {goal.title or 'apply changes'}"
                rc, _out, err = _run(repo, ["git", "commit", "-m", msg])
                if rc != 0:
                    # allow "nothing to commit" to still succeed
                    if "nothing to commit" not in (err or "").lower():
                        raise RuntimeError(f"git commit failed: {err}")
                _release_lock(ctx, "repo-write", goal.id)
                step = self._finish_ok(goal, step, ctx, f"files={len(files)} committed")
                return step

            if op == "apply_patch":
                if not _has_lock(ctx, "repo-write", goal.id):
                    return self._defer(goal, step, "repo-write lock unavailable")
                diff = step.action.get("diff")
                if not diff:
                    raise ValueError("diff is empty")
                rc, _out, err = _run(repo, ["git", "apply", "-p0", "-"], input_text=diff)
                if rc != 0:
                    raise RuntimeError(f"git apply failed: {err}")
                _run(repo, ["git", "add", "-A"])
                msg = step.action.get("commit_message") or f"feat(orrin): {goal.title or 'apply patch'}"
                rc2, _out2, err2 = _run(repo, ["git", "commit", "-m", msg])
                if rc2 != 0 and "nothing to commit" not in (err2 or "").lower():
                    raise RuntimeError(f"git commit failed: {err2}")
                _release_lock(ctx, "repo-write", goal.id)
                step = self._finish_ok(goal, step, ctx, "patch applied & committed")
                return step

            if op == "run_cmd":
                cmd = step.action.get("cmd")
                if not cmd:
                    raise ValueError("cmd is empty")
                timeout = step.action.get("timeout")
                shell_cmd = ["bash", "-lc", cmd]
                rc, out, err = _run(repo, shell_cmd, timeout=timeout)
                art = _write_artifact(ctx, goal, f"{step.id}_run.txt", f"$ {cmd}\n\n[stdout]\n{out}\n\n[stderr]\n{err}")
                step.artifacts.append(art)
                if rc != 0:
                    raise RuntimeError(f"command failed ({rc})")
                step = self._finish_ok(goal, step, ctx, f"ran: {cmd}")
                return step

            if op == "summarize":
                rc1, diffstat, _ = _run(repo, ["git", "diff", "--staged", "--stat"])
                if rc1 != 0 or not diffstat.strip():
                    rc1b, diffstat, _ = _run(repo, ["git", "diff", "--stat"])
                rc2, head, _ = _run(repo, ["git", "log", "-1", "--pretty=oneline"])
                summary = {
                    "goal": asdict(goal),
                    "head": head.strip(),
                    "diffstat": diffstat.strip(),
                    "finished_at": UTCNOW().isoformat(),
                }
                art = _write_artifact(ctx, goal, f"{step.id}_summary.json", json.dumps(summary, indent=2))
                step.artifacts.append(art)
                step = self._finish_ok(goal, step, ctx, "summary written")
                return step

            # Unknown op
            raise ValueError(f"unknown op: {op!r}")

        except Exception as e:
            step.last_error = f"{type(e).__name__}: {e}"
            step.attempts += 1
            # If we just started and failed immediately, clear started_at so runner can reschedule cleanly
            if started_now:
                step.started_at = None
                step.status = Status.READY
            # Exhausted?
            if step.attempts >= step.max_attempts:
                step.status = Status.FAILED
                step.finished_at = UTCNOW()
            # Release lock if we hold it and failed
            if op in {"apply_files", "apply_patch", "git_branch"}:
                _release_lock(ctx, "repo-write", goal.id)
            return step

    # ---------- Helpers ----------

    def _finish_ok(self, goal: Goal, step: Step, ctx: HandlerContext, note: str) -> Step:
        step.status = Status.DONE
        step.finished_at = UTCNOW()
        # Attach a tiny artifact with the note for quick auditing
        art = _write_artifact(ctx, goal, f"{step.id}_ok.txt", note)
        step.artifacts.append(art)
        step.last_error = None
        return step

    def _defer(self, goal: Goal, step: Step, reason: str) -> Step:
        # Put the step back to READY with a note; runner/policy will retry later.
        step.status = Status.READY
        step.last_error = f"DEFERRED: {reason}"
        # Do not increment attempts for defers; it's not a failure.
        # Clear started_at so scheduling can pick it up fresh.
        step.started_at = None
        return step


__all__ = ["CodingHandler"]
