# goals/cli.py
# Command-line interface for the Goals subsystem (create/list/describe/update/cancel/submit)

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .api import GoalsAPI
from .model import Goal, Status, Priority

# Optional store/daemon imports (provide clear error if missing)
try:
    from .store import FileGoalsStore  # type: ignore
except Exception as e:  # pragma: no cover
    FileGoalsStore = None  # type: ignore
    _STORE_IMPORT_ERR = e

try:
    from .goals_daemon import GoalsDaemon  # type: ignore
except Exception:
    GoalsDaemon = None  # type: ignore


# ----------------------------- globals/helpers -----------------------------

# Keep track of the resolved data dir so helper functions (like JSON arg handling)
# can place auto-created files in a predictable location.
_DEFAULT_DATA_DIR: Optional[Path] = None

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _ensure_json_dir() -> Path:
    """
    Ensure and return the directory where CLI-owned JSON files should live.
    Prefer the current _DEFAULT_DATA_DIR (set in _build_api), otherwise resolve
    from env ORRIN_GOALS_DIR or default data/goals.
    Final path: <data_dir>/json
    """
    base = _DEFAULT_DATA_DIR or Path(os.environ.get("ORRIN_GOALS_DIR") or "data/goals").resolve()
    json_dir = base / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir

def _goal_to_jsonable(g: Goal) -> Dict[str, Any]:
    """Convert a Goal to a plain JSON-serializable dict."""
    d = g.__dict__.copy()
    # enums -> names
    d["status"] = getattr(g.status, "name", str(g.status))
    d["priority"] = getattr(g.priority, "name", str(g.priority))
    # datetimes -> ISO
    if d.get("deadline"):    d["deadline"]    = g.deadline.isoformat()
    if d.get("created_at"):  d["created_at"]  = g.created_at.isoformat()
    if d.get("updated_at"):  d["updated_at"]  = g.updated_at.isoformat()
    # dataclass progress -> dict
    pr = d.get("progress")
    if is_dataclass(pr):
        d["progress"] = asdict(pr)
    # acceptance/spec already dict-like; ensure plain dict
    if d.get("acceptance") is not None:
        d["acceptance"] = dict(d["acceptance"])
    if d.get("spec") is not None:
        d["spec"] = dict(d["spec"])
    return d

def _parse_json_arg(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON string or @path reference. If the reference file does not exist,
    auto-create a minimal `{}` JSON file in our dedicated folder:
        <DATA_DIR or $ORRIN_GOALS_DIR>/json/<name>.json
    and return {}.
    """
    if not s:
        return None
    s = s.strip()
    if s.startswith("@"):
        name = s[1:].strip()
        # If user didn't provide a directory, place under <data_dir>/json/
        p = Path(name)
        if not p.is_absolute() and (p.parent == Path(".") or str(p.parent) == ""):
            p = _ensure_json_dir() / p.name
        else:
            # Ensure parent exists if user gave a relative/absolute path with dirs
            p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("{}\n", encoding="utf-8")
            return {}
        text = p.read_text(encoding="utf-8").strip()
        if text == "":
            # Make empty files valid JSON
            p.write_text("{}\n", encoding="utf-8")
            return {}
        return json.loads(text)
    return json.loads(s)

def _priority_from_arg(x: Optional[str]) -> Priority:
    if x is None:
        return Priority.NORMAL
    s = x.strip().upper()
    mapping = {
        "ROUTINE": Priority.LOW,
        "LOW": Priority.LOW,
        "IMPORTANT": Priority.NORMAL,
        "NORMAL": Priority.NORMAL,
        "HIGH": Priority.HIGH,
        "CRITICAL": Priority.CRITICAL,
    }
    if s in mapping:
        return mapping[s]
    try:
        return Priority(int(s))
    except Exception:
        return Priority.NORMAL

def _status_from_arg(x: Optional[str]) -> Optional[Status]:
    if x is None:
        return None
    s = x.strip().upper()
    for st in Status:
        if s == st.name or s == getattr(st, "value", s):
            return st
    return None

def _deadline_from_arg(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    s = x.strip()
    if s.endswith("Z"):
        return s  # ISO with Z
    # Accept naive "YYYY-MM-DD HH:MM" and add Z
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo:
            return dt.isoformat()
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return s  # let API try to parse

def _print_table(goals: Sequence[Goal]) -> None:
    if not goals:
        print("(no goals)")
        return
    cols = ["id", "kind", "priority", "status", "title", "deadline", "updated_at"]
    widths = {c: len(c) for c in cols}
    rows: List[Dict[str, str]] = []
    for g in goals:
        row = {
            "id": g.id,
            "kind": g.kind,
            "priority": getattr(g.priority, "name", str(int(g.priority))),
            "status": getattr(g.status, "name", str(g.status)),
            "title": (g.title or "")[:60],
            "deadline": g.deadline.isoformat() if g.deadline else "-",
            "updated_at": getattr(g, "updated_at", _utcnow()).isoformat(timespec="seconds"),
        }
        rows.append(row)
        for k, v in row.items():
            widths[k] = max(widths[k], len(str(v)))
    # header
    hdr = "  ".join(k.ljust(widths[k]) for k in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

def _ensure_store(path: Path):
    if FileGoalsStore is None:  # pragma: no cover
        raise RuntimeError(f"Cannot import FileGoalsStore: {_STORE_IMPORT_ERR}")
    path.mkdir(parents=True, exist_ok=True)
    return FileGoalsStore(data_dir=path)

def _build_api(args) -> GoalsAPI:
    # Resolve and remember data dir for helper functions
    global _DEFAULT_DATA_DIR
    data_dir = Path(args.data_dir or os.environ.get("ORRIN_GOALS_DIR") or "data/goals").resolve()
    _DEFAULT_DATA_DIR = data_dir

    store = _ensure_store(data_dir)

    daemon = None
    if getattr(args, "with_daemon", False) and GoalsDaemon is not None:
        # Minimal daemon bootstrap; caller is a CLI, so we won't keep it running—just create to allow submit()
        daemon = GoalsDaemon(store=store, registry=None, workers=0)  # registry not required for submit()
        # Do not start threads for CLI usage

    api = GoalsAPI(
        store=store,
        daemon=daemon,
        reaper_sink=None,
        memory_writer=None,
        plan_on_create=not args.no_plan_on_create,
    )
    return api


# ----------------------------- commands -----------------------------

def cmd_add(args) -> int:
    api = _build_api(args)
    spec = _parse_json_arg(args.spec)
    acceptance = _parse_json_arg(args.acceptance)
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] if args.tags else []
    goal = api.create_goal(
        title=args.title,
        kind=args.kind,
        spec=spec,
        priority=_priority_from_arg(args.priority),
        deadline=_deadline_from_arg(args.deadline),
        tags=tags,
        parent_id=args.parent_id,
        acceptance=acceptance,
    )
    if args.json:
        print(json.dumps(_goal_to_jsonable(goal), indent=2))
    else:
        print(f"created {goal.id} [{goal.kind}/{goal.priority.name}] {goal.title}")
    # Nudge daemon if present
    api.submit(goal.id)
    return 0

def cmd_list(args) -> int:
    api = _build_api(args)
    kinds = args.kinds
    statuses = [_status_from_arg(s) for s in (args.statuses or [])]
    statuses = [s for s in statuses if s]
    priorities = [_priority_from_arg(p) for p in (args.priorities or [])]
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] if args.tags else None

    goals = api.list_goals(
        kinds=kinds,
        statuses=statuses or None,
        priorities=priorities or None,
        tags=tags,
        text=args.text,
        limit=args.limit,
        sort=args.sort,
    )
    if args.json:
        print(json.dumps([_goal_to_jsonable(g) for g in goals], indent=2))
    else:
        _print_table(goals)
    return 0

def cmd_describe(args) -> int:
    api = _build_api(args)
    g = api.get_goal(args.goal_id)
    if not g:
        print(f"goal not found: {args.goal_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_goal_to_jsonable(g), indent=2))
        return 0
    print(f"{g.id} [{g.kind}/{g.priority.name}] {g.title}")
    print(f"status:    {g.status.name}")
    print(f"deadline:  {g.deadline.isoformat() if g.deadline else '-'}")
    print(f"tags:      {', '.join(g.tags) if g.tags else '-'}")
    print(f"parent:    {g.parent_id or '-'}")
    print(f"created:   {g.created_at.isoformat()}")
    print(f"updated:   {g.updated_at.isoformat()}")
    print("spec:")
    print(json.dumps(g.spec or {}, indent=2))
    if g.acceptance:
        print("acceptance:")
        print(json.dumps(g.acceptance, indent=2))
    if g.last_error:
        print(f"last_error: {g.last_error}")
    return 0

def cmd_update(args) -> int:
    api = _build_api(args)
    fields: Dict[str, Any] = {}
    if args.title is not None:
        fields["title"] = args.title
    if args.priority is not None:
        fields["priority"] = _priority_from_arg(args.priority)
    if args.deadline is not None:
        fields["deadline"] = _deadline_from_arg(args.deadline)
    if args.status is not None:
        st = _status_from_arg(args.status)
        if st:
            fields["status"] = st
    if args.tags is not None:
        fields["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.spec is not None:
        fields["spec"] = _parse_json_arg(args.spec) or {}
    g = api.update_goal(args.goal_id, **fields)
    if not g:
        print(f"goal not found: {args.goal_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_goal_to_jsonable(g), indent=2))
    else:
        print(f"updated {g.id} [{g.kind}/{g.priority.name}] {g.title}")
    return 0

def cmd_cancel(args) -> int:
    api = _build_api(args)
    g = api.cancel_goal(args.goal_id, reason=args.reason or "api.cancel")
    if not g:
        print(f"goal not found: {args.goal_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_goal_to_jsonable(g), indent=2))
    else:
        print(f"cancelled {g.id} {g.title}")
    return 0

def cmd_submit(args) -> int:
    api = _build_api(args)
    api.submit(args.goal_id)
    if not args.quiet:
        print(f"submitted {args.goal_id}")
    return 0

def cmd_pause(args) -> int:
    api = _build_api(args)
    g = api.update_goal(args.goal_id, status=Status.PAUSED)
    if not g:
        print(f"goal not found: {args.goal_id}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"paused {g.id}")
    return 0

def cmd_resume(args) -> int:
    api = _build_api(args)
    g = api.update_goal(args.goal_id, status=Status.READY)
    if not g:
        print(f"goal not found: {args.goal_id}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"resumed {g.id}")
    return 0


# ----------------------------- argparse -----------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m goals.cli", description="Goals subsystem CLI")
    p.add_argument("--data-dir", default=None, help="Directory for goals data (default: data/goals or $ORRIN_GOALS_DIR)")
    p.add_argument("--json", action="store_true", help="Output JSON when applicable")
    p.add_argument("--with-daemon", action="store_true", help="Initialize a daemon object for submit() (no threads started)")
    p.add_argument("--no-plan-on-create", action="store_true", help="Do not auto-submit newly created goals")

    sp = p.add_subparsers(dest="cmd", required=True)

    # add
    s = sp.add_parser("add", help="Create a new goal")
    s.add_argument("kind", help="Goal kind (e.g., coding, research, housekeeping)")
    s.add_argument("title", help="Goal title")
    s.add_argument("--spec", help='JSON string or @path/to/file.json with the spec')
    s.add_argument("--priority", help="ROUTINE|IMPORTANT|CRITICAL or 0..3")
    s.add_argument("--deadline", help="ISO timestamp (e.g., 2025-09-20T23:59:00Z)")
    s.add_argument("--tags", help="Comma-separated tags")
    s.add_argument("--parent-id", help="Parent goal/epic ID")
    s.add_argument("--acceptance", help='JSON string or @path/to/file.json acceptance criteria')
    s.set_defaults(func=cmd_add)

    # list
    s = sp.add_parser("list", help="List goals")
    s.add_argument("--kinds", nargs="*", help="Filter by kind(s)")
    s.add_argument("--statuses", nargs="*", help="Filter by status names")
    s.add_argument("--priorities", nargs="*", help="Filter by priorities")
    s.add_argument("--tags", help="Require these comma-separated tags")
    s.add_argument("--text", help="Substring match in title/error")
    s.add_argument("--limit", type=int, help="Limit results")
    s.add_argument("--sort", default="-updated_at", help="Sort by field (e.g., -updated_at, created_at, -priority)")
    s.set_defaults(func=cmd_list)

    # describe
    s = sp.add_parser("describe", help="Describe one goal in detail")
    s.add_argument("goal_id")
    s.set_defaults(func=cmd_describe)

    # update
    s = sp.add_parser("update", help="Update fields on a goal")
    s.add_argument("goal_id")
    s.add_argument("--title")
    s.add_argument("--priority")
    s.add_argument("--deadline")
    s.add_argument("--status")
    s.add_argument("--tags")
    s.add_argument("--spec", help='JSON string or @path/to/file.json')
    s.set_defaults(func=cmd_update)

    # cancel
    s = sp.add_parser("cancel", help="Cancel a goal")
    s.add_argument("goal_id")
    s.add_argument("--reason")
    s.set_defaults(func=cmd_cancel)

    # submit
    s = sp.add_parser("submit", help="Submit/kick a goal to the daemon")
    s.add_argument("goal_id")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_submit)

    # pause/resume
    s = sp.add_parser("pause", help="Pause a goal (status=PAUSED)")
    s.add_argument("goal_id")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_pause)

    s = sp.add_parser("resume", help="Resume a goal (status=READY)")
    s.add_argument("goal_id")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_resume)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
