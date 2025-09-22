# goals/snapshot.py
# Create/restore compact JSONL snapshots of goals (and steps) with optional WAL rotation

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .model import Goal, Step  # only for type hints; code is duck-typed at runtime

UTCNOW = lambda: datetime.now(timezone.utc)


# -----------------------------
# Public API
# -----------------------------

def snapshot_state(
    store: Any,
    *,
    out_path: str | Path = "data/goals/state.jsonl",
    include_steps: bool = True,
    atomic: bool = True,
) -> Path:
    """
    Write a newline-delimited JSON snapshot of current goals (and steps).

    File format (JSONL):
      {"type":"goal", ...Goal fields...}
      {"type":"step", ...Step fields...}      # only if include_steps=True

    The function is duck-typed and works with any store exposing:
      - iter_goals() | list_goals() | all()
      - iter_steps() | list_steps()            (optional)

    Returns the final Path to the written snapshot.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Gather items
    goals = list(_iter_goals(store))
    steps = list(_iter_steps(store)) if include_steps else []

    # Write atomically (tmp + replace) to avoid partial snapshots
    if atomic:
        fd, tmpname = tempfile.mkstemp(prefix="goals_state_", suffix=".jsonl", dir=str(out.parent))
        os.close(fd)
        tmp = Path(tmpname)
        try:
            _write_jsonl(tmp, ({"type": "goal", **_jsonable(g)} for g in goals))
            if steps:
                _write_jsonl(tmp, ({"type": "step", **_jsonable(s)} for s in steps), append=True)
            tmp.replace(out)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    else:
        _write_jsonl(out, ({"type": "goal", **_jsonable(g)} for g in goals))
        if steps:
            _write_jsonl(out, ({"type": "step", **_jsonable(s)} for s in steps), append=True)

    return out


def load_state(path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Stream records from a JSONL snapshot (as dicts). Caller can filter by 'type'.
    """
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def rotate_wal(
    wal_path: str | Path = "data/goals/wal.log",
    *,
    rotated_dir: str | Path = "data/goals/wal-rotated",
    keep_tail_lines: int = 5_000,
) -> Optional[Path]:
    """
    Compact the WAL by moving all but the last `keep_tail_lines` lines into a gzipped rotation file.
    Returns the Path to the created .gz (or None if nothing rotated).
    """
    wal = Path(wal_path)
    if not wal.exists():
        return None

    rotated = Path(rotated_dir)
    rotated.mkdir(parents=True, exist_ok=True)

    text = wal.read_text(encoding="utf-8").splitlines()
    if len(text) <= keep_tail_lines:
        return None

    ts = UTCNOW().strftime("%Y%m%d-%H%M%S")
    gz = rotated / f"wal_{ts}.log.gz"

    # Write old prefix into gz
    with gzip.open(gz, "wb") as f:
        f.write(("\n".join(text[:-keep_tail_lines]) + "\n").encode("utf-8"))

    # Rewrite wal with tail
    wal.write_text("\n".join(text[-keep_tail_lines:]) + "\n", encoding="utf-8")
    return gz


def checkpoint(
    store: Any,
    *,
    state_path: str | Path = "data/goals/state.jsonl",
    wal_path: str | Path = "data/goals/wal.log",
    rotate_keep_tail: int = 5_000,
    include_steps: bool = True,
) -> Dict[str, Any]:
    """
    One-shot maintenance:
      1) Write a fresh snapshot from `store` to `state_path`.
      2) Rotate/compact the WAL at `wal_path` (keeping last N lines).

    Returns a small report dict.
    """
    state = snapshot_state(store, out_path=state_path, include_steps=include_steps, atomic=True)
    gz = rotate_wal(wal_path, keep_tail_lines=rotate_keep_tail)
    return {
        "ts": UTCNOW().isoformat(),
        "state": str(state),
        "wal_rotated": str(gz) if gz else None,
        "keep_tail_lines": rotate_keep_tail,
    }


# -----------------------------
# Internals
# -----------------------------

def _iter_goals(store: Any) -> Iterable[Goal]:
    if hasattr(store, "iter_goals"):
        return store.iter_goals()
    if hasattr(store, "list_goals"):
        return store.list_goals()
    if hasattr(store, "all"):
        return store.all()
    return []


def _iter_steps(store: Any) -> Iterable[Step]:
    if hasattr(store, "iter_steps"):
        return store.iter_steps()
    if hasattr(store, "list_steps"):
        return store.list_steps()
    if hasattr(store, "steps_for"):
        # Some stores want a goal_id; we fall back to listing all by passing None if allowed
        try:
            return store.steps_for(None)  # type: ignore[arg-type]
        except Exception:
            return []
    return []


def _jsonable(obj: Any) -> Dict[str, Any]:
    """
    Convert Goal/Step (dataclass or object) into a JSON-serializable dict.
    - Datetimes → ISO strings
    - Enums → their .name (fallback to str)
    """
    if is_dataclass(obj):
        d = asdict(obj)
    elif hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
    else:
        # Last resort: try to coerce to dict via json round-trip
        try:
            return json.loads(json.dumps(obj))  # type: ignore[arg-type]
        except Exception:
            return {"value": str(obj)}

    def conv(x: Any) -> Any:
        if isinstance(x, datetime):
            return x.replace(tzinfo=x.tzinfo or timezone.utc).isoformat()
        # enum-ish (has name attribute)
        if hasattr(x, "name") and isinstance(getattr(x, "name"), str):
            try:
                return x.name
            except Exception:
                pass
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        return x

    return {k: conv(v) for k, v in d.items()}


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]], *, append: bool = False) -> None:
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


__all__ = ["snapshot_state", "load_state", "rotate_wal", "checkpoint"]
