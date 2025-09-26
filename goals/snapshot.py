# goals/snapshot.py
# Snapshot and WAL rotation utilities.

from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

UTCNOW = lambda: datetime.now(timezone.utc)

def rotate_wal(
    wal_path: Union[str, Path],
    *,
    rotated_dir: Union[str, Path] = "data/goals/wal-rotated",
    keep_tail_lines: int = 5_000,
) -> Optional[Path]:
    """
    Compact the WAL by gzipping everything except the last `keep_tail_lines` lines.
    Returns the Path to the created .gz (or None if nothing rotated).
    """
    p = Path(wal_path)
    rd = Path(rotated_dir)
    rd.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        return None

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    keep_tail_lines = max(0, int(keep_tail_lines))
    if len(lines) <= keep_tail_lines:
        return None

    head = lines[:-keep_tail_lines]
    tail = lines[-keep_tail_lines:]

    ts = UTCNOW().strftime("%Y%m%dT%H%M%SZ")
    # Name like wal.<timestamp>.log.gz if source ends with .log, else <name>.<timestamp>.gz
    if p.suffix == ".log":
        gz_name = f"{p.stem}.{ts}.log.gz"
    else:
        gz_name = f"{p.name}.{ts}.gz"
    gz_path = rd / gz_name

    # Write the head to the rotated gzip
    with gzip.open(gz_path, "wt", encoding="utf-8") as zf:
        for s in head:
            zf.write(s + "\n")

    # Rewrite the active WAL with just the tail
    p.write_text(("\n".join(tail) + ("\n" if tail else "")), encoding="utf-8")

    return gz_path

__all__ = ["rotate_wal"]
