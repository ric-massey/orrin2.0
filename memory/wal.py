# memory/wal.py
# Append-only Write-Ahead Log (WAL) for Orrin2.0 memory: JSONL logs for Events and MemoryItems, safe rotation, replay helpers, and basic stats.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Dict, Any, Tuple
import json
import gzip
import os
import io
import time
import threading
import uuid
from datetime import datetime, timezone

from .config import MEMCFG
from .models import Event, MemoryItem


ISO = "%Y-%m-%dT%H:%M:%SZ"
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _json_dumps(obj: Any) -> str:
    # Compact but readable enough; preserve Unicode.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0

def _sanitize_event_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop heavy/transient fields before logging (e.g., precomputed vectors)."""
    if not isinstance(meta, dict):
        return {}
    out = dict(meta)
    out.pop("_vec", None)  # large float arrays, carried only in-memory
    return out


@dataclass
class WalStats:
    events_written: int = 0
    items_written: int = 0
    rotate_events: int = 0
    rotate_items: int = 0
    write_failures: int = 0


class WAL:
    """
    Simple, thread-safe JSONL write-ahead log. Two streams:
      - events.jsonl  : raw ingest events (sanitized)
      - items.jsonl   : MemoryItem upserts (no vectors; just metadata)

    Rotation is size-based; rotated files optionally gzipped.
    """

    def __init__(
        self,
        events_path: Path = MEMCFG.WAL_EVENTS_PATH,
        items_path: Path = MEMCFG.WAL_ITEMS_PATH,
        *,
        max_bytes: int = 64 * 1024 * 1024,   # 64 MB per file
        gzip_rotate: bool = True,
        fsync_every: bool = False,
    ):
        self.events_path = Path(events_path)
        self.items_path = Path(items_path)
        self.max_bytes = int(max_bytes)
        self.gzip_rotate = bool(gzip_rotate)
        self.fsync_every = bool(fsync_every)

        _ensure_parent(self.events_path)
        _ensure_parent(self.items_path)

        self._lock_ev = threading.RLock()
        self._lock_it = threading.RLock()

        self._fh_ev = open(self.events_path, "a", encoding="utf-8", buffering=1)
        self._fh_it = open(self.items_path, "a", encoding="utf-8", buffering=1)

        self.stats = WalStats()

    # -------------------- Public API --------------------

    def append_event(self, ev: Event) -> None:
        """
        Append a sanitized Event record to events.jsonl.
        Adds a generated 'id' and a 'ts' if not present in meta.
        """
        rec = {
            "id": _gen_id("ev"),
            "ts": _now_iso(),
            "kind": ev.kind,
            "content": ev.content or "",
            "meta": _sanitize_event_meta(ev.meta),
        }
        line = _json_dumps(rec) + "\n"

        with self._lock_ev:
            try:
                self._maybe_rotate(self.events_path, self._fh_ev, which="events", next_line_bytes=len(line))
                self._fh_ev.write(line)
                if self.fsync_every:
                    self._fh_ev.flush()
                    os.fsync(self._fh_ev.fileno())
                self.stats.events_written += 1
            except Exception:
                self.stats.write_failures += 1

    def append_items(self, items: Iterable[MemoryItem]) -> int:
        """
        Append MemoryItem upserts to items.jsonl (one line per item).
        Uses MemoryItem.to_dict(); vectors are not included by design.
        Returns the number of items written.
        """
        n = 0
        with self._lock_it:
            for it in items:
                try:
                    rec = it.to_dict()
                    line = _json_dumps(rec) + "\n"
                    self._maybe_rotate(self.items_path, self._fh_it, which="items", next_line_bytes=len(line))
                    self._fh_it.write(line)
                    n += 1
                except Exception:
                    self.stats.write_failures += 1
            try:
                if n and self.fsync_every:
                    self._fh_it.flush()
                    os.fsync(self._fh_it.fileno())
            except Exception:
                self.stats.write_failures += 1

        self.stats.items_written += n
        return n

    def flush(self) -> None:
        """Flush both streams."""
        try:
            with self._lock_ev:
                self._fh_ev.flush()
                if self.fsync_every:
                    os.fsync(self._fh_ev.fileno())
        except Exception:
            self.stats.write_failures += 1
        try:
            with self._lock_it:
                self._fh_it.flush()
                if self.fsync_every:
                    os.fsync(self._fh_it.fileno())
        except Exception:
            self.stats.write_failures += 1

    def close(self) -> None:
        """Close file handles (idempotent)."""
        try:
            with self._lock_ev:
                try:
                    self._fh_ev.flush()
                finally:
                    self._fh_ev.close()
        except Exception:
            pass
        try:
            with self._lock_it:
                try:
                    self._fh_it.flush()
                finally:
                    self._fh_it.close()
        except Exception:
            pass

    # ---------- Replay (read) helpers ----------

    @staticmethod
    def replay_events(path: Path) -> Iterator[Dict[str, Any]]:
        """Yield raw dict records from an events.jsonl (or rotated .gz)."""
        p = Path(path)
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    @staticmethod
    def replay_items(path: Path) -> Iterator[Dict[str, Any]]:
        """Yield raw dict records from an items.jsonl (or rotated .gz)."""
        return WAL.replay_events(path)  # same line format; reuse

    @staticmethod
    def record_to_event(rec: Dict[str, Any]) -> Event:
        """Convert an events.jsonl record back to an Event."""
        kind = str(rec.get("kind") or "unknown")
        content = str(rec.get("content") or "")
        meta = dict(rec.get("meta") or {})
        return Event(kind=kind, content=content, meta=meta)

    @staticmethod
    def record_to_item(rec: Dict[str, Any]) -> MemoryItem:
        """
        Convert an items.jsonl record to a MemoryItem.
        Assumes record keys match MemoryItem fields (as produced by to_dict()).
        """
        # to_dict keys match dataclass fields; unpack directly
        return MemoryItem(**rec)  # type: ignore[arg-type]

    # -------------------- Internal helpers --------------------

    def _maybe_rotate(self, path: Path, fh, *, which: str, next_line_bytes: int) -> None:
        """
        If current file size plus next_line_bytes exceeds max_bytes, rotate:
          path -> path.TIMESTAMP
          (optional) gzip the rotated file and remove plain text
        """
        try:
            cur_size = _size(path)
            if cur_size + int(next_line_bytes) <= self.max_bytes:
                return
        except Exception:
            # If we cannot stat, attempt rotation anyway
            pass

        # Close current handle
        try:
            fh.flush()
        except Exception:
            self.stats.write_failures += 1
        try:
            if which == "events":
                self._fh_ev.close()
            else:
                self._fh_it.close()
        except Exception:
            pass

        # Prepare rotated filename with UTC timestamp
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rotated = path.with_name(f"{path.stem}.{ts}{path.suffix}")
        try:
            os.replace(str(path), str(rotated))  # atomic if same filesystem
        except FileNotFoundError:
            # Nothing to rotate; continue
            pass
        except Exception:
            self.stats.write_failures += 1

        # Optional gzip compression
        if self.gzip_rotate and rotated.exists():
            try:
                gz = rotated.with_suffix(rotated.suffix + ".gz")
                with open(rotated, "rb") as src, gzip.open(gz, "wb") as dst:
                    # stream copy
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(chunk)
                rotated.unlink(missing_ok=True)  # py3.8+: ok param added in 3.8
            except Exception:
                self.stats.write_failures += 1

        # Reopen a fresh file handle
        try:
            if which == "events":
                self._fh_ev = open(path, "a", encoding="utf-8", buffering=1)
                self.stats.rotate_events += 1
            else:
                self._fh_it = open(path, "a", encoding="utf-8", buffering=1)
                self.stats.rotate_items += 1
        except Exception:
            self.stats.write_failures += 1


# -------------------- Module-level singleton (optional) --------------------

# Create a convenient global WAL instance you can import and use directly.
DEFAULT_WAL = WAL()

def append_event(ev: Event) -> None:
    DEFAULT_WAL.append_event(ev)

def append_items(items: Iterable[MemoryItem]) -> int:
    return DEFAULT_WAL.append_items(items)

def flush() -> None:
    DEFAULT_WAL.flush()

def stats() -> WalStats:
    return DEFAULT_WAL.stats
