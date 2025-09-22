# goals/utils.py
# Common helpers for Orrin Goals: time/ids/paths, JSON/JSONL I/O, slugging, hashing, atomic writes, tiny thread group

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

# ---------- time ----------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    ss = s.strip()
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ss)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def human_secs(sec: float) -> str:
    sec = float(max(0, sec))
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"

# ---------- ids & hashing ----------

def short_uid(n: int = 10) -> str:
    return hashlib.sha1(f"{time.time_ns()}-{os.getpid()}".encode()).hexdigest()[:max(4, n)]

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_of_file(path: Union[str, Path], *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()

# ---------- strings / paths ----------

def slug(s: str, *, keep: str = "-_.") -> str:
    return "".join(c if c.isalnum() or c in keep else "-" for c in (s or "").lower()).strip("-")

def ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

# ---------- I/O: JSON / JSONL / text (atomic) ----------

def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return iso(obj)
    return obj

def write_json(path: Union[str, Path], data: Any, *, indent: Optional[int] = 2, atomic: bool = True) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    text = json.dumps(_jsonable(data), ensure_ascii=False, indent=indent)
    return write_text(p, text + ("\n" if not text.endswith("\n") else ""), atomic=atomic)

def read_json(path: Union[str, Path], *, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def append_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_jsonable(rec), ensure_ascii=False, separators=(",", ":")) + "\n")
    return p

def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_text(path: Union[str, Path], text: str, *, atomic: bool = True) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if not atomic:
        p.write_text(text, encoding="utf-8")
        return p
    fd, tmpname = tempfile.mkstemp(prefix="._tmp_", dir=str(p.parent))
    os.close(fd)
    tmp = Path(tmpname)
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
    finally:
        with contextlib.suppress(Exception):
            if tmp.exists():
                tmp.unlink()
    return p

def write_bytes(path: Union[str, Path], data: bytes, *, atomic: bool = True) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if not atomic:
        p.write_bytes(data)
        return p
    fd, tmpname = tempfile.mkstemp(prefix="._tmp_", dir=str(p.parent))
    os.close(fd)
    tmp = Path(tmpname)
    try:
        tmp.write_bytes(data)
        tmp.replace(p)
    finally:
        with contextlib.suppress(Exception):
            if tmp.exists():
                tmp.unlink()
    return p

# ---------- misc FS helpers ----------

@contextlib.contextmanager
def temp_dir(prefix: str = "orrin_") -> Generator[Path, None, None]:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield d
    finally:
        for root, dirs, files in os.walk(d, topdown=False):
            for name in files:
                with contextlib.suppress(Exception):
                    os.remove(Path(root) / name)
            for name in dirs:
                with contextlib.suppress(Exception):
                    os.rmdir(Path(root) / name)
        with contextlib.suppress(Exception):
            os.rmdir(d)

# ---------- small concurrency helpers ----------

class ThreadGroup:
    """
    Tiny helper to start/join named daemon threads and track exceptions.
    """
    def __init__(self) -> None:
        self._threads: List[threading.Thread] = []
        self._errors: List[str] = []

    def start(self, name: str, target, *, daemon: bool = True) -> None:
        def _runner():
            try:
                target()
            except Exception as e:
                self._errors.append(f"{name}: {type(e).__name__}: {e}")
        t = threading.Thread(target=_runner, name=name, daemon=daemon)
        t.start()
        self._threads.append(t)

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

# ---------- retry ----------

def retry(fn, *, attempts: int = 3, delay_seconds: float = 0.1, backoff: float = 2.0, retry_on: Tuple[type, ...] = (Exception,)):
    """
    Decorator-like helper:

      @retry
      def op(): ...

      @retry(attempts=5, delay_seconds=0.5)
      def op2(): ...
    """
    def _wrap(*args, **kwargs):
        d = float(delay_seconds)
        for i in range(1, int(attempts) + 1):
            try:
                return fn(*args, **kwargs)
            except retry_on as e:  # type: ignore[arg-type]
                if i >= attempts:
                    raise
                time.sleep(d)
                d *= float(backoff)
    return _wrap

__all__ = [
    "utcnow", "iso", "parse_iso", "human_secs",
    "short_uid", "sha256_of_bytes", "sha256_of_file",
    "slug", "ensure_dir",
    "write_json", "read_json", "append_jsonl", "iter_jsonl", "write_text", "write_bytes",
    "temp_dir",
    "ThreadGroup",
    "retry",
]
