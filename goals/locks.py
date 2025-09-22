# goals/locks.py
# Cooperative in-process lock manager for named exclusive resources (TTL, reentrancy, simple fairness)

from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, List, Any


def _now() -> float:
    return time.monotonic()


@dataclass
class _LockState:
    holder: str
    acquired_at: float
    renew_at: float  # timestamp when we consider it stale/expired (if ttl_seconds is set)


class LockManager:
    """
    Minimal, thread-safe lock manager for *named* exclusive resources.
    Intended for single-process use (daemon + handlers). For multi-process, wrap with file/db-based locks.

    Semantics:
      - acquire(name, holder_id) -> bool
          True if lock was obtained or is already held by the same holder (reentrant).
          If a TTL is configured and the existing lock is expired, we steal it.

      - release(name, holder_id) -> None
          Idempotent: only the current holder (or an expired one) will clear the lock.

      - renew(name, holder_id) -> bool
          Extend TTL if you still hold the lock. Returns False if not the holder.

      - acquire_blocking(name, holder_id, timeout=None, poll_interval=0.05) -> bool
          Spin-waits until the lock is acquired or timeout elapses.

      - session(name, holder_id):
          Context manager that acquires then releases, renewing automatically between long steps is up to caller.

    Notes:
      - Fairness: best-effort via FIFO waiters list; non-blocking acquire() does not reorder, but blocking acquire
        will append to waiters and check turn order. TTL helps avoid dead holders.
      - TTL: if ttl_seconds is None, locks don't expire automatically.
    """

    def __init__(self, *, ttl_seconds: Optional[float] = 60.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._mu = threading.Lock()
        self._locks: Dict[str, _LockState] = {}
        self._waiters: Dict[str, List[str]] = {}  # lock name -> list of holder_ids (FIFO expectation)

    # -------- core ops --------

    def acquire(self, name: str, holder_id: str) -> bool:
        now = _now()
        with self._mu:
            st = self._locks.get(name)

            # Lock is free
            if st is None:
                self._locks[name] = _LockState(holder=holder_id, acquired_at=now, renew_at=self._new_renew_at(now))
                self._pop_waiter_if_head(name, holder_id)  # claim our spot if present
                return True

            # Reentrant by same holder
            if st.holder == holder_id:
                st.renew_at = self._new_renew_at(now)
                return True

            # Expired lock can be stolen
            if self._expired(st, now):
                self._locks[name] = _LockState(holder=holder_id, acquired_at=now, renew_at=self._new_renew_at(now))
                self._pop_waiter_if_head(name, holder_id)
                return True

            # Respect FIFO if we are not at the head (for blocking callers)
            # Non-blocking path: just fail
            return False

    def release(self, name: str, holder_id: str) -> None:
        now = _now()
        with self._mu:
            st = self._locks.get(name)
            if st is None:
                return
            if st.holder == holder_id or self._expired(st, now):
                self._locks.pop(name, None)
                self._pop_waiter_if_head(name, holder_id)

    def renew(self, name: str, holder_id: str) -> bool:
        now = _now()
        with self._mu:
            st = self._locks.get(name)
            if st and st.holder == holder_id:
                st.renew_at = self._new_renew_at(now)
                return True
            return False

    # -------- blocking acquire with simple fairness --------

    def acquire_blocking(
        self,
        name: str,
        holder_id: str,
        *,
        timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> bool:
        deadline = None if timeout is None else (_now() + max(0.0, timeout))
        self._enqueue_waiter(name, holder_id)
        try:
            while True:
                # If there's a waiter queue and we're not at head, wait our turn
                if not self._is_head_waiter(name, holder_id):
                    if deadline is not None and _now() >= deadline:
                        return False
                    time.sleep(poll_interval)
                    continue

                if self.acquire(name, holder_id):
                    return True

                if deadline is not None and _now() >= deadline:
                    return False
                time.sleep(poll_interval)
        finally:
            # If we exit without acquiring, ensure we remove our waiter entry
            if not self.is_held_by(name, holder_id):
                self._remove_waiter(name, holder_id)

    # -------- queries & admin --------

    def is_held(self, name: str) -> bool:
        with self._mu:
            st = self._locks.get(name)
            return bool(st and not self._expired(st, _now()))

    def is_held_by(self, name: str, holder_id: str) -> bool:
        with self._mu:
            st = self._locks.get(name)
            return bool(st and st.holder == holder_id and not self._expired(st, _now()))

    def held_by(self, name: str) -> Optional[str]:
        with self._mu:
            st = self._locks.get(name)
            if st and not self._expired(st, _now()):
                return st.holder
            return None

    def owned(self, holder_id: str) -> List[str]:
        now = _now()
        with self._mu:
            return [n for n, st in self._locks.items() if st.holder == holder_id and not self._expired(st, now)]

    def cleanup(self) -> int:
        """Remove expired locks. Returns number of locks cleared."""
        now = _now()
        cleared = 0
        with self._mu:
            for n, st in list(self._locks.items()):
                if self._expired(st, now):
                    self._locks.pop(n, None)
                    cleared += 1
        return cleared

    def force_release(self, name: str) -> None:
        with self._mu:
            self._locks.pop(name, None)
            self._waiters.pop(name, None)

    def health(self) -> Dict[str, Any]:
        now = _now()
        with self._mu:
            active = {n: {"holder": st.holder, "age_sec": round(now - st.acquired_at, 3)} for n, st in self._locks.items() if not self._expired(st, now)}
            waiters = {n: list(q) for n, q in self._waiters.items() if q}
        return {"active": active, "waiters": waiters, "ttl_seconds": self.ttl_seconds}

    # -------- context manager --------

    @contextmanager
    def session(self, name: str, holder_id: str, *, timeout: Optional[float] = None, poll_interval: float = 0.05):
        ok = self.acquire_blocking(name, holder_id, timeout=timeout, poll_interval=poll_interval)
        if not ok:
            raise TimeoutError(f"timeout acquiring lock '{name}' for holder '{holder_id}'")
        try:
            yield
        finally:
            self.release(name, holder_id)

    # -------- internals --------

    def _expired(self, st: _LockState, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return now >= st.renew_at

    def _new_renew_at(self, now: float) -> float:
        if self.ttl_seconds is None:
            # Far future
            return now + 10**9
        return now + float(self.ttl_seconds)

    def _enqueue_waiter(self, name: str, holder_id: str) -> None:
        with self._mu:
            q = self._waiters.setdefault(name, [])
            if holder_id not in q:
                q.append(holder_id)

    def _remove_waiter(self, name: str, holder_id: str) -> None:
        with self._mu:
            q = self._waiters.get(name)
            if not q:
                return
            try:
                q.remove(holder_id)
            except ValueError:
                pass
            if not q:
                self._waiters.pop(name, None)

    def _is_head_waiter(self, name: str, holder_id: str) -> bool:
        with self._mu:
            q = self._waiters.get(name)
            return bool(q and q[0] == holder_id)

    def _pop_waiter_if_head(self, name: str, holder_id: str) -> None:
        with self._mu:
            q = self._waiters.get(name)
            if q and q[0] == holder_id:
                q.pop(0)
            if q == []:
                self._waiters.pop(name, None)


__all__ = ["LockManager"]
