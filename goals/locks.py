# goals/locks.py
# Cooperative in-process lock manager for named exclusive resources (TTL, reentrancy, simple fairness)

from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple

def _now() -> float:
    return time.monotonic()

@dataclass
class _LockState:
    holder: str
    acquired_at: float
    renew_at: float  # timestamp when we consider it stale/expired (if ttl_seconds is set)

# Grace windows
STALE_WAITER_S = 5.0    # evict a dead/stalled head waiter after this many seconds
STALE_HOLDER_S = 15.0   # if TTL=None and holder lives longer than this, head waiter may steal

class LockManager:
    """
    Minimal, thread-safe lock manager for *named* exclusive resources.
    Intended for single-process use (daemon + handlers). For multi-process, wrap with file/db-based locks.
    """

    def __init__(self, *, ttl_seconds: Optional[float] = 60.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._mu = threading.Lock()
        self._locks: Dict[str, _LockState] = {}
        self._waiters: Dict[str, List[str]] = {}               # lock name -> FIFO holder_ids
        self._waiter_enq_ts: Dict[Tuple[str, str], float] = {} # (name, holder_id) -> enqueue time

    # -------- core ops --------

    def acquire(self, name: str, holder_id: str) -> bool:
        now = _now()
        with self._mu:
            st = self._locks.get(name)

            # Lock is free
            if st is None:
                self._locks[name] = _LockState(holder=holder_id, acquired_at=now, renew_at=self._new_renew_at(now))
                self._pop_waiter_if_head_locked(name, holder_id)  # NOTE: locked variant
                return True

            # Reentrant by same holder
            if st.holder == holder_id:
                st.renew_at = self._new_renew_at(now)
                return True

            # Expired lock can be stolen
            if self._expired(st, now):
                self._locks[name] = _LockState(holder=holder_id, acquired_at=now, renew_at=self._new_renew_at(now))
                self._pop_waiter_if_head_locked(name, holder_id)  # NOTE: locked variant
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
                self._pop_waiter_if_head_locked(name, holder_id)  # NOTE: locked variant

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
                # If there's a waiter queue and we're not at head, wait our turn (and evict a stale head if needed)
                if not self._is_head_waiter(name, holder_id):
                    self._evict_stale_head_if_needed(name)  # prevent permanent stalls when the head died
                    if deadline is not None and _now() >= deadline:
                        return False
                    time.sleep(poll_interval)
                    continue

                # We are head: try to acquire
                if self.acquire(name, holder_id):
                    return True

                # Dead-holder rescue for TTL=None: if holder is stuck too long, steal it
                self._steal_if_dead_holder(name, holder_id, max_age_s=STALE_HOLDER_S)

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
                self._waiter_enq_ts[(name, holder_id)] = _now()  # track enqueue time

    def _remove_waiter(self, name: str, holder_id: str) -> None:
        with self._mu:
            q = self._waiters.get(name)
            if not q:
                self._waiter_enq_ts.pop((name, holder_id), None)
                return
            try:
                q.remove(holder_id)
            except ValueError:
                pass
            finally:
                self._waiter_enq_ts.pop((name, holder_id), None)
            if not q:
                self._waiters.pop(name, None)

    def _is_head_waiter(self, name: str, holder_id: str) -> bool:
        with self._mu:
            q = self._waiters.get(name)
            return bool(q and q[0] == holder_id)

    # NOTE: locked variant to avoid re-entrance deadlock
    def _pop_waiter_if_head_locked(self, name: str, holder_id: str) -> None:
        q = self._waiters.get(name)
        if q and q[0] == holder_id:
            q.pop(0)
            self._waiter_enq_ts.pop((name, holder_id), None)
        if q == []:
            self._waiters.pop(name, None)

    def _evict_stale_head_if_needed(self, name: str) -> None:
        """If the head waiter appears stuck (likely died), evict it after a grace window."""
        with self._mu:
            q = self._waiters.get(name)
            if not q:
                return
            head = q[0]
            ts = self._waiter_enq_ts.get((name, head))
            if ts is None:
                # No timestamp? Evict conservatively.
                q.pop(0)
                self._waiter_enq_ts.pop((name, head), None)
                if not q:
                    self._waiters.pop(name, None)
                return
            if (_now() - ts) >= STALE_WAITER_S:
                q.pop(0)
                self._waiter_enq_ts.pop((name, head), None)
                if not q:
                    self._waiters.pop(name, None)

    def _steal_if_dead_holder(self, name: str, head_holder_id: str, *, max_age_s: float) -> None:
        """
        When TTL is None (no automatic expiry) and the current holder appears dead or wedged,
        allow the HEAD WAITER to forcibly steal after max_age_s. Preserves fairness.
        """
        if self.ttl_seconds is not None:
            return  # normal TTL-based expiry/steal applies
        now = _now()
        with self._mu:
            st = self._locks.get(name)
            if st is None:
                return
            # Only the current head waiter gets to consider a steal
            q = self._waiters.get(name)
            if not q or q[0] != head_holder_id:
                return
            # If the holder is the same as head, acquire() would have succeeded. So holder is someone else.
            age = now - st.acquired_at
            if age >= float(max_age_s):
                self._locks[name] = _LockState(holder=head_holder_id, acquired_at=now, renew_at=self._new_renew_at(now))
                # We've taken the lock; drop our waiter head entry.
                q.pop(0)
                self._waiter_enq_ts.pop((name, head_holder_id), None)
                if not q:
                    self._waiters.pop(name, None)

__all__ = ["LockManager"]
