# reaper/error_checker.py
# Tracks error events, applies thresholds and rate limits, triggers Reaper on violations.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional, Tuple
from collections import defaultdict, deque
import time
from .errors import ErrorEvent

# METRICS: errors and trips
try:
    from observability.metrics import (
        errors_total, error_rate_trips_total, error_threshold_trips_total
    )
except Exception:
    errors_total = error_rate_trips_total = error_threshold_trips_total = None  # type: ignore

OnViolation = Callable[[str], None]

@dataclass
class ErrorChecker:
    on_violation: OnViolation
    thresholds: Dict[int, int] = field(default_factory=lambda: {1: 10, 2: 25, 3: 50})
    window_s: Optional[float] = None
    now_fn: Callable[[], float] = time.monotonic

    any_rate_count: Optional[int] = None
    any_rate_window_s: Optional[float] = None
    per_key_rate_limits: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    _events: Dict[Tuple[str, int], Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _any_events: Deque[float] = field(default_factory=deque)
    _per_key_events: Dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def observe(self, event: ErrorEvent, *, details: Optional[str] = None) -> None:
        key = getattr(event, "key", None) or "unknown_error"
        sev = int(getattr(event, "severity", 1) or 1)
        if sev not in self.thresholds:
            sev = 1

        now = self.now_fn()

        # metrics: count every observed error
        if errors_total is not None:
            try: errors_total.labels(key=key, severity=str(sev)).inc()
            except Exception: pass

        # --- Repetition thresholds ---
        dq = self._events[(key, sev)]
        dq.append(now)
        if self.window_s is not None:
            cutoff = now - self.window_s
            while dq and dq[0] < cutoff:
                dq.popleft()

        limit = int(self.thresholds[sev])
        if len(dq) >= limit:
            msg = f"HARD:error_threshold key={key} sev={sev} count={len(dq)}"
            if self.window_s is not None:
                msg += f" window_s={self.window_s:.1f}"
            if details:
                msg += f" details={details}"
            self.on_violation(msg)
            # metrics: threshold trip
            if error_threshold_trips_total is not None:
                try: error_threshold_trips_total.labels(key=key, severity=str(sev)).inc()
                except Exception: pass
            dq.clear()

        # --- Global ANY rate limit ---
        if self.any_rate_count and self.any_rate_window_s:
            self._any_events.append(now)
            cutoff_any = now - self.any_rate_window_s
            while self._any_events and self._any_events[0] < cutoff_any:
                self._any_events.popleft()
            if len(self._any_events) >= self.any_rate_count:
                msg = (f"HARD:error_rate_limit scope=any "
                       f"count={len(self._any_events)} window_s={self.any_rate_window_s:.1f}")
                if details:
                    msg += f" details={details}"
                self.on_violation(msg)
                if error_rate_trips_total is not None:
                    try: error_rate_trips_total.labels(scope="any", key="").inc()
                    except Exception: pass
                self._any_events.clear()

        # --- Per-key rate limit ---
        if key in self.per_key_rate_limits:
            k_count, k_window = self.per_key_rate_limits[key]
            kdq = self._per_key_events[key]
            kdq.append(now)
            cutoff_k = now - k_window
            while kdq and kdq[0] < cutoff_k:
                kdq.popleft()
            if len(kdq) >= k_count:
                msg = (f"HARD:error_rate_limit scope=key key={key} "
                       f"count={len(kdq)} window_s={k_window:.1f}")
                if details:
                    msg += f" details={details}"
                self.on_violation(msg)
                if error_rate_trips_total is not None:
                    try: error_rate_trips_total.labels(scope="key", key=key).inc()
                    except Exception: pass
                kdq.clear()

    def report(self, key: str, severity: int, *, details: Optional[str] = None) -> None:
        class _Evt:
            __slots__ = ("key", "severity")
            def __init__(self, k, s): self.key, self.severity = k, s
        self.observe(_Evt(key, severity), details=details)

    def set_any_rate_limit(self, count: int, window_s: float) -> None:
        self.any_rate_count = int(count)
        self.any_rate_window_s = float(window_s)

    def set_key_rate_limit(self, key: str, count: int, window_s: float) -> None:
        self.per_key_rate_limits[key] = (int(count), float(window_s))

    def clear(self, key: Optional[str] = None, severity: Optional[int] = None) -> None:
        if key is None and severity is None:
            self._events.clear(); self._any_events.clear(); self._per_key_events.clear()
            return
        if key is not None and severity is None:
            for (k, s) in list(self._events.keys()):
                if k == key:
                    self._events.pop((k, s), None)
            self._per_key_events.pop(key, None)
            return
        if key is None and severity is not None:
            for (k, s) in list(self._events.keys()):
                if s == severity:
                    self._events.pop((k, s), None)
            return
        self._events.pop((key, severity), None)
