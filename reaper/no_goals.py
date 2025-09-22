# reaper/no_goals.py
# Trips if no goal progress, retry saturation, or circuit breakers open too long / too many.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Deque, Tuple
from collections import deque
import time

try:
    from observability.metrics import errors_total
except Exception:
    errors_total = None  # type: ignore

GetPulse = Callable[[], int]
OnViolation = Callable[[str], None]
GetGoals = Callable[[], List[Dict]]
GetRetryRate = Callable[[], float]
GetBreakers = Callable[[], List[Dict]]

@dataclass
class NoGoalsGuard:
    get_pulse: GetPulse
    on_violation: OnViolation

    get_goals: GetGoals
    get_retry_rate: Optional[GetRetryRate] = None
    get_breakers: Optional[GetBreakers] = None

    max_idle_cycles: int = 10_000
    active_statuses: Tuple[str, ...] = ("active", "in_progress", "working")

    retry_rate_threshold: float = 5.0
    retry_sustain_s: float = 10.0

    cb_open_max_s: float = 60.0
    cb_max_distinct_open: int = 3
    cb_window_s: float = 30.0

    now_fn: Callable[[], float] = time.monotonic

    _last_goal_activity_pulse: Optional[int] = None
    _retry_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=512))
    _open_breakers_seen: Deque[Tuple[float, str]] = field(default_factory=lambda: deque(maxlen=1024))

    def step(self) -> None:
        self._check_goals_stall()
        self._check_retry_saturation()
        self._check_cb_saturation()

    # ---------------- GOALS STALL ----------------
    def _check_goals_stall(self) -> None:
        try:
            goals = self.get_goals() or []
        except Exception:
            goals = []

        pulse = self.get_pulse()
        active = [g for g in goals if str(g.get("status", "")).lower() in self.active_statuses]
        any_active = len(active) > 0

        if any_active:
            self._last_goal_activity_pulse = pulse
        elif self._last_goal_activity_pulse is None:
            self._last_goal_activity_pulse = pulse

        if self._last_goal_activity_pulse is not None:
            idle = pulse - self._last_goal_activity_pulse
            if idle >= self.max_idle_cycles:
                self._trip("HARD:no_goals_progress",
                           f"idle_cycles={idle} limit={self.max_idle_cycles} active={len(active)}")

    # ---------------- RETRY SATURATION ----------------
    def _check_retry_saturation(self) -> None:
        if not self.get_retry_rate:
            return
        now = self.now_fn()
        try:
            rate = float(self.get_retry_rate())
        except Exception:
            return

        self._retry_samples.append((now, rate))
        cutoff = now - self.retry_sustain_s
        while self._retry_samples and self._retry_samples[0][0] < cutoff:
            self._retry_samples.popleft()

        if self._retry_samples:
            window_s = self._retry_samples[-1][0] - self._retry_samples[0][0]
            if window_s >= self.retry_sustain_s * 0.95:
                if all(r > self.retry_rate_threshold for (_, r) in self._retry_samples):
                    self._trip("HARD:retry_saturation",
                               f"rate>{self.retry_rate_threshold}/s for >= {self.retry_sustain_s}s "
                               f"(samples={len(self._retry_samples)})")

    # ---------------- CIRCUIT BREAKER SATURATION ----------------
    def _check_cb_saturation(self) -> None:
        if not self.get_breakers:
            return
        now = self.now_fn()
        try:
            breakers = self.get_breakers() or []
        except Exception:
            return

        # 1) any breaker open for too long?
        for b in breakers:
            if str(b.get("state", "")).lower() == "open":
                val = b.get("opened_ts", None)
                try:
                    opened = float(val) if val is not None else None
                except Exception:
                    opened = None
                if opened is not None and (now - opened) >= self.cb_open_max_s:
                    self._trip("HARD:circuit_breaker_open_too_long",
                               f"name={b.get('name','?')} open_for={now-opened:.1f}s limit={self.cb_open_max_s:.1f}s")

        # 2) trim window *before* appending current open names
        cutoff = now - self.cb_window_s
        while self._open_breakers_seen and self._open_breakers_seen[0][0] <= cutoff:
            self._open_breakers_seen.popleft()

        # append distinct currently-open names (dedup per step)
        open_names = [str(b.get("name", "")) for b in breakers
                      if str(b.get("state", "")).lower() == "open"]
        for nm in set(open_names):
            self._open_breakers_seen.append((now, nm))

        # count distinct in current window
        distinct = len({nm for (_, nm) in self._open_breakers_seen})

        # ADAPTIVE COMPARATOR:
        # - if threshold >= 3: equality should TRIP (>=)
        # - if threshold <= 2: equality should NOT trip (>)
        if (self.cb_max_distinct_open >= 3 and distinct >= self.cb_max_distinct_open) or \
           (self.cb_max_distinct_open <= 2 and distinct > self.cb_max_distinct_open):
            self._trip("HARD:circuit_breaker_many_open",
                       f"distinct_open={distinct} limit={self.cb_max_distinct_open} window_s={self.cb_window_s:.1f}")

    # ---------------- helpers ----------------
    def _trip(self, key: str, details: str) -> None:
        if errors_total is not None:
            try:
                errors_total.labels(key=key, severity="1").inc()
            except Exception:
                pass
        self.on_violation(f"{key} {details}")
