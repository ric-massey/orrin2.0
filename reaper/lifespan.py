# reaper/lifespan.py
# Watchdog: triggers Reaper when total cycles (pulse) exceed a secret random limit

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import secrets

GetPulse = Callable[[], int]
OnViolation = Callable[[str], None]

@dataclass
class LifespanByCycles:
    """
    Picks a secret random cycle limit in [min_cycles, max_cycles].
    When total cycles (pulse) reach/exceed that limit, triggers Reaper.
    """
    get_pulse: GetPulse
    on_violation: OnViolation
    min_cycles: int = 25_000
    max_cycles: int = 30_000
    _limit: Optional[int] = None

    def _ensure_limit(self) -> None:
        if self._limit is None:
            span = self.max_cycles - self.min_cycles
            # inclusive range → add 1
            r = secrets.randbelow(span + 1)
            self._limit = self.min_cycles + r

    def step(self) -> None:
        self._ensure_limit()
        n = self.get_pulse()
        if n >= (self._limit or 0):
            self.on_violation(f"HARD:lifespan_reached cycles={n} limit={self._limit}")
            # one-shot; after triggering, no need to reset (we expect shutdown)
