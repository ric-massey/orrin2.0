# reaper/liveness_cycle.py
# Watchdog: registered sections must be 'touched' at least once


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from functools import wraps

# optional metrics hook (safe if not present)
try:
    from observability.metrics import errors_total
except Exception:
    errors_total = None  # type: ignore

GetPulse = Callable[[], int]
OnViolation = Callable[[str], None]

DEFAULT_MAX_MISSED_CYCLES = 10_000  # <- your new global expectation

@dataclass
class _Section:
    name: str
    max_missed_cycles: int
    last_pulse_seen: Optional[int] = None
    tripped: bool = False  # one-shot until touched again

@dataclass
class LivenessByCycles:
    """
    Watchdog: registered sections must be 'touched' at least once every
    `max_missed_cycles` main-loop iterations (cycles). If not, Reaper triggers.

    Use one of:
      - .touch("name") in the code path
      - @liveness.required(name="name")    # uses DEFAULT_MAX_MISSED_CYCLES
      - @liveness.required(1234, "name")   # custom max
      - with liveness.alive("name"): ...   # auto-touch after the block

    Call .step() once per loop to enforce.
    """
    get_pulse: GetPulse
    on_violation: OnViolation
    _sections: Dict[str, _Section] = field(default_factory=dict)

    # --- register / mark / enforce ---

    def register(self, name: str, max_missed_cycles: int = DEFAULT_MAX_MISSED_CYCLES) -> None:
        self._sections[name] = _Section(name=name, max_missed_cycles=int(max_missed_cycles))

    def touch(self, name: str) -> None:
        sec = self._sections.get(name)
        if not sec:
            return
        sec.last_pulse_seen = self.get_pulse()
        if sec.tripped:
            sec.tripped = False

    def step(self) -> None:
        current = self.get_pulse()
        for sec in self._sections.values():
            if sec.last_pulse_seen is None:
                # first observation initializes baseline without tripping
                sec.last_pulse_seen = current
                continue
            missed = current - sec.last_pulse_seen
            if missed >= sec.max_missed_cycles and not sec.tripped:
                reason = (f"HARD:liveness_missed section={sec.name} "
                          f"missed_cycles={missed} limit={sec.max_missed_cycles}")
                if errors_total is not None:
                    try:
                        errors_total.labels(key="liveness_missed", severity="1").inc()
                    except Exception:
                        pass
                self.on_violation(reason)
                sec.tripped = True  # one-shot until touched again

    # --- ergonomics: decorator & context manager ---

    def required(self, max_missed_cycles: Optional[int] = None, name: Optional[str] = None):
        """
        Decorate a function that should run at least every N cycles.
        If max_missed_cycles is None, uses DEFAULT_MAX_MISSED_CYCLES (10k).
        """
        def deco(fn):
            sec_name = name or f"{fn.__module__}.{fn.__qualname__}"
            self.register(sec_name, max_missed_cycles or DEFAULT_MAX_MISSED_CYCLES)
            @wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.touch(sec_name)
            return wrapper
        return deco

    class _AliveCtx:
        def __init__(self, outer: "LivenessByCycles", name: str):
            self.outer = outer; self.name = name
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb):
            self.outer.touch(self.name)
            return False  # don't swallow exceptions

    def alive(self, name: str, *, max_missed_cycles: Optional[int] = None):
        """
        Context manager: guarantees a touch after the block.
        If name not registered, you can set max_missed_cycles here.
        """
        if name not in self._sections:
            self.register(name, max_missed_cycles or DEFAULT_MAX_MISSED_CYCLES)
        return LivenessByCycles._AliveCtx(self, name)
