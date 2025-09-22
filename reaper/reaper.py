# reaper/reaper.py
# the kill switch for the main loop

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import os
import signal
import sys

# METRICS: count reaper triggers
try:
    from observability.metrics import reaper_trips_total
except Exception:  # metrics optional
    reaper_trips_total = None  # type: ignore

KillFn = Callable[[str], None]

@dataclass
class Reaper:
    kill: KillFn

    def trigger(self, reason: str) -> None:
        # Metrics
        if reaper_trips_total is not None:
            try:
                # use first token (e.g., "HARD:pulse_too_fast") to reduce label cardinality
                reaper_trips_total.labels(reason=reason.split()[0]).inc()
            except Exception:
                pass
        # Log + execute the kill behavior
        print(f"[REAPER] Shutting down: {reason}", file=sys.stderr)
        self.kill(reason)

# --- ready-to-use kill strategies ---

def kill_current_process(_: str) -> None:
    # Exit this process immediately (useful if reaper runs inside main loop proc)
    os._exit(1)

def signal_pid(pid: int, sig: int = signal.SIGTERM) -> KillFn:
    # Return a function that signals another process by PID
    def _kill(reason: str) -> None:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            print(f"[REAPER] PID {pid} not found", file=sys.stderr)
    return _kill
