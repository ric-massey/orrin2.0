# watchdogs.py
import threading
import time
from typing import Tuple, Callable, Dict, List, Optional

from reaper.reaper import Reaper, kill_current_process
from reaper.heartbeatdetector import HeartbeatDetector
from reaper.error_checker import ErrorChecker
from reaper.liveness_cycle import LivenessByCycles, DEFAULT_MAX_MISSED_CYCLES
from reaper.lifespan import LifespanByCycles
from reaper.no_goals import NoGoalsGuard
from reaper.memory import MemoryHealthGuard
from reaper.repeat import RepeatLoopGuard  # ← NEW

# Provider type hints (optional, just for clarity)
GetGoals = Callable[[], List[Dict]]
GetRetryRate = Callable[[], float]
GetBreakers = Callable[[], List[Dict]]

# Memory/FD/CPU provider hints
GetRssMb = Callable[[], float]
GetFdOpen = Callable[[], int]
GetFdLimit = Callable[[], int]
GetSockOpen = Callable[[], int]
GetSockLimit = Callable[[], int]
GetCpuUtil = Callable[[], float]
GetStepLatencyMs = Callable[[], float]
GetMemoryHealth = Callable[[], Dict[str, float | int]]  # ← NEW

class Pulse:
    """Thread-safe pulse counter that the main loop updates."""
    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self._n += 1

    def read(self) -> int:
        with self._lock:
            return self._n

def start_watchdogs(
    pulse: Pulse,
    *,
    # Heartbeat thresholds
    min_period_ms: float = 5.0,
    max_period_ms: float = 10_000.0,  # 10s slow cap
    sustain_checks_fast: int = 100,
    sustain_checks_slow: int = 10,
    window: int = 64,
    # Heartbeat/Liveness polling rate (background thread)
    hb_poll_interval_s: float = 0.010,  # 100 Hz
    # Error checker defaults
    error_window_s: float = 60.0,
    any_rate_limit: Tuple[int, float] | None = (100, 30.0),  # (count, window_s)
    per_key_limits: dict[str, Tuple[int, float]] | None = None,
    # Liveness-by-cycles (section freshness)
    liveness_max_missed_cycles: int = DEFAULT_MAX_MISSED_CYCLES,  # 10_000
    # Lifespan (random hard cutoff)
    lifespan_min_cycles: int = 25_000,
    lifespan_max_cycles: int = 30_000,
    # --------- NO-GOALS / SATURATION GUARD (providers + tunables) ---------
    goals_provider: Optional[GetGoals] = None,
    retry_rate_provider: Optional[GetRetryRate] = None,
    breakers_provider: Optional[GetBreakers] = None,
    # goals idleness (cycles)
    goals_max_idle_cycles: int = 10_000,
    # retry saturation (R/sec over T seconds)
    retry_rate_threshold: float = 5.0,
    retry_sustain_s: float = 10.0,
    # circuit breaker saturation
    cb_open_max_s: float = 60.0,
    cb_max_distinct_open: int = 3,
    cb_window_s: float = 30.0,
    # --------- MEMORY / FD / CPU GUARD (providers + tunables) ---------
    get_rss_mb: Optional[GetRssMb] = None,
    get_fd_open: Optional[GetFdOpen] = None,
    get_fd_limit: Optional[GetFdLimit] = None,
    get_sock_open: Optional[GetSockOpen] = None,
    get_sock_limit: Optional[GetSockLimit] = None,
    get_cpu_util: Optional[GetCpuUtil] = None,
    get_step_latency_ms: Optional[GetStepLatencyMs] = None,
    get_memory_health: Optional[GetMemoryHealth] = None,  # ← NEW
    # thresholds/windows
    mem_slope_mb_per_s: float = 1.0,
    mem_sustain_s: float = 30.0,
    fd_pct_threshold: float = 0.90,
    fd_sustain_s: float = 10.0,
    cpu_util_threshold: float = 0.95,
    cpu_sustain_s: float = 10.0,
    latency_slope_ms_per_s: float = 0.5,
    latency_mean_ms_threshold: float = 50.0,
    # --------- REPEAT-LOOP GUARD (tunables) ---------
    enable_repeat_guard: bool = True,
    action_window_n: int = 50,
    same_call_k: int = 4,
    same_call_t: float = 30.0,
    breaker_cool_s: float = 60.0,
    pingpong_k: int = 6,
    pingpong_t: float = 30.0,
    no_progress_t: float = 60.0,
    no_progress_min_actions: int = 20,
    retry_k: int = 5,
    retry_w: float = 30.0,
    retry_escalate_k: int = 8,
):
    """
    Spin up a daemon thread that continuously checks watchdogs.
    Returns:
      (reaper, detector, errors, liveness, lifespan, no_goals, mem_guard, repeat_guard, stop_evt)
    """
    reaper = Reaper(kill=kill_current_process)

    detector = HeartbeatDetector(
        get_pulse=pulse.read,
        on_violation=reaper.trigger,
        min_period_ms=min_period_ms,
        max_period_ms=max_period_ms,
        sustain_checks_fast=sustain_checks_fast,
        sustain_checks_slow=sustain_checks_slow,
        window=window,
    )

    errors = ErrorChecker(on_violation=reaper.trigger, window_s=error_window_s)

    if any_rate_limit:
        count, window_s = any_rate_limit
        errors.set_any_rate_limit(count=count, window_s=window_s)

    per_key_limits = per_key_limits or {}
    for key, (count, window_s) in per_key_limits.items():
        errors.set_key_rate_limit(key, count=count, window_s=window_s)

    # Liveness & lifespan
    liveness = LivenessByCycles(get_pulse=pulse.read, on_violation=reaper.trigger)

    lifespan = LifespanByCycles(
        get_pulse=pulse.read,
        on_violation=reaper.trigger,
        min_cycles=lifespan_min_cycles,
        max_cycles=lifespan_max_cycles,
    )

    # No-goals / saturation guard (optional)
    no_goals = None
    if goals_provider is not None:
        no_goals = NoGoalsGuard(
            get_pulse=pulse.read,
            on_violation=reaper.trigger,
            get_goals=goals_provider,
            get_retry_rate=retry_rate_provider,
            get_breakers=breakers_provider,
            max_idle_cycles=goals_max_idle_cycles,
            retry_rate_threshold=retry_rate_threshold,
            retry_sustain_s=retry_sustain_s,
            cb_open_max_s=cb_open_max_s,
            cb_max_distinct_open=cb_max_distinct_open,
            cb_window_s=cb_window_s,
        )

    # Memory/FD/CPU guard (optional providers; skipped if None)
    mem_guard = MemoryHealthGuard(
        on_violation=reaper.trigger,
        get_rss_mb=get_rss_mb,
        get_fd_open=get_fd_open, get_fd_limit=get_fd_limit,
        get_sock_open=get_sock_open, get_sock_limit=get_sock_limit,
        get_cpu_util=get_cpu_util, get_step_latency_ms=get_step_latency_ms,
        get_memory_health=get_memory_health,  # ← NEW
        mem_slope_mb_per_s=mem_slope_mb_per_s, mem_sustain_s=mem_sustain_s,
        fd_pct_threshold=fd_pct_threshold, fd_sustain_s=fd_sustain_s,
        cpu_util_threshold=cpu_util_threshold, cpu_sustain_s=cpu_sustain_s,
        latency_slope_ms_per_s=latency_slope_ms_per_s,
        latency_mean_ms_threshold=latency_mean_ms_threshold,
    )

    # Repeat-loop guard (optional)
    repeat_guard: Optional[RepeatLoopGuard] = None
    if enable_repeat_guard:
        repeat_guard = RepeatLoopGuard(
            on_violation=reaper.trigger,
            action_window_n=action_window_n,
            same_call_k=same_call_k,
            same_call_t=same_call_t,
            breaker_cool_s=breaker_cool_s,
            pingpong_k=pingpong_k,
            pingpong_t=pingpong_t,
            no_progress_t=no_progress_t,
            no_progress_min_actions=no_progress_min_actions,
            retry_k=retry_k,
            retry_w=retry_w,
            retry_escalate_k=retry_escalate_k,
        )

    stop_evt = threading.Event()

    def watchdog_thread():
        # background watchdog loop
        while not stop_evt.is_set():
            detector.step()
            liveness.step()
            lifespan.step()
            if no_goals is not None:
                no_goals.step()
            mem_guard.step()
            if repeat_guard is not None:
                repeat_guard.step()
            time.sleep(hb_poll_interval_s)

    t = threading.Thread(target=watchdog_thread, name="watchdogs", daemon=True)
    t.start()

    return (
        reaper,
        detector,
        errors,
        liveness,
        lifespan,
        no_goals,
        mem_guard,
        repeat_guard,
        stop_evt,
    )
