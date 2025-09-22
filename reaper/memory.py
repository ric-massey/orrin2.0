# reaper/memory.py
# Resource & subsystem health watchdog. Trips Reaper if any sustained condition holds.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple, Dict, Any
from collections import deque
import time

# Optional metrics (safe if missing)
try:
    from observability.metrics import errors_total
except Exception:
    errors_total = None  # type: ignore

OnViolation = Callable[[str], None]
NowFn = Callable[[], float]

# Providers you pass from your app (existing):
GetRssMb = Callable[[], float]                        # Resident set size in MB
GetFdOpen = Callable[[], int]                         # current open FDs
GetFdLimit = Callable[[], int]                        # hard or soft limit for FDs
GetSockOpen = Callable[[], int]                       # optional, else reuse FDs
GetSockLimit = Callable[[], int]                      # optional
GetCpuUtil = Callable[[], float]                      # 0..1 fraction (or 0..100; we normalize)
GetStepLatencyMs = Callable[[], float]                # recent loop/step latency (ms)

# New (optional) provider that "looks at the memory file" (i.e., your memory subsystem):
# Return a lightweight dict of signals; keys are best-effort and optional.
# Example producer is shown below the class.
GetMemoryHealth = Callable[[], Dict[str, Any]]


@dataclass
class MemoryHealthGuard:
    """
    Resource & subsystem watchdog. Trips Reaper if any sustained condition holds:

      A) Generic process signals (what you already had):
         1) Memory slope leak: d(RSS_MB)/dt > mem_slope_mb_per_s for ≥ mem_sustain_s seconds.
         2) FD/socket pressure: open/limit > fd_pct_threshold for ≥ fd_sustain_s seconds.
         3) CPU starvation: cpu_util sustained high AND (latency slope rising OR mean high).

      B) Orrin Memory subsystem (optional — if you pass get_memory_health()):
         4) Index lag sustained high (items with missing vectors).
         5) Working cache sustained above a threshold (compaction not keeping up).
         6) Compaction stale for too long (no promotion/summary activity).
         7) Vector bytes over a limit (backends running large).
         8) WAL write failures rising too quickly or total over a limit.

    All checks are rolling windows except compaction-stale / bytes-limit which are thresholded.
    """
    on_violation: OnViolation
    now_fn: NowFn = time.monotonic

    # ---- providers (pass the ones you can measure) ----
    get_rss_mb: Optional[GetRssMb] = None
    get_fd_open: Optional[GetFdOpen] = None
    get_fd_limit: Optional[GetFdLimit] = None
    get_sock_open: Optional[GetSockOpen] = None
    get_sock_limit: Optional[GetSockLimit] = None
    get_cpu_util: Optional[GetCpuUtil] = None
    get_step_latency_ms: Optional[GetStepLatencyMs] = None

    # New: memory subsystem provider
    get_memory_health: Optional[GetMemoryHealth] = None

    # ---- thresholds / windows (existing) ----
    mem_slope_mb_per_s: float = 1.0
    mem_sustain_s: float = 30.0

    fd_pct_threshold: float = 0.90
    fd_sustain_s: float = 10.0

    cpu_util_threshold: float = 0.95         # 95%+
    cpu_sustain_s: float = 10.0
    latency_slope_ms_per_s: float = 0.5      # rising latency >= 0.5 ms/sec
    latency_mean_ms_threshold: float = 50.0  # or mean >= 50 ms

    # ---- thresholds / windows (memory subsystem) ----
    idx_lag_threshold: int = 200             # items with embeddings pending
    idx_lag_sustain_s: float = 20.0

    working_cache_threshold: int = 2000      # items stuck in working cache
    working_cache_sustain_s: float = 30.0

    compaction_stale_s: float = 60.0 * 60.0  # 1 hour without compaction

    vector_bytes_limit: int = 512 * 1024 * 1024  # 512 MB across vectors

    wal_error_rate_per_min: float = 5.0      # failures/min considered bad
    wal_error_sustain_s: float = 60.0
    wal_error_total_limit: int = 100         # hard cap

    # ---- internal rolling buffers ----
    _mem_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1024))  # (t, rss_mb)
    _fd_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1024))   # (t, pct_open)
    _sock_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1024)) # (t, pct_open)
    _cpu_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))  # (t, util 0..1)
    _lat_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))  # (t, ms)

    # memory subsystem rolling buffers
    _idxlag_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))      # (t, lag)
    _workcache_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))   # (t, working_cache)
    _vector_bytes_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))# (t, bytes)
    _wal_fail_samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))    # (t, failures_total)
    _last_compaction_ts: Optional[float] = None

    def step(self) -> None:
        """Call this periodically (e.g., in your watchdog thread)."""
        now = self.now_fn()

        # --- collect generic process samples ---
        if self.get_rss_mb:
            try:
                self._mem_samples.append((now, float(self.get_rss_mb())))
            except Exception:
                pass

        if self.get_fd_open and self.get_fd_limit:
            try:
                open_fd = float(self.get_fd_open())
                lim_fd = float(self.get_fd_limit())
                pct = (open_fd / lim_fd) if lim_fd > 0 else 0.0
                self._fd_samples.append((now, pct))
            except Exception:
                pass

        if self.get_sock_open and self.get_sock_limit:
            try:
                open_sk = float(self.get_sock_open())
                lim_sk = float(self.get_sock_limit())
                pct = (open_sk / lim_sk) if lim_sk > 0 else 0.0
                self._sock_samples.append((now, pct))
            except Exception:
                pass

        if self.get_cpu_util:
            try:
                util = float(self.get_cpu_util())
                if util > 1.5:   # normalize 0..100 -> 0..1 if needed
                    util = util / 100.0
                self._cpu_samples.append((now, min(max(util, 0.0), 1.0)))
            except Exception:
                pass

        if self.get_step_latency_ms:
            try:
                self._lat_samples.append((now, float(self.get_step_latency_ms())))
            except Exception:
                pass

        # --- collect memory subsystem samples (optional) ---
        if self.get_memory_health:
            try:
                h = self.get_memory_health() or {}
                sig = h.get("signals", h) or {}

                # Prefer explicit keys, but be forgiving:
                idx_lag = float(sig.get("memory.index_lag", sig.get("index_lag", 0)))
                self._idxlag_samples.append((now, idx_lag))

                working_cache = float(sig.get("memory.working_cache", sig.get("working_cache", 0)))
                self._workcache_samples.append((now, working_cache))

                vbytes = float(sig.get("memory.vectors.bytes", sig.get("vector_bytes_total", 0)))
                self._vector_bytes_samples.append((now, vbytes))

                # Compaction last timestamp (epoch seconds)
                lcts = sig.get("memory.last_compaction_ts", sig.get("last_compaction_ts"))
                if isinstance(lcts, (int, float)) and lcts > 0:
                    self._last_compaction_ts = float(lcts)

                # WAL failures (monotonic counter)
                wal_fail = float(sig.get("memory.wal.write_failures", sig.get("wal_write_failures", 0)))
                self._wal_fail_samples.append((now, wal_fail))
            except Exception:
                # do not break the watchdog if snapshot fails
                pass

        # --- evaluate everything ---
        self._check_memory(now)
        self._check_fd_socket(now)
        self._check_cpu_starvation(now)
        self._check_memory_subsystem(now)

    # ------------------- checks (generic) -------------------

    def _check_memory(self, now: float) -> None:
        if not self._mem_samples:
            return
        self._trim(self._mem_samples, now - self.mem_sustain_s)
        if self._window_ok(self._mem_samples, self.mem_sustain_s):
            slope = self._slope(self._mem_samples)  # MB / sec
            if slope is not None and slope > self.mem_slope_mb_per_s:
                self._trip("HARD:memory_leak_slope",
                           f"slope={slope:.3f} MB/s limit={self.mem_slope_mb_per_s:.3f} sustain={self.mem_sustain_s:.1f}s")

    def _check_fd_socket(self, now: float) -> None:
        # FDs
        if self._fd_samples:
            self._trim(self._fd_samples, now - self.fd_sustain_s)
            if self._window_ok(self._fd_samples, self.fd_sustain_s):
                if all(pct > self.fd_pct_threshold for _, pct in self._fd_samples):
                    last = self._fd_samples[-1][1]
                    self._trip("HARD:fd_pressure",
                               f"pct={last:.2%} threshold={self.fd_pct_threshold:.2%} sustain={self.fd_sustain_s:.1f}s")

        # Sockets (optional)
        if self._sock_samples:
            self._trim(self._sock_samples, now - self.fd_sustain_s)
            if self._window_ok(self._sock_samples, self.fd_sustain_s):
                if all(pct > self.fd_pct_threshold for _, pct in self._sock_samples):
                    last = self._sock_samples[-1][1]
                    self._trip("HARD:socket_pressure",
                               f"pct={last:.2%} threshold={self.fd_pct_threshold:.2%} sustain={self.fd_sustain_s:.1f}s")

    def _check_cpu_starvation(self, now: float) -> None:
        if not self._cpu_samples:
            return
        # Trim both CPU and latency with the same sustain window
        self._trim(self._cpu_samples, now - self.cpu_sustain_s)
        if self._lat_samples:
            self._trim(self._lat_samples, now - self.cpu_sustain_s)

        # Require CPU sustained high
        if not self._window_ok(self._cpu_samples, self.cpu_sustain_s):
            return
        if not all(util >= self.cpu_util_threshold for _, util in self._cpu_samples):
            return

        # Then require latency either rising (slope) OR high mean
        lat_ok = False
        if self._lat_samples and self._window_ok(self._lat_samples, self.cpu_sustain_s):
            lat_slope = self._slope(self._lat_samples) or 0.0  # ms / sec
            lat_mean = sum(v for _, v in self._lat_samples) / max(1, len(self._lat_samples))
            if lat_slope >= self.latency_slope_ms_per_s or lat_mean >= self.latency_mean_ms_threshold:
                lat_ok = True

        if lat_ok:
            util_mean = sum(u for _, u in self._cpu_samples) / max(1, len(self._cpu_samples))
            self._trip("HARD:cpu_starvation",
                       (f"cpu_mean={util_mean:.2%}≥{self.cpu_util_threshold:.2%} "
                        f"lat_slope≥{self.latency_slope_ms_per_s:.2f}ms/s or "
                        f"lat_mean≥{self.latency_mean_ms_threshold:.1f}ms "
                        f"sustain={self.cpu_sustain_s:.1f}s"))

    # ------------------- checks (memory subsystem) -------------------

    def _check_memory_subsystem(self, now: float) -> None:
        if not self.get_memory_health:
            return

        # Index lag sustained high
        if self._idxlag_samples:
            self._trim(self._idxlag_samples, now - self.idx_lag_sustain_s)
            if self._window_ok(self._idxlag_samples, self.idx_lag_sustain_s):
                if all(v > float(self.idx_lag_threshold) for _, v in self._idxlag_samples):
                    last = self._idxlag_samples[-1][1]
                    self._trip("HARD:mem_index_lag",
                               f"lag={last:.0f} threshold>{self.idx_lag_threshold} sustain={self.idx_lag_sustain_s:.1f}s")

        # Working cache sustained high
        if self._workcache_samples:
            self._trim(self._workcache_samples, now - self.working_cache_sustain_s)
            if self._window_ok(self._workcache_samples, self.working_cache_sustain_s):
                if all(v > float(self.working_cache_threshold) for _, v in self._workcache_samples):
                    last = self._workcache_samples[-1][1]
                    self._trip("HARD:mem_working_cache_pressure",
                               f"size={last:.0f} threshold>{self.working_cache_threshold} sustain={self.working_cache_sustain_s:.1f}s")

        # Compaction stale (threshold check, no sustain window)
        if self._last_compaction_ts:
            age = max(0.0, now - float(self._last_compaction_ts))
            if age >= self.compaction_stale_s:
                self._trip("HARD:mem_compaction_stale",
                           f"age={age:.0f}s limit={self.compaction_stale_s:.0f}s")

        # Vector bytes limit (threshold check)
        if self._vector_bytes_samples:
            vb = self._vector_bytes_samples[-1][1]
            if vb > float(self.vector_bytes_limit):
                self._trip("HARD:mem_vector_bytes_limit",
                           f"bytes={int(vb)} limit={int(self.vector_bytes_limit)}")

        # WAL failures (rate or total)
        if len(self._wal_fail_samples) >= 2:
            self._trim(self._wal_fail_samples, now - self.wal_error_sustain_s)
            if self._window_ok(self._wal_fail_samples, self.wal_error_sustain_s):
                first_t, first_v = self._wal_fail_samples[0]
                last_t, last_v = self._wal_fail_samples[-1]
                dt = max(1e-6, last_t - first_t)
                rate_per_min = ((last_v - first_v) / dt) * 60.0
                if rate_per_min >= self.wal_error_rate_per_min or last_v >= self.wal_error_total_limit:
                    self._trip("HARD:mem_wal_write_failures",
                               f"rate={rate_per_min:.1f}/min limit={self.wal_error_rate_per_min:.1f}/min total={last_v:.0f} max_total={self.wal_error_total_limit}")

    # ------------------- helpers -------------------

    @staticmethod
    def _trim(dq: Deque[Tuple[float, float]], cutoff: float) -> None:
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    @staticmethod
    def _window_ok(dq: Deque[Tuple[float, float]], need_s: float) -> bool:
        if len(dq) < 2:
            return False
        span = dq[-1][0] - dq[0][0]
        return span >= need_s * 0.95  # slack against sampling jitter

    @staticmethod
    def _slope(dq: Deque[Tuple[float, float]]) -> Optional[float]:
        # simple least-squares slope over (t, v)
        n = len(dq)
        if n < 2:
            return None
        sx = sy = sxx = sxy = 0.0
        for t, v in dq:
            sx += t; sy += v
            sxx += t * t; sxy += t * v
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return None
        return (n * sxy - sx * sy) / denom

    def _trip(self, key: str, details: str) -> None:
        if errors_total is not None:
            try:
                errors_total.labels(key=key, severity="1").inc()
            except Exception:
                pass
        self.on_violation(f"{key} {details}")
