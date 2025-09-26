# observability/nervous_system.py
from __future__ import annotations
import time, threading, math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Dict, Any, Callable

try:
    from memory.models import Event
except Exception:
    Event = None  # type: ignore

def _clip01(x: float) -> float:
    return 0.0 if math.isnan(x) else max(0.0, min(1.0, x))

def _ewma(prev: float, x: float, alpha: float) -> float:
    return (1 - alpha) * prev + alpha * x

@dataclass
class HealthSample:
    ts: float
    # raw
    cpu_util: float = 0.0         # 0..1
    rss_mb: float = 0.0
    hb_avg_period_ms: float = 0.0
    errors_any_rate: float = 0.0
    index_lag: float = 0.0
    working_cache: float = 0.0
    # scaled 0..1
    cpu_s: float = 0.0
    rss_s: float = 0.0
    hb_s: float = 0.0
    err_s: float = 0.0
    idx_s: float = 0.0
    wcache_s: float = 0.0
    # ewm
    cpu_ewm: float = 0.0
    hb_ewm: float = 0.0
    err_ewm: float = 0.0

class HealthBus:
    def __init__(self, maxlen: int = 1200, alpha: float = 0.25):
        self.buf: Deque[HealthSample] = deque(maxlen=maxlen)  # ~4 min @5Hz
        self.alpha = float(alpha)
        self._lock = threading.Lock()

    def add(self, s: HealthSample) -> None:
        with self._lock:
            if self.buf:
                p = self.buf[-1]
                s.cpu_ewm = _ewma(p.cpu_ewm, s.cpu_s, self.alpha)
                s.hb_ewm  = _ewma(p.hb_ewm,  s.hb_s,  self.alpha)
                s.err_ewm = _ewma(p.err_ewm, s.err_s, self.alpha)
            else:
                s.cpu_ewm, s.hb_ewm, s.err_ewm = s.cpu_s, s.hb_s, s.err_s
            self.buf.append(s)

    def latest(self) -> Optional[HealthSample]:
        with self._lock:
            return self.buf[-1] if self.buf else None

    def summary(self) -> Dict[str, float]:
        with self._lock:
            if not self.buf:
                return {}
            s = self.buf[-1]
            return {
                "cpu_s": s.cpu_s, "cpu_ewm": s.cpu_ewm,
                "hb_s": s.hb_s,   "hb_ewm": s.hb_ewm,
                "err_s": s.err_s, "err_ewm": s.err_ewm,
                "rss_s": s.rss_s, "idx_s": s.idx_s, "wcache_s": s.wcache_s,
            }

def _scale(v: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    return _clip01((v - lo) / (hi - lo))

def _build_sample(raw: Dict[str, Any], ranges: Dict[str, tuple]) -> HealthSample:
    ts = time.time()
    cpu = float(raw.get("cpu_util", 0.0) or 0.0)
    rss = float(raw.get("rss_mb", 0.0) or 0.0)
    hb  = float(raw.get("hb_avg_period_ms", 0.0) or 0.0)
    err = float(raw.get("errors_any_rate", 0.0) or 0.0)
    idx = float(raw.get("index_lag", raw.get("mem_guard.index_lag", 0.0)) or 0.0)
    wc  = float(raw.get("working_cache", raw.get("mem_guard.working_cache", 0.0)) or 0.0)

    cpu_s = _scale(cpu, *ranges.get("cpu_util", (0.05, 0.85)))
    rss_s = _scale(rss, *ranges.get("rss_mb",  (300.0, 1500.0)))
    hb_s  = _scale(hb,  *ranges.get("hb_ms",   (5.0,  50.0)))
    err_s = _scale(err, *ranges.get("err_rps", (0.0,  0.5)))
    idx_s = _scale(idx, *ranges.get("index_lag",(0.0, 200.0)))
    wc_s  = _scale(wc,  *ranges.get("working_cache",(0.0, 512.0)))

    return HealthSample(
        ts=ts, cpu_util=cpu, rss_mb=rss, hb_avg_period_ms=hb, errors_any_rate=err,
        index_lag=idx, working_cache=wc,
        cpu_s=cpu_s, rss_s=rss_s, hb_s=hb_s, err_s=err_s, idx_s=idx_s, wcache_s=wc_s,
    )

class NervousSystem:
    """
    5 Hz sampler -> scaled state; optional memory summaries.
    """
    def __init__(self, *,
                 get_raw: Callable[[], Dict[str, Any]],
                 bus: HealthBus,
                 daemon=None,
                 sample_interval_s: float = 0.2,     # << 5 Hz
                 summary_interval_s: float = 5.0,     # write to memory less often
                 ranges: Optional[Dict[str, tuple]] = None):
        self.get_raw = get_raw
        self.bus = bus
        self.daemon = daemon
        self.sample_interval_s = float(sample_interval_s)
        self.summary_interval_s = float(summary_interval_s)
        self.ranges = ranges or {}
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._last_summary = 0.0

    def start(self) -> None:
        if self._thr: return
        self._thr = threading.Thread(target=self._loop, name="nervous-system", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                raw = self.get_raw() or {}
                s = _build_sample(raw, self.ranges)
                self.bus.add(s)

                # Periodic compact memory summary
                now = time.time()
                if self.daemon is not None and Event is not None and (now - self._last_summary) >= self.summary_interval_s:
                    self._last_summary = now
                    latest = self.bus.summary()
                    if latest:
                        content = f"Health summary: cpu={latest['cpu_s']:.2f}, hb={latest['hb_s']:.2f}, err={latest['err_s']:.2f}"
                        meta = {"latest": latest, "ts": now}
                        try:
                            self.daemon.ingest(Event(kind="health_summary", content=content, meta=meta))
                        except Exception:
                            pass
            except Exception:
                pass
            # maintain ~5 Hz; compensate some jitter
            dt = time.time() - t0
            time.sleep(max(0.0, self.sample_interval_s - dt))

    # scheduler gate helpers
    def get_state(self) -> Dict[str, float]:
        return self.bus.summary()

    def load_scalar(self) -> float:
        """Single 0..1 scalar (max of smoothed drivers)."""
        s = self.bus.summary()
        return max(s.get("cpu_ewm",0), s.get("hb_ewm",0), s.get("err_ewm",0))

    def should_run_heavy(self) -> bool:
        """Policy: allow heavy ops when load < 0.6."""
        return self.load_scalar() < 0.60

