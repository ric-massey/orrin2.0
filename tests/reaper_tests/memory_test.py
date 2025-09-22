# tests/reaper_tests/memory_test.py
from reaper.memory import MemoryHealthGuard

# ---------- helpers ----------

class FakeClock:
    def __init__(self, t=0.0): self.t = t
    def now(self): return self.t
    def step(self, dt): self.t += dt  # seconds

class KillRecorder:
    def __init__(self): self.reasons = []
    def __call__(self, reason: str): self.reasons.append(reason)

# These mutable providers let us change the returned value over time
class Box:
    def __init__(self, v): self.v = v
    def set(self, v): self.v = v

def make_guard(clock, **kw):
    kills = KillRecorder()
    guard = MemoryHealthGuard(on_violation=kills, now_fn=clock.now, **kw)
    return guard, kills

# ---------- MEMORY SLOPE ----------

def test_memory_slope_trips_when_above_threshold_and_sustained():
    clk = FakeClock()
    rss = Box(100.0)

    guard, kills = make_guard(
        clk,
        get_rss_mb=lambda: rss.v,
        mem_slope_mb_per_s=1.0,   # threshold 1 MB/s
        mem_sustain_s=10.0,       # need ~≥ 9.5s span
    )

    # Build ~ 1.2 MB/s slope over 10s: start 100, +1.2 each second
    for i in range(11):  # 0..10 => span 10s
        rss.set(100.0 + 1.2 * i)
        guard.step()
        clk.step(1.0)

    assert kills.reasons, "expected memory leak slope trip"
    msg = kills.reasons[-1]
    assert "HARD:memory_leak_slope" in msg and "slope=" in msg and "sustain=10.0s" in msg

def test_memory_slope_does_not_trip_if_below_threshold():
    clk = FakeClock()
    rss = Box(100.0)
    guard, kills = make_guard(
        clk,
        get_rss_mb=lambda: rss.v,
        mem_slope_mb_per_s=1.0,
        mem_sustain_s=10.0,
    )
    # ~0.5 MB/s slope → below threshold
    for i in range(11):
        rss.set(100.0 + 0.5 * i)
        guard.step()
        clk.step(1.0)
    assert kills.reasons == []

def test_memory_slope_does_not_trip_if_window_too_short():
    clk = FakeClock()
    rss = Box(0.0)
    guard, kills = make_guard(
        clk,
        get_rss_mb=lambda: rss.v,
        mem_slope_mb_per_s=0.1,
        mem_sustain_s=10.0,
    )
    # Large slope but only 5s span (needs ~≥9.5s)
    for i in range(6):  # span 5s
        rss.set(10.0 * i)
        guard.step()
        clk.step(1.0)
    assert kills.reasons == []

# ---------- FD PRESSURE ----------

def test_fd_pressure_trips_when_strictly_over_threshold_and_sustained():
    clk = FakeClock()
    open_fd = Box(91.0); lim_fd = Box(100.0)  # 91% > 90%
    guard, kills = make_guard(
        clk,
        get_fd_open=lambda: open_fd.v,
        get_fd_limit=lambda: lim_fd.v,
        fd_pct_threshold=0.90,
        fd_sustain_s=5.0,
    )
    for _ in range(6):  # span 5s
        guard.step()
        clk.step(1.0)
    assert any("HARD:fd_pressure" in r for r in kills.reasons)

def test_fd_pressure_does_not_trip_at_boundary_equal_threshold():
    clk = FakeClock()
    open_fd = Box(90.0); lim_fd = Box(100.0)  # 90% == threshold (code uses >, not >=)
    guard, kills = make_guard(
        clk,
        get_fd_open=lambda: open_fd.v,
        get_fd_limit=lambda: lim_fd.v,
        fd_pct_threshold=0.90,
        fd_sustain_s=5.0,
    )
    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert not any("HARD:fd_pressure" in r for r in kills.reasons)

def test_fd_pressure_no_trip_if_window_too_short():
    clk = FakeClock()
    open_fd = Box(99.0); lim_fd = Box(100.0)  # 99% > threshold
    guard, kills = make_guard(
        clk,
        get_fd_open=lambda: open_fd.v,
        get_fd_limit=lambda: lim_fd.v,
        fd_pct_threshold=0.90,
        fd_sustain_s=10.0,
    )
    for _ in range(5):  # span 4s < 9.5s needed
        guard.step(); clk.step(1.0)
    assert not any("HARD:fd_pressure" in r for r in kills.reasons)

# ---------- SOCKET PRESSURE ----------

def test_socket_pressure_trips_when_strictly_over_threshold_and_sustained():
    clk = FakeClock()
    open_sk = Box(95.0); lim_sk = Box(100.0)  # 95% > 90%
    guard, kills = make_guard(
        clk,
        get_sock_open=lambda: open_sk.v,
        get_sock_limit=lambda: lim_sk.v,
        fd_pct_threshold=0.90,
        fd_sustain_s=5.0,
    )
    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert any("HARD:socket_pressure" in r for r in kills.reasons)

def test_socket_pressure_boundary_equal_threshold_no_trip():
    clk = FakeClock()
    open_sk = Box(90.0); lim_sk = Box(100.0)  # == threshold
    guard, kills = make_guard(
        clk,
        get_sock_open=lambda: open_sk.v,
        get_sock_limit=lambda: lim_sk.v,
        fd_pct_threshold=0.90,
        fd_sustain_s=5.0,
    )
    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert not any("HARD:socket_pressure" in r for r in kills.reasons)

# ---------- CPU STARVATION ----------

def test_cpu_starvation_trips_with_high_cpu_and_latency_slope():
    clk = FakeClock()
    cpu = Box(0.0); lat = Box(10.0)
    guard, kills = make_guard(
        clk,
        get_cpu_util=lambda: cpu.v,            # will use 0..1 directly
        get_step_latency_ms=lambda: lat.v,
        cpu_util_threshold=0.95,
        cpu_sustain_s=5.0,
        latency_slope_ms_per_s=0.2,
        latency_mean_ms_threshold=50.0,
    )
    # build high CPU and steadily rising latency
    for i in range(6):  # span 5s
        cpu.set(0.98)
        lat.set(10.0 + 1.0 * i)   # slope = 1.0 ms/s
        guard.step(); clk.step(1.0)
    assert any("HARD:cpu_starvation" in r for r in kills.reasons)

def test_cpu_starvation_trips_with_high_cpu_and_high_mean_latency():
    clk = FakeClock()
    cpu = Box(0.0); lat = Box(0.0)
    guard, kills = make_guard(
        clk,
        get_cpu_util=lambda: cpu.v,
        get_step_latency_ms=lambda: lat.v,
        cpu_util_threshold=0.90,      # slightly lower to make easier
        cpu_sustain_s=5.0,
        latency_slope_ms_per_s=5.0,   # huge slope to force reliance on mean instead
        latency_mean_ms_threshold=20.0,
    )
    for _ in range(6):  # span 5s
        cpu.set(0.95)
        lat.set(25.0)   # constant, mean=25 > 20
        guard.step(); clk.step(1.0)
    assert any("HARD:cpu_starvation" in r for r in kills.reasons)

def test_cpu_starvation_requires_latency_data():
    clk = FakeClock()
    cpu = Box(1.0)  # high CPU
    guard, kills = make_guard(
        clk,
        get_cpu_util=lambda: cpu.v,
        # no latency provider -> should NOT trip
        cpu_util_threshold=0.95,
        cpu_sustain_s=5.0,
        latency_slope_ms_per_s=0.2,
        latency_mean_ms_threshold=20.0,
    )
    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert not any("HARD:cpu_starvation" in r for r in kills.reasons)

def test_cpu_high_but_latency_not_rising_and_mean_low_no_trip():
    clk = FakeClock()
    cpu = Box(0.99); lat = Box(10.0)
    guard, kills = make_guard(
        clk,
        get_cpu_util=lambda: cpu.v,
        get_step_latency_ms=lambda: lat.v,
        cpu_util_threshold=0.95,
        cpu_sustain_s=5.0,
        latency_slope_ms_per_s=1.0,
        latency_mean_ms_threshold=50.0,
    )
    # latency constant (slope=0), mean=10 (below threshold)
    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert not any("HARD:cpu_starvation" in r for r in kills.reasons)

def test_cpu_percentage_normalization_from_0_100():
    clk = FakeClock()
    cpu = Box(99.0)     # 99% -> normalized 0.99
    lat = Box(60.0)     # high mean latency
    guard, kills = make_guard(
        clk,
        get_cpu_util=lambda: cpu.v,            # returns 0..100
        get_step_latency_ms=lambda: lat.v,
        cpu_util_threshold=0.95,
        cpu_sustain_s=5.0,
        latency_slope_ms_per_s=100.0,         # ignore slope path
        latency_mean_ms_threshold=50.0,       # rely on mean path
    )
    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert any("HARD:cpu_starvation" in r for r in kills.reasons)

# ---------- METRICS HOOK & PROVIDER SAFETY ----------

class _FakeMetric:
    def __init__(self):
        self.calls = 0
        self.kw = None
    def labels(self, **kw):
        self.kw = kw; return self
    def inc(self, n: float = 1.0):
        self.calls += 1

def test_metrics_hook_increments_on_trip(monkeypatch):
    clk = FakeClock()
    rss = Box(100.0)
    # patch module-level errors_total
    import reaper.memory as mod
    fake = _FakeMetric()
    monkeypatch.setattr(mod, "errors_total", fake, raising=True)

    guard, kills = make_guard(
        clk,
        get_rss_mb=lambda: rss.v,
        mem_slope_mb_per_s=0.1,
        mem_sustain_s=5.0,
    )
    # Force big slope over 5s
    for i in range(6):
        rss.set(100.0 + 5.0 * i)
        guard.step(); clk.step(1.0)

    assert kills.reasons
    assert fake.calls >= 1
    assert fake.kw == {"key": "HARD:memory_leak_slope", "severity": "1"}

def test_missing_providers_are_safe_no_crash():
    clk = FakeClock()
    guard, kills = make_guard(clk)  # no providers at all
    for _ in range(5):
        guard.step(); clk.step(1.0)
    # Nothing to assert except that it didn't crash and no reasons appeared
    assert kills.reasons == []
