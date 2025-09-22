# tests/reaper_tests/no_goals_test.pyPYTHO
from reaper.no_goals import NoGoalsGuard

# --------- helpers ----------
class FakeClock:
    def __init__(self, t=0.0): self.t = t
    def now(self): return self.t
    def step(self, dt): self.t += dt  # seconds

class Pulse:
    def __init__(self, n=0): self.n = n
    def read(self): return self.n
    def tick(self, k=1): self.n += k
    def set(self, n): self.n = n

class KillRecorder:
    def __init__(self): self.reasons = []
    def __call__(self, reason: str): self.reasons.append(reason)

# Providers we can mutate on the fly
class Box:
    def __init__(self, v): self.v = v
    def set(self, v): self.v = v

# -------------------- GOALS STALL --------------------

def test_startup_seeds_no_instant_trip():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    def get_goals():
        return []  # nothing yet at startup

    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=get_goals, now_fn=clk.now,
        max_idle_cycles=5,
    )

    # First step seeds last activity pulse; no trip
    guard.step()
    assert kills.reasons == []

def test_goals_stall_trips_after_max_idle_cycles():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    goals = Box([])  # start with no goals
    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: goals.v, now_fn=clk.now,
        max_idle_cycles=5,
    )

    # Seed
    guard.step()
    # Advance 4 cycles: under limit
    for _ in range(4):
        pulse.tick(); guard.step()
    assert kills.reasons == []

    # 5th missed cycle reaches limit → trip
    pulse.tick(); guard.step()
    assert any("HARD:no_goals_progress" in r for r in kills.reasons)

def test_any_active_goal_prevents_stall():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    def get_goals():
        # Any active status should count as activity each step (per current implementation)
        return [{"id":"g1","status":"active","updated_ts":clk.now()}]

    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=get_goals, now_fn=clk.now,
        max_idle_cycles=5,
    )

    # Many cycles pass, but always at least one active → no trip
    for _ in range(20):
        pulse.tick(); guard.step()
    assert kills.reasons == []

def test_unknown_status_does_not_count_as_active():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    def get_goals():
        # not in ("active","in_progress","working")
        return [{"id":"g1","status":"paused","updated_ts":clk.now()}]

    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=get_goals, now_fn=clk.now,
        max_idle_cycles=3,
    )

    guard.step()  # seed
    for _ in range(3):
        pulse.tick(); guard.step()
    assert any("HARD:no_goals_progress" in r for r in kills.reasons)

# -------------------- RETRY SATURATION --------------------

def test_retry_saturation_trips_when_rate_above_threshold_and_sustained():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    rate = Box(10.0)  # retries/sec
    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        get_retry_rate=lambda: rate.v,
        retry_rate_threshold=5.0, retry_sustain_s=5.0,
        max_idle_cycles=999999,  # disable stall path in this test
    )

    # Build ≥ ~5s window with all rates > threshold
    for _ in range(6):   # 0..5 => span 5s
        guard.step(); clk.step(1.0)

    assert any("HARD:retry_saturation" in r for r in kills.reasons)

def test_retry_equal_to_threshold_does_not_trip():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    rate = Box(5.0)  # equals threshold
    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        get_retry_rate=lambda: rate.v,
        retry_rate_threshold=5.0, retry_sustain_s=5.0,
        max_idle_cycles=999999,
    )

    for _ in range(6):
        guard.step(); clk.step(1.0)
    assert not any("HARD:retry_saturation" in r for r in kills.reasons)

def test_retry_window_too_short_no_trip():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    rate = Box(100.0)
    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        get_retry_rate=lambda: rate.v,
        retry_rate_threshold=1.0, retry_sustain_s=10.0,
        max_idle_cycles=999999,
    )

    for _ in range(5):  # span 4s < 9.5s slack requirement
        guard.step(); clk.step(1.0)
    assert not any("HARD:retry_saturation" in r for r in kills.reasons)

# -------------------- CIRCUIT BREAKERS --------------------

def test_single_breaker_open_too_long_trips():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    opened = clk.now() - 61.0
    breakers = Box([{"name":"db","state":"open","opened_ts":opened}])

    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        get_breakers=lambda: breakers.v,
        cb_open_max_s=60.0, cb_window_s=30.0, cb_max_distinct_open=3,
        max_idle_cycles=999999,
    )

    guard.step()
    assert any("HARD:circuit_breaker_open_too_long" in r for r in kills.reasons)

def test_many_distinct_breakers_open_within_window_trips():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    # Start with none open, then open different ones inside the window
    bxs = [
        Box([{"name":"a","state":"open","opened_ts":clk.now()}]),
        Box([{"name":"b","state":"open","opened_ts":clk.now()}]),
        Box([{"name":"c","state":"open","opened_ts":clk.now()}]),
    ]
    idx = 0
    def get_breakers():
        nonlocal idx
        return bxs[min(idx, len(bxs)-1)].v

    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        get_breakers=get_breakers,
        cb_open_max_s=9999.0,           # ignore long-open path
        cb_window_s=10.0, cb_max_distinct_open=3,
        max_idle_cycles=999999,
    )

    # Open 'a', step for a bit
    idx = 0; guard.step(); clk.step(1.0)
    # Now 'b'
    idx = 1; guard.step(); clk.step(1.0)
    # Now 'c' -> should trip (3 distinct in window)
    idx = 2; guard.step()
    assert any("HARD:circuit_breaker_many_open" in r for r in kills.reasons)

def test_distinct_breakers_window_trimming():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    # We'll push breakers 'a' and 'b', then let window expire, then 'c'
    seq = [
        [{"name":"a","state":"open","opened_ts":clk.now()}],
        [{"name":"b","state":"open","opened_ts":clk.now()}],
        [],  # none open while time advances
        [{"name":"c","state":"open","opened_ts":clk.now()}],
    ]
    idx = 0
    def get_breakers():
        nonlocal idx
        return seq[min(idx, len(seq)-1)]

    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        get_breakers=get_breakers,
        cb_open_max_s=9999.0,
        cb_window_s=5.0, cb_max_distinct_open=2,
        max_idle_cycles=999999,
    )

    # See 'a' then 'b'
    idx = 0; guard.step(); clk.step(1.0)
    idx = 1; guard.step(); clk.step(1.0)

    # With threshold 2, equality (exactly two distinct) must NOT trip
    assert not any("HARD:circuit_breaker_many_open" in r for r in kills.reasons)

    # Advance past window so 'a' & 'b' fall out
    idx = 2
    for _ in range(6):
        guard.step(); clk.step(1.0)

    # Now only 'c' should be counted; distinct=1 < 2 -> no trip
    idx = 3; guard.step()
    assert not any("HARD:circuit_breaker_many_open" in r for r in kills.reasons)

# -------------------- METRICS HOOK & PROVIDER SAFETY --------------------

class _FakeMetric:
    def __init__(self):
        self.calls = 0
        self.kw = None
    def labels(self, **kw):
        self.kw = kw; return self
    def inc(self, n: float = 1.0):
        self.calls += 1

def test_metrics_hook_called_on_trip(monkeypatch):
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    import reaper.no_goals as mod
    fake = _FakeMetric()
    monkeypatch.setattr(mod, "errors_total", fake, raising=True)

    # Force a goals stall trip
    goals = Box([])
    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: goals.v, now_fn=clk.now,
        max_idle_cycles=2,
    )
    guard.step()        # seed
    pulse.tick(2); guard.step()

    assert kills.reasons
    assert fake.calls >= 1
    assert fake.kw == {"key": "HARD:no_goals_progress", "severity": "1"}

def test_missing_providers_are_safe():
    clk = FakeClock()
    pulse = Pulse(0)
    kills = KillRecorder()

    # Only goals provider; no retry/breakers — should not crash
    guard = NoGoalsGuard(
        get_pulse=pulse.read, on_violation=kills,
        get_goals=lambda: [], now_fn=clk.now,
        max_idle_cycles=3,
    )
    guard.step()  # seed
    pulse.tick(3); guard.step()
    assert any("HARD:no_goals_progress" in r for r in kills.reasons)
