# tests/reaper_tests/liveness_cycle_test.py
from reaper.liveness_cycle import LivenessByCycles, DEFAULT_MAX_MISSED_CYCLES

# --- helpers ---
class Pulse:
    def __init__(self, n=0):
        self.n = n
    def read(self):
        return self.n
    def tick(self, k=1):
        self.n += k
    def set(self, n):
        self.n = n

class KillRecorder:
    def __init__(self):
        self.reasons = []
    def __call__(self, reason: str):
        self.reasons.append(reason)

# -------------------- core behavior --------------------

def test_first_step_seeds_baseline_no_trip():
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)

    live.register("planner", max_missed_cycles=5)

    # first step just seeds last_pulse_seen
    live.step()
    assert kills.reasons == []

    # advance fewer than limit -> still fine
    pulse.tick(4)
    live.step()
    assert kills.reasons == []

def test_trips_when_missed_cycles_reach_limit():
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)

    live.register("retrieval", max_missed_cycles=5)
    live.step()          # seed baseline at 0

    pulse.tick(5)        # missed = 5 (== limit) -> trip
    live.step()
    assert kills.reasons, "expected liveness trip at limit"
    msg = kills.reasons[-1]
    assert "HARD:liveness_missed" in msg and "section=retrieval" in msg
    assert "missed_cycles=5" in msg and "limit=5" in msg

def test_one_shot_until_touched_again():
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)

    live.register("io", max_missed_cycles=3)
    live.step()          # seed @ 0

    pulse.tick(3)
    live.step()
    assert len(kills.reasons) == 1

    # keep missing; shouldn't retrip until a touch
    pulse.tick(10)
    live.step()
    assert len(kills.reasons) == 1

    # touch resets tripped state and last_pulse_seen
    live.touch("io")
    pulse.tick(3)
    live.step()
    assert len(kills.reasons) == 2  # trips again after reset

def test_touch_unknown_section_is_noop():
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)
    # touching an unregistered section should not raise or change anything
    live.touch("not_registered")
    live.step()
    assert kills.reasons == []

# -------------------- decorator behavior --------------------

def test_required_decorator_default_max(monkeypatch):
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)

    @live.required()  # default uses DEFAULT_MAX_MISSED_CYCLES
    def periodic():
        # simulate work
        return 42

    # first enforce step seeds
    live.step()
    # tick just below the huge default limit
    pulse.tick(DEFAULT_MAX_MISSED_CYCLES - 1)
    live.step()
    assert kills.reasons == []

    # calling the function should touch and reset the counter
    periodic()
    live.step()
    assert kills.reasons == []

def test_required_decorator_custom_max():
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)

    @live.required(5, name="scheduler.tick")
    def tick_fn():
        return "ok"

    # seed baseline
    live.step()

    # miss under the limit → no trip
    pulse.tick(4)
    live.step()
    assert kills.reasons == []

    # calling function touches and resets
    tick_fn()
    pulse.tick(5)
    live.step()
    assert kills.reasons, "expected trip after custom limit"
    assert "section=scheduler.tick" in kills.reasons[-1]

# -------------------- context manager behavior --------------------

def test_alive_context_auto_register_and_touch():
    pulse = Pulse(0)
    kills = KillRecorder()
    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)

    # Section is not registered yet; 'alive' should register with default limit
    with live.alive("worker"):
        pass  # touching happens on exit
    live.step()  # seed baseline at current pulse

    # Miss fewer than default -> ok
    pulse.tick(10)
    live.step()
    assert kills.reasons == []

    # Context again -> touch resets
    with live.alive("worker"):
        pass
    # Jump a big amount to force a trip against default
    pulse.tick(DEFAULT_MAX_MISSED_CYCLES)
    live.step()
    assert kills.reasons, "expected trip after exceeding default missed cycles"
    assert "section=worker" in kills.reasons[-1]

# -------------------- metrics hook (optional) --------------------

class _FakeMetric:
    def __init__(self):
        self.calls = 0
        self.last_labels = None
    def labels(self, **kw):
        self.last_labels = kw
        return self
    def inc(self, n: float = 1.0):
        self.calls += 1

def test_metrics_hook_called_if_present(monkeypatch):
    pulse = Pulse(0)
    kills = KillRecorder()
    fake = _FakeMetric()
    # monkeypatch the module-level errors_total symbol used inside LivenessByCycles.step
    import reaper.liveness_cycle as mod
    monkeypatch.setattr(mod, "errors_total", fake, raising=True)

    live = LivenessByCycles(get_pulse=pulse.read, on_violation=kills)
    live.register("net", max_missed_cycles=3)

    live.step()       # seed
    pulse.tick(3)
    live.step()       # trip

    assert kills.reasons, "should have tripped"
    assert fake.calls >= 1
    assert fake.last_labels == {"key": "liveness_missed", "severity": "1"}
