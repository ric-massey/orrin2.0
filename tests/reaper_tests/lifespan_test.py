# tests/reaper_tests/lifespan_test.py
import secrets
import types
from reaper.lifespan import LifespanByCycles

# --- helpers ---
class Pulse:
    def __init__(self, n=0):
        self.n = n
    def read(self):
        return self.n
    def set(self, n):
        self.n = n
    def tick(self, k=1):
        self.n += k

class KillRecorder:
    def __init__(self):
        self.reasons = []
    def __call__(self, reason: str):
        self.reasons.append(reason)

# --- tests ---

def test_triggers_at_min_edge(monkeypatch):
    """
    Force randbelow() to return 0 so limit == min_cycles.
    Verify: no trigger below, trigger exactly at min, message contains limit.
    """
    # randbelow(span+1) -> 0  => limit = min
    monkeypatch.setattr(secrets, "randbelow", lambda _n: 0)

    pulse = Pulse(0)
    kills = KillRecorder()
    ls = LifespanByCycles(get_pulse=pulse.read, on_violation=kills, min_cycles=100, max_cycles=200)

    # seed + below threshold
    pulse.set(99); ls.step()
    assert kills.reasons == []

    # hit exactly at min -> trigger
    pulse.set(100); ls.step()
    assert kills.reasons, "expected lifespan trigger at min"
    assert "lifespan_reached" in kills.reasons[0]
    assert "limit=100" in kills.reasons[0]  # explicit edge check

def test_triggers_at_max_edge(monkeypatch):
    """
    Force randbelow() to return span so limit == max_cycles.
    Verify: no trigger below, trigger at max.
    """
    min_cycles, max_cycles = 50, 75
    span = max_cycles - min_cycles
    monkeypatch.setattr(secrets, "randbelow", lambda _n: span)

    pulse = Pulse(0)
    kills = KillRecorder()
    ls = LifespanByCycles(get_pulse=pulse.read, on_violation=kills, min_cycles=min_cycles, max_cycles=max_cycles)

    pulse.set(74); ls.step()
    assert kills.reasons == []

    pulse.set(75); ls.step()
    assert kills.reasons, "expected lifespan trigger at max"
    assert "limit=75" in kills.reasons[0]

def test_no_trigger_below_limit_then_trigger(monkeypatch):
    """
    Generic flow: pick some mid limit, ensure < limit => no trigger, >= limit => trigger.
    """
    # choose r=5; with min=10,max=30 => span=20 => limit=15
    monkeypatch.setattr(secrets, "randbelow", lambda _n: 5)

    pulse = Pulse(0)
    kills = KillRecorder()
    ls = LifespanByCycles(get_pulse=pulse.read, on_violation=kills, min_cycles=10, max_cycles=30)

    pulse.set(14); ls.step()
    assert kills.reasons == []

    pulse.set(15); ls.step()
    assert any("lifespan_reached" in r for r in kills.reasons)

def test_min_equals_max_is_deterministic():
    """
    When min==max, limit must equal that value; trigger only at/after it.
    """
    pulse = Pulse(0)
    kills = KillRecorder()
    ls = LifespanByCycles(get_pulse=pulse.read, on_violation=kills, min_cycles=10, max_cycles=10)

    pulse.set(9); ls.step()
    assert kills.reasons == []

    pulse.set(10); ls.step()
    assert kills.reasons
    assert "limit=10" in kills.reasons[0]

def test_after_threshold_step_keeps_triggering_if_not_exited(monkeypatch):
    """
    This implementation will trigger on every step once pulse >= limit.
    We assert repeated triggers (since real app would exit on first).
    """
    monkeypatch.setattr(secrets, "randbelow", lambda _n: 0)  # limit = min
    pulse = Pulse(0)
    kills = KillRecorder()
    ls = LifespanByCycles(get_pulse=pulse.read, on_violation=kills, min_cycles=3, max_cycles=5)

    pulse.set(2); ls.step()
    assert kills.reasons == []

    pulse.set(3); ls.step()       # first trigger
    ls.step()                     # still >= limit -> trigger again
    assert len(kills.reasons) >= 2
    assert all("lifespan_reached" in r for r in kills.reasons)

def test_limit_computed_once_only(monkeypatch):
    """
    Ensure _ensure_limit is sticky: randbelow() is called exactly once.
    """
    call_count = {"n": 0}
    def rb(n):
        call_count["n"] += 1
        return 2  # arbitrary within span
    monkeypatch.setattr(secrets, "randbelow", rb)

    pulse = Pulse(0)
    kills = KillRecorder()
    ls = LifespanByCycles(get_pulse=pulse.read, on_violation=kills, min_cycles=100, max_cycles=200)

    # multiple steps below limit should only call randbelow once
    for v in (0, 10, 50, 90, 120):  # last one might trigger depending on chosen limit
        pulse.set(v)
        ls.step()
        if kills.reasons:
            break

    assert call_count["n"] == 1, "randbelow should be called once to choose limit"
