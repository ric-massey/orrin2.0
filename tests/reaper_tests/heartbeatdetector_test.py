import time
from reaper.heartbeatdetector import HeartbeatDetector
from reaper.reaper import Reaper

# --------- helpers ----------
class FakeClock:
    def __init__(self, start=0.0): self.t = start
    def now(self): return self.t
    def step(self, dt): self.t += dt  # seconds

class KillRecorder:
    def __init__(self): self.reasons = []
    def __call__(self, reason: str): self.reasons.append(reason)

def run_beats(det, pulse, clk, period_s, beats):
    """Increment pulse each 'beat', advance time, tick detector once per beat."""
    for _ in range(beats):
        pulse["n"] += 1
        det.step()
        clk.step(period_s)

def mk(clk, pulse, fast_checks=100, slow_checks=10, window=20,
       min_ms=5.0, max_ms=10_000.0):
    """Factory with your limits (5ms fast / 10s slow)."""
    kills = KillRecorder()
    reaper = Reaper(kill=kills)
    det = HeartbeatDetector(
        get_pulse=lambda: pulse["n"],
        on_violation=reaper.trigger,
        min_period_ms=min_ms,
        max_period_ms=max_ms,
        sustain_checks_fast=fast_checks,
        sustain_checks_slow=slow_checks,
        window=window,
    )
    return det, kills

# --------- tests ----------

def test_normal_no_kill(monkeypatch):
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    det, kills = mk(clk, pulse)  # defaults: 5ms fast / 10s slow, 100/10 checks

    run_beats(det, pulse, clk, 0.020, 200)  # 20ms avg is healthy
    assert kills.reasons == []

def test_too_fast_trips(monkeypatch):
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    # smaller window so avg flips quickly; short fast streak for speed
    det, kills = mk(clk, pulse, fast_checks=8, slow_checks=999, window=5)

    run_beats(det, pulse, clk, 0.020, 10)   # warm up normal
    run_beats(det, pulse, clk, 0.001, 20)   # ~1ms avg → too fast
    assert kills.reasons, "expected a fast violation"
    assert any("pulse_too_fast" in r for r in kills.reasons)

def test_too_slow_trips(monkeypatch):
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    # smaller window so avg flips quickly; short slow streak for speed
    det, kills = mk(clk, pulse, fast_checks=999, slow_checks=5, window=5)

    run_beats(det, pulse, clk, 0.020, 10)   # warm up
    run_beats(det, pulse, clk, 10.5, 10)    # >10s avg → too slow
    assert kills.reasons, "expected a slow violation"
    assert any("pulse_too_slow" in r for r in kills.reasons)

def test_fast_blip_doesnt_kill(monkeypatch):
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    det, kills = mk(clk, pulse, fast_checks=8)

    run_beats(det, pulse, clk, 0.020, 10)
    run_beats(det, pulse, clk, 0.003, 1)    # one too-fast blip
    run_beats(det, pulse, clk, 0.020, 20)
    assert kills.reasons == []

def test_slow_blip_doesnt_kill(monkeypatch):
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    det, kills = mk(clk, pulse, slow_checks=5)

    run_beats(det, pulse, clk, 0.020, 10)
    run_beats(det, pulse, clk, 10.5, 1)     # one too-slow blip
    run_beats(det, pulse, clk, 0.020, 10)
    assert kills.reasons == []

def test_streak_resets_when_back_to_normal(monkeypatch):
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    det, kills = mk(clk, pulse, fast_checks=8, slow_checks=8)

    # build some fast streak, then return to normal before threshold
    run_beats(det, pulse, clk, 0.003, 5)    # fast but < 8 checks
    assert det._fast_streak > 0
    run_beats(det, pulse, clk, 0.020, 3)    # back to normal → resets
    assert det._fast_streak == 0
    assert kills.reasons == []

def test_no_pulse_change_trips_slow_by_age(monkeypatch):
    # Simulate detector polling while pulse never increments.
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    # Lower slow threshold and shorten sustain so it's deterministic & quick
    det, kills = mk(clk, pulse, slow_checks=3, max_ms=1_000.0)  # 1s threshold, need 3 checks

    # Seed first observation (sets last_ts without increment)
    det.step()

    # Advance time in 2s chunks without incrementing pulse → age grows.
    # Every check is > 1s, so the slow streak climbs to 3 and trips.
    for _ in range(5):
        clk.step(2.0)
        det.step()

    assert kills.reasons, "expected slow due to stale age"
    assert any("pulse_too_slow" in r for r in kills.reasons)

def test_mutual_exclusion_of_reasons(monkeypatch):
    # Ensure we never get both fast & slow at the same time
    clk = FakeClock(); monkeypatch.setattr(time, "monotonic", clk.now)
    pulse = {"n": 0}
    det, kills = mk(clk, pulse, fast_checks=5, slow_checks=5)

    run_beats(det, pulse, clk, 0.001, 20)   # force fast case
    assert kills.reasons
    assert any("pulse_too_fast" in r for r in kills.reasons)
    assert not any("pulse_too_slow" in r for r in kills.reasons)
