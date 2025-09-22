# tests/reaper_tests/error_test.py
import time

from reaper.error_checker import ErrorChecker
from reaper.reaper import Reaper
from reaper.errors import (
    ErrorEvent,
    make_event_from_key,
    make_event,
    severity_for_exception,
    SEVERITY_BY_KEY,
    Severity,
)

# --------- helpers ----------
class FakeClock:
    def __init__(self, start=0.0): self.t = start
    def now(self): return self.t
    def step(self, dt): self.t += dt  # seconds

class KillRecorder:
    def __init__(self): self.reasons = []
    def __call__(self, reason: str): self.reasons.append(reason)

def make_checker(clock, thresholds=None, window_s=None,
                 any_rate=None, per_key=None):
    """
    Build a checker with injectable clock.
    any_rate: (count, window_s) or None
    per_key: dict key -> (count, window_s)
    """
    kills = KillRecorder()
    reaper = Reaper(kill=kills)
    checker = ErrorChecker(
        on_violation=reaper.trigger,
        thresholds=thresholds or {1: 10, 2: 25, 3: 50},
        window_s=window_s,
        now_fn=clock.now
    )
    if any_rate:
        checker.set_any_rate_limit(any_rate[0], any_rate[1])
    for k, (cnt, win) in (per_key or {}).items():
        checker.set_key_rate_limit(k, cnt, win)
    return checker, kills

# --------- errors.py coverage ----------

def test_error_event_normalizes_severity():
    e = ErrorEvent(key="x", severity=999)
    assert e.severity == 1  # unknown → worst
    e2 = ErrorEvent(key="y", severity=Severity.SEV2)
    assert e2.severity == 2
    e3 = ErrorEvent(key="z", severity="2")
    assert e3.severity == 2

def test_make_event_and_from_key_registry_default():
    # explicit make_event honors provided severity
    ev = make_event("custom", Severity.SEV3, message="ok")
    assert ev.key == "custom" and ev.severity == 3 and ev.message == "ok"

    # from_key pulls from registry; unknown → worst (1)
    known_key = next(iter(SEVERITY_BY_KEY.keys()))
    ev2 = make_event_from_key(known_key)
    assert ev2.severity == SEVERITY_BY_KEY[known_key]

    ev3 = make_event_from_key("totally_unknown_key")
    assert ev3.severity == 1

def test_severity_for_exception_heuristic():
    class CorruptError(Exception): pass
    class Timeoutish(Exception): pass
    class Minor(Exception): pass

    assert severity_for_exception(CorruptError("fatal data corrupt")) == 1
    assert severity_for_exception(Timeoutish("timeout while calling api")) == 2
    assert severity_for_exception(Minor("some random note")) == 3

# --------- error_checker.py coverage ----------

def test_severity_thresholds_cumulative_no_window():
    clock = FakeClock()
    checker, kills = make_checker(clock)

    # sev3 -> 50
    for _ in range(49):
        checker.observe(ErrorEvent(key="soft_warn", severity=3))
    assert kills.reasons == []
    checker.observe(ErrorEvent(key="soft_warn", severity=3))
    assert any("sev=3" in r for r in kills.reasons)

    # sev2 -> 25
    for _ in range(24):
        checker.observe(ErrorEvent(key="mid_issue", severity=2))
    assert not any("mid_issue" in r for r in kills.reasons)
    checker.observe(ErrorEvent(key="mid_issue", severity=2))
    assert any("mid_issue" in r for r in kills.reasons)

    # sev1 -> 10
    for _ in range(9):
        checker.observe(ErrorEvent(key="critical", severity=1))
    assert not any("critical" in r for r in kills.reasons)
    checker.observe(ErrorEvent(key="critical", severity=1))
    assert any("critical" in r for r in kills.reasons)

def test_after_trip_resets_counter():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 3, 2: 25, 3: 50})

    # Trip once
    for _ in range(3):
        checker.observe(ErrorEvent(key="flaky", severity=1))
    assert any("flaky" in r for r in kills.reasons)
    first = len(kills.reasons)

    # Needs 3 again
    checker.observe(ErrorEvent(key="flaky", severity=1))
    assert len(kills.reasons) == first
    checker.observe(ErrorEvent(key="flaky", severity=1))
    checker.observe(ErrorEvent(key="flaky", severity=1))
    assert len(kills.reasons) == first + 1

def test_windowed_repetition_counts():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 3, 2: 25, 3: 50}, window_s=5.0)

    checker.observe(ErrorEvent(key="glitch", severity=1)); clock.step(2.0)
    checker.observe(ErrorEvent(key="glitch", severity=1)); clock.step(2.9)
    assert kills.reasons == []

    # Third still within 5s window -> trip
    checker.observe(ErrorEvent(key="glitch", severity=1))
    assert any("glitch" in r for r in kills.reasons)

    # Let prior window expire; 2 more not enough
    clock.step(5.1)
    checker.observe(ErrorEvent(key="glitch", severity=1))
    checker.observe(ErrorEvent(key="glitch", severity=1))
    assert sum("glitch" in r for r in kills.reasons) == 1

def test_window_boundary_behavior_inclusive_on_cutoff():
    # Items with timestamp == cutoff should be retained (code prunes while < cutoff)
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 3, 2: 25, 3: 50}, window_s=5.0)

    # t=0,2,5 exactly → at t=5, cutoff = 0; events at 0 NOT pruned? (0 < 0 is False) so retained
    checker.observe(ErrorEvent(key="edge", severity=1)); clock.step(2.0)   # t=0
    checker.observe(ErrorEvent(key="edge", severity=1)); clock.step(3.0)   # t=2
    checker.observe(ErrorEvent(key="edge", severity=1))                     # t=5
    # Three within window (by inclusive boundary) → trip
    assert any("edge" in r for r in kills.reasons)

def test_keys_and_severities_are_isolated():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 3, 2: 2, 3: 2})

    checker.observe(ErrorEvent(key="same_key", severity=2))
    checker.observe(ErrorEvent(key="same_key", severity=3))
    assert kills.reasons == []

    checker.observe(ErrorEvent(key="same_key", severity=2))
    assert any("same_key" in r for r in kills.reasons)
    # sev=3 stream still separate
    checker.observe(ErrorEvent(key="same_key", severity=3))
    assert sum("same_key" in r for r in kills.reasons) == 2

def test_unknown_severity_treated_as_worst():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 2, 2: 999, 3: 999})
    checker.observe(ErrorEvent(key="odd", severity=99))
    assert kills.reasons == []
    checker.observe(ErrorEvent(key="odd", severity=99))
    assert any("odd" in r for r in kills.reasons)

def test_details_are_included_in_violation_message():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 2, 2: 25, 3: 50})
    checker.observe(ErrorEvent(key="with_details", severity=1), details="hello-world")
    checker.observe(ErrorEvent(key="with_details", severity=1), details="hello-world")
    assert any("with_details" in r and "hello-world" in r for r in kills.reasons)

def test_global_any_error_rate_limit():
    clock = FakeClock()
    # Any 5 errors within 10s should trip
    checker, kills = make_checker(clock, any_rate=(5, 10.0))
    for i in range(4):
        checker.observe(ErrorEvent(key=f"k{i}", severity=3))
        clock.step(1.0)
    assert kills.reasons == []

    # 5th within the 10s window -> trip
    checker.observe(ErrorEvent(key="k4", severity=3))
    assert any("scope=any" in r for r in kills.reasons)

    # After trip, window cleared
    assert kills.reasons  # one trip
    clock.step(10.1)
    for i in range(4):
        checker.observe(ErrorEvent(key=f"k{i}", severity=3))
    assert sum("scope=any" in r for r in kills.reasons) == 1

def test_per_key_rate_limit():
    clock = FakeClock()
    # key 'llm_timeout': 3 in 5s trips
    checker, kills = make_checker(clock, per_key={"llm_timeout": (3, 5.0)})

    checker.observe(make_event_from_key("llm_timeout")); clock.step(1.0)
    checker.observe(make_event_from_key("llm_timeout")); clock.step(1.0)
    assert not any("key=llm_timeout" in r for r in kills.reasons)

    checker.observe(make_event_from_key("llm_timeout"))
    assert any("scope=key" in r and "key=llm_timeout" in r for r in kills.reasons)

def test_rate_limits_are_independent_of_repetition_thresholds():
    clock = FakeClock()
    # repetition thresholds high; rely on rate limit
    checker, kills = make_checker(clock,
                                  thresholds={1: 999, 2: 999, 3: 999},
                                  any_rate=(3, 2.0))

    checker.observe(ErrorEvent(key="a", severity=3)); clock.step(0.5)
    checker.observe(ErrorEvent(key="b", severity=3)); clock.step(0.5)
    assert kills.reasons == []
    checker.observe(ErrorEvent(key="c", severity=3))  # 3 in 1s -> trip
    assert any("scope=any" in r for r in kills.reasons)

def test_unknown_key_from_registry_trips_with_low_threshold():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 2, 2: 25, 3: 50})
    # unknown key → sev1 by default → threshold=2
    checker.observe(make_event_from_key("totally_unknown_key"))
    assert kills.reasons == []
    checker.observe(make_event_from_key("totally_unknown_key"))
    assert any("key=totally_unknown_key" in r or "error_threshold" in r for r in kills.reasons)

def test_clear_all_by_key_by_severity_and_pair():
    clock = FakeClock()
    checker, kills = make_checker(clock, thresholds={1: 2, 2: 2, 3: 2})

    # Build up some counters
    checker.observe(ErrorEvent(key="K1", severity=1))
    checker.observe(ErrorEvent(key="K2", severity=2))
    checker.observe(ErrorEvent(key="K1", severity=3))

    # Clear by (key, severity) pair
    checker.clear(key="K1", severity=1)
    checker.observe(ErrorEvent(key="K1", severity=1))
    assert kills.reasons == []  # needs 1 more to trip again; counter was cleared

    # Clear by key
    checker.clear(key="K2")
    checker.observe(ErrorEvent(key="K2", severity=2))
    assert not any("K2" in r for r in kills.reasons)

    # Clear by severity
    checker.clear(severity=3)
    checker.observe(ErrorEvent(key="K1", severity=3))
    assert not any("sev=3" in r for r in kills.reasons)

    # Clear all
    checker.clear()
    checker.observe(ErrorEvent(key="K3", severity=1))
    assert kills.reasons == []

def test_when_both_global_and_per_key_limits_fire_same_event():
    clock = FakeClock()
    # Configure so the 3rd 'hot' event hits BOTH:
    # - any_rate: 3 events in 5s
    # - per_key: 'hot' 3 events in 5s
    checker, kills = make_checker(clock, any_rate=(3, 5.0), per_key={"hot": (3, 5.0)})

    checker.observe(ErrorEvent(key="hot", severity=3)); clock.step(1.0)
    checker.observe(ErrorEvent(key="hot", severity=3)); clock.step(1.0)
    assert kills.reasons == []

    # Third 'hot' should trigger BOTH global and per-key in current implementation
    checker.observe(ErrorEvent(key="hot", severity=3))
    # Expect at least two violation messages (order doesn't matter)
    assert sum("scope=any" in r for r in kills.reasons) >= 1
    assert sum("scope=key" in r and "key=hot" in r for r in kills.reasons) >= 1
