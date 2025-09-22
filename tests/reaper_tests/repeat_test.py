# tests/reaper_tests/repeat_test.py
import itertools

from reaper.repeat import RepeatLoopGuard

# ---------- helpers ----------

class FakeClock:
    def __init__(self, t=0.0): self.t = t
    def now(self): return self.t
    def step(self, dt): self.t += dt  # seconds

class KillRecorder:
    def __init__(self): self.reasons = []
    def __call__(self, reason: str): self.reasons.append(reason)

def mk_guard(clk, **overrides):
    rec = KillRecorder()
    guard = RepeatLoopGuard(on_violation=rec, now_fn=clk.now, **overrides)
    return guard, rec

# ---------- SAME-CALL LOOP ----------

def test_same_call_soft_then_hard_escalation_without_progress():
    clk = FakeClock()
    # lower thresholds so test runs fast
    guard, rec = mk_guard(clk, same_call_k=3, same_call_t=5.0, breaker_cool_s=10.0)

    # three identical actions within 5s, no success/progress -> soft breaker open
    for _ in range(3):
        guard.observe_action("retrieval.search", {"q": "foo"})
        guard.step(); clk.step(1.0)

    assert any(r.startswith("SOFT:breaker_open") and "reason=same_call_loop" in r for r in rec.reasons)

    # While breaker is still open, keep repeating → escalates to HARD
    before = len(rec.reasons)
    for _ in range(2):
        guard.observe_action("retrieval.search", {"q": "foo"})
        guard.step(); clk.step(1.0)

    assert len(rec.reasons) > before
    assert any(r.startswith("HARD:repeat_same_call_loop") for r in rec.reasons)

def test_same_call_does_not_trip_if_progress_present():
    clk = FakeClock()
    guard, rec = mk_guard(clk, same_call_k=3, same_call_t=5.0)

    # identical calls but with progress
    for _ in range(3):
        guard.observe_action("writer.emit", {"path": "/tmp/x"}, progress_delta=1.0)
        guard.step(); clk.step(1.0)

    assert not rec.reasons

def test_same_call_resets_after_window_expires():
    clk = FakeClock()
    guard, rec = mk_guard(clk, same_call_k=3, same_call_t=3.0)

    # two calls, then wait past window, then one call → no trip
    guard.observe_action("f.x"); guard.step(); clk.step(1.0)
    guard.observe_action("f.x"); guard.step(); clk.step(3.1)  # past window
    guard.observe_action("f.x"); guard.step()
    assert not rec.reasons

# ---------- PING-PONG LOOP ----------

def test_ping_pong_soft_then_hard_escalation_without_progress():
    clk = FakeClock()
    guard, rec = mk_guard(clk, pingpong_k=6, pingpong_t=10.0, breaker_cool_s=10.0)

    # Build A,B,A,B,A,B within window with no progress
    for name in ["A","B"] * 3:
        guard.observe_action("step", {"n": name})
        guard.step(); clk.step(1.0)
    assert any(r.startswith("SOFT:breaker_open") and "reason=ping_pong" in r for r in rec.reasons)

    # Repeat while breakers are open → HARD escalation
    before = len(rec.reasons)
    for name in ["A","B"] * 3:
        guard.observe_action("step", {"n": name})
        guard.step(); clk.step(0.5)
    assert len(rec.reasons) > before
    assert any(r.startswith("HARD:repeat_ping_pong_loop") for r in rec.reasons)

def test_ping_pong_no_trip_if_progress_happens():
    clk = FakeClock()
    guard, rec = mk_guard(clk, pingpong_k=6, pingpong_t=10.0)

    # same A/B alternation but inject a success once
    seq = ["A","B","A","B","A","B"]
    for i, name in enumerate(seq):
        guard.observe_action("act", {"n": name}, success=(i == 3))
        guard.step(); clk.step(1.0)

    assert not rec.reasons

# ---------- NO-PROGRESS LOOP ----------

def test_no_progress_loop_trips_after_enough_actions_in_window():
    clk = FakeClock()
    guard, rec = mk_guard(clk, no_progress_t=10.0, no_progress_min_actions=5)

    # 5 actions in window, no success/progress -> HARD
    for _ in range(5):
        guard.observe_action("work.unit")
        guard.step(); clk.step(2.0)
    assert any(r.startswith("HARD:no_progress_loop") for r in rec.reasons)

def test_no_progress_loop_not_tripped_if_actions_too_few_or_progress_present():
    clk = FakeClock()
    guard, rec = mk_guard(clk, no_progress_t=10.0, no_progress_min_actions=5)

    # Not enough actions in window
    for _ in range(4):
        guard.observe_action("work.unit")
        guard.step(); clk.step(1.0)
    assert not rec.reasons

    # Add actions with progress -> still no trip
    for _ in range(3):
        guard.observe_action("work.unit", progress_delta=0.5)
        guard.step(); clk.step(1.0)
    assert not rec.reasons

# ---------- RETRY SATURATION ----------

def test_retry_saturation_soft_then_hard_after_breaker_reset():
    clk = FakeClock()
    guard, rec = mk_guard(
        clk, retry_k=3, retry_w=10.0, retry_escalate_k=5, breaker_cool_s=8.0
    )

    # 3 quick retries -> opens breaker (SOFT)
    for _ in range(3):
        guard.report_retry("llm_timeout")
        guard.step(); clk.step(1.0)
    assert any(r.startswith("SOFT:breaker_open") and "reason=retry_saturation" in r for r in rec.reasons)

    # While breaker open, produce more retries to meet escalate_k -> HARD
    before = len(rec.reasons)
    for _ in range(5):
        guard.report_retry("llm_timeout")
        guard.step(); clk.step(1.0)
    assert len(rec.reasons) > before
    assert any(r.startswith("HARD:retry_saturation") for r in rec.reasons)

def test_retry_saturation_window_pruning_prevents_trip():
    clk = FakeClock()
    guard, rec = mk_guard(clk, retry_k=3, retry_w=5.0)

    # Two retries, then let them fall out of window, then one -> never reaches 3 in window
    guard.report_retry("db"); guard.step(); clk.step(2.6)
    guard.report_retry("db"); guard.step(); clk.step(2.6)  # first falls out
    guard.report_retry("db"); guard.step()
    assert not rec.reasons

# ---------- BREAKER / BLOCKING & PRUNING ----------

def test_is_blocked_true_while_breaker_open():
    clk = FakeClock()
    guard, rec = mk_guard(clk, same_call_k=2, same_call_t=10.0, breaker_cool_s=5.0)

    # trip the same-call soft breaker
    guard.observe_action("planner.plan", {"m": 1}); guard.step()
    guard.observe_action("planner.plan", {"m": 1}); guard.step()

    assert any(r.startswith("SOFT:breaker_open") for r in rec.reasons)
    assert guard.is_blocked("planner.plan", {"m": 1}) is True

    # After cool-off (advance time), breaker closes
    clk.step(6.0)
    assert guard.is_blocked("planner.plan", {"m": 1}) is False

def test_pruning_old_actions():
    clk = FakeClock()
    # shrink windows to exercise pruning path
    guard, rec = mk_guard(clk, same_call_t=3.0, pingpong_t=3.0, no_progress_t=3.0, no_progress_min_actions=3)

    # add actions then move far beyond the longest window so they are pruned
    for _ in range(3):
        guard.observe_action("x"); guard.step(); clk.step(0.5)
    clk.step(10.0)  # well past windows
    guard.step()

    # Now add fresh actions; previous ones shouldn't influence any rule
    for _ in range(2):
        guard.observe_action("x"); guard.step(); clk.step(0.5)
    assert not rec.reasons

# ---------- EDGE: action_window_n cap respected ----------

def test_action_window_cap_does_not_crash_or_overgrow():
    clk = FakeClock()
    guard, rec = mk_guard(clk, action_window_n=5, same_call_k=5, same_call_t=100.0)

    # Push many actions; deque has its own maxlen (512), but we rely on time pruning too.
    # Ensure same-call condition holds only when last 5 are identical with no progress.
    for i in range(20):
        guard.observe_action("foo", {"i": i})  # all different -> no trip
        guard.step(); clk.step(0.1)
    assert not rec.reasons

    # now 5 identical within window -> trip soft
    for _ in range(5):
        guard.observe_action("foo", {"i": 42})
        guard.step(); clk.step(0.1)
    assert any("SOFT:breaker_open" in r for r in rec.reasons)
