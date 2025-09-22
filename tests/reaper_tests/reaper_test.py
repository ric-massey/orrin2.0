# tests/reaper_tests/reaper_test.py
import signal
import types

import reaper.reaper as reaper_mod
from reaper.reaper import Reaper, kill_current_process, signal_pid


class KillRecorder:
    def __init__(self):
        self.reasons = []
    def __call__(self, reason: str):
        self.reasons.append(reason)


class FakeCounter:
    """Minimal prometheus-like metric stub."""
    def __init__(self):
        self.calls = []  # list of {"reason": <label>}
        self.count = 0
    def labels(self, *, reason: str):
        # record label and return self (like a Counter with bound labels)
        self.calls.append({"reason": reason})
        return self
    def inc(self, n: float = 1.0):
        self.count += n


def test_trigger_calls_kill_logs_and_metrics(monkeypatch, capsys):
    # install a fake metric object in the module under test
    fake = FakeCounter()
    monkeypatch.setattr(reaper_mod, "reaper_trips_total", fake, raising=True)

    # kill fn recorder
    k = KillRecorder()
    r = Reaper(kill=k)

    # reason includes a first token + extra details
    reason = "HARD:pulse_too_fast avg_ms=1.00"
    r.trigger(reason)

    # kill fn invoked with full reason
    assert k.reasons == [reason]

    # stderr contains the log line
    out = capsys.readouterr()
    assert "[REAPER] Shutting down: HARD:pulse_too_fast avg_ms=1.00" in out.err

    # metrics got only the first token as the label
    assert fake.count == 1
    assert fake.calls and fake.calls[0]["reason"] == "HARD:pulse_too_fast"


def test_trigger_no_metrics_does_not_crash(monkeypatch, capsys):
    # simulate metrics unavailable
    monkeypatch.setattr(reaper_mod, "reaper_trips_total", None, raising=True)

    k = KillRecorder()
    r = Reaper(kill=k)
    r.trigger("HARD:some_reason extra")

    # kill still called; no exception; stderr has a line
    assert k.reasons == ["HARD:some_reason extra"]
    assert "Shutting down: HARD:some_reason extra" in capsys.readouterr().err


def test_kill_current_process_exits_monkeypatched(monkeypatch):
    # prevent the test process from exiting; capture the code instead
    called = {}
    def fake_exit(code):
        called["code"] = code
        # do not actually exit

    monkeypatch.setattr(reaper_mod.os, "_exit", fake_exit, raising=True)

    # call the strategy
    kill_current_process("ignored")
    assert called.get("code") == 1


def test_signal_pid_happy_path(monkeypatch, capsys):
    calls = []

    def fake_kill(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(reaper_mod.os, "kill", fake_kill, raising=True)

    fn = signal_pid(4321, sig=signal.SIGUSR1)
    fn("any-reason")
    assert calls == [(4321, signal.SIGUSR1)]

    # should not print errors in happy path
    assert capsys.readouterr().err == ""


def test_signal_pid_process_lookup_error(monkeypatch, capsys):
    def fake_kill(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(reaper_mod.os, "kill", fake_kill, raising=True)

    fn = signal_pid(999999, sig=signal.SIGTERM)
    # should not raise; should print a message about PID not found
    fn("whatever")

    err = capsys.readouterr().err
    assert "PID 999999 not found" in err
