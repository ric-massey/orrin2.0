# reaper/repeat.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional, Tuple, Any, List
from collections import deque, Counter
import time
import json

OnViolation = Callable[[str], None]
NowFn = Callable[[], float]

def _fingerprint(name: str, key_args: Optional[Dict[str, Any]] = None) -> str:
    if not key_args:
        return name
    try:
        blob = json.dumps(key_args, sort_keys=True, separators=(",", ":"))
    except Exception:
        blob = str(sorted(key_args.items()))
    return f"{name}|{blob}"

@dataclass
class Breaker:
    open_until: float = 0.0
    trips: int = 0
    def is_open(self, now: float) -> bool:
        return now < self.open_until

@dataclass
class RepeatLoopGuard:
    on_violation: OnViolation
    now_fn: NowFn = time.monotonic

    # Windows & thresholds
    action_window_n: int = 50
    same_call_k: int = 5              # ← was 4; tests expect no trip at 4
    same_call_t: float = 30.0
    breaker_cool_s: float = 60.0

    pingpong_k: int = 6
    pingpong_t: float = 30.0

    no_progress_t: float = 60.0
    no_progress_min_actions: int = 20
    no_progress_min_span_ratio: float = 0.80  # ← require window span >= 80% of no_progress_t

    retry_k: int = 5
    retry_w: float = 30.0
    retry_escalate_k: int = 8

    _actions: Deque[Tuple[float, str, bool, float]] = field(default_factory=lambda: deque(maxlen=512))
    _retries: Dict[str, Deque[float]] = field(default_factory=dict)
    _breakers_fp: Dict[str, Breaker] = field(default_factory=dict)
    _breakers_retry: Dict[str, Breaker] = field(default_factory=dict)

    def observe_action(self, func_name: str, key_args: Optional[Dict[str, Any]] = None, *,
                       success: bool = False, progress_delta: float = 0.0) -> None:
        now = self.now_fn()
        fp = _fingerprint(func_name, key_args)
        self._actions.append((now, fp, bool(success), float(progress_delta)))

    def report_retry(self, error_key: str) -> None:
        now = self.now_fn()
        dq = self._retries.setdefault(error_key, deque(maxlen=1024))
        dq.append(now)

    def is_blocked(self, func_name: str, key_args: Optional[Dict[str, Any]] = None) -> bool:
        now = self.now_fn()
        fp = _fingerprint(func_name, key_args)
        br = self._breakers_fp.get(fp)
        return bool(br and br.is_open(now))

    def step(self) -> None:
        now = self.now_fn()
        self._prune_retries(now)
        self._prune_actions(now)
        self._check_same_call_loop(now)
        self._check_ping_pong_loop(now)
        self._check_no_progress(now)
        self._check_retry_saturation(now)

    def _check_same_call_loop(self, now: float) -> None:
        if not self._actions:
            return
        recent = [a for a in self._actions if now - a[0] <= self.same_call_t]
        if len(recent) < self.same_call_k:
            return
        counts = Counter(fp for _, fp, _, _ in recent)
        fp, freq = counts.most_common(1)[0]
        progress = any(s or (pd > 0) for (_, fp2, s, pd) in recent if fp2 == fp)
        if freq >= self.same_call_k and not progress:
            br = self._breakers_fp.setdefault(fp, Breaker())
            if br.is_open(now):
                self._trip(f"HARD:repeat_same_call_loop fp={fp} freq={freq} window_s={self.same_call_t}")
            else:
                br.open_until = now + self.breaker_cool_s
                br.trips += 1
                self._trip_soft(f"SOFT:breaker_open fp={fp} reason=same_call_loop "
                                f"freq={freq} window_s={self.same_call_t} cool_s={self.breaker_cool_s}")

    def _check_ping_pong_loop(self, now: float) -> None:
        seq: List[str] = [fp for (ts, fp, _, _) in self._actions if now - ts <= self.pingpong_t]
        if len(seq) < self.pingpong_k:
            return
        tail = seq[-self.pingpong_k:]
        if len(set(tail)) <= 2 and self._is_alternating(tail):
            progress = any((s or pd > 0) for (ts, fp, s, pd) in self._actions if now - ts <= self.pingpong_t)
            if not progress:
                fps = list(set(tail))
                escalated = False
                for f in fps:
                    br = self._breakers_fp.setdefault(f, Breaker())
                    if br.is_open(now):
                        escalated = True
                    br.open_until = now + self.breaker_cool_s
                    br.trips += 1
                if escalated:
                    self._trip(f"HARD:repeat_ping_pong_loop fps={fps} cycles={self.pingpong_k} window_s={self.pingpong_t}")
                else:
                    self._trip_soft(f"SOFT:breaker_open fps={fps} reason=ping_pong "
                                    f"cycles={self.pingpong_k} window_s={self.pingpong_t} cool_s={self.breaker_cool_s}")

    def _check_no_progress(self, now: float) -> None:
        window = [(ts, fp, s, pd) for (ts, fp, s, pd) in self._actions if now - ts <= self.no_progress_t]
        if len(window) < self.no_progress_min_actions:
            return
        span = window[-1][0] - window[0][0]
        if span < self.no_progress_t * self.no_progress_min_span_ratio:
            # window didn’t actually cover enough time → don’t trip
            return
        progressed = any(s or (pd > 0) for (_, _, s, pd) in window)
        if not progressed:
            self._trip(f"HARD:no_progress_loop actions={len(window)} window_s={self.no_progress_t}")

    def _check_retry_saturation(self, now: float) -> None:
        if not self._retries:
            return
        for key, dq in self._retries.items():
            cutoff = now - self.retry_w
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.retry_k:
                br = self._breakers_retry.setdefault(key, Breaker())
                if br.is_open(now):
                    if len(dq) >= self.retry_escalate_k:
                        self._trip(f"HARD:retry_saturation key={key} count={len(dq)} window_s={self.retry_w}")
                else:
                    br.open_until = now + self.breaker_cool_s
                    br.trips += 1
                    self._trip_soft(f"SOFT:breaker_open key={key} reason=retry_saturation "
                                    f"count={len(dq)} window_s={self.retry_w} cool_s={self.breaker_cool_s}")

    @staticmethod
    def _is_alternating(seq: List[str]) -> bool:
        if len(seq) < 4:
            return False
        a = seq[0]
        b = next((x for x in seq[1:] if x != a), None)
        if b is None:
            return False
        expect = a
        for x in seq:
            if x != expect:
                return False
            expect = b if expect == a else a
        return True

    def _prune_actions(self, now: float) -> None:
        max_t = max(self.same_call_t, self.pingpong_t, self.no_progress_t)
        cutoff = now - max_t * 1.5
        while self._actions and self._actions[0][0] < cutoff:
            self._actions.popleft()

    def _prune_retries(self, now: float) -> None:
        cutoff = now - self.retry_w * 1.5
        for dq in self._retries.values():
            while dq and dq[0] < cutoff:
                dq.popleft()

    def _trip(self, reason: str) -> None:
        self.on_violation(reason)

    def _trip_soft(self, note: str) -> None:
        self.on_violation(note)
