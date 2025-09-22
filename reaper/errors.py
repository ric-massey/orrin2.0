# reaper/errors.py
# Defines ErrorEvent and related helpers for structured error reporting.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional
import time


class Severity(IntEnum):
    """Error severity: 1 = worst (trip at 10), 2 = medium (25), 3 = least severe (50)."""
    SEV1 = 1  # worst
    SEV2 = 2
    SEV3 = 3  # least severe


# Starter registry of common AI-system error keys → default severity
SEVERITY_BY_KEY: Dict[str, int] = {
    # --- Sev1: critical correctness / safety / integrity ---
    "data_corruption": 1,
    "train_data_leakage": 1,
    "bad_checkpoint": 1,
    "model_not_loaded": 1,
    "tensor_shape_mismatch": 1,
    "dtype_mismatch": 1,
    "nan_inference": 1,
    "nan_loss": 1,
    "security_violation": 1,
    "pii_leak": 1,
    "auth_denied": 1,
    "secret_missing": 1,
    "cuda_driver_mismatch": 1,
    "gpu_init_failed": 1,
    "nccl_rendezvous_failed": 1,
    "disk_full": 1,
    "fs_unwritable": 1,
    "fd_exhausted": 1,
    "prompt_injection_detected": 1,
    "safety_violation_blocked": 1,

    # --- Sev2: availability / reliability / performance ---
    "llm_timeout": 2,
    "provider_rate_limited": 2,
    "quota_exhausted": 2,
    "retrieval_zero_results": 2,
    "retrieval_latency_high": 2,
    "embedding_service_timeout": 2,
    "vectordb_unavailable": 2,
    "index_stale": 2,
    "queue_overflow": 2,
    "consumer_lag_high": 2,
    "orchestrator_deadlock": 2,
    "oom_warning": 2,
    "memory_leak_suspected": 2,
    "cpu_hot": 2,
    "gpu_oom": 2,
    "checkpoint_save_failed": 2,
    "checkpoint_load_slow": 2,
    "metrics_emitter_down": 2,
    "log_sink_unavailable": 2,

    # --- Sev3: quality / UX / soft errors ---
    "hallucination_suspected": 3,
    "grounding_missing_citations": 3,
    "tool_call_schema_error": 3,
    "function_json_invalid": 3,
    "tokenizer_mismatch_warning": 3,
    "context_truncated": 3,
    "json_parse_warning": 3,
    "formatting_warning": 3,
    "low_similarity_retrieval": 3,
    "cost_spike_warning": 3,
    "token_usage_spike": 3,
}


@dataclass
class ErrorEvent:
    """
    A single error occurrence.

    Required:
      - key: stable identifier for the error type (e.g., "llm_timeout", "db_down")
      - severity: 1 (worst), 2, or 3 (least severe)

    Optional:
      - message: human-friendly description
      - context: extra structured data (ids, payload sizes, etc.)
      - ts: timestamp (monotonic seconds)
    """
    key: str
    severity: int
    message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # Normalize severity to 1..3; anything else becomes worst (1)
        try:
            sev = int(self.severity)
        except Exception:
            sev = 1
        if sev not in (1, 2, 3):
            sev = 1
        self.severity = sev


# --- Helpers ---

def make_event(key: str, severity: int | Severity, *,
               message: Optional[str] = None,
               context: Optional[Dict[str, Any]] = None) -> ErrorEvent:
    """Convenience factory to create an ErrorEvent with optional message/context."""
    sev = int(severity)
    return ErrorEvent(key=key, severity=sev, message=message, context=context or {})


def make_event_from_key(key: str, *,
                        message: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> ErrorEvent:
    """
    Create an ErrorEvent using SEVERITY_BY_KEY to choose severity.
    Unknown keys default to worst (1) so they don't get ignored.
    """
    sev = SEVERITY_BY_KEY.get(key, 1)
    return ErrorEvent(key=key, severity=sev, message=message, context=context or {})


def severity_for_exception(exc: BaseException) -> int:
    """
    Example heuristic to map exceptions to severities.
    Tweak as you like or delete if not needed.
    """
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()

    # Worst (1): data corruption, auth failures, hard "down"
    if any(t in name or t in msg for t in ("corrupt", "integrity", "unauthorized", "forbidden", "fatal", "panic")):
        return int(Severity.SEV1)

    # Medium (2): timeouts, rate limits, temporary unavailability
    if any(t in name or t in msg for t in ("timeout", "rate", "throttle", "unavailable", "retry")):
        return int(Severity.SEV2)

    # Least (3): minor parsing/validation or expected soft errors
    return int(Severity.SEV3)