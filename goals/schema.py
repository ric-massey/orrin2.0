# goals/schema.py
# JSON Schemas + validators for Goal.spec and Step.action (kind/op-aware, permissive by default)

from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, List

try:
    # Prefer modern draft validator if present
    from jsonschema import Draft202012Validator as _Validator  # type: ignore
    _JSONSCHEMA_AVAILABLE = True
except Exception:  # pragma: no cover
    _JSONSCHEMA_AVAILABLE = False
    _Validator = None  # type: ignore


# ---------------------------
# Public API
# ---------------------------

def get_goal_schema(kind: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a JSON Schema dict for Goal.spec based on goal kind.
    If kind is unknown/None, returns a permissive generic schema.
    """
    kind = (kind or "").strip().lower()
    if kind == "coding":
        return _SCHEMA_GOAL_SPEC_CODING
    if kind == "research":
        return _SCHEMA_GOAL_SPEC_RESEARCH
    if kind == "housekeeping":
        return _SCHEMA_GOAL_SPEC_HOUSEKEEPING
    return _SCHEMA_GOAL_SPEC_GENERIC


def get_step_action_schema(kind: Optional[str] = None, op: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a JSON Schema dict for a Step.action payload.
    Uses kind/op when known; otherwise returns a generic action schema.
    """
    k = (kind or "").strip().lower()
    o = (op or "").strip().lower()
    if k == "coding":
        return _CODING_OP_SCHEMAS.get(o, _SCHEMA_ACTION_GENERIC)
    if k == "research":
        return _RESEARCH_OP_SCHEMAS.get(o, _SCHEMA_ACTION_GENERIC)
    if k == "housekeeping":
        return _HOUSEKEEPING_OP_SCHEMAS.get(o, _SCHEMA_ACTION_GENERIC)
    return _SCHEMA_ACTION_GENERIC


def validate_goal_spec(spec: Dict[str, Any], *, kind: Optional[str] = None) -> None:
    """
    Validate a Goal.spec dict. Raises ValueError with a readable message on failure.
    No-op (permissive) if jsonschema is not installed.
    """
    _validate_with_schema(get_goal_schema(kind), spec, where=f"goal.spec (kind={kind or 'generic'})")


def validate_step_action(action: Dict[str, Any], *, kind: Optional[str] = None, op: Optional[str] = None) -> None:
    """
    Validate a Step.action dict. Raises ValueError on failure.
    If 'op' is not provided, we read it from the action.
    """
    op = op or str(action.get("op") or "").strip()
    if not op:
        raise ValueError("step.action must include an 'op' field (non-empty string)")
    _validate_with_schema(get_step_action_schema(kind, op), action, where=f"step.action (kind={kind or 'generic'}, op={op})")


# ---------------------------
# Internal: schemas
# ---------------------------

# Generic (fallback) goal.spec — permissive but with common patterns
_SCHEMA_GOAL_SPEC_GENERIC: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "triggers": {"type": "array", "items": {"type": "object"}},
        "locks": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

# Coding goal.spec
_SCHEMA_GOAL_SPEC_CODING: Dict[str, Any] = {
    **_SCHEMA_GOAL_SPEC_GENERIC,
    "properties": {
        **_SCHEMA_GOAL_SPEC_GENERIC["properties"],  # type: ignore[index]
        "repo": {"type": "string"},
        "branch": {"type": "string"},
        "allow_dirty": {"type": "boolean"},
        "files": {
            "type": "object",
            "patternProperties": {".+": {"type": "string"}},
            "additionalProperties": False,
        },
        "diff": {"type": "string"},
        "commit_message": {"type": "string"},
        "tests": {"oneOf": [{"type": "boolean"}, {"type": "string"}]},
        "tests_timeout": {"type": "integer", "minimum": 1},
        "summary": {"type": "boolean"},
    },
    "allOf": [
        # don't allow both files and diff together
        {"not": {"required": ["files", "diff"]}},
    ],
}

# Research goal.spec
_SCHEMA_GOAL_SPEC_RESEARCH: Dict[str, Any] = {
    **_SCHEMA_GOAL_SPEC_GENERIC,
    "properties": {
        **_SCHEMA_GOAL_SPEC_GENERIC["properties"],  # type: ignore[index]
        "queries": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
        "urls": {"type": "array", "items": {"type": "string", "format": "uri"}},
        "per_query_k": {"type": "integer", "minimum": 1, "maximum": 25},
        "fetch_limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "synth_kind": {"type": "string", "enum": ["memo", "bullets", "report"]},
        "output_name": {"type": "string"},
        "include_citations": {"type": "boolean"},
        "style": {"type": "string"},
    },
}

# Housekeeping goal.spec
_SCHEMA_GOAL_SPEC_HOUSEKEEPING: Dict[str, Any] = {
    **_SCHEMA_GOAL_SPEC_GENERIC,
    "properties": {
        **_SCHEMA_GOAL_SPEC_GENERIC["properties"],  # type: ignore[index]
        "tasks": {"type": "array", "items": {"type": "string"}},
        "opts": {"type": "object"},
        "repo": {"type": "string"},
    },
}

# Generic step.action (must have op; anything else is allowed)
_SCHEMA_ACTION_GENERIC: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["op"],
    "properties": {
        "op": {"type": "string", "minLength": 1},
        "locks": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

# Coding ops
_CODING_OP_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "git_status_clean": {
        **_SCHEMA_ACTION_GENERIC,
        "properties": {**_SCHEMA_ACTION_GENERIC["properties"]},  # type: ignore[index]
    },
    "git_branch": {
        **_SCHEMA_ACTION_GENERIC,
        "required": ["op", "branch"],
        "properties": {
            **_SCHEMA_ACTION_GENERIC["properties"],  # type: ignore[index]
            "branch": {"type": "string", "minLength": 1},
        },
    },
    "apply_files": {
        **_SCHEMA_ACTION_GENERIC,
        "required": ["op", "files"],
        "properties": {
            **_SCHEMA_ACTION_GENERIC["properties"],  # type: ignore[index]
            "files": {"type": "object", "patternProperties": {".+": {"type": "string"}}, "additionalProperties": False},
            "commit_message": {"type": "string"},
        },
    },
    "apply_patch": {
        **_SCHEMA_ACTION_GENERIC,
        "required": ["op", "diff"],
        "properties": {
            **_SCHEMA_ACTION_GENERIC["properties"],  # type: ignore[index]
            "diff": {"type": "string", "minLength": 1},
            "commit_message": {"type": "string"},
        },
    },
    "run_cmd": {
        **_SCHEMA_ACTION_GENERIC,
        "required": ["op", "cmd"],
        "properties": {
            **_SCHEMA_ACTION_GENERIC["properties"],  # type: ignore[index]
            "cmd": {"type": "string", "minLength": 1},
            "timeout": {"type": "integer", "minimum": 1},
        },
    },
    "summarize": {
        **_SCHEMA_ACTION_GENERIC,
        "properties": {**_SCHEMA_ACTION_GENERIC["properties"]},  # type: ignore[index]
    },
}

# Research ops
_RESEARCH_OP_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "draft_queries": {**_SCHEMA_ACTION_GENERIC},
    "search": {**_SCHEMA_ACTION_GENERIC},
    "fetch": {**_SCHEMA_ACTION_GENERIC},
    "synthesize": {
        **_SCHEMA_ACTION_GENERIC,
        "properties": {**_SCHEMA_ACTION_GENERIC["properties"]},  # extensible
    },
}

# Housekeeping ops
_HOUSEKEEPING_OP_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "snapshot_goals": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "prune_goals_wal": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "snapshot_memory": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "prune_memory_wal": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "vacuum_logs": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "clean_tmp": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "pip_check": {**_SCHEMA_ACTION_GENERIC},
    "pytest_smoke": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "ruff_lint": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "ruff_format": {**_SCHEMA_ACTION_GENERIC, "properties": {**_SCHEMA_ACTION_GENERIC["properties"], "opts": {"type": "object"}}},  # type: ignore[index]
    "reindex_memory": {**_SCHEMA_ACTION_GENERIC},
}


# ---------------------------
# Internal: validation runner
# ---------------------------

def _validate_with_schema(schema: Dict[str, Any], instance: Dict[str, Any], *, where: str) -> None:
    if not _JSONSCHEMA_AVAILABLE:
        # Soft fallback: ensure it's a dict and required fields are present
        if not isinstance(instance, dict):
            raise ValueError(f"{where}: expected object/dict, got {type(instance).__name__}")
        req = schema.get("required", [])
        missing = [k for k in req if k not in instance]
        if missing:
            raise ValueError(f"{where}: missing required field(s): {', '.join(missing)}")
        return

    validator = _Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if not errors:
        return

    msgs: List[str] = []
    for e in errors[:5]:  # cap to first five
        path = "$" + "".join(f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in e.path)
        msgs.append(f"{path}: {e.message}")
    more = "" if len(errors) <= 5 else f" (+{len(errors)-5} more)"
    raise ValueError(f"{where}: schema validation failed: " + "; ".join(msgs) + more)


__all__ = [
    "get_goal_schema",
    "get_step_action_schema",
    "validate_goal_spec",
    "validate_step_action",
]
