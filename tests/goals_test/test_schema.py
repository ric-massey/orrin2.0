# tests/goals/test_schema.py
# Pytest for goals.schema (schemas + validators for goal.spec and step.action)

from __future__ import annotations

import pytest

from goals import schema as S


def test_get_goal_schema_by_kind_and_generic():
    g_generic = S.get_goal_schema()
    g_coding = S.get_goal_schema("coding")
    g_research = S.get_goal_schema("research")
    g_house = S.get_goal_schema("housekeeping")

    # basic shape
    assert isinstance(g_generic, dict) and isinstance(g_coding, dict)

    # spot-check kind-specific fields
    assert "triggers" in g_generic.get("properties", {})
    assert "repo" in g_coding.get("properties", {})
    assert "queries" in g_research.get("properties", {})
    assert "tasks" in g_house.get("properties", {})


def test_get_step_action_schema_by_kind_and_op():
    # Known op
    a_patch = S.get_step_action_schema("coding", "apply_patch")
    assert a_patch.get("required") and "diff" in a_patch["required"]

    # Unknown kind/op → generic schema (only requires 'op')
    a_unknown = S.get_step_action_schema("wut", "nope")
    assert a_unknown.get("required") == ["op"]


def test_validate_step_action_enforces_required_fields():
    # Missing 'op' is always an error (pre-check), independent of jsonschema presence
    with pytest.raises(ValueError):
        S.validate_step_action({}, kind="coding")

    # For a known op with extra required fields (e.g., 'diff' for apply_patch)…
    with pytest.raises(ValueError):
        S.validate_step_action({"op": "apply_patch"}, kind="coding")  # missing 'diff'

    # …and succeeds when required fields are present
    S.validate_step_action({"op": "apply_patch", "diff": "--- a\n+++ b\n"}, kind="coding")


def test_validate_goal_spec_permissive_minimal_ok():
    # Minimal empty specs should validate for all kinds (schemas are permissive)
    for k in (None, "coding", "research", "housekeeping"):
        S.validate_goal_spec({}, kind=k)


def test_validate_goal_spec_allof_conflict_requires_jsonschema():
    # This test exercises a rule that's only enforced when jsonschema is available.
    pytest.importorskip("jsonschema")
    # Coding goal forbids having both 'files' and 'diff' via allOf/not rule.
    bad_spec = {"files": {"foo.py": "print(1)"}, "diff": "--- a\n+++ b\n"}
    with pytest.raises(ValueError):
        S.validate_goal_spec(bad_spec, kind="coding")
