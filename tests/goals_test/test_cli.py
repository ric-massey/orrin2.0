# tests/goals/test_cli.py
# Pytest suite for the Goals CLI (create/list/describe/update/cancel/submit/pause/resume)

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from goals import cli


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Fresh data directory per test."""
    d = tmp_path / "goals-data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_cli(args, capsys) -> tuple[int, str, str]:
    """Invoke cli.main(argv) and capture stdout/stderr."""
    rc = cli.main(args)
    out, err = capsys.readouterr()
    return rc, out.strip(), err.strip()


def _created_id_from(out: str) -> str:
    m = re.search(r"created\s+(\S+)", out)
    assert m, f"could not find created id in output: {out!r}"
    return m.group(1)


def test_add_list_describe_update_cancel_flow(data_dir: Path, capsys: pytest.CaptureFixture) -> None:
    # add
    rc, out, err = _run_cli(
        ["--data-dir", str(data_dir), "add", "research", "Test goal end-to-end"],
        capsys,
    )
    assert rc == 0 and not err
    gid = _created_id_from(out)

    # list (json)
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "list"], capsys)
    assert rc == 0 and not err
    items = json.loads(out)
    assert any(g["id"] == gid for g in items), f"goal {gid} not in list"

    # describe (json)
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "describe", gid], capsys)
    assert rc == 0 and not err
    g = json.loads(out)
    assert g["id"] == gid
    assert g["kind"] == "research"
    assert g["status"] in {"NEW", "READY"}  # planner may mark READY

    # update title + priority
    rc, out, err = _run_cli(
        ["--data-dir", str(data_dir), "update", gid, "--title", "Renamed goal", "--priority", "CRITICAL"],
        capsys,
    )
    assert rc == 0 and "updated" in out and not err

    # verify update
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "describe", gid], capsys)
    g2 = json.loads(out)
    assert g2["title"] == "Renamed goal"
    assert g2["priority"] in {"CRITICAL", 3}

    # cancel
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "cancel", gid, "--reason", "tests"], capsys)
    assert rc == 0 and "cancelled" in out

    # verify cancelled
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "describe", gid], capsys)
    g3 = json.loads(out)
    assert g3["status"] == "CANCELLED"

    # basic files exist
    assert (data_dir / "wal.log").exists()
    assert (data_dir / "state.jsonl").exists()


def test_pause_then_resume(data_dir: Path, capsys: pytest.CaptureFixture) -> None:
    # add a coding goal
    rc, out, err = _run_cli(
        ["--data-dir", str(data_dir), "add", "coding", "Implement feature X"],
        capsys,
    )
    gid = _created_id_from(out)

    # pause
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "pause", gid], capsys)
    assert rc == 0 and "paused" in out

    # verify PAUSED
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "describe", gid], capsys)
    st = json.loads(out)["status"]
    assert st == "PAUSED"

    # resume
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "resume", gid], capsys)
    assert rc == 0 and "resumed" in out

    # verify READY
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "describe", gid], capsys)
    st2 = json.loads(out)["status"]
    assert st2 in {"READY", "NEW"}  # may be NEW until daemon plans


def test_add_with_spec_file_and_list_filters(data_dir: Path, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    # prepare a spec file on disk
    spec = {"queries": ["site:example.com foo", "bar overview"], "fetch_limit": 3}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    # add research goal with @spec.json and tags
    rc, out, err = _run_cli(
        [
            "--data-dir",
            str(data_dir),
            "add",
            "research",
            "Investigate foo",
            "--spec",
            f"@{spec_path}",
            "--tags",
            "lab,urgent",
            "--priority",
            "HIGH",
        ],
        capsys,
    )
    gid = _created_id_from(out)

    # describe and verify spec was ingested
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "describe", gid], capsys)
    g = json.loads(out)
    assert g["spec"]["fetch_limit"] == 3
    assert "urgent" in g.get("tags", []) or "urgent" in (g.get("tags") or [])
    assert g["priority"] in {"HIGH", 2}

    # list filter by kind
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "list", "--kinds", "research"], capsys)
    items = json.loads(out)
    assert all(x["kind"] == "research" for x in items)

    # list filter by text
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "--json", "list", "--text", "Investigate"], capsys)
    items2 = json.loads(out)
    assert any(x["id"] == gid for x in items2)


def test_submit_quiet_flag(data_dir: Path, capsys: pytest.CaptureFixture) -> None:
    # create a goal
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "add", "housekeeping", "Nightly maintenance"], capsys)
    gid = _created_id_from(out)

    # submit (quiet) — ensures command works and prints nothing
    rc, out, err = _run_cli(["--data-dir", str(data_dir), "submit", gid, "--quiet"], capsys)
    assert rc == 0
    assert out == "" and err == ""
