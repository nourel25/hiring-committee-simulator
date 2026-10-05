"""Tests for hcs.runlog: writing and reading run logs. Uses tmp_path; never touches runs/."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hcs.runlog import RunRecord, append_run, read_runs
from hcs.schemas import Check, Review


def make_record(**overrides) -> RunRecord:
    """Build a valid ok record; each test overrides only the fields it cares about."""
    fields = {
        "run_id": "run-1",
        "started_at": datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc),
        "agent": "tech_lead",
        "prompt_version": "reviewer-v1",
        "model": "fake",
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 500,
        "max_retries": 1,
        "messages": [{"role": "user", "content": "Review this CV."}],
        "status": "ok",
        "attempts": 1,
        "raw_reply": "{}",
        "review": Review(checks=[Check(id="T1", check="x", evidence=None, passed=False)]),
        "score": 0.0,
        "decision": "strong_no",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_ok_and_error_records_round_trip(tmp_path):
    path = tmp_path / "logs" / "runs.jsonl"
    ok = make_record()
    error = make_record(
        run_id="run-2",
        status="error",
        error="Cut off at max_tokens (50): the JSON is incomplete.",
        raw_reply='{"checks": [',
        review=None,
        score=None,
        decision=None,
    )

    append_run(ok, path)
    append_run(error, path)

    assert read_runs(path) == [ok, error]


def test_each_record_is_one_line(tmp_path):
    path = tmp_path / "runs.jsonl"

    append_run(make_record(), path)
    append_run(make_record(run_id="run-2"), path)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_read_runs_rejects_a_damaged_line(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text('{"run_id": "run-1", "started_at":\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        read_runs(path)
