"""Tests for the SQLite traces built from the run log."""

import sqlite3
from datetime import datetime, timezone

from hcs.runlog import RunRecord, append_run
from hcs.tracing import build, summary


def record(agent: str, status: str = "ok", seconds: float | None = 10.0) -> RunRecord:
    return RunRecord(
        run_id=f"{agent}-{status}-{seconds}",
        started_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
        agent=agent,
        prompt_version="v1",
        model="m",
        temperature=0,
        seed=42,
        max_tokens=10,
        max_retries=1,
        messages=[{"role": "user", "content": "x"}],
        status=status,
        attempts=1,
        raw_reply="{}",
        seconds=seconds,
        prompt_tokens=100,
        completion_tokens=50,
    )


def test_every_run_becomes_a_row_and_failures_are_counted(tmp_path):
    runs, traces = tmp_path / "runs.jsonl", tmp_path / "traces.db"
    for item in [
        record("tech_lead"),
        record("tech_lead", "error", None),
        record("recruiter", seconds=4.0),
    ]:
        append_run(item, runs)

    assert build(runs, traces) == 3
    assert summary(traces) == [
        ("recruiter", "m", 1, 0, 4.0, 100.0, 50.0),
        ("tech_lead", "m", 2, 1, 10.0, 100.0, 50.0),
    ]
    # Rebuilding replaces the table instead of adding to it.
    build(runs, traces)
    with sqlite3.connect(traces) as db:
        assert db.execute("select count(*) from runs").fetchone() == (3,)
