"""Traces in SQLite: every logged run as a row, so cost and latency can be queried (Step 7).

The run log (runs/runs.jsonl) stays the source of truth: one validated line per model call, with
the exact messages and reply. This module rebuilds runs/traces.db from it and prints the cost and
latency per agent and model, the design's "tokens and seconds per session on local hardware".

    python -m hcs.tracing                # rebuild and print the summary
    sqlite3 runs/traces.db "select ..."  # anything else
"""

import sqlite3
from pathlib import Path

from hcs.runlog import RUNS_PATH, read_runs

TRACES_PATH = Path("runs") / "traces.db"

COLUMNS = {
    "run_id": "text primary key",
    "started_at": "text",
    "agent": "text",
    "case_id": "text",
    "checklist": "text",
    "prompt_version": "text",
    "model": "text",
    "temperature": "real",
    "seed": "integer",
    "status": "text",
    "attempts": "integer",
    "seconds": "real",
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "score": "real",
    "decision": "text",
}


def build(runs_path: Path = RUNS_PATH, traces_path: Path = TRACES_PATH) -> int:
    """Write every run in the log to a fresh traces table; returns the number of rows."""
    records = read_runs(runs_path)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(traces_path) as db:
        db.execute("drop table if exists runs")
        db.execute(f"create table runs ({', '.join(f'{k} {v}' for k, v in COLUMNS.items())})")
        rows = []
        for record in records:
            data = record.model_dump(mode="json", include=set(COLUMNS))
            rows.append(tuple(data[column] for column in COLUMNS))
        db.executemany(f"insert into runs values ({', '.join('?' for _ in COLUMNS)})", rows)
    return len(records)


SUMMARY = """
select agent, model, count(*) as runs,
       sum(status != 'ok') as failed,
       avg(seconds) as seconds,
       avg(prompt_tokens) as tokens_in,
       avg(completion_tokens) as tokens_out
from runs
group by agent, model
order by agent, runs desc
"""


def summary(traces_path: Path = TRACES_PATH) -> list[tuple]:
    with sqlite3.connect(traces_path) as db:
        return db.execute(SUMMARY).fetchall()


def main() -> None:
    count = build()
    print(f"{count} runs written to {TRACES_PATH}\n")
    print(f"{'agent':<18}{'model':<24}{'runs':>6}{'failed':>8}{'sec':>7}{'in':>7}{'out':>6}")
    for agent, model, runs, failed, seconds, tokens_in, tokens_out in summary():
        print(
            f"{agent:<18}{model:<24}{runs:>6}{failed:>8}{seconds or 0:>7.1f}"
            f"{tokens_in or 0:>7.0f}{tokens_out or 0:>6.0f}"
        )


if __name__ == "__main__":
    main()
