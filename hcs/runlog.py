"""Run logs: one JSON line per run, success or failure, so runs can be replayed and compared."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from hcs.schemas import ClaimJudgement, Decision, Objection, Review, SummaryLine

RUNS_PATH = Path("runs") / "runs.jsonl"


class RunRecord(BaseModel):
    """Everything needed to replay and explain one run. Read back with validation."""

    # Identity
    run_id: str
    started_at: datetime
    agent: str
    case_id: str | None = None
    checklist: str | None = None

    # Versions and settings
    prompt_version: str
    model: str
    temperature: float
    # None where the provider takes no seed (Gemini).
    seed: int | None
    max_tokens: int
    max_retries: int

    # Inputs: what the model actually saw
    messages: list[dict[str, str]]

    # Outcome
    status: Literal["ok", "error"]
    error: str | None = None

    # The call
    attempts: int
    raw_reply: str
    seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # Policies and result (None when the run failed)
    evidence_policy_changes: int | None = None
    truth_check_changes: int | None = None
    missing_quotes: list[str] = []
    failed_must_haves: list[str] = []
    review: Review | None = None
    score: float | None = None
    decision: Decision | None = None

    # The devil's advocate's result instead of a review (None for other reviewers and failed runs)
    objections: list[Objection] | None = None
    ungrounded_objections: list[int] = []

    # The grounding judge's result (None for other agents and failed runs)
    judgements: list[ClaimJudgement] | None = None
    # The refiner's summary lines before the grounding check (None for other agents)
    summary: list[SummaryLine] | None = None


def append_run(record: RunRecord, path: Path = RUNS_PATH) -> None:
    """Add one record as a single line at the end of the log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")


def read_runs(path: Path = RUNS_PATH) -> list[RunRecord]:
    """Read every record back, validating each line."""
    with path.open(encoding="utf-8") as file:
        return [RunRecord.model_validate_json(line) for line in file if line.strip()]
