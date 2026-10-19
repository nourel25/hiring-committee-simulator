"""One review of one case: call the agent, apply the policies, score, and log the record."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from hcs.agents import tech_lead
from hcs.cases import Case, Checklist
from hcs.llm import LLMOutputError, get_structured
from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    decide,
    failed_must_haves,
)
from hcs.runlog import RUNS_PATH, RunRecord, append_run
from hcs.schemas import Review, review_model_for


def review_case(
    client: OpenAI,
    case: Case,
    checklist: Checklist,
    runs_path: Path = RUNS_PATH,
    model: str = tech_lead.MODEL,
    temperature: float = tech_lead.TEMPERATURE,
    seed: int = tech_lead.SEED,
    max_tokens: int = tech_lead.MAX_TOKENS,
    strict: bool = False,
) -> RunRecord:
    """Run the tech lead on one case, log the result, and return the logged record."""
    messages = tech_lead.build_messages(case.cv, checklist.items)
    response_model = review_model_for([item.id for item in checklist.items])
    common = {
        "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc),
        "agent": "tech_lead",
        "case_id": case.id,
        "checklist": f"{checklist.name}@v{checklist.version}",
        "prompt_version": tech_lead.PROMPT_VERSION,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "max_retries": tech_lead.MAX_RETRIES,
        "messages": messages,
    }

    try:
        result = get_structured(
            client,
            messages,
            response_model=response_model,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            max_retries=tech_lead.MAX_RETRIES,
            strict=strict,
        )
    except LLMOutputError as error:
        record = RunRecord(
            **common,
            status="error",
            error=str(error),
            attempts=error.attempts,
            raw_reply=error.raw_reply,
        )
        append_run(record, runs_path)
        return record

    # Validated against this checklist's ids; stored as a plain Review like every other record.
    review = Review.model_validate(result.parsed.model_dump())
    policy_changes = apply_evidence_policy(review)
    truth_changes, missing_quotes = apply_truth_check(review, case.cv)
    score = compute_score(review, checklist.items)
    failed = failed_must_haves(review, checklist.items)

    record = RunRecord(
        **common,
        status="ok",
        attempts=result.attempts,
        seconds=result.seconds,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        raw_reply=result.raw_reply,
        evidence_policy_changes=policy_changes,
        truth_check_changes=truth_changes,
        missing_quotes=missing_quotes,
        failed_must_haves=failed,
        review=review,
        score=score,
        decision=decide(score, failed),
    )
    append_run(record, runs_path)
    return record
