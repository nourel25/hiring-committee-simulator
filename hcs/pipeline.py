"""One review of one case: call the agent, apply the policies, score, and log the record."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from hcs.agents import hiring_manager, recruiter, tech_lead
from hcs.cases import Case, Checklist, Posting
from hcs.llm import LLMOutputError, get_structured
from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    decide,
    failed_must_haves,
)
from hcs.runlog import RUNS_PATH, RunRecord, append_run
from hcs.schemas import Check, Review, review_model_for
from hcs.screening import recruiter_view, screening_checks


def run_review(
    client: OpenAI,
    *,
    agent: str,
    prompt_version: str,
    case_id: str,
    checklist: Checklist,
    messages: list[dict[str, str]],
    model_ids: list[str],
    truth_text: str,
    code_checks: list[Check] | None = None,
    runs_path: Path,
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
    max_retries: int,
    strict: bool,
) -> RunRecord:
    """Any reviewer's run: the model answers model_ids, the policies check its quotes against
    truth_text, code checks join unchanged, and the whole checklist is scored, gated and logged."""
    common = {
        "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc),
        "agent": agent,
        "case_id": case_id,
        "checklist": f"{checklist.name}@v{checklist.version}",
        "prompt_version": prompt_version,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "messages": messages,
    }

    try:
        result = get_structured(
            client,
            messages,
            response_model=review_model_for(model_ids),
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            max_retries=max_retries,
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

    # Validated against these ids; stored as a plain Review like every other record.
    model_review = Review.model_validate(result.parsed.model_dump())
    policy_changes = apply_evidence_policy(model_review)
    truth_changes, missing_quotes = apply_truth_check(model_review, truth_text)
    # Code checks skip the policies: their evidence is what code found, not a quote.
    review = Review(checks=[*(code_checks or []), *model_review.checks])
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
    return run_review(
        client,
        agent="tech_lead",
        prompt_version=tech_lead.PROMPT_VERSION,
        case_id=case.id,
        checklist=checklist,
        messages=tech_lead.build_messages(case.cv, checklist.items),
        model_ids=[item.id for item in checklist.items],
        truth_text=case.cv,
        runs_path=runs_path,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_retries=tech_lead.MAX_RETRIES,
        strict=strict,
    )


def review_recruiter(
    client: OpenAI,
    case: Case,
    posting: Posting,
    checklist: Checklist,
    this_year: int,
    runs_path: Path = RUNS_PATH,
    model: str = recruiter.MODEL,
    temperature: float = recruiter.TEMPERATURE,
    seed: int = recruiter.SEED,
    max_tokens: int = recruiter.MAX_TOKENS,
    strict: bool = False,
) -> RunRecord:
    """Run the recruiter on one case: code screens, the model judges the top third's clarity."""
    cv_top = recruiter_view(case.cv)
    checks = recruiter.model_items(checklist.items)
    return run_review(
        client,
        agent="recruiter",
        prompt_version=recruiter.PROMPT_VERSION,
        case_id=case.id,
        checklist=checklist,
        messages=recruiter.build_messages(cv_top, checks),
        model_ids=[item.id for item in checks],
        truth_text=cv_top,
        code_checks=screening_checks(case.cv, posting.recruiter, this_year),
        runs_path=runs_path,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_retries=recruiter.MAX_RETRIES,
        strict=strict,
    )


def review_hiring_manager(
    client: OpenAI,
    case: Case,
    posting: Posting,
    checklist: Checklist,
    runs_path: Path = RUNS_PATH,
    model: str = hiring_manager.MODEL,
    temperature: float = hiring_manager.TEMPERATURE,
    seed: int = hiring_manager.SEED,
    max_tokens: int = hiring_manager.MAX_TOKENS,
    strict: bool = False,
) -> RunRecord:
    """Run the hiring manager on one case: the full CV judged against the posting's brief."""
    return run_review(
        client,
        agent="hiring_manager",
        prompt_version=hiring_manager.PROMPT_VERSION,
        case_id=case.id,
        checklist=checklist,
        messages=hiring_manager.build_messages(posting.brief, case.cv, checklist.items),
        model_ids=[item.id for item in checklist.items],
        # Quotes must come from the CV: a quote from the brief proves nothing about the candidate.
        truth_text=case.cv,
        runs_path=runs_path,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_retries=hiring_manager.MAX_RETRIES,
        strict=strict,
    )
