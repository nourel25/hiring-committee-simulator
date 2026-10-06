"""Send a CV snippet to the local model and get back a validated Review."""

import uuid
from datetime import datetime, timezone

from hcs.agents import tech_lead
from hcs.cases import load_case, load_checklist
from hcs.llm import LLMOutputError, get_structured, make_client
from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    decide,
    failed_must_haves,
)
from hcs.runlog import RUNS_PATH, RunRecord, append_run
from hcs.schemas import Review

CASE_ID = "good-1"


def main() -> None:
    case = load_case(CASE_ID)
    checklist = load_checklist(case.checklist)
    client = make_client()
    messages = tech_lead.build_messages(case.cv, checklist.items)

    common = {
        "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc),
        "agent": "tech_lead",
        "prompt_version": tech_lead.PROMPT_VERSION,
        "model": tech_lead.MODEL,
        "temperature": tech_lead.TEMPERATURE,
        "seed": tech_lead.SEED,
        "max_tokens": tech_lead.MAX_TOKENS,
        "max_retries": tech_lead.MAX_RETRIES,
        "messages": messages,
    }

    try:
        result = get_structured(
            client,
            messages,
            response_model=Review,
            model=tech_lead.MODEL,
            temperature=tech_lead.TEMPERATURE,
            seed=tech_lead.SEED,
            max_tokens=tech_lead.MAX_TOKENS,
            max_retries=tech_lead.MAX_RETRIES,
        )
    except LLMOutputError as error:
        print(f"Gave up after {error.attempts} attempt(s): {error}")
        print(f"Last reply: {error.raw_reply!r}")
        append_run(
            RunRecord(
                **common,
                status="error",
                error=str(error),
                attempts=error.attempts,
                raw_reply=error.raw_reply,
            )
        )
        print(f"Logged failed run {common['run_id']} to {RUNS_PATH}")
        return

    print(f"Done in {result.attempts} attempt(s), {result.seconds:.1f}s")
    print(f"Tokens: {result.prompt_tokens} in, {result.completion_tokens} out")

    review = result.parsed

    policy_changes = apply_evidence_policy(review)
    print(f"Evidence policy: {policy_changes} pass(es) without evidence changed to fail")

    truth_changes, missing_quotes = apply_truth_check(review, case.cv)
    print(f"Truth check: {truth_changes} pass(es) with a quote not in the CV changed to fail")
    if missing_quotes:
        print(f"Quotes not found in the CV: {', '.join(missing_quotes)}")

    for check in review.checks:
        print(f"{check.id}: passed={check.passed}, evidence={check.evidence!r}")

    score = compute_score(review, checklist.items)
    failed = failed_must_haves(review, checklist.items)
    decision = decide(score, failed)
    print(f"Score: {score:.1f}/10 -> {decision}")
    if failed:
        print(f"Failed must-haves: {', '.join(failed)}")

    append_run(
        RunRecord(
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
            decision=decision,
        )
    )
    print(f"Logged run {common['run_id']} to {RUNS_PATH}")


if __name__ == "__main__":
    main()
