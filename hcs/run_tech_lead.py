"""Send a CV snippet to the local model and get back a validated Review."""

import uuid
from datetime import datetime, timezone

from hcs.agents import tech_lead
from hcs.llm import LLMOutputError, get_structured, make_client
from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    decide,
    failed_must_haves,
)
from hcs.runlog import RUNS_PATH, RunRecord, append_run
from hcs.schemas import ChecklistItem, Review

CV_TEXT = """
Backend Developer, fintech startup (2022-present)
- Built and maintain 6 REST APIs in Python (FastAPI) serving 50,000 daily users
- Designed the PostgreSQL schema for the payments service and led its migration from MySQL
- Cut average API response time by 40% by adding Redis caching
- Containerised services with Docker and deployed them to a single cloud VM

Junior Python Developer, digital agency (2021-2022)
- Fixed bugs and added features to Django websites for client projects
""".strip()

# Built from the "Backend Engineer (Python), Payments Platform" posting.
CHECKLIST = [
    ChecklistItem(
        id="TL1",
        text="Has built backend services in Python in a professional role",
        importance="must",
    ),
    ChecklistItem(
        id="TL2",
        text="Has maintained REST APIs that serve real users in production",
        importance="must",
    ),
    ChecklistItem(
        id="TL3",
        text="Has designed (not just queried) a relational database schema",
        importance="must",
    ),
    ChecklistItem(
        id="TL4",
        text="Has deployed services to production themselves",
        importance="must",
    ),
    ChecklistItem(
        id="TL5",
        text="Has monitored production services or taken part in an on-call rotation",
        importance="nice",
    ),
    ChecklistItem(
        id="TL6",
        text="Has run containerised services on an orchestration platform such as Kubernetes",
        importance="nice",
    ),
    ChecklistItem(
        id="TL7",
        text="Has measurably improved the performance of a system",
        importance="nice",
    ),
    ChecklistItem(
        id="TL8",
        text="Has mentored or led other engineers (people leadership, not technical ownership)",
        importance="nice",
    ),
]


APIS_LINE = "Built and maintain 6 REST APIs in Python (FastAPI) serving 50,000 daily users"
DOCKER_LINE = "Containerised services with Docker and deployed them to a single cloud VM"


# The answer key, written before any model run.
# Scoring rule: passed is always scored; evidence only when expected passed is true.
# Expected: 13/16 -> 8.1 -> strong_hire (no must-have fails).
EXPECTED = {
    "TL1": {"passed": True, "evidence": APIS_LINE},
    "TL2": {"passed": True, "evidence": APIS_LINE},
    "TL3": {"passed": True, "evidence": "Designed the PostgreSQL schema for the payments service"},
    "TL4": {"passed": True, "evidence": DOCKER_LINE},
    "TL5": {"passed": False, "evidence": None},
    "TL6": {"passed": False, "evidence": DOCKER_LINE},
    "TL7": {
        "passed": True,
        "evidence": "Cut average API response time by 40% by adding Redis caching",
    },
    "TL8": {"passed": False, "evidence": None},
}


def main() -> None:
    client = make_client()
    messages = tech_lead.build_messages(CV_TEXT, CHECKLIST)

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

    truth_changes, missing_quotes = apply_truth_check(review, CV_TEXT)
    print(f"Truth check: {truth_changes} pass(es) with a quote not in the CV changed to fail")
    if missing_quotes:
        print(f"Quotes not found in the CV: {', '.join(missing_quotes)}")

    for check in review.checks:
        print(f"{check.id}: passed={check.passed}, evidence={check.evidence!r}")

    score = compute_score(review, CHECKLIST)
    failed = failed_must_haves(review, CHECKLIST)
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
