"""Send a CV snippet to the local model and get back a validated Review."""

from hcs.agents import tech_lead
from hcs.llm import LLMOutputError, get_structured, make_client
from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    score_to_decision,
)
from hcs.schemas import Review

CV_TEXT = """
Backend Developer, fintech startup (2022-present)
- Built and maintain 6 REST APIs in Python (FastAPI) serving 50,000 daily users
- Designed the PostgreSQL schema for the payments service and led its migration from MySQL
- Cut average API response time by 40% by adding Redis caching
- Containerised services with Docker and deployed them to a single cloud VM

Junior Python Developer, digital agency (2021-2022)
- Fixed bugs and added features to Django websites for client projects
""".strip()

CHECKS = [
    {"id": "TL1", "text": "Has designed (not just used) a database schema for a production system"},
    {
        "id": "TL2",
        "text": "Has led or mentored other engineers (people leadership, not technical ownership)",
    },
    {"id": "TL3", "text": "Has run services on Kubernetes in production"},
]

# The answer key, written before any model run.
# Scoring rule: passed is always scored; evidence only when expected passed is true.
EXPECTED = {
    "TL1": {
        "passed": True,
        "evidence": "Designed the PostgreSQL schema for the payments service",
    },
    "TL2": {
        "passed": False,
        "evidence": None,
    },
    "TL3": {
        "passed": False,
        "evidence": "Containerised services with Docker and deployed them to a single cloud VM",
    },
}


def main() -> None:
    client = make_client()
    messages = tech_lead.build_messages(CV_TEXT, CHECKS)
    try:
        result = get_structured(
            client,
            messages,
            response_model=Review,
            model=tech_lead.MODEL,
            temperature=tech_lead.TEMPERATURE,
            seed=tech_lead.SEED,
            max_tokens=tech_lead.MAX_TOKENS,
        )
    except LLMOutputError as error:
        print(f"Gave up after {error.attempts} attempt(s): {error}")
        print(f"Last reply: {error.raw_reply!r}")
        return

    print(f"Done in {result.attempts} attempt(s), {result.seconds:.1f}s")
    print(f"Tokens: {result.prompt_tokens} in, {result.completion_tokens} out")

    review = result.parsed

    policy_changes = apply_evidence_policy(review)
    print(f"Evidence policy: {policy_changes} pass(es) without evidence changed to fail")

    truth_changes = apply_truth_check(review, CV_TEXT)
    print(f"Truth check: {truth_changes} pass(es) with a quote not in the CV changed to fail")

    for check in review.checks:
        print(f"{check.id}: passed={check.passed}, evidence={check.evidence!r}")

    score = compute_score(review)
    decision = score_to_decision(score)
    print(f"Score: {score:.1f}/10 -> {decision}")


if __name__ == "__main__":
    main()
