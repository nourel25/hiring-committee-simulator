"""Stage 1: send a CV snippet to the local model and get back a validated Review."""

from pydantic import ValidationError

from hcs.agents import tech_lead
from hcs.llm import call_model, make_client
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
    {"id": "TL2", "text": "Has led or mentored other engineers (people leadership, not technical ownership)"},
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
    response, seconds = call_model(
        client, 
        messages,
        response_model=Review,
        model=tech_lead.MODEL,
        temperature=tech_lead.TEMPERATURE,
        seed=tech_lead.SEED,
        max_tokens=tech_lead.MAX_TOKENS
    )
    
    choice = response.choices[0]
    print(f"Finished: {choice.finish_reason} in {seconds:.1f}s")
    print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
    
    if choice.finish_reason == "length":   
        print(f"Cut off at max_tokens ({tech_lead.MAX_TOKENS}): the JSON is incomplete, so it was not validated.")
        return 
    
    try:
        review = Review.model_validate_json(choice.message.content)
    except ValidationError as error:
        print("The reply did not match the Review schema:")
        print(error)
        return
    
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
    