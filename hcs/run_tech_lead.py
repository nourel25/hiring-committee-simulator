"""Review one labelled case with the tech lead and print the result."""

from hcs.cases import load_case, load_checklist
from hcs.llm import make_client
from hcs.pipeline import review_case
from hcs.runlog import RUNS_PATH

CASE_ID = "good-1"


def main() -> None:
    case = load_case(CASE_ID)
    checklist = load_checklist(case.checklist)
    record = review_case(make_client(), case, checklist)

    if record.status == "error":
        print(f"Gave up after {record.attempts} attempt(s): {record.error}")
        print(f"Last reply: {record.raw_reply!r}")
        print(f"Logged failed run {record.run_id} to {RUNS_PATH}")
        return

    print(f"Done in {record.attempts} attempt(s), {record.seconds:.1f}s")
    print(f"Tokens: {record.prompt_tokens} in, {record.completion_tokens} out")
    print(
        f"Evidence policy: {record.evidence_policy_changes} pass(es) without evidence changed to fail"
    )
    print(
        f"Truth check: {record.truth_check_changes} pass(es) with a quote not in the CV changed to fail"
    )
    if record.missing_quotes:
        print(f"Quotes not found in the CV: {', '.join(record.missing_quotes)}")

    for check in record.review.checks:
        print(f"{check.id}: passed={check.passed}, evidence={check.evidence!r}")

    print(f"Score: {record.score:.1f}/10 -> {record.decision}")
    if record.failed_must_haves:
        print(f"Failed must-haves: {', '.join(record.failed_must_haves)}")
    print(f"Logged run {record.run_id} to {RUNS_PATH}")


if __name__ == "__main__":
    main()
