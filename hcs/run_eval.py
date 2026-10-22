"""Run one reviewer over every labelled case and compare each result with its answer key."""

import argparse
from datetime import date

from openai import OpenAI

from hcs.cases import Case, load_cases, load_checklist, load_posting
from hcs.evaluation import HEADER, CaseReport, compare, format_row
from hcs.llm import make_client
from hcs.pipeline import review_case, review_hiring_manager, review_recruiter

AGENTS = ["tech_lead", "recruiter", "hiring_manager"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read which reviewer to evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=AGENTS, default="tech_lead", help="the reviewer to run")
    return parser.parse_args(argv)


def evaluate(client: OpenAI, case: Case, agent: str) -> CaseReport:
    """Run the agent on one case and compare the result with that agent's key."""
    if agent == "recruiter":
        record = review_recruiter(
            client,
            case,
            load_posting(case.checklist),
            load_checklist("recruiter"),
            date.today().year,
        )
        return compare(case, record, case.recruiter)
    if agent == "hiring_manager":
        record = review_hiring_manager(
            client, case, load_posting(case.checklist), load_checklist("hiring_manager")
        )
        return compare(case, record, case.hiring_manager)
    return compare(case, review_case(client, case, load_checklist(case.checklist)))


def main() -> None:
    args = parse_args()
    client = make_client()
    reports = []

    print(f"Agent: {args.agent}")
    print(HEADER)
    for case in load_cases():
        report = evaluate(client, case, args.agent)
        reports.append(report)
        print(format_row(report))
        if report.passed_wrong:
            print(f"    passed differs from the key: {', '.join(report.passed_wrong)}")
        if report.evidence_wrong:
            print(f"    evidence differs from the key: {', '.join(report.evidence_wrong)}")

    decisions_ok = sum(1 for report in reports if report.decision_ok)
    agreed = sum(report.total_checks - len(report.passed_wrong) for report in reports)
    total = sum(report.total_checks for report in reports)
    seconds = sum(report.seconds or 0 for report in reports)
    print()
    print(f"Decisions as expected: {decisions_ok}/{len(reports)}")
    print(f"Checks agreeing with the keys: {agreed}/{total}")
    print(f"Model time: {seconds:.0f}s")


if __name__ == "__main__":
    main()
