"""Run the tech lead over every labelled case and compare each result with its answer key."""

from hcs.cases import load_cases, load_checklist
from hcs.evaluation import HEADER, compare, format_row
from hcs.llm import make_client
from hcs.pipeline import review_case


def main() -> None:
    client = make_client()
    reports = []

    print(HEADER)
    for case in load_cases():
        record = review_case(client, case, load_checklist(case.checklist))
        report = compare(case, record)
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
