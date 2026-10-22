"""Run the devil's advocate over every labelled case and compare its objections with the gaps the
tech lead's key knows. Prints every objection, because there is no score to summarise them."""

from hcs.cases import load_cases, load_checklist, load_posting
from hcs.evaluation import OBJECTION_HEADER, compare_objections, format_objection_row
from hcs.llm import make_client
from hcs.pipeline import review_devils_advocate


def main() -> None:
    client = make_client()
    reports = []
    downgraded = 0

    print(OBJECTION_HEADER)
    for case in load_cases():
        record = review_devils_advocate(client, case, load_posting(case.checklist))
        report = compare_objections(case, record, load_checklist(case.checklist))
        reports.append(report)
        downgraded += record.truth_check_changes or 0
        print(format_objection_row(report))
        for objection in record.objections or []:
            print(f"    [{objection.severity}] {objection.reason}")
            print(f"        posting: {objection.requirement}")
            print(f"        cv:      {objection.evidence}")

    with_gaps = [report for report in reports if report.hit is not None]
    without_gaps = [report for report in reports if report.hit is None]
    total = sum(len(report.severities) for report in reports)
    print()
    print(f"Failed runs: {sum(1 for report in reports if report.status != 'ok')}/{len(reports)}")
    print(f"Objections: {total} (at most 3 per case)")
    print(f"Ratings turned to 'not' for lack of a real quote: {downgraded}")
    print(
        f"Gaps found (an objection on a failed must-have's line): "
        f"{sum(1 for report in with_gaps if report.hit)}/{len(with_gaps)}"
    )
    print(
        f"False alarms (high, on a must-have the key says is met): "
        f"{sum(report.false_alarms for report in reports)}"
    )
    print(
        f"Cases without a gap that still got a high objection: "
        f"{sum(1 for report in without_gaps if 'high' in report.severities)}/{len(without_gaps)}"
    )
    print(f"Model time: {sum(report.seconds or 0 for report in reports):.0f}s")


if __name__ == "__main__":
    main()
