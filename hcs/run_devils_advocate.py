"""Run the devil's advocate over every labelled case and compare its objections with the gaps the
tech lead's key knows. Prints every objection, because there is no score to summarise them."""

import argparse
from pathlib import Path

from hcs.cases import CASES_DIR, load_cases, load_checklist, load_posting
from hcs.config import API_MAX_TOKENS
from hcs.evaluation import OBJECTION_HEADER, compare_objections, format_objection_row
from hcs.llm import make_client
from hcs.pipeline import review_devils_advocate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="the devil's advocate's model (default: judge A)")
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        default="ollama",
        help="where the model runs (groq: a larger max_tokens for reasoning models)",
    )
    parser.add_argument("--retries", type=int, help="repair retries per call (default: 1)")
    parser.add_argument(
        "--dir", type=Path, default=CASES_DIR, help="where the cases are (default: data/cases)"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    client = make_client(args.provider)
    settings = {"model": args.model} if args.model else {}
    if args.provider != "ollama":
        settings["max_tokens"] = API_MAX_TOKENS
    if args.retries is not None:
        settings["max_retries"] = args.retries
    reports = []
    downgraded = 0

    print(OBJECTION_HEADER)
    for case in load_cases(args.dir):
        record = review_devils_advocate(client, case, load_posting(case.checklist), **settings)
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
    print(
        f"Cases without a gap that got any objection: "
        f"{sum(1 for report in without_gaps if report.severities)}/{len(without_gaps)}"
    )
    explained = sum(report.mediums_explained for report in reports)
    mediums = explained + sum(report.mediums_unexplained for report in reports)
    print(f"Medium objections explained by a part the key fails: {explained}/{mediums}")
    print(f"Model time: {sum(report.seconds or 0 for report in reports):.0f}s")


if __name__ == "__main__":
    main()
