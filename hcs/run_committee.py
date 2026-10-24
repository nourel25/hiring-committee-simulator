"""Run the whole committee on the labelled cases and compare the chair's verdicts with the labels."""

import argparse
from datetime import date

from hcs.cases import load_case, load_cases
from hcs.chair import LABEL_VERDICT
from hcs.committee import format_transcript, run_committee
from hcs.config import devils_advocate_endpoint
from hcs.evaluation import CASE_WIDTH
from hcs.llm import make_client

HEADER = (
    f"{'case':<{CASE_WIDTH}}{'label':<12}{'tech lead':<13}{'DA high':>8}  "
    f"{'verdict':<12}{'confidence':<11}{'sec':>5}  ok"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case ids (default: every case)")
    parser.add_argument("--transcript", action="store_true", help="print each meeting in full")
    parser.add_argument(
        "--local-devil",
        action="store_true",
        help="run the devil's advocate locally (judge A) instead of config.DEVILS_ADVOCATE",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    client = make_client()
    devil = None if args.local_devil else devils_advocate_endpoint()
    cases = [load_case(case_id) for case_id in args.cases] if args.cases else load_cases()
    rows = []

    print(HEADER)
    for case in cases:
        round_ = run_committee(client, case, date.today().year, devil=devil)
        verdict = round_.verdict
        expected = LABEL_VERDICT[case.label]
        tech_lead = round_.records["tech_lead"].decision or "failed"
        objections = round_.records["devils_advocate"].objections or []
        highs = sum(1 for objection in objections if objection.severity == "high")
        ok = verdict.verdict == expected
        rows.append((case, verdict.verdict, round_))
        print(
            f"{case.id:<{CASE_WIDTH}}{case.label:<12}{tech_lead:<13}{highs:>8}  "
            f"{verdict.verdict:<12}{verdict.confidence:<11}{round_.seconds:>5.0f}  "
            f"{'yes' if ok else 'NO (' + expected + ')'}"
        )
        if args.transcript:
            print()
            print(format_transcript(round_))
            print()

    agreed = sum(1 for case, verdict, _ in rows if verdict == LABEL_VERDICT[case.label])
    poor = [verdict for case, verdict, _ in rows if case.label == "poor"]
    failed = sum(
        1 for _, _, round_ in rows for record in round_.records.values() if record.status != "ok"
    )
    seconds = [round_.seconds for _, _, round_ in rows]
    print()
    print(f"Verdict agreement with the labels: {agreed}/{len(rows)}")
    print(f"Leniency (hire on a poor CV): {sum(1 for v in poor if v == 'hire')}/{len(poor)}")
    print(f"Failed member runs: {failed}/{len(rows) * 4}")
    print(f"Seconds per session: mean {sum(seconds) / len(seconds):.0f}, max {max(seconds):.0f}")


if __name__ == "__main__":
    main()
