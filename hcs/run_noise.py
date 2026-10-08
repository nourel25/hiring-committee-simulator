"""Run every labelled case several times at a higher temperature to measure score noise."""

from hcs.cases import load_cases, load_checklist
from hcs.evaluation import (
    NOISE_HEADER,
    compare,
    format_row,
    format_summary,
    noise_verdict,
    summarize_repeats,
)
from hcs.llm import make_client
from hcs.pipeline import review_case

TEMPERATURE = 0.7
SEEDS = [1, 2, 3, 4, 5]


def main() -> None:
    client = make_client()
    summaries = []

    for case in load_cases():
        checklist = load_checklist(case.checklist)
        reports = []
        for seed in SEEDS:
            record = review_case(client, case, checklist, temperature=TEMPERATURE, seed=seed)
            report = compare(case, record)
            reports.append(report)
            print(f"seed {seed}  {format_row(report)}")
        summaries.append(summarize_repeats(reports))

    print()
    print(f"{len(SEEDS)} runs per case at temperature {TEMPERATURE}")
    print(NOISE_HEADER)
    for summary in summaries:
        print(format_summary(summary))

    verdict = noise_verdict(summaries)
    if verdict:
        gap, widest = verdict
        result = "meets" if widest < gap else "MISSES"
        print()
        print(
            f"Good-poor gap {gap:.1f} vs widest range {widest:.1f}: {result} the design doc's target"
        )


if __name__ == "__main__":
    main()
