"""Compare models as the devil's advocate on every case, from the run log, and apply the rule.

The rule (written before the comparison runs, 2026-10-24):
1. A model is eligible only with 0 failed runs, every must-have gap found and 0 false alarms
   (a high objection on a must-have the key says is met): what the chair reads.
2. Among eligible models, the highest share of explained mediums wins; ties go to fewer
   unexplained mediums, then to the faster model.
3. No eligible model -> the devil's advocate stays on the local model.

    python -m hcs.run_compare_da qwen2.5:7b qwen/qwen3.8-27b gemini-3.8-flash gemma-4-31b-it
"""

import argparse
from dataclasses import dataclass

from hcs.agents import devils_advocate
from hcs.cases import CASES_DIR, load_cases, load_checklist
from hcs.evaluation import compare_objections
from hcs.runlog import read_runs

DA_CASES_DIR = CASES_DIR.parent / "da_cases"


@dataclass
class ModelResult:
    model: str
    cases: int
    failed: int
    gaps_found: int
    gaps: int
    false_alarms: int
    explained: int
    unexplained: int
    seconds: float

    @property
    def eligible(self) -> bool:
        return self.failed == 0 and self.gaps_found == self.gaps and self.false_alarms == 0

    @property
    def share(self) -> float:
        mediums = self.explained + self.unexplained
        return self.explained / mediums if mediums else 1.0


def has_gap(case) -> bool:
    """The case's key fails at least one must-have: there is a gap to find."""
    checklist = load_checklist(case.checklist)
    return any(
        not case.expected[item.id].passed for item in checklist.items if item.importance == "must"
    )


def summarise(model: str, records: dict, cases: list) -> ModelResult:
    """One model's results over the cases; a case with no record counts as a failed run."""
    reports = [
        compare_objections(case, records[case.id], load_checklist(case.checklist))
        for case in cases
        if case.id in records
    ]
    with_gaps = [report for report in reports if report.hit is not None]
    return ModelResult(
        model=model,
        cases=len(reports),
        failed=sum(1 for report in reports if report.status != "ok") + len(cases) - len(reports),
        gaps_found=sum(1 for report in with_gaps if report.hit),
        gaps=sum(1 for case in cases if has_gap(case)),
        false_alarms=sum(report.false_alarms for report in reports),
        explained=sum(report.mediums_explained for report in reports),
        unexplained=sum(report.mediums_unexplained for report in reports),
        seconds=sum(report.seconds or 0 for report in reports) / max(len(reports), 1),
    )


def winner(results: list[ModelResult]) -> ModelResult | None:
    """The rule: eligible models only, then the best explained share, fewer unexplained, faster."""
    eligible = [result for result in results if result.eligible]
    if not eligible:
        return None
    return min(eligible, key=lambda result: (-result.share, result.unexplained, result.seconds))


def latest_records(model: str, case_ids: set[str]) -> dict:
    """The most recent devils-advocate-v3 record per case for one model."""
    records = {}
    for record in read_runs():
        if (
            record.agent == "devils_advocate"
            and record.prompt_version == devils_advocate.PROMPT_VERSION
            and record.model == model
            and record.case_id in case_ids
        ):
            records[record.case_id] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="model names as logged")
    args = parser.parse_args()
    cases = load_cases() + load_cases(DA_CASES_DIR)
    ids = {case.id for case in cases}
    results = [summarise(model, latest_records(model, ids), cases) for model in args.models]

    print(
        f"{'model':<22}{'cases':>6}{'failed':>7}{'gaps':>8}{'false':>6}{'explained':>11}{'sec':>6}"
    )
    for r in results:
        print(
            f"{r.model:<22}{r.cases:>6}{r.failed:>7}{f'{r.gaps_found}/{r.gaps}':>8}"
            f"{r.false_alarms:>6}{f'{r.explained}/{r.explained + r.unexplained}':>11}"
            f"{r.seconds:>6.0f}  {'eligible' if r.eligible else 'not eligible'}"
        )
    best = winner(results)
    print()
    print(f"Winner by the rule: {best.model if best else 'none eligible: stays local'}")


if __name__ == "__main__":
    main()
