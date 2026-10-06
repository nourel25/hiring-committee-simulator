"""Compare a run with its case's answer key. Pure functions: no model calls, no files."""

import statistics
from dataclasses import dataclass

from hcs.cases import Case
from hcs.policies import normalize
from hcs.runlog import RunRecord
from hcs.schemas import Decision


def evidence_matches(got: str | None, expected: str | None) -> bool:
    """True if one quote contains the other once both are normalised; False if either is missing."""
    if got is None or expected is None:
        return False
    got, expected = normalize(got), normalize(expected)
    return expected in got or got in expected


@dataclass
class CaseReport:
    """How one run compares with its case's answer key."""

    case_id: str
    label: str
    status: str
    seconds: float | None
    score: float | None
    decision: Decision | None
    expected_decision: Decision
    decision_ok: bool
    total_checks: int
    passed_wrong: list[str]
    evidence_wrong: list[str]


def compare(case: Case, record: RunRecord) -> CaseReport:
    """Judge the post-policy run against the key: passed on every check, evidence on expected passes."""
    checks = {check.id: check for check in record.review.checks} if record.review else {}

    passed_wrong = []
    evidence_wrong = []
    for check_id, key in case.expected.items():
        check = checks.get(check_id)
        got_passed = check.passed if check else False
        got_evidence = check.evidence if check else None

        if got_passed != key.passed:
            passed_wrong.append(check_id)
        if key.passed and not evidence_matches(got_evidence, key.evidence):
            evidence_wrong.append(check_id)

    return CaseReport(
        case_id=case.id,
        label=case.label,
        status=record.status,
        seconds=record.seconds,
        score=record.score,
        decision=record.decision,
        expected_decision=case.expected_decision,
        decision_ok=record.decision == case.expected_decision,
        total_checks=len(case.expected),
        passed_wrong=passed_wrong,
        evidence_wrong=evidence_wrong,
    )


HEADER = f"{'case':<14}{'label':<12}{'score':>6}  {'decision':<12}{'expected':<12}{'checks':>7}  ok"


def format_row(report: CaseReport) -> str:
    """One line of the summary table, aligned under HEADER."""
    score = "-" if report.score is None else f"{report.score:.1f}"
    decision = report.decision or report.status
    agreed = report.total_checks - len(report.passed_wrong)
    checks = f"{agreed}/{report.total_checks}"
    ok = "yes" if report.decision_ok else "NO"
    return (
        f"{report.case_id:<14}{report.label:<12}{score:>6}  "
        f"{decision:<12}{report.expected_decision:<12}{checks:>7}  {ok}"
    )


@dataclass
class RepeatSummary:
    """How one case behaved across repeated runs."""

    case_id: str
    label: str
    runs: int
    errors: int
    scores: list[float]
    decisions_ok: int
    check_misses: dict[str, int]


def summarize_repeats(reports: list[CaseReport]) -> RepeatSummary:
    """Combine the reports of repeated runs of one case."""
    check_misses: dict[str, int] = {}
    for report in reports:
        for check_id in report.passed_wrong:
            check_misses[check_id] = check_misses.get(check_id, 0) + 1

    return RepeatSummary(
        case_id=reports[0].case_id,
        label=reports[0].label,
        runs=len(reports),
        errors=sum(1 for report in reports if report.status == "error"),
        scores=[report.score for report in reports if report.score is not None],
        decisions_ok=sum(1 for report in reports if report.decision_ok),
        check_misses=check_misses,
    )


NOISE_HEADER = (
    f"{'case':<14}{'label':<12}{'mean':>6}{'sd':>6}{'min-max':>12}  {'decisions':>9}  misses"
)


def format_summary(summary: RepeatSummary) -> str:
    """One line of the noise table: score spread, decision stability, checks that disagreed."""
    if summary.scores:
        mean = statistics.mean(summary.scores)
        sd = statistics.stdev(summary.scores) if len(summary.scores) > 1 else 0.0
        span = f"{min(summary.scores):.1f}-{max(summary.scores):.1f}"
        stats = f"{mean:>6.1f}{sd:>6.1f}{span:>12}"
    else:
        stats = f"{'-':>6}{'-':>6}{'-':>12}"

    decisions = f"{summary.decisions_ok}/{summary.runs}"
    misses = ", ".join(
        f"{check_id} {count}/{summary.runs}"
        for check_id, count in sorted(summary.check_misses.items())
    )
    failed = f" ({summary.errors} failed)" if summary.errors else ""
    return (
        f"{summary.case_id:<14}{summary.label:<12}{stats}  "
        f"{decisions:>9}  {misses or 'none'}{failed}"
    )
