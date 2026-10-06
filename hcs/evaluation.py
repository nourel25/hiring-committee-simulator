"""Compare a run with its case's answer key. Pure functions: no model calls, no files."""

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
