"""Compare a run with its case's answer key. Pure functions: no model calls, no files."""

import statistics
from dataclasses import dataclass

from hcs.cases import AgentKey, Case, Checklist
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
    attempts: int
    score: float | None
    decision: Decision | None
    expected_decision: Decision
    decision_ok: bool
    total_checks: int
    passed_wrong: list[str]
    lenient: list[str]
    evidence_wrong: list[str]


def compare(case: Case, record: RunRecord, agent_key: AgentKey | None = None) -> CaseReport:
    """Judge the post-policy run against the key: passed on every check, evidence on expected passes.

    The key is the tech lead's unless another reviewer's key is given. A key without evidence on a
    pass (a check computed by code) is judged on passed only.
    """
    expected = agent_key.expected if agent_key else case.expected
    expected_decision = agent_key.expected_decision if agent_key else case.expected_decision
    checks = {check.id: check for check in record.review.checks} if record.review else {}

    passed_wrong = []
    evidence_wrong = []
    lenient = []
    for check_id, key in expected.items():
        check = checks.get(check_id)
        got_passed = check.passed if check else False
        got_evidence = check.evidence if check else None

        if got_passed != key.passed:
            passed_wrong.append(check_id)
        if got_passed and not key.passed:
            lenient.append(check_id)
        if (
            key.passed
            and key.evidence is not None
            and not evidence_matches(got_evidence, key.evidence)
        ):
            evidence_wrong.append(check_id)

    return CaseReport(
        case_id=case.id,
        label=case.label,
        status=record.status,
        seconds=record.seconds,
        attempts=record.attempts,
        score=record.score,
        decision=record.decision,
        expected_decision=expected_decision,
        decision_ok=record.decision == expected_decision,
        total_checks=len(expected),
        passed_wrong=passed_wrong,
        lenient=lenient,
        evidence_wrong=evidence_wrong,
    )


# Wide enough for ids like "data-borderline-1"; tables stay aligned as postings add longer ids.
CASE_WIDTH = 20

HEADER = f"{'case':<{CASE_WIDTH}}{'label':<12}{'score':>6}  {'decision':<12}{'expected':<12}{'checks':>7}  ok"


def format_row(report: CaseReport) -> str:
    """One line of the summary table, aligned under HEADER."""
    score = "-" if report.score is None else f"{report.score:.1f}"
    decision = report.decision or report.status
    agreed = report.total_checks - len(report.passed_wrong)
    checks = f"{agreed}/{report.total_checks}"
    ok = "yes" if report.decision_ok else "NO"
    return (
        f"{report.case_id:<{CASE_WIDTH}}{report.label:<12}{score:>6}  "
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


NOISE_HEADER = f"{'case':<{CASE_WIDTH}}{'label':<12}{'mean':>6}{'sd':>6}{'min-max':>12}  {'decisions':>9}  misses"


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
        f"{summary.case_id:<{CASE_WIDTH}}{summary.label:<12}{stats}  "
        f"{decisions:>9}  {misses or 'none'}{failed}"
    )


def noise_verdict(summaries: list[RepeatSummary]) -> tuple[float, float] | None:
    """The design doc's noise test: (good-poor gap in mean score, widest range); None without both."""
    means = {s.label: statistics.mean(s.scores) for s in summaries if s.scores}
    ranges = [max(s.scores) - min(s.scores) for s in summaries if s.scores]
    if "good" not in means or "poor" not in means:
        return None
    return means["good"] - means["poor"], max(ranges)


@dataclass
class ModelSummary:
    """One model's results across the comparison."""

    model: str
    decisions_at_zero: int
    cases: int
    runs: int
    errors: int
    retried: int
    lenient: int
    agreed: int
    checks: int
    gap: float | None
    widest: float | None
    seconds: float | None
    gpu_share: float | None


def summarize_model(
    model: str,
    zero_reports: list[CaseReport],
    noisy_reports: list[CaseReport],
    gpu_share: float | None,
) -> ModelSummary:
    """Combine one model's temperature-0 runs (one per case) with its repeated noisy runs."""
    by_case: dict[str, list[CaseReport]] = {}
    for report in noisy_reports:
        by_case.setdefault(report.case_id, []).append(report)
    verdict = noise_verdict([summarize_repeats(reports) for reports in by_case.values()])
    all_reports = zero_reports + noisy_reports
    seconds = [report.seconds for report in all_reports if report.seconds is not None]

    return ModelSummary(
        model=model,
        decisions_at_zero=sum(1 for report in zero_reports if report.decision_ok),
        cases=len(zero_reports),
        runs=len(all_reports),
        errors=sum(1 for report in all_reports if report.status == "error"),
        retried=sum(1 for report in all_reports if report.attempts > 1),
        lenient=sum(len(report.lenient) for report in noisy_reports),
        agreed=sum(report.total_checks - len(report.passed_wrong) for report in noisy_reports),
        checks=sum(report.total_checks for report in noisy_reports),
        gap=verdict[0] if verdict else None,
        widest=verdict[1] if verdict else None,
        seconds=statistics.mean(seconds) if seconds else None,
        gpu_share=gpu_share,
    )


MODEL_HEADER = (
    f"{'model':<15}{'dec@0':>6}{'failed':>8}{'retried':>8}{'lenient':>8}{'agree':>8}"
    f"{'gap':>6}{'range':>6}{'sec':>6}{'gpu':>6}"
)


def format_or_dash(value: float | None, spec: str) -> str:
    """Format a number with a format spec, or show '-' when it is missing."""
    return "-" if value is None else format(value, spec)


def format_model_row(summary: ModelSummary) -> str:
    """One line of the model comparison table, aligned under MODEL_HEADER."""
    decisions = f"{summary.decisions_at_zero}/{summary.cases}"
    failed = f"{summary.errors}/{summary.runs}"
    retried = f"{summary.retried}/{summary.runs}"
    agree = f"{summary.agreed}/{summary.checks}"
    gap = format_or_dash(summary.gap, ".1f")
    widest = format_or_dash(summary.widest, ".1f")
    seconds = format_or_dash(summary.seconds, ".0f")
    gpu = format_or_dash(summary.gpu_share, ".0%")
    return (
        f"{summary.model:<15}{decisions:>6}{failed:>8}{retried:>8}{summary.lenient:>8}{agree:>8}"
        f"{gap:>6}{widest:>6}{seconds:>6}{gpu:>6}"
    )


# The devil's advocate has no checks to compare, so it is judged against the gaps the tech lead's key
# already knows: the must-haves the key fails, traced to the posting lines they come from.


def must_haves_named(requirement: str | None, checklist: Checklist) -> list[str]:
    """Ids of the must-haves whose posting lines the quoted requirement is part of (or contains)."""
    return [
        item.id
        for item in checklist.items
        if item.importance == "must"
        and any(evidence_matches(requirement, source) for source in item.sources)
    ]


@dataclass
class ObjectionReport:
    """How one devil's-advocate run compares with the tech lead's key for the same CV."""

    case_id: str
    expected_decision: Decision
    gaps: list[str]
    status: str
    severities: list[str]
    # None when the key has no failed must-have: there is no gap to find.
    hit: bool | None
    # High objections ("not shown at all") that name only must-haves the key says the CV meets.
    false_alarms: int
    ungrounded: int
    seconds: float | None


def compare_objections(case: Case, record: RunRecord, checklist: Checklist) -> ObjectionReport:
    """Did an objection name a must-have the key fails, and did a high one claim the CV lacks a
    must-have the key says it has?

    v2 (Step 4.3f): a hit is an objection of any severity, because code only picks requirement lines
    the model rated as not fully shown; a line with several parts (scheduling, quality checks and
    on-call) shown only in part is "partly", so medium, and still a real finding. v1 counted highs only.
    """
    must_ids = {item.id for item in checklist.items if item.importance == "must"}
    gaps = sorted(check_id for check_id in must_ids if not case.expected[check_id].passed)
    objections = [
        objection
        for position, objection in enumerate(record.objections or [])
        if position not in record.ungrounded_objections
    ]
    named = [must_haves_named(objection.requirement, checklist) for objection in objections]
    named_by_highs = [
        ids
        for ids, objection in zip(named, objections, strict=True)
        if objection.severity == "high"
    ]
    return ObjectionReport(
        case_id=case.id,
        expected_decision=case.expected_decision,
        gaps=gaps,
        status=record.status,
        severities=[objection.severity for objection in record.objections or []],
        hit=any(set(ids) & set(gaps) for ids in named) if gaps else None,
        false_alarms=sum(1 for ids in named_by_highs if ids and not set(ids) & set(gaps)),
        ungrounded=len(record.ungrounded_objections),
        seconds=record.seconds,
    )


OBJECTION_HEADER = (
    f"{'case':<{CASE_WIDTH}}{'tech lead':<12}{'gaps':<14}{'severity':<10}"
    f"{'hit':<5}{'false':>6}{'ungrounded':>11}"
)


def format_objection_row(report: ObjectionReport) -> str:
    """One line of the devil's-advocate table, aligned under OBJECTION_HEADER."""
    gaps = ",".join(report.gaps) or "-"
    # H M L: high, medium, low, strongest first.
    severities = " ".join(s[0].upper() for s in report.severities) or report.status
    hit = "-" if report.hit is None else ("yes" if report.hit else "NO")
    return (
        f"{report.case_id:<{CASE_WIDTH}}{report.expected_decision:<12}{gaps:<14}{severities:<10}"
        f"{hit:<5}{report.false_alarms:>6}{report.ungrounded:>11}"
    )
