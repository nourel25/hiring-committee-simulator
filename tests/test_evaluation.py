"""Tests for hcs.evaluation: comparing runs with answer keys. Fake client; logs go to tmp_path."""

from pathlib import Path

import pytest

from hcs.cases import AgentKey, ExpectedCheck
from hcs.evaluation import (
    CaseReport,
    RepeatSummary,
    compare,
    evidence_matches,
    noise_verdict,
    summarize_model,
    summarize_repeats,
)
from hcs.pipeline import review_case
from hcs.runlog import RunRecord
from tests.fakes import TINY_CASE, TINY_CHECKLIST, TINY_REPLY, FakeClient, make_response


@pytest.mark.parametrize(
    ("got", "expected", "matches"),
    [
        ("Designed the schema for payments and more", "Designed the schema for payments", True),
        ("Designed the schema", "Designed the schema for payments", True),
        ("designed  THE schema", "Designed the schema", True),
        ("Ran Kubernetes clusters", "Designed the schema", False),
        (None, "Designed the schema", False),
    ],
)
def test_evidence_matches(got, expected, matches):
    assert evidence_matches(got, expected) is matches


def run_tiny(reply: str, tmp_path: Path, finish_reason: str = "stop") -> RunRecord:
    """Review TINY_CASE with one scripted reply, logging to tmp_path."""
    client = FakeClient([make_response(reply, finish_reason)])
    return review_case(client, TINY_CASE, TINY_CHECKLIST, tmp_path / "runs.jsonl")


def test_a_run_that_matches_the_key(tmp_path):
    report = compare(TINY_CASE, run_tiny(TINY_REPLY, tmp_path))

    assert report.passed_wrong == []
    assert report.evidence_wrong == []
    assert report.decision_ok is True


def test_an_invented_quote_is_caught_by_the_truth_check_and_the_key(tmp_path):
    reply = TINY_REPLY.replace("Built the payments API", "Led the payments team")

    report = compare(TINY_CASE, run_tiny(reply, tmp_path))

    assert report.passed_wrong == ["M1"]
    assert report.evidence_wrong == ["M1"]
    assert report.decision == "strong_no"
    assert report.decision_ok is False


def test_a_pass_the_key_says_should_fail_is_counted_as_lenient(tmp_path):
    reply = TINY_REPLY.replace(
        '"evidence": null, "passed": false', '"evidence": "Built the payments API", "passed": true'
    )

    report = compare(TINY_CASE, run_tiny(reply, tmp_path))

    assert report.passed_wrong == ["N1"]
    assert report.lenient == ["N1"]
    assert report.decision == "strong_hire"


def test_a_failed_run_is_reported_not_skipped(tmp_path):
    report = compare(TINY_CASE, run_tiny('{"checks": [', tmp_path, finish_reason="length"))

    assert report.status == "error"
    assert report.passed_wrong == ["M1"]
    assert report.evidence_wrong == ["M1"]
    assert report.decision is None
    assert report.decision_ok is False


def make_report(
    score: float | None,
    decision: str | None,
    passed_wrong: list[str],
    lenient: list[str] | None = None,
    attempts: int = 1,
) -> CaseReport:
    """A CaseReport for TINY_CASE (expected lean_hire) with only the fields a test cares about."""
    return CaseReport(
        case_id="case-1",
        label="good",
        status="ok" if decision else "error",
        seconds=1.0,
        attempts=attempts,
        score=score,
        decision=decision,
        expected_decision="lean_hire",
        decision_ok=decision == "lean_hire",
        total_checks=2,
        passed_wrong=passed_wrong,
        lenient=lenient or [],
        evidence_wrong=[],
    )


def test_summarize_repeats_counts_scores_decisions_failures_and_misses():
    reports = [
        make_report(7.5, "lean_hire", []),
        make_report(0.0, "strong_no", ["M1"]),
        make_report(None, None, ["M1"]),
    ]

    summary = summarize_repeats(reports)

    assert summary.runs == 3
    assert summary.errors == 1
    assert summary.scores == [7.5, 0.0]
    assert summary.decisions_ok == 1
    assert summary.check_misses == {"M1": 2}


def repeat(label: str, scores: list[float]) -> RepeatSummary:
    """A RepeatSummary with only a label and scores."""
    return RepeatSummary(
        case_id=label,
        label=label,
        runs=len(scores),
        errors=0,
        scores=scores,
        decisions_ok=0,
        check_misses={},
    )


def test_noise_verdict_is_the_good_poor_gap_and_the_widest_range():
    summaries = [
        repeat("good", [8.0, 9.0]),
        repeat("borderline", [5.0, 7.0]),
        repeat("poor", [1.0, 3.0]),
    ]

    assert noise_verdict(summaries) == (6.5, 2.0)
    assert noise_verdict(summaries[:2]) is None


def test_summarize_model_combines_zero_and_noisy_runs():
    zero = [make_report(7.5, "lean_hire", [])]
    noisy = [
        make_report(7.5, "lean_hire", [], attempts=2),
        make_report(10.0, "strong_hire", ["N1"], lenient=["N1"]),
        make_report(None, None, ["M1"]),
    ]

    summary = summarize_model("fake-model", zero, noisy, gpu_share=0.8)

    assert summary.decisions_at_zero == 1
    assert (summary.runs, summary.errors, summary.retried) == (4, 1, 1)
    assert summary.lenient == 1
    assert (summary.agreed, summary.checks) == (4, 6)
    assert summary.gap is None


def test_compare_uses_another_reviewers_key_and_skips_evidence_on_code_checks(tmp_path):
    # M1 stands in for a code check: the key has no evidence, so only passed is compared.
    key = AgentKey(
        expected_decision="lean_hire",
        expected={"M1": ExpectedCheck(passed=True), "N1": ExpectedCheck(passed=False)},
    )

    report = compare(TINY_CASE, run_tiny(TINY_REPLY, tmp_path), key)

    assert report.decision_ok is True
    assert report.total_checks == 2
    assert report.passed_wrong == []
    assert report.evidence_wrong == []
