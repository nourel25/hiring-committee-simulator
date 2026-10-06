"""Tests for hcs.evaluation: comparing runs with answer keys. Fake client; logs go to tmp_path."""

from pathlib import Path

import pytest

from hcs.evaluation import CaseReport, compare, evidence_matches, summarize_repeats
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


def test_a_failed_run_is_reported_not_skipped(tmp_path):
    report = compare(TINY_CASE, run_tiny('{"checks": [', tmp_path, finish_reason="length"))

    assert report.status == "error"
    assert report.passed_wrong == ["M1"]
    assert report.evidence_wrong == ["M1"]
    assert report.decision is None
    assert report.decision_ok is False


def make_report(score: float | None, decision: str | None, passed_wrong: list[str]) -> CaseReport:
    """A CaseReport for TINY_CASE (expected lean_hire) with only the fields a test cares about."""
    return CaseReport(
        case_id="case-1",
        label="good",
        status="ok" if decision else "error",
        seconds=1.0,
        score=score,
        decision=decision,
        expected_decision="lean_hire",
        decision_ok=decision == "lean_hire",
        total_checks=2,
        passed_wrong=passed_wrong,
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
