"""Tests for hcs.evaluation: comparing runs with answer keys. Fake client; logs go to tmp_path."""

from pathlib import Path

import pytest

from hcs.evaluation import compare, evidence_matches
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
