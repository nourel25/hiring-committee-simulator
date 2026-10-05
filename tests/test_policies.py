"""Tests for the agent-independent rules in hcs.policies. No model calls."""

import pytest

from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    normalize,
    score_to_decision,
)
from hcs.schemas import Check, Review

CV = "Designed the PostgreSQL schema for the payments service (2022-present)"


def make_check(evidence: str | None, passed: bool) -> Check:
    """Build a Check with placeholder id and text, so each test states only what matters."""
    return Check(id="T1", check="placeholder", evidence=evidence, passed=passed)


def test_pass_without_evidence_becomes_fail():
    review = Review(checks=[make_check(None, True), make_check("   ", True)])

    changed = apply_evidence_policy(review)

    assert changed == 2
    assert [c.passed for c in review.checks] == [False, False]


def test_pass_with_evidence_is_kept():
    review = Review(checks=[make_check("Designed the PostgreSQL schema", True)])

    changed = apply_evidence_policy(review)

    assert changed == 0
    assert review.checks[0].passed is True


def test_normalize_ignores_case_spacing_and_lookalike_dashes():
    assert normalize("Designed  the\u2013SCHEMA") == normalize("designed the-schema")


def test_truth_check_fails_a_pass_with_a_paraphrase():
    review = Review(checks=[make_check("Made the database design", True)])

    changed, missing = apply_truth_check(review, CV)

    assert changed == 1
    assert missing == ["T1"]
    assert review.checks[0].passed is False


def test_truth_check_flags_but_keeps_a_failed_check():
    review = Review(checks=[make_check("Ran Kubernetes clusters", False)])

    changed, missing = apply_truth_check(review, CV)

    assert changed == 0
    assert missing == ["T1"]
    assert review.checks[0].passed is False


def test_truth_check_accepts_a_real_quote():
    review = Review(checks=[make_check("Designed the PostgreSQL schema", True)])

    result = apply_truth_check(review, CV)

    assert result == (0, [])
    assert review.checks[0].passed is True


def test_compute_score_is_share_passed_times_ten():
    review = Review(
        checks=[make_check("x", True), make_check(None, False), make_check(None, False)]
    )

    assert compute_score(review) == pytest.approx(10 / 3)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "strong_no"),
        (3.99, "strong_no"),
        (4, "lean_no"),
        (5.99, "lean_no"),
        (6, "lean_hire"),
        (7.99, "lean_hire"),
        (8, "strong_hire"),
        (10, "strong_hire"),
    ],
)
def test_score_to_decision_bands(score, expected):
    assert score_to_decision(score) == expected
