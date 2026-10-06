"""Tests for the agent-independent rules in hcs.policies. No model calls."""

import pytest

from hcs.policies import (
    apply_evidence_policy,
    apply_truth_check,
    compute_score,
    decide,
    failed_must_haves,
    normalize,
    score_to_decision,
)
from hcs.schemas import Check, ChecklistItem, Review

CV = "Designed the PostgreSQL schema for the payments service (2022-present)"


# Total weight 3 + 3 + 1 = 7.
CHECKLIST = [
    ChecklistItem(id="M1", text="must one", importance="must"),
    ChecklistItem(id="M2", text="must two", importance="must"),
    ChecklistItem(id="N1", text="nice one", importance="nice"),
]


def make_check(evidence: str | None, passed: bool, check_id: str = "T1") -> Check:
    """Build a Check with placeholder text, so each test states only what matters."""
    return Check(id=check_id, check="placeholder", evidence=evidence, passed=passed)


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


def test_compute_score_weights_must_haves_three_times():
    review = Review(
        checks=[
            make_check("x", True, "M1"),
            make_check(None, False, "M2"),
            make_check("x", True, "N1"),
        ]
    )

    assert compute_score(review, CHECKLIST) == pytest.approx(40 / 7)


def test_compute_score_counts_a_missing_item_as_failed():
    review = Review(checks=[make_check("x", True, "M1"), make_check("x", True, "N1")])

    assert compute_score(review, CHECKLIST) == pytest.approx(40 / 7)


def test_compute_score_counts_a_repeated_item_once():
    review = Review(
        checks=[
            make_check("x", True, "M1"),
            make_check("x", True, "M1"),
            make_check("x", True, "M2"),
            make_check("x", True, "N1"),
        ]
    )

    assert compute_score(review, CHECKLIST) == pytest.approx(10)


def test_failed_must_haves_lists_failed_and_missing_must_haves_only():
    review = Review(checks=[make_check(None, False, "M1"), make_check(None, False, "N1")])

    assert failed_must_haves(review, CHECKLIST) == ["M1", "M2"]


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


@pytest.mark.parametrize(
    ("score", "failed", "expected"),
    [
        (8.1, [], "strong_hire"),
        (8.1, ["M2"], "lean_no"),
        (6.5, ["M2"], "lean_no"),
        (6.5, [], "lean_hire"),
        (3.0, ["M2"], "strong_no"),
    ],
)
def test_decide_caps_at_lean_no_when_a_must_have_failed(score, failed, expected):
    assert decide(score, failed) == expected
