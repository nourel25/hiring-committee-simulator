"""Tests for hcs.cases: loading from TOML, and checking that the real answer keys are consistent."""

import pytest
from pydantic import ValidationError

from hcs.cases import (
    key_problems,
    load_case,
    load_cases,
    load_checklist,
    load_posting,
    save_checklist,
)
from hcs.policies import apply_truth_check, compute_score, decide, failed_must_haves, quote_in_cv
from hcs.schemas import Check, Review
from hcs.screening import recruiter_view
from tests.fakes import TINY_CHECKLIST

CHECKLIST_TOML = """
name = "tiny"
posting = "Test posting"

[[items]]
id = "M1"
text = "must one"
importance = "must"

[[items]]
id = "N1"
text = "nice one"
importance = "nice"
"""

CASE_TOML = '''
id = "case-1"
label = "good"
checklist = "tiny"
expected_decision = "strong_hire"
cv = """
Line one
"""

[expected.M1]
passed = true
evidence = "Line one"

[expected.N1]
passed = false
'''


def test_checklist_and_case_load_from_toml(tmp_path):
    (tmp_path / "tiny.toml").write_text(CHECKLIST_TOML, encoding="utf-8")
    (tmp_path / "case-1.toml").write_text(CASE_TOML, encoding="utf-8")

    checklist = load_checklist("tiny", tmp_path)
    case = load_case("case-1", tmp_path)

    assert [item.id for item in checklist.items] == ["M1", "N1"]
    assert checklist.version == 1
    assert case.split == "dev"
    assert case.cv == "Line one"
    assert case.expected["N1"].evidence is None
    assert key_problems(case, checklist) == []


def test_case_with_an_unknown_label_is_rejected(tmp_path):
    (tmp_path / "case-1.toml").write_text(CASE_TOML.replace('"good"', '"great"'), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_case("case-1", tmp_path)


def test_key_problems_names_missing_and_unknown_ids(tmp_path):
    (tmp_path / "tiny.toml").write_text(CHECKLIST_TOML, encoding="utf-8")
    (tmp_path / "case-1.toml").write_text(CASE_TOML.replace("N1", "X9"), encoding="utf-8")

    problems = key_problems(load_case("case-1", tmp_path), load_checklist("tiny", tmp_path))

    assert problems == ["missing: N1", "unknown: X9"]


def test_a_saved_checklist_loads_back_unchanged_with_its_header(tmp_path):
    path = save_checklist(TINY_CHECKLIST, "Draft by a test.\nReview it.", tmp_path)

    assert path == tmp_path / "tiny.toml"
    assert path.read_text(encoding="utf-8").startswith("# Draft by a test.\n# Review it.\n")
    assert load_checklist("tiny", tmp_path) == TINY_CHECKLIST


def test_saving_never_overwrites_a_file_that_may_hold_edits(tmp_path):
    (tmp_path / "tiny.toml").write_text("# edited by hand\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_checklist(TINY_CHECKLIST, "Draft.", tmp_path)
    assert (tmp_path / "tiny.toml").read_text(encoding="utf-8") == "# edited by hand\n"


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.id)
def test_real_answer_key_is_consistent(case):
    checklist = load_checklist(case.checklist)
    review = Review(
        checks=[
            Check(id=check_id, check="", evidence=key.evidence, passed=key.passed)
            for check_id, key in case.expected.items()
        ]
    )

    _, missing_quotes = apply_truth_check(review, case.cv)
    score = compute_score(review, checklist.items)
    decision = decide(score, failed_must_haves(review, checklist.items))

    assert key_problems(case, checklist) == []
    assert missing_quotes == []
    assert decision == case.expected_decision


@pytest.mark.parametrize(
    "case", [case for case in load_cases() if case.recruiter], ids=lambda case: case.id
)
def test_real_recruiter_key_is_consistent(case):
    # The recruiter model reads only the top third without its dated role lines, so its quotes
    # must come from there.
    checklist = load_checklist("recruiter")
    review = Review(
        checks=[
            Check(id=check_id, check="", evidence=key.evidence, passed=key.passed)
            for check_id, key in case.recruiter.expected.items()
        ]
    )

    _, missing_quotes = apply_truth_check(review, recruiter_view(case.cv))
    score = compute_score(review, checklist.items)
    decision = decide(score, failed_must_haves(review, checklist.items))

    assert sorted(case.recruiter.expected) == sorted(item.id for item in checklist.items)
    assert missing_quotes == []
    assert decision == case.recruiter.expected_decision


def test_every_case_has_a_recruiter_key():
    assert [case.id for case in load_cases() if case.recruiter is None] == []


@pytest.mark.parametrize("name", ["backend_payments", "data_logistics"])
def test_every_must_have_names_posting_lines_that_exist(name):
    checklist = load_checklist(name)
    posting = load_posting(name)

    for item in checklist.items:
        if item.importance == "must":
            assert item.sources, item.id
        assert all(quote_in_cv(source, posting.text) for source in item.sources), item.id


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.id)
def test_real_hiring_manager_key_is_consistent(case):
    # The hiring manager reads the full CV; quotes from the brief would prove nothing.
    checklist = load_checklist("hiring_manager")
    review = Review(
        checks=[
            Check(id=check_id, check="", evidence=key.evidence, passed=key.passed)
            for check_id, key in case.hiring_manager.expected.items()
        ]
    )

    _, missing_quotes = apply_truth_check(review, case.cv)
    score = compute_score(review, checklist.items)
    decision = decide(score, failed_must_haves(review, checklist.items))

    assert sorted(case.hiring_manager.expected) == sorted(item.id for item in checklist.items)
    assert missing_quotes == []
    assert decision == case.hiring_manager.expected_decision
