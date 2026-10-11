"""Tests for hcs.cases: loading from TOML, and checking that the real answer keys are consistent."""

import pytest
from pydantic import ValidationError

from hcs.cases import key_problems, load_case, load_cases, load_checklist
from hcs.policies import apply_truth_check, compute_score, decide, failed_must_haves
from hcs.schemas import Check, Review

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
