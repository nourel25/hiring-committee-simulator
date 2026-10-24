"""Tests for the UI's views: plain data built from records and logged sessions."""

import json

from hcs.cases import load_case, load_cases
from hcs.chair import ChairObjection, ChairVerdict
from hcs.views import (
    chair_rows,
    check_rows,
    example_name,
    examples_for,
    known_invented,
    load_sessions,
    member_summary,
    objection_rows,
    postings,
    session_rows,
)
from tests.test_chair import check, objection, record


def test_every_posting_is_offered_by_its_title():
    assert postings()["backend_payments"].startswith("Backend Engineer")


def test_a_review_becomes_one_row_per_check_with_its_quote():
    reviewed = record(
        "tech_lead", "lean_no", 5.0, [check("TL1", True, "Built APIs"), check("TL2", False)]
    )

    assert check_rows(reviewed) == [
        {"": "✓", "id": "TL1", "check": "tl1", "evidence": "Built APIs"},
        {"": "✗", "id": "TL2", "check": "tl2", "evidence": ""},
    ]
    assert member_summary("tech_lead", reviewed) == "Tech lead: 5.0/10, lean_no"


def test_the_devils_advocate_shows_objections_not_checks():
    devil = record("devils_advocate", objections=[objection("Schema design", "high")])

    assert check_rows(devil) == []
    assert objection_rows(devil)[0]["severity"] == "high"
    assert member_summary("devils_advocate", devil) == "Devil's advocate: 1 objection(s)"


def test_a_failed_review_says_so():
    failed = record("recruiter").model_copy(update={"status": "error"})

    assert member_summary("recruiter", failed) == "Recruiter: the review failed"


def test_the_chairs_objections_name_members_by_title():
    verdict = ChairVerdict(
        "borderline",
        "medium",
        [ChairObjection("schema", "high", "gap", ["tech_lead", "devils_advocate"])],
        [],
        {},
        [],
    )

    assert chair_rows(verdict)[0]["raised by"] == "Tech lead, Devil's advocate"


def test_logged_sessions_become_table_rows(tmp_path):
    path = tmp_path / "sessions.jsonl"
    session = {
        "case_id": "good-1",
        "label": "good",
        "verdicts": ["hire", "hire"],
        "recruiter_scores": [5.8, 10.0],
        "kept": [{"facts": ["F2"], "text": "x"}],
        "rejected": [],
        "stop_reason": "hire",
    }
    path.write_text(json.dumps(session) + "\n", encoding="utf-8")

    assert session_rows(load_sessions(path)) == [
        {
            "case": "good-1",
            "label": "good",
            "verdicts": "hire → hire",
            "recruiter": "5.8 → 10.0",
            "lines kept": 1,
            "rejected": 0,
            "stop": "hire",
        }
    ]
    assert load_sessions(tmp_path / "missing.jsonl") == []


def test_an_example_is_named_by_its_fit_its_latest_role_and_its_years():
    assert example_name(load_case("good-1"), 2026) == (
        "Strong fit: Backend Developer, fintech startup, 5 years"
    )


def test_an_example_opening_with_a_summary_is_named_by_its_first_dated_role():
    assert example_name(load_case("present-1"), 2026) == (
        "Strong fit: Backend Developer, online pharmacy, 7 years"
    )


def test_examples_list_one_name_per_case_of_the_posting_strong_fits_first():
    cases = load_cases()
    examples = examples_for("backend_payments", cases, 2026)

    assert len(examples) == sum(1 for case in cases if case.checklist == "backend_payments")
    labels = [case.label for case in examples.values()]
    assert labels == sorted(labels, key=["good", "borderline", "poor"].index)


def test_only_an_unedited_example_counts_as_invented():
    example = load_case("good-1")

    assert known_invented(example.cv, example, uploaded=False)
    assert not known_invented(example.cv + "\n- Added a line", example, uploaded=False)
    assert not known_invented(example.cv, example, uploaded=True)
    assert not known_invented("Engineer, shop (2020-present)", None, uploaded=False)
