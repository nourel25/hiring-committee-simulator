"""Tests for the UI's views: plain data built from records and logged sessions."""

import json

from hcs.chair import ChairObjection, ChairVerdict
from hcs.views import (
    chair_rows,
    check_rows,
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
