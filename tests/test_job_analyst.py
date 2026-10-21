"""Tests for the job analyst, with no model: the schema, the source check and the checklist it builds."""

import pytest
from pydantic import ValidationError

from hcs.agents import job_analyst
from hcs.cases import load_posting
from hcs.schemas import DraftCheck, JobAnalysis

POSTING = """
Requirements
- Strong Python skills, 3+ years
- Team player
""".strip()

DRAFT_PYTHON = {
    "source": "- Strong Python skills, 3+ years",
    "check": "Has built services in Python",
    "does_not_count": "scripts do not count",
    "importance": "must",
}
DRAFT_TESTS = {
    "source": "Strong Python skills",
    "check": "Has written automated tests",
    "does_not_count": None,
    "importance": "nice",
}
DROPPED = {"source": "Team player", "reason": "a personality trait"}


def analysis(checks: list[dict], dropped: list[dict]) -> dict:
    return {"checks": checks, "lookups": ["Python, 3+ years"], "dropped": dropped}


def test_the_model_picks_its_source_before_writing_the_check():
    assert list(DraftCheck.model_fields) == ["source", "check", "does_not_count", "importance"]


def test_sources_from_the_posting_are_accepted():
    model = job_analyst.analysis_model_for(POSTING)
    result = model.model_validate(analysis([DRAFT_PYTHON, DRAFT_TESTS], [DROPPED]))
    assert isinstance(result, JobAnalysis)


@pytest.mark.parametrize(
    "checks, dropped",
    [
        ([{**DRAFT_PYTHON, "source": "5+ years of Go"}], []),
        ([DRAFT_PYTHON], [{**DROPPED, "source": "Loves ping-pong"}]),
    ],
    ids=["invented check source", "invented dropped source"],
)
def test_an_invented_source_is_rejected(checks, dropped):
    model = job_analyst.analysis_model_for(POSTING)
    with pytest.raises(ValidationError, match="not in the posting"):
        model.model_validate(analysis(checks, dropped))


def test_the_checklist_is_numbered_in_order_and_joins_what_does_not_count():
    parsed = JobAnalysis.model_validate(analysis([DRAFT_PYTHON, DRAFT_TESTS], []))
    checklist = job_analyst.to_checklist(parsed, "tiny", "Tiny posting")

    assert [item.id for item in checklist.items] == ["TL1", "TL2"]
    assert checklist.items[0].text == "Has built services in Python; scripts do not count"
    assert checklist.items[1].text == "Has written automated tests"
    assert [item.importance for item in checklist.items] == ["must", "nice"]
    assert checklist.posting == "Tiny posting"


def test_the_real_posting_loads():
    posting = load_posting("backend_payments")
    assert posting.title == "Backend Engineer (Python), Payments Platform (mid-level)"
    assert posting.text.startswith("Requirements")
