"""Tests for the job analyst, with no model: the schema, the source check and the checklist it builds."""

import pytest
from pydantic import ValidationError

from hcs.agents import job_analyst
from hcs.cases import load_posting
from hcs.policies import quote_in_cv
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


@pytest.mark.parametrize(
    "clause, stored",
    [
        ("scripts do not count", "scripts do not count"),
        ("Querying a schema does not count.", "Querying a schema does not count"),
        (None, None),
    ],
)
def test_an_exclusion_clause_is_accepted(clause, stored):
    draft = DraftCheck.model_validate({**DRAFT_PYTHON, "does_not_count": clause})
    assert draft.does_not_count == stored


def test_a_clause_that_reads_as_a_requirement_is_rejected():
    # Run 1 of job-analyst-v1 wrote this for TL3; joined to the check, it asked for queries.
    with pytest.raises(ValidationError, match="does not count"):
        DraftCheck.model_validate({**DRAFT_PYTHON, "does_not_count": "Has written queries"})


def test_the_worked_example_is_a_valid_reply_for_its_own_posting():
    model = job_analyst.analysis_model_for(job_analyst.EXAMPLE_POSTING)
    model.model_validate(job_analyst.EXAMPLE_REPLY)


def test_the_worked_example_does_not_come_from_the_real_posting():
    posting = load_posting("backend_payments")
    sources = [check["source"] for check in job_analyst.EXAMPLE_REPLY["checks"]]
    assert not any(quote_in_cv(source, posting.text) for source in sources)


def test_a_clean_draft_has_no_warnings():
    parsed = JobAnalysis.model_validate(analysis([DRAFT_PYTHON, DRAFT_TESTS], [DROPPED]))
    assert job_analyst.draft_warnings(parsed) == []


def test_too_many_must_haves_are_flagged():
    musts = [DRAFT_PYTHON] * (job_analyst.MAX_MUST_HAVES + 1)
    parsed = JobAnalysis.model_validate(analysis(musts, []))

    [warning] = job_analyst.draft_warnings(parsed)
    assert warning.startswith(f"{job_analyst.MAX_MUST_HAVES + 1} must-haves")


def test_a_line_that_is_both_a_check_and_dropped_is_flagged():
    # Run 1 of job-analyst-v3 (qwen) made "Team player" a check and dropped it too.
    team_check = {**DRAFT_TESTS, "source": "- Team player"}
    parsed = JobAnalysis.model_validate(analysis([DRAFT_PYTHON, team_check], [DROPPED]))

    assert job_analyst.draft_warnings(parsed) == ["Both a check and dropped: 'Team player'"]


def test_the_real_posting_loads():
    posting = load_posting("backend_payments")
    assert posting.title == "Backend Engineer (Python), Payments Platform (mid-level)"
    assert posting.text.startswith("Requirements")
