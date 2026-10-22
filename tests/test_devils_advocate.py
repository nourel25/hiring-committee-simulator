"""Tests for the devil's advocate: reading the requirements, the ratings schema and policy, picking
objections, the pipeline, and the comparison with the tech lead's key. Fake client; logs to tmp_path."""

import json

import pytest
from pydantic import ValidationError

from hcs.agents.devils_advocate import pick_objections, requirement_lines, without_year_counts
from hcs.cases import (
    Case,
    Checklist,
    DevilsAdvocateLines,
    ExpectedCheck,
    HiringManagerTargets,
    Posting,
    RecruiterScreen,
    load_posting,
)
from hcs.evaluation import compare_objections, must_haves_named
from hcs.pipeline import review_devils_advocate
from hcs.policies import apply_rating_policy
from hcs.runlog import read_runs
from hcs.schemas import ChecklistItem, LineRating, RequirementRatings, ratings_model_for
from tests.fakes import FakeClient, make_response

POSTING = Posting(
    title="Backend",
    text=(
        "Intro line\n\nRequirements\n- Strong Python skills, 3+ years\n- Schema design, not just "
        "queries\n- Team player\n- On-call\n\nNice to have\n- Kubernetes"
    ),
    brief="x",
    recruiter=RecruiterScreen(titles=["backend engineer"], keywords=[], min_years=0),
    hiring_manager=HiringManagerTargets(system="x", goals=["x"]),
    devils_advocate=DevilsAdvocateLines(ignore=["Team player"]),
)
LINES = {"D1": "Strong Python skills", "D2": "Schema design, not just queries", "D3": "On-call"}
CV = "Backend Engineer, shop (2020-present)\n- Built the shop API in Python\n- Wrote SQL queries"
CHECKLIST = Checklist(
    name="tiny",
    posting="Backend",
    items=[
        ChecklistItem(id="M1", text="python", importance="must", sources=["Strong Python skills"]),
        ChecklistItem(
            id="M2", text="schema", importance="must", sources=["Schema design, not just queries"]
        ),
        ChecklistItem(id="N1", text="on-call", importance="nice", sources=["On-call"]),
    ],
)


def make_case(m2_passed: bool) -> Case:
    return Case(
        id="da-1",
        label="borderline",
        checklist="tiny",
        expected_decision="lean_no",
        cv=CV,
        expected={
            "M1": ExpectedCheck(passed=True, evidence="Built the shop API in Python"),
            "M2": ExpectedCheck(passed=m2_passed),
            "N1": ExpectedCheck(passed=False),
        },
    )


def rating(line_id: str, evidence: str | list[str] | None, shown: str) -> dict:
    return {"id": line_id, "evidence": evidence, "reason": f"about {line_id}", "shown": shown}


def reply(*ratings: dict) -> str:
    return json.dumps({"ratings": list(ratings)})


PYTHON_SHOWN = rating("D1", "Built the shop API in Python", "fully")
SCHEMA_MISSING = rating("D2", "Wrote SQL queries", "not")
ON_CALL_MISSING = rating("D3", None, "not")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Strong Python skills, 3+ years", "Strong Python skills"),
        ("3+ years of Python and SQL", "Python and SQL"),
        ("Experience building REST APIs", "Experience building REST APIs"),
    ],
)
def test_year_counts_are_removed_because_code_computes_years(line, expected):
    assert without_year_counts(line) == expected


def test_only_requirements_lines_are_read_numbered_without_ignored_ones():
    assert requirement_lines(POSTING.text, POSTING.devils_advocate.ignore) == LINES


@pytest.mark.parametrize("name", ["backend_payments", "data_logistics"])
def test_the_real_postings_ignore_only_lines_they_have(name):
    posting = load_posting(name)
    lines = requirement_lines(posting.text, [])

    assert all(
        without_year_counts(line) in lines.values() for line in posting.devils_advocate.ignore
    )
    ignored = len(posting.devils_advocate.ignore)
    assert (
        len(requirement_lines(posting.text, posting.devils_advocate.ignore)) == len(lines) - ignored
    )


def test_every_requirement_must_be_rated_exactly_once():
    model = ratings_model_for(["D1", "D2"])

    with pytest.raises(ValidationError):
        model.model_validate_json(reply(PYTHON_SHOWN))
    with pytest.raises(ValidationError):
        model.model_validate_json(reply(PYTHON_SHOWN, PYTHON_SHOWN))
    assert len(model.model_validate_json(reply(PYTHON_SHOWN, SCHEMA_MISSING)).ratings) == 2


@pytest.mark.parametrize("written", ["null", "None", " NULL ", "", None, ["null"]])
def test_null_written_as_a_word_is_missing_evidence(written):
    assert LineRating(**rating("D1", written, "not")).evidence == []


def test_a_single_quote_becomes_a_list_of_one():
    assert LineRating(**rating("D1", "Wrote SQL queries", "not")).evidence == ["Wrote SQL queries"]


def test_a_rating_above_not_needs_a_real_quote():
    ratings = RequirementRatings.model_validate_json(
        reply(
            PYTHON_SHOWN,
            rating("D2", "Designed a 40-table schema", "partly"),
            rating("D3", None, "fully"),
        )
    )

    assert apply_rating_policy(ratings, CV) == 2
    assert [item.shown for item in ratings.ratings] == ["fully", "not", "not"]


def test_quotes_not_in_the_cv_are_dropped_and_real_ones_kept():
    # A requirement with several parts is shown on several lines; one invented quote among real
    # ones is dropped without changing the rating.
    ratings = RequirementRatings.model_validate_json(
        reply(
            rating("D1", ["Built the shop API in Python", "Wrote SQL queries"], "fully"),
            rating("D2", ["Wrote SQL queries", "Designed a 40-table schema"], "partly"),
            ON_CALL_MISSING,
        )
    )

    assert apply_rating_policy(ratings, CV) == 0
    assert [item.evidence for item in ratings.ratings] == [
        ["Built the shop API in Python", "Wrote SQL queries"],
        ["Wrote SQL queries"],
        [],
    ]


def test_code_picks_up_to_three_objections_not_shown_first():
    ratings = RequirementRatings.model_validate_json(
        reply(rating("D1", "x", "partly"), SCHEMA_MISSING, ON_CALL_MISSING)
    )

    objections = pick_objections(ratings, LINES)

    assert [(item.requirement, item.severity) for item in objections] == [
        ("Schema design, not just queries", "high"),
        ("On-call", "high"),
        ("Strong Python skills", "medium"),
    ]


def test_a_cv_that_shows_everything_gets_no_objection():
    ratings = RequirementRatings.model_validate_json(
        reply(PYTHON_SHOWN, rating("D2", "x", "fully"), rating("D3", "x", "fully"))
    )

    assert pick_objections(ratings, LINES) == []


def test_the_devils_advocate_rates_the_requirements_and_logs_its_objections(tmp_path):
    path = tmp_path / "runs.jsonl"
    invented = rating("D3", "Carried the pager for two years", "fully")
    client = FakeClient([make_response(reply(PYTHON_SHOWN, SCHEMA_MISSING, invented))])

    record = review_devils_advocate(client, make_case(False), POSTING, path)

    user_message = client.calls[0][1]["content"]
    assert "- D1: Strong Python skills\n" in user_message and "3+ years" not in user_message
    assert "Team player" not in user_message and "Kubernetes" not in user_message
    assert "<cv>\nBackend Engineer" in user_message
    # The invented on-call quote fails the rating policy, so on-call becomes an objection too.
    assert record.truth_check_changes == 1
    assert [item.requirement for item in record.objections] == [
        "Schema design, not just queries",
        "On-call",
    ]
    assert (record.agent, record.checklist, record.score) == ("devils_advocate", None, None)
    assert read_runs(path) == [record]


def test_a_failed_devils_advocate_call_is_logged(tmp_path):
    client = FakeClient([make_response('{"ratings": [', finish_reason="length")])

    record = review_devils_advocate(client, make_case(False), POSTING, tmp_path / "runs.jsonl")

    assert record.status == "error"
    assert record.objections is None


def test_a_quoted_requirement_line_names_the_must_have_it_comes_from():
    assert must_haves_named("Schema design, not just queries", CHECKLIST) == ["M2"]
    assert must_haves_named("On-call", CHECKLIST) == []  # a nice-to-have
    assert must_haves_named(None, CHECKLIST) == []


def test_an_objection_on_a_failed_must_have_is_a_hit(tmp_path):
    client = FakeClient([make_response(reply(PYTHON_SHOWN, SCHEMA_MISSING, ON_CALL_MISSING))])
    case = make_case(False)

    report = compare_objections(
        case, review_devils_advocate(client, case, POSTING, tmp_path / "r.jsonl"), CHECKLIST
    )

    assert (report.gaps, report.hit, report.false_alarms) == (["M2"], True, 0)


def test_a_partly_shown_failed_must_have_is_still_a_hit(tmp_path):
    partly = rating("D2", "Wrote SQL queries", "partly")
    client = FakeClient([make_response(reply(PYTHON_SHOWN, partly, ON_CALL_MISSING))])
    case = make_case(False)

    report = compare_objections(
        case, review_devils_advocate(client, case, POSTING, tmp_path / "r.jsonl"), CHECKLIST
    )

    assert (report.hit, report.severities) == (True, ["high", "medium"])


def test_a_high_objection_on_a_must_have_the_cv_meets_is_a_false_alarm(tmp_path):
    client = FakeClient([make_response(reply(PYTHON_SHOWN, SCHEMA_MISSING, ON_CALL_MISSING))])
    case = make_case(True)

    report = compare_objections(
        case, review_devils_advocate(client, case, POSTING, tmp_path / "r.jsonl"), CHECKLIST
    )

    # No failed must-have, so there is no gap to find; the schema objection is a false alarm.
    assert (report.gaps, report.hit, report.false_alarms) == ([], None, 1)
