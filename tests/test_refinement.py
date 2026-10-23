"""Tests for the refinement loop: the summary, the grounding of each line, and the stop conditions."""

import json

import pytest
from pydantic import ValidationError

from hcs.cases import load_case, load_posting
from hcs.committee import Checklists
from hcs.profile import build_facts
from hcs.refinement import ground, run_session, with_summary, write_summary
from hcs.runlog import read_runs
from hcs.schemas import SummaryLine, refinement_model_for
from tests.fakes import FakeClient, make_response

CASE = load_case("present-1")
POSTING = load_posting(CASE.checklist)
FACTS = build_facts(CASE.cv)
TRUE_LINE = {"facts": ["F4"], "text": "Builds Python REST APIs serving 30,000 customers a day"}
PADDED_LINE = {"facts": ["F4"], "text": "Builds Python REST APIs on AWS"}
INFLATED_LINE = {"facts": ["F8"], "text": "Leads a team of two developers"}


def summary(*lines: dict) -> str:
    return json.dumps({"summary": list(lines)})


def judged(*supported: bool) -> str:
    return json.dumps(
        {
            "judgements": [
                {
                    "id": f"G{n}",
                    "unsupported": None if ok else "leads a team",
                    "supported": ok,
                }
                for n, ok in enumerate(supported, start=1)
            ]
        }
    )


def review(ids: list[str], passed: bool, evidence: str) -> str:
    checks = [{"id": i, "check": i, "evidence": evidence, "passed": passed} for i in ids]
    return json.dumps({"checks": checks})


def test_the_summary_goes_under_a_headline_with_the_current_title():
    cv = with_summary(CASE.cv, ["Builds Python REST APIs"])

    # The recruiter's title check reads the first line, so the title stays first.
    assert cv.splitlines()[:4] == [
        "Backend Developer",
        "Builds Python REST APIs",
        "",
        "Backend Developer, online pharmacy (2021-present)",
    ]
    assert "Passionate" not in cv


def test_a_line_that_restates_one_short_fact_needs_no_judge(tmp_path):
    line = SummaryLine(facts=["F5"], text="Designed the PostgreSQL schema for prescriptions")

    kept, rejected, record = ground(FakeClient([]), [line], FACTS, [], "x", tmp_path / "r.jsonl")

    assert (kept, rejected, record) == ([line], [], None)


def test_each_summary_line_cites_exactly_one_fact():
    model = refinement_model_for(["F1", "F2"])

    with pytest.raises(ValidationError):
        model.model_validate({"summary": [{"facts": ["F1", "F2"], "text": "x"}]})


def test_rules_reject_before_the_judge_and_the_judge_rejects_meaning(tmp_path):
    lines = [SummaryLine(**TRUE_LINE), SummaryLine(**PADDED_LINE), SummaryLine(**INFLATED_LINE)]
    client = FakeClient([make_response(judged(True, False))])

    kept, rejected, record = ground(client, lines, FACTS, ["AWS"], "x", tmp_path / "r.jsonl")

    assert [line.text for line in kept] == [TRUE_LINE["text"]]
    assert [(item.line.text, item.problems[0]) for item in rejected] == [
        (PADDED_LINE["text"], "name 'AWS' is not in the cited facts"),
        (INFLATED_LINE["text"], "not supported: leads a team"),
    ]
    # Only the two lines the rules passed reach the judge.
    assert "G3" not in client.calls[0][1]["content"] and "AWS" not in client.calls[0][1]["content"]


def test_rejected_lines_go_back_once_with_the_reasons(tmp_path):
    client = FakeClient(
        [
            make_response(summary(TRUE_LINE, PADDED_LINE)),
            make_response(judged(True)),
            make_response(summary(TRUE_LINE)),
            make_response(judged(True)),
        ]
    )

    draft = write_summary(client, FACTS, ["C1"], ["AWS"], "x", tmp_path / "r.jsonl")

    assert [line.text for line in draft.kept] == [TRUE_LINE["text"]]
    assert draft.rejected == []
    retry = client.calls[2][1]["content"]
    assert "name 'AWS' is not in the cited facts" in retry
    assert [record.agent for record in read_runs(tmp_path / "r.jsonl")] == [
        "refiner",
        "grounding_judge",
        "refiner",
        "grounding_judge",
    ]


def committee(recruiter_passes: bool) -> list:
    """The four member replies for one round of present-1: the tech lead passes everything it can."""
    checklists = Checklists.for_posting(CASE.checklist)
    quote = "Built and maintain the Python REST APIs for orders and prescriptions, serving 30,000 customers a day"
    ratings = {
        "ratings": [
            {"id": f"D{n}", "evidence": [quote], "reason": "r", "shown": "fully"} for n in (1, 2, 3)
        ]
    }
    top = (
        "Builds Python REST APIs serving 30,000 customers a day"
        if recruiter_passes
        else "Passionate"
    )
    return [
        make_response(review(["C1", "C2"], recruiter_passes, top)),
        make_response(review(["HM1", "HM2", "HM3", "HM4"], False, quote)),
        make_response(review([item.id for item in checklists.tech_lead.items], True, quote)),
        make_response(json.dumps(ratings)),
    ]


def test_a_session_refines_until_nothing_fixable_is_left(tmp_path):
    client = FakeClient(
        [
            *committee(recruiter_passes=False),
            make_response(summary(TRUE_LINE)),
            make_response(judged(True)),
            # Round 2: only the recruiter reads the CV again.
            committee(recruiter_passes=True)[0],
        ]
    )

    session = run_session(client, CASE, 2026, POSTING, runs_path=tmp_path / "r.jsonl")

    assert [round_.number for round_ in session.rounds] == [1, 2]
    assert session.stop_reason == "hire"
    assert session.final_cv.splitlines()[:2] == ["Backend Developer", TRUE_LINE["text"]]
    assert TRUE_LINE["text"] in client.calls[6][1]["content"]
    first, second = (round_.committee.records for round_ in session.rounds)
    assert second["tech_lead"] is first["tech_lead"]
    assert second["recruiter"] is not first["recruiter"]


def test_a_session_stops_when_no_line_survives_the_grounding_check(tmp_path):
    client = FakeClient(
        [
            *committee(recruiter_passes=False),
            make_response(summary(INFLATED_LINE)),
            make_response(judged(False)),
            make_response(summary(INFLATED_LINE)),
            make_response(judged(False)),
        ]
    )

    session = run_session(client, CASE, 2026, POSTING, runs_path=tmp_path / "r.jsonl")

    assert session.stop_reason == "no grounded summary"
    assert session.final_cv == CASE.cv
