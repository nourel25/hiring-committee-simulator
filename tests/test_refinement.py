"""Tests for the refinement loop: triage, the summary, the grounding of each line, and the stops."""

import json

import pytest
from pydantic import ValidationError

from hcs.cases import load_case, load_cases, load_posting
from hcs.committee import Checklists, CommitteeRound
from hcs.profile import build_facts
from hcs.refinement import (
    Judge,
    ground,
    own_summary,
    run_session,
    triage,
    with_summary,
    write_summary,
)
from hcs.runlog import read_runs
from hcs.schemas import SummaryLine, refinement_model_for
from tests.fakes import FakeClient, make_response
from tests.test_chair import check, record

CASE = load_case("present-1")
POSTING = load_posting(CASE.checklist)
FACTS = build_facts(CASE.cv)
F4 = FACTS[3]
TRUE_LINE = {"facts": ["F4"], "text": "Builds Python REST APIs serving 30,000 customers a day"}
PADDED_LINE = {"facts": ["F4"], "text": "Builds Python REST APIs on AWS"}
INFLATED_LINE = {"facts": ["F4"], "text": "Leads the team that builds Python REST APIs"}


def summary(*lines: dict) -> str:
    return json.dumps({"summary": list(lines)})


def judged(*supported: bool) -> str:
    return json.dumps(
        {
            "judgements": [
                {"id": f"G{n}", "unsupported": None if ok else "leads a team", "supported": ok}
                for n, ok in enumerate(supported, start=1)
            ]
        }
    )


def review(ids: list[str], passed: bool, evidence: str) -> str:
    checks = [{"id": i, "check": i, "evidence": evidence, "passed": passed} for i in ids]
    return json.dumps({"checks": checks})


def recruiter_round(*failed: str) -> CommitteeRound:
    checks = [check(i, i not in failed) for i in ("R1", "R2", "R3", "R4", "C1", "C2")]
    return CommitteeRound("x", {"recruiter": record("recruiter", checks=checks)}, None, 0.0)


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


def test_a_cv_with_its_own_summary_section_is_known_and_the_labelled_ones_are_not():
    own = (
        "Software Engineer\nAbout Me\nBackend engineer.\nEngineer, shop (2024-present)\n- Built it"
    )
    later = "Engineer, shop (2024-present)\n- Built it\nProfile\nNot above the jobs"

    assert own_summary(own)
    assert not own_summary(later)
    assert not any(own_summary(case.cv) for case in load_cases())


def test_the_summary_never_drops_a_section_of_an_uploaded_cv():
    cv = "\n".join(
        [
            "Software Engineer",
            "About Me",
            "Backend engineer who builds compliance software.",
            "Education",
            "B.Sc. Computer Science, Example University (2021-2025)",
            "Experience",
            "Software Engineer—Northwind, Lisbon (Mar 2026-present)",
            "- Built the rights module",
        ]
    )

    lines = with_summary(cv, ["Cut search time 5x"]).splitlines()

    assert lines[:3] == ["Software Engineer", "Cut search time 5x", ""]
    assert lines[3:] == cv.splitlines()[1:]


def test_triage_picks_a_fact_for_each_fixable_check():
    cv = "Engineer, shop (2020-present)\n- Built the shop API\n- Wrote Python jobs\n- Cut costs by 30%"
    facts = build_facts(cv)

    round_ = recruiter_round("R1", "R3", "C2")
    round_.records["tech_lead"] = record(
        "tech_lead", checks=[check("TL7", True, "Cut costs by 30%")]
    )

    plan = triage(cv, round_, facts, ["Python", "Kafka"])

    # Python is in a bullet; Kafka is in none, but one found keyword is worth a summary line.
    assert [fact.text for fact in plan.facts] == ["Wrote Python jobs", "Cut costs by 30%"]
    assert (plan.fixable, plan.gaps) == (["R3", "C2"], [])


def test_a_check_no_fact_can_fix_is_a_gap():
    cv = "Engineer, shop (2020-present)\n- Built the shop API in 2021"
    facts = build_facts(cv)

    plan = triage(cv, recruiter_round("R3", "C2"), facts, ["Kafka"])

    # A year is not a result, and no bullet names Kafka.
    assert (plan.facts, plan.fixable, plan.gaps) == ([], [], ["R3", "C2"])


def test_a_line_that_restates_one_short_fact_needs_no_judge(tmp_path):
    line = SummaryLine(facts=["F5"], text="Designed the PostgreSQL schema for prescriptions")

    kept, rejected, judge = ground(
        Judge(FakeClient([])), [line], FACTS, [], "x", tmp_path / "r.jsonl"
    )

    assert (kept, rejected, judge) == ([line], [], None)


def test_each_chosen_fact_gets_exactly_one_line():
    model = refinement_model_for(["F1", "F2"])

    with pytest.raises(ValidationError):
        model.model_validate({"summary": [{"facts": ["F1", "F2"], "text": "x"}]})
    with pytest.raises(ValidationError):
        model.model_validate({"summary": [{"facts": ["F1"], "text": "x"}]})
    assert model.model_validate(
        {"summary": [{"facts": ["F2"], "text": "x"}, {"facts": ["F1"], "text": "y"}]}
    )


def test_rules_reject_before_the_judge_and_the_judge_rejects_meaning(tmp_path):
    lines = [SummaryLine(**TRUE_LINE), SummaryLine(**PADDED_LINE), SummaryLine(**INFLATED_LINE)]
    client = FakeClient([make_response(judged(True, False))])

    kept, rejected, judge = ground(Judge(client), lines, FACTS, ["AWS"], "x", tmp_path / "r.jsonl")

    assert [line.text for line in kept] == [TRUE_LINE["text"]]
    assert [(item.line.text, item.problems[0]) for item in rejected] == [
        (PADDED_LINE["text"], "name 'AWS' is not in the cited facts"),
        (INFLATED_LINE["text"], "not supported: leads a team"),
    ]
    # Only the two lines the rules passed reach the judge.
    assert "G3" not in client.calls[0][1]["content"] and "AWS" not in client.calls[0][1]["content"]


def test_a_rejected_line_goes_back_once_with_the_reasons(tmp_path):
    client = FakeClient(
        [
            make_response(summary(PADDED_LINE)),
            make_response(summary(TRUE_LINE)),
            make_response(judged(True)),
        ]
    )

    draft = write_summary(client, FACTS, [F4], ["AWS"], "x", tmp_path / "r.jsonl")

    assert [line.text for line in draft.kept] == [TRUE_LINE["text"]]
    assert "name 'AWS' is not in the cited facts" in client.calls[1][1]["content"]
    assert [item.agent for item in read_runs(tmp_path / "r.jsonl")] == [
        "refiner",
        "refiner",
        "grounding_judge",
    ]


def test_a_short_fact_whose_lines_keep_failing_is_used_as_written(tmp_path):
    client = FakeClient(
        [
            make_response(summary(INFLATED_LINE)),
            make_response(judged(False)),
            make_response(summary(INFLATED_LINE)),
            make_response(judged(False)),
        ]
    )

    draft = write_summary(client, FACTS, [F4], [], "x", tmp_path / "r.jsonl")

    assert [line.text for line in draft.kept] == [F4.text]
    assert draft.verbatim == ["F4"]


def committee(recruiter_passes: bool) -> list:
    """The four member replies for one round of present-1: the tech lead passes everything it can."""
    checklists = Checklists.for_posting(CASE.checklist)
    quote = F4.text
    ratings = {
        "ratings": [
            {"id": f"D{n}", "evidence": [quote], "reason": "r", "shown": "fully"} for n in (1, 2, 3)
        ]
    }
    top = TRUE_LINE["text"] if recruiter_passes else "Passionate"
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
    # C1 and C2 failed; F4 says what the work is and has a number, so it is the one fact chosen.
    assert [fact.id for fact in session.rounds[0].plan.facts] == ["F4"]
    assert session.stop_reason == "hire"
    assert session.final_cv.splitlines()[:2] == ["Backend Developer", TRUE_LINE["text"]]
    assert TRUE_LINE["text"] in client.calls[6][1]["content"]
    first, second = (round_.committee.records for round_ in session.rounds)
    assert second["tech_lead"] is first["tech_lead"]
    assert second["recruiter"] is not first["recruiter"]


def test_the_grounding_judge_uses_its_own_client(tmp_path):
    committee_client = FakeClient([make_response(summary(TRUE_LINE))])
    judge_client = FakeClient([make_response(judged(True))])
    judge = Judge(judge_client, {"model": "api-model", "max_retries": 3})

    draft = write_summary(committee_client, FACTS, [F4], [], "x", tmp_path / "r.jsonl", judge=judge)

    assert [line.text for line in draft.kept] == [TRUE_LINE["text"]]
    assert len(committee_client.calls) == 1 and len(judge_client.calls) == 1
    assert read_runs(tmp_path / "r.jsonl")[-1].model == "api-model"


def test_c1_takes_the_line_the_tech_lead_quoted_not_personal_qualities():
    cv = (
        "Engineer, shop (2020-present)\n- Hard-working and reliable\n- Built the shop API in Python"
    )
    round_ = recruiter_round("C1")
    round_.records["tech_lead"] = record(
        "tech_lead", checks=[check("TL1", True, "Built the shop API in Python")]
    )

    plan = triage(cv, round_, build_facts(cv), [])

    assert [fact.text for fact in plan.facts] == ["Built the shop API in Python"]


def test_c2_takes_a_numbered_line_a_reviewer_passed_never_the_employers_numbers():
    cv = (
        "Engineer, PayCo (2020-present)\n- PayCo serves 4 million users in 12 countries\n"
        "- Built the payouts API in Python, used by 9,000 merchants\n- Cut payout time from 3 days to 1"
    )
    round_ = recruiter_round("C2")
    round_.records["tech_lead"] = record(
        "tech_lead",
        checks=[
            check("TL1", True, "Built the payouts API in Python, used by 9,000 merchants"),
            check("TL7", True, "Cut payout time from 3 days to 1"),
        ],
    )

    plan = triage(cv, round_, build_facts(cv), [])

    # A measured result (TL7) comes before a scale; the employer's numbers are never chosen.
    assert [fact.text for fact in plan.facts] == ["Cut payout time from 3 days to 1"]


def test_c2_with_no_reviewed_numbered_line_is_a_gap():
    cv = "Engineer, PayCo (2020-present)\n- PayCo serves 4 million users in 12 countries"

    plan = triage(cv, recruiter_round("C2"), build_facts(cv), [])

    assert (plan.facts, plan.gaps) == ([], ["C2"])


def test_facts_already_in_the_summary_stay_and_only_free_lines_are_filled():
    cv = "Engineer, shop (2020-present)\n- Built APIs in Python\n- Wrote SQL\n- Ran Kafka\n- Used REST"
    facts = build_facts(cv)

    plan = triage(
        cv, recruiter_round("R3"), facts, ["Python", "SQL", "Kafka", "REST"], ["F2", "F3"]
    )

    # F2 and F3 are already summary lines: of the two keywords left, one line is free.
    assert [fact.id for fact in plan.facts] == ["F4"]


def test_a_session_keeps_earlier_summary_lines_and_uses_the_given_judge(tmp_path):
    committee_client = FakeClient(
        [
            *committee(recruiter_passes=False),
            make_response(summary(TRUE_LINE)),
            committee(recruiter_passes=True)[0],
        ]
    )
    judge_client = FakeClient([make_response(judged(True))])
    judge = Judge(judge_client, {"model": "api-model"})

    session = run_session(
        committee_client, CASE, 2026, POSTING, runs_path=tmp_path / "r.jsonl", judge=judge
    )

    assert len(judge_client.calls) == 1
    assert TRUE_LINE["text"] in session.final_cv


def test_a_cv_with_its_own_summary_gets_suggestions_and_is_left_unchanged(tmp_path):
    # "About Me" replaces the first line, so the fact ids (F4 = the REST bullet) stay the same.
    own = CASE.model_copy(update={"cv": "About Me\n" + CASE.cv.split("\n", 1)[1]})
    committee_client = FakeClient(
        [*committee(recruiter_passes=False), make_response(summary(TRUE_LINE))]
    )
    judge = Judge(FakeClient([make_response(judged(True))]), {"model": "api-model"})

    session = run_session(
        committee_client, own, 2026, POSTING, runs_path=tmp_path / "r.jsonl", judge=judge
    )

    assert session.final_cv == own.cv
    assert session.suggestions == [TRUE_LINE["text"]]
    assert session.stop_reason.startswith("the CV has its own summary")


def recruiter_reply(c1: str | None, c2: str | None) -> object:
    """The recruiter model's two checks; a check passes with the quote given, fails with None."""
    checks = [
        {"id": "C1", "check": "C1", "evidence": c1, "passed": c1 is not None},
        {"id": "C2", "check": "C2", "evidence": c2, "passed": c2 is not None},
    ]
    return make_response(json.dumps({"checks": checks}))


def test_each_round_adds_to_the_summary_and_never_drops_earlier_lines(tmp_path):
    deploy, schema = FACTS[5], FACTS[4]  # F6 (no number) and F5 (9 tables)
    deploy_line = {
        "facts": ["F6"],
        "text": "Deploy services to production with GitHub Actions and Kubernetes",
    }
    schema_line = {
        "facts": ["F5"],
        "text": "Designed the PostgreSQL schema for prescriptions: 9 tables",
    }
    checklists = Checklists.for_posting(CASE.checklist)
    tech_lead = [
        {"id": item.id, "check": item.id, "evidence": quote, "passed": True}
        for item, quote in zip(
            checklists.tech_lead.items,
            [deploy.text, F4.text, schema.text, deploy.text, None, None, schema.text, None],
            strict=True,
        )
    ]
    for item in tech_lead:
        item["passed"] = item["evidence"] is not None
    client = FakeClient(
        [
            # Round 1: C1 fails. The tech lead's first passing quote (F6) says what the work is.
            recruiter_reply(None, F4.text),
            committee(recruiter_passes=False)[1],
            make_response(json.dumps({"checks": tech_lead})),
            committee(recruiter_passes=False)[3],
            make_response(summary(deploy_line)),
            # Round 2: C2 fails. TL7's quote (F5, 9 tables) is a measured line a reviewer passed.
            recruiter_reply(deploy_line["text"], None),
            make_response(summary(schema_line)),
            # Round 3: the last round only reviews.
            recruiter_reply(deploy_line["text"], schema_line["text"]),
        ]
    )

    session = run_session(client, CASE, 2026, POSTING, runs_path=tmp_path / "r.jsonl")

    lines = session.final_cv.splitlines()
    assert lines[:3] == ["Backend Developer", deploy_line["text"], schema_line["text"]]
    assert [[fact.id for fact in round_.plan.facts] for round_ in session.rounds[:2]] == [
        ["F6"],
        ["F5"],
    ]
