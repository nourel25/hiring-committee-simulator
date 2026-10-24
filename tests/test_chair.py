"""Tests for the chair: the hire rule, dissent and confidence, and merging the members' objections."""

import json
from datetime import datetime, timezone

import pytest

from hcs.cases import Checklist, load_case, load_cases, load_posting
from hcs.chair import LABEL_VERDICT, chair, confidence_from, decide_verdict, find_dissent
from hcs.committee import Checklists, format_transcript, run_committee, weigh
from hcs.config import Endpoint
from hcs.runlog import RunRecord
from hcs.schemas import Check, ChecklistItem, Objection, Review
from tests.fakes import FakeClient, make_response

TECH = Checklist(
    name="tiny",
    posting="Backend",
    items=[
        ChecklistItem(id="M1", text="api", importance="must", sources=["Build APIs"]),
        ChecklistItem(id="M2", text="schema", importance="must", sources=["Schema design"]),
        ChecklistItem(id="N1", text="k8s", importance="nice"),
    ],
)
RECRUITER = Checklist(
    name="recruiter",
    posting="any",
    items=[
        ChecklistItem(id="C1", text="clear", importance="must"),
        ChecklistItem(id="C2", text="number", importance="nice"),
    ],
)
MANAGER = Checklist(
    name="hiring_manager",
    posting="any",
    items=[ChecklistItem(id="HM1", text="fit", importance="nice")],
)


def record(agent: str, decision=None, score=None, checks=None, objections=None) -> RunRecord:
    return RunRecord(
        run_id="x",
        started_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
        agent=agent,
        prompt_version="v",
        model="m",
        temperature=0,
        seed=42,
        max_tokens=10,
        max_retries=1,
        messages=[],
        status="ok",
        attempts=1,
        raw_reply="{}",
        review=Review(checks=checks) if checks else None,
        score=score,
        decision=decision,
        objections=objections,
    )


def check(check_id: str, passed: bool, evidence: str | None = None) -> Check:
    return Check(id=check_id, check=check_id.lower(), evidence=evidence, passed=passed)


def objection(requirement: str, severity: str) -> Objection:
    return Objection(requirement=requirement, evidence=[], reason="r", severity=severity)


@pytest.mark.parametrize(
    ("tech_lead", "highs", "verdict"),
    [
        ("strong_hire", 0, "hire"),
        ("lean_hire", 0, "hire"),
        ("strong_hire", 1, "borderline"),
        ("lean_no", 0, "borderline"),
        ("strong_no", 0, "no_hire"),
        ("strong_no", 2, "no_hire"),
        (None, 0, "borderline"),
    ],
)
def test_the_hire_rule(tech_lead, highs, verdict):
    assert decide_verdict(tech_lead, highs)[0] == verdict


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.id)
def test_the_rule_on_each_key_gives_the_label(case):
    # With the tech lead exactly as its key says and no false high, the rule gives the case's label:
    # the labels are about the requirements, which only the tech lead's key encodes.
    assert decide_verdict(case.expected_decision, 0)[0] == LABEL_VERDICT[case.label]


def test_dissent_is_a_member_on_the_other_side_of_the_verdict():
    decisions = {"tech_lead": "strong_hire", "recruiter": "strong_no", "hiring_manager": None}

    assert find_dissent("hire", decisions) == ["recruiter"]
    assert find_dissent("borderline", decisions) == ["tech_lead"]


def test_confidence_falls_with_dissent():
    assert confidence_from([], 3) == "high"
    assert confidence_from(["recruiter"], 3) == "medium"
    assert confidence_from(["recruiter", "hiring_manager"], 3) == "low"


def test_a_devils_advocate_objection_joins_the_must_have_it_names():
    verdict = chair(
        tech_lead=record(
            "tech_lead", "lean_no", 6.0, [check("M1", True, "Built APIs"), check("M2", False)]
        ),
        tech_checklist=TECH,
        recruiter=record("recruiter", "strong_hire", 10.0, [check("C1", True), check("C2", False)]),
        recruiter_checklist=RECRUITER,
        hiring_manager=None,
        hiring_manager_checklist=MANAGER,
        devil=record(
            "devils_advocate",
            objections=[objection("Schema design", "high"), objection("Build APIs", "medium")],
        ),
    )

    assert verdict.verdict == "borderline"
    assert verdict.dissent == ["recruiter"]
    # The medium objection on the API line (which the tech lead passed) is noise and left out.
    first, second = verdict.objections
    assert (first.text, first.severity, first.members) == (
        "m2",
        "high",
        ["tech_lead", "devils_advocate"],
    )
    assert (second.text, second.severity, second.kind) == ("c2", "low", "presentation")


def test_a_high_objection_on_a_line_the_tech_lead_passed_blocks_hire_and_is_listed():
    verdict = chair(
        tech_lead=record("tech_lead", "strong_hire", 10.0, [check("M1", True), check("M2", True)]),
        tech_checklist=TECH,
        recruiter=None,
        recruiter_checklist=RECRUITER,
        hiring_manager=None,
        hiring_manager_checklist=MANAGER,
        devil=record("devils_advocate", objections=[objection("Schema design", "high")]),
    )

    assert verdict.verdict == "borderline"
    assert [(item.text, item.members) for item in verdict.objections] == [
        ("Schema design", ["devils_advocate"])
    ]


def test_a_failed_member_run_is_left_out():
    failed = record("tech_lead").model_copy(update={"status": "error", "error": "cut off"})

    verdict = weigh({"tech_lead": failed}, Checklists(TECH, RECRUITER, MANAGER))

    assert verdict.verdict == "borderline"
    assert verdict.scores["tech_lead"] is None


def test_the_committee_runs_four_independent_members_then_the_chair(tmp_path):
    case = load_case("good-1")
    posting = load_posting(case.checklist)
    checklists = Checklists.for_posting(case.checklist)
    line = (
        "Built and maintained Python REST APIs"  # not in the CV: every pass fails the truth check
    )

    def review(ids):
        checks = [{"id": i, "check": i, "evidence": line, "passed": True} for i in ids]
        return make_response(json.dumps({"checks": checks}))

    ratings = {
        "ratings": [
            {"id": f"D{n}", "evidence": [], "reason": "r", "shown": "not"} for n in (1, 2, 3)
        ]
    }
    client = FakeClient(
        [
            review(["C1", "C2"]),
            review(["HM1", "HM2", "HM3", "HM4"]),
            review([item.id for item in checklists.tech_lead.items]),
            make_response(json.dumps(ratings)),
        ]
    )

    round_ = run_committee(client, case, 2026, posting, checklists, tmp_path / "runs.jsonl")

    # No member's messages contain another member's output.
    assert len(client.calls) == 4
    assert round_.records["tech_lead"].decision == "strong_no"
    assert round_.verdict.verdict == "no_hire"
    assert "Chair: no_hire" in format_transcript(round_)


def test_the_devils_advocate_can_run_on_its_own_endpoint(tmp_path):
    case = load_case("good-1")
    checklists = Checklists.for_posting(case.checklist)
    ratings = {
        "ratings": [
            {"id": f"D{n}", "evidence": [], "reason": "r", "shown": "not"} for n in (1, 2, 3)
        ]
    }

    def review(ids):
        checks = [{"id": i, "check": i, "evidence": None, "passed": False} for i in ids]
        return make_response(json.dumps({"checks": checks}))

    local = FakeClient(
        [
            review(["C1", "C2"]),
            review(["HM1", "HM2", "HM3", "HM4"]),
            review([item.id for item in checklists.tech_lead.items]),
        ]
    )
    remote = FakeClient([make_response(json.dumps(ratings))])

    round_ = run_committee(
        local,
        case,
        2026,
        load_posting(case.checklist),
        checklists,
        tmp_path / "runs.jsonl",
        devil=Endpoint(remote, {"model": "api-model", "seed": None}),
    )

    assert len(local.calls) == 3 and len(remote.calls) == 1
    assert round_.records["devils_advocate"].model == "api-model"
    assert "seed" not in remote.kwargs[0]
