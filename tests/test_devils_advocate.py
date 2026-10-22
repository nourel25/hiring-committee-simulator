"""Tests for the devil's advocate: its schema, the grounding policy, the pipeline and the comparison
with the tech lead's key. Fake client; logs go to tmp_path."""

import pytest
from pydantic import ValidationError

from hcs.cases import Case, Checklist, ExpectedCheck, HiringManagerTargets, Posting, RecruiterScreen
from hcs.evaluation import compare_objections, must_haves_named
from hcs.pipeline import review_devils_advocate
from hcs.policies import ungrounded_objections
from hcs.runlog import read_runs
from hcs.schemas import ChecklistItem, Objection, Objections
from tests.fakes import FakeClient, make_response

POSTING = Posting(
    title="Backend",
    text="Requirements\n- Schema design, not just queries\n- On-call\n\nNice to have\n- Kubernetes",
    brief="x",
    recruiter=RecruiterScreen(titles=["backend engineer"], keywords=[], min_years=0),
    hiring_manager=HiringManagerTargets(system="x", goals=["x"]),
)
CV = "Backend Engineer, shop (2020-present)\n- Built the shop API\n- Wrote SQL queries"
CHECKLIST = Checklist(
    name="tiny",
    posting="Backend",
    items=[
        ChecklistItem(
            id="M1", text="api", importance="must", sources=["Built the shop API (not in posting)"]
        ),
        ChecklistItem(
            id="M2", text="schema", importance="must", sources=["Schema design, not just queries"]
        ),
        ChecklistItem(id="N1", text="k8s", importance="nice", sources=["Kubernetes"]),
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
            "M1": ExpectedCheck(passed=True, evidence="Built the shop API"),
            "M2": ExpectedCheck(passed=m2_passed),
            "N1": ExpectedCheck(passed=False),
        },
    )


def objection(requirement, evidence, severity="high") -> dict:
    return {"requirement": requirement, "evidence": evidence, "reason": "r", "severity": severity}


def reply(*objections: dict) -> str:
    return Objections(objections=[Objection(**item) for item in objections]).model_dump_json()


SCHEMA_GAP = objection("Schema design", "Wrote SQL queries")
NO_KUBERNETES = objection("Kubernetes", None, "medium")
SMALL_TEAM = objection(None, "Built the shop API", "low")


def test_exactly_three_objections_with_the_quotes_before_the_verdict():
    with pytest.raises(ValidationError):
        Objections(objections=[Objection(**SCHEMA_GAP)] * 2)
    assert list(Objection.model_fields) == ["requirement", "evidence", "reason", "severity"]


def test_an_objection_is_ungrounded_when_it_cites_nothing_or_quotes_the_wrong_text():
    objections = Objections(
        objections=[
            Objection(**SCHEMA_GAP),
            Objection(**objection(None, None)),
            # A posting line quoted as CV evidence: not in the CV.
            Objection(**objection("On-call", "Schema design, not just queries")),
        ]
    )

    assert ungrounded_objections(objections, POSTING.text, CV) == [1, 2]


def test_the_devils_advocate_sees_the_posting_and_the_cv_and_logs_its_objections(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response(reply(SCHEMA_GAP, NO_KUBERNETES, SMALL_TEAM))])

    record = review_devils_advocate(client, make_case(False), POSTING, path)

    user_message = client.calls[0][1]["content"]
    assert "<posting>\nRequirements" in user_message and "<cv>\nBackend Engineer" in user_message
    assert [item.severity for item in record.objections] == ["high", "medium", "low"]
    assert record.ungrounded_objections == []
    assert (record.agent, record.checklist, record.score) == ("devils_advocate", None, None)
    assert read_runs(path) == [record]


def test_a_failed_devils_advocate_call_is_logged(tmp_path):
    client = FakeClient([make_response('{"objections": [', finish_reason="length")])

    record = review_devils_advocate(client, make_case(False), POSTING, tmp_path / "runs.jsonl")

    assert record.status == "error"
    assert record.objections is None


def test_a_quoted_part_of_a_posting_line_names_the_must_have_it_comes_from():
    assert must_haves_named("Schema design", CHECKLIST) == ["M2"]
    assert must_haves_named("Kubernetes", CHECKLIST) == []  # a nice-to-have
    assert must_haves_named(None, CHECKLIST) == []


def test_a_high_objection_on_a_failed_must_have_is_a_hit(tmp_path):
    client = FakeClient([make_response(reply(SCHEMA_GAP, NO_KUBERNETES, SMALL_TEAM))])
    case = make_case(False)

    report = compare_objections(
        case, review_devils_advocate(client, case, POSTING, tmp_path / "r.jsonl"), CHECKLIST
    )

    assert (report.gaps, report.hit, report.false_alarms) == (["M2"], True, 0)


def test_a_high_objection_on_a_must_have_the_cv_meets_is_a_false_alarm(tmp_path):
    client = FakeClient([make_response(reply(SCHEMA_GAP, NO_KUBERNETES, SMALL_TEAM))])
    case = make_case(True)

    report = compare_objections(
        case, review_devils_advocate(client, case, POSTING, tmp_path / "r.jsonl"), CHECKLIST
    )

    # No failed must-have, so there is no gap to find; the schema objection is a false alarm.
    assert (report.gaps, report.hit, report.false_alarms) == ([], None, 1)


def test_an_ungrounded_high_objection_counts_for_nothing(tmp_path):
    invented = objection("Schema design", "Designed a 40-table schema")
    client = FakeClient([make_response(reply(invented, NO_KUBERNETES, SMALL_TEAM))])
    case = make_case(False)

    report = compare_objections(
        case, review_devils_advocate(client, case, POSTING, tmp_path / "r.jsonl"), CHECKLIST
    )

    assert (report.hit, report.ungrounded) == (False, 1)
