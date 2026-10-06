"""Tests for hcs.pipeline.review_case with a fake client. No model calls; logs go to tmp_path."""

from hcs.cases import Case, Checklist, ExpectedCheck
from hcs.pipeline import review_case
from hcs.runlog import read_runs
from hcs.schemas import ChecklistItem
from tests.fakes import FakeClient, make_response

CHECKLIST = Checklist(
    name="tiny",
    posting="Test posting",
    items=[
        ChecklistItem(id="M1", text="must one", importance="must"),
        ChecklistItem(id="N1", text="nice one", importance="nice"),
    ],
)
CASE = Case(
    id="case-1",
    label="good",
    checklist="tiny",
    expected_decision="lean_hire",
    cv="Built the payments API",
    expected={
        "M1": ExpectedCheck(passed=True, evidence="Built the payments API"),
        "N1": ExpectedCheck(passed=False),
    },
)

# M1 passes with a real quote (3 of 4 points), N1 fails: 7.5 -> lean_hire.
REPLY = (
    '{"checks": ['
    '{"id": "M1", "check": "must one", "evidence": "Built the payments API", "passed": true}, '
    '{"id": "N1", "check": "nice one", "evidence": null, "passed": false}]}'
)


def test_review_case_scores_logs_and_returns_the_record(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response(REPLY)])

    record = review_case(client, CASE, CHECKLIST, path)

    assert record.status == "ok"
    assert record.case_id == "case-1"
    assert record.score == 7.5
    assert record.decision == "lean_hire"
    assert read_runs(path) == [record]


def test_review_case_logs_a_failed_call(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response('{"checks": [', finish_reason="length")])

    record = review_case(client, CASE, CHECKLIST, path)

    assert record.status == "error"
    assert record.decision is None
    assert read_runs(path) == [record]
