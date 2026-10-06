"""Tests for hcs.pipeline.review_case with a fake client. No model calls; logs go to tmp_path."""

from hcs.pipeline import review_case
from hcs.runlog import read_runs
from tests.fakes import TINY_CASE, TINY_CHECKLIST, TINY_REPLY, FakeClient, make_response


def test_review_case_scores_logs_and_returns_the_record(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response(TINY_REPLY)])

    record = review_case(client, TINY_CASE, TINY_CHECKLIST, path)

    assert record.status == "ok"
    assert record.case_id == "case-1"
    assert record.score == 7.5
    assert record.decision == "lean_hire"
    assert read_runs(path) == [record]


def test_review_case_logs_a_failed_call(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response('{"checks": [', finish_reason="length")])

    record = review_case(client, TINY_CASE, TINY_CHECKLIST, path)

    assert record.status == "error"
    assert record.decision is None
    assert read_runs(path) == [record]
