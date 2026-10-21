"""Tests for hcs.pipeline (tech lead and recruiter) with a fake client. No model calls; logs go to tmp_path."""

from hcs.cases import Case, Checklist, ExpectedCheck, Posting, RecruiterScreen
from hcs.pipeline import review_case, review_recruiter
from hcs.runlog import read_runs
from hcs.schemas import ChecklistItem
from tests.fakes import TINY_CASE, TINY_CHECKLIST, TINY_REPLY, FakeClient, make_response

RECRUITER_CHECKLIST = Checklist(
    name="recruiter",
    posting="any",
    items=[
        ChecklistItem(id="R1", text="title", importance="must"),
        ChecklistItem(id="C1", text="clear first lines", importance="must"),
    ],
)
RECRUITER_POSTING = Posting(
    title="Backend",
    text="x",
    brief="x",
    recruiter=RecruiterScreen(titles=["backend engineer"], keywords=[], min_years=0),
)
# Six lines, so the top third is the title and the first bullet.
RECRUITER_CASE = Case(
    id="r-1",
    label="good",
    checklist="tiny",
    expected_decision="strong_hire",
    cv="Backend Engineer, shop (2020-present)\n- Built the shop API\n- a\n- b\n- c\n- Hidden line",
    expected={"M1": ExpectedCheck(passed=True, evidence="Built the shop API")},
)
RECRUITER_REPLY = (
    '{"checks": [{"id": "C1", "check": "clear first lines", '
    '"evidence": "Built the shop API", "passed": true}]}'
)


def test_review_case_scores_logs_and_returns_the_record(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response(TINY_REPLY)])

    record = review_case(client, TINY_CASE, TINY_CHECKLIST, path)

    assert record.status == "ok"
    assert record.case_id == "case-1"
    assert record.checklist == "tiny@v1"
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


def test_review_case_uses_and_logs_the_model_temperature_and_seed_it_is_given(tmp_path):
    client = FakeClient([make_response(TINY_REPLY)])

    record = review_case(
        client,
        TINY_CASE,
        TINY_CHECKLIST,
        tmp_path / "runs.jsonl",
        model="fake-model",
        temperature=0.7,
        seed=3,
    )

    settings = client.kwargs[0]
    assert (settings["model"], settings["temperature"], settings["seed"]) == ("fake-model", 0.7, 3)
    assert (record.model, record.temperature, record.seed) == ("fake-model", 0.7, 3)


def test_review_case_passes_max_tokens_and_strict_to_the_call_and_logs_max_tokens(tmp_path):
    client = FakeClient([make_response(TINY_REPLY)])

    record = review_case(
        client, TINY_CASE, TINY_CHECKLIST, tmp_path / "runs.jsonl", max_tokens=2000, strict=True
    )

    settings = client.kwargs[0]
    assert settings["max_tokens"] == 2000
    assert settings["response_format"]["json_schema"]["strict"] is True
    assert record.max_tokens == 2000


def test_review_case_reports_wrong_ids_as_a_failed_run_not_a_zero_score(tmp_path):
    wrong_ids = TINY_REPLY.replace('"id": "M1"', '"id": "M1: must one"')
    client = FakeClient([make_response(wrong_ids), make_response(wrong_ids)])

    record = review_case(client, TINY_CASE, TINY_CHECKLIST, tmp_path / "runs.jsonl")

    assert record.status == "error"
    assert record.score is None
    assert len(client.calls) == 2


def test_the_recruiter_scores_code_checks_and_model_checks_together(tmp_path):
    path = tmp_path / "runs.jsonl"
    client = FakeClient([make_response(RECRUITER_REPLY)])

    record = review_recruiter(
        client, RECRUITER_CASE, RECRUITER_POSTING, RECRUITER_CHECKLIST, 2026, path
    )

    # Code checks come first; R2-R4 are not in this checklist, so scoring ignores them.
    assert [check.id for check in record.review.checks] == ["R1", "R2", "R3", "R4", "C1"]
    # R1 (code) passes: Backend Engineer; C1 (model) passes with a quote from the top third.
    assert record.score == 10.0
    assert record.agent == "recruiter"
    assert read_runs(path) == [record]


def test_the_recruiter_model_sees_only_the_top_third_and_only_its_own_checks(tmp_path):
    client = FakeClient([make_response(RECRUITER_REPLY)])

    review_recruiter(
        client, RECRUITER_CASE, RECRUITER_POSTING, RECRUITER_CHECKLIST, 2026, tmp_path / "r.jsonl"
    )

    user_message = client.calls[0][1]["content"]
    assert "Backend Engineer, shop (2020-present)" in user_message
    assert "Hidden line" not in user_message
    assert "- C1:" in user_message and "R1" not in user_message and "R4" not in user_message
    schema = client.kwargs[0]["response_format"]["json_schema"]["schema"]
    # One allowed id is written as "const", several as "enum"; both bind the grammar.
    id_schema = schema["$defs"]["Check"]["properties"]["id"]
    assert id_schema.get("enum", [id_schema.get("const")]) == ["C1"]


def test_a_recruiter_quote_from_below_the_top_third_fails_the_truth_check(tmp_path):
    below = RECRUITER_REPLY.replace("Built the shop API", "Hidden line")
    client = FakeClient([make_response(below)])

    record = review_recruiter(
        client, RECRUITER_CASE, RECRUITER_POSTING, RECRUITER_CHECKLIST, 2026, tmp_path / "r.jsonl"
    )

    assert record.truth_check_changes == 1
    assert record.failed_must_haves == ["C1"]
    assert record.decision == "lean_no"
