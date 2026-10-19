"""Tests for hcs.llm.get_structured using a fake client. No model calls."""

import copy

import pytest
from pydantic import ValidationError

from hcs.llm import (
    REPAIR_PROMPT,
    CallResult,
    LLMOutputError,
    forbid_extra_keys,
    get_structured,
    make_client,
)
from hcs.schemas import Review
from tests.fakes import FakeClient, make_response

MESSAGES = [
    {"role": "system", "content": "You are a test reviewer."},
    {"role": "user", "content": "Review this CV."},
]
VALID = '{"checks": [{"id": "T1", "check": "x", "evidence": null, "passed": false}]}'
BROKEN = '{"checks": []}'
CUT_OFF = '{"checks": [{"id": "T1", "che'


def run(client: FakeClient) -> CallResult:
    """Call get_structured with dummy settings; the fake ignores them."""
    return get_structured(
        client,
        MESSAGES,
        response_model=Review,
        model="fake",
        temperature=0,
        seed=0,
        max_tokens=500,
    )


def test_valid_first_reply_needs_one_call():
    client = FakeClient([make_response(VALID)])

    result = run(client)

    assert result.attempts == 1
    assert isinstance(result.parsed, Review)
    assert len(client.calls) == 1


def test_broken_reply_is_retried_with_repair_messages():
    client = FakeClient([make_response(BROKEN), make_response(VALID)])

    result = run(client)

    assert result.attempts == 2
    second_call = client.calls[1]
    assert len(second_call) == len(MESSAGES) + 2
    assert second_call[-2] == {"role": "assistant", "content": BROKEN}
    assert second_call[-1]["role"] == "user"
    assert REPAIR_PROMPT.splitlines()[0] in second_call[-1]["content"]
    assert result.prompt_tokens == 20


def test_broken_twice_raises_with_cause():
    client = FakeClient([make_response(BROKEN), make_response(BROKEN)])

    with pytest.raises(LLMOutputError) as exc_info:
        run(client)

    assert exc_info.value.attempts == 2
    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert len(client.calls) == 2


def test_cut_off_raises_without_retry():
    client = FakeClient([make_response(CUT_OFF, finish_reason="length"), make_response(VALID)])

    with pytest.raises(LLMOutputError) as exc_info:
        run(client)

    assert len(client.calls) == 1
    assert "max_tokens" in str(exc_info.value)


def test_callers_messages_are_not_changed():
    before = copy.deepcopy(MESSAGES)
    client = FakeClient([make_response(BROKEN), make_response(VALID)])

    run(client)

    assert MESSAGES == before


def test_strict_sends_a_closed_schema_and_the_strict_flag():
    client = FakeClient([make_response(VALID)])

    get_structured(
        client,
        MESSAGES,
        response_model=Review,
        model="fake",
        temperature=0,
        seed=0,
        max_tokens=500,
        strict=True,
    )

    json_schema = client.kwargs[0]["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["$defs"]["Check"]["additionalProperties"] is False


def test_without_strict_the_schema_is_sent_as_before():
    client = FakeClient([make_response(VALID)])

    run(client)

    json_schema = client.kwargs[0]["response_format"]["json_schema"]
    assert "strict" not in json_schema
    assert json_schema["schema"] == Review.model_json_schema()


def test_forbid_extra_keys_leaves_the_original_schema_unchanged():
    schema = Review.model_json_schema()

    forbid_extra_keys(schema)

    assert "additionalProperties" not in schema["$defs"]["Check"]


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        make_client("openrouter")
