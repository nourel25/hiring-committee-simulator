"""Tests for hcs.llm.get_structured using a fake client. No model calls."""

import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hcs.llm import REPAIR_PROMPT, CallResult, LLMOutputError, get_structured
from hcs.schemas import Review

MESSAGES = [
    {"role": "system", "content": "You are a test reviewer."},
    {"role": "user", "content": "Review this CV."},
]
VALID = '{"checks": [{"id": "T1", "check": "x", "evidence": null, "passed": false}]}'
BROKEN = '{"checks": []}'
CUT_OFF = '{"checks": [{"id": "T1", "che'


def make_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    """Build an object shaped like the parts of a ChatCompletion that hcs.llm reads."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeClient:
    """Stands in for OpenAI: returns scripted responses and records the messages it was sent."""

    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> SimpleNamespace:
        if not self.replies:
            raise AssertionError("Unexpected extra call to the model")
        self.calls.append(list(kwargs["messages"]))
        return self.replies.pop(0)


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
