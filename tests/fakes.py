"""Test doubles and sample data shared by several test files."""

from types import SimpleNamespace

from hcs.cases import Case, Checklist, ExpectedCheck
from hcs.schemas import ChecklistItem

TINY_CHECKLIST = Checklist(
    name="tiny",
    posting="Test posting",
    items=[
        ChecklistItem(id="M1", text="must one", importance="must"),
        ChecklistItem(id="N1", text="nice one", importance="nice"),
    ],
)
TINY_CASE = Case(
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
TINY_REPLY = (
    '{"checks": ['
    '{"id": "M1", "check": "must one", "evidence": "Built the payments API", "passed": true}, '
    '{"id": "N1", "check": "nice one", "evidence": null, "passed": false}]}'
)


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
