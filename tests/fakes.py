"""Test doubles shared by several test files: a fake OpenAI client with scripted replies."""

from types import SimpleNamespace


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
