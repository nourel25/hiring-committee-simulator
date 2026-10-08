"""Talking to the local model: one place for the client and the call."""

import json
import time
import urllib.request
from dataclasses import dataclass

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_BASE_URL = f"{OLLAMA_HOST}/v1"

REPAIR_PROMPT = (
    "Your previous reply did not match the required JSON schema.\n"
    "Validation errors:\n{error}\n"
    "Reply again with only the corrected JSON."
)


class LLMOutputError(Exception):
    """The model's reply could not be turned into the requested schema."""

    def __init__(self, message: str, attempts: int, raw_reply: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.raw_reply = raw_reply


@dataclass
class CallResult:
    """A validated reply plus what it cost to get it."""

    parsed: BaseModel
    attempts: int
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    raw_reply: str


def make_client() -> OpenAI:
    """Create a client pointed at the local Ollama server."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def call_model(
    client: OpenAI,
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
) -> tuple[ChatCompletion, float]:
    """Send one request constrained to response_model's schema; return the response and seconds taken."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        },
    )
    return response, time.perf_counter() - start


def get_structured(
    client: OpenAI,
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
    max_retries: int = 1,
) -> CallResult:
    """Call the model and validate the reply; on a schema error, retry with the error attached."""
    conversation = list(messages)
    total_attempts = max_retries + 1
    total_seconds = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for attempt in range(1, total_attempts + 1):
        response, seconds = call_model(
            client,
            conversation,
            response_model,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        total_seconds += seconds
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        raw_reply = choice.message.content or ""

        if choice.finish_reason == "length":
            raise LLMOutputError(
                f"Cut off at max_tokens ({max_tokens}): the JSON is incomplete.",
                attempts=attempt,
                raw_reply=raw_reply,
            )

        try:
            parsed = response_model.model_validate_json(raw_reply)
        except ValidationError as error:
            if attempt == total_attempts:
                raise LLMOutputError(
                    f"Reply still invalid after {attempt} attempt(s).",
                    attempts=attempt,
                    raw_reply=raw_reply,
                ) from error
            conversation.append({"role": "assistant", "content": raw_reply})
            conversation.append({"role": "user", "content": REPAIR_PROMPT.format(error=error)})
            continue

        return CallResult(
            parsed=parsed,
            attempts=attempt,
            seconds=total_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_reply=raw_reply,
        )


def gpu_share(model: str) -> float | None:
    """Share of a loaded model's memory on the GPU, from Ollama's native /api/ps; None if not loaded."""
    with urllib.request.urlopen(f"{OLLAMA_HOST}/api/ps", timeout=5) as response:
        loaded = json.load(response)["models"]
    for entry in loaded:
        if entry["name"] == model:
            return entry["size_vram"] / entry["size"]
    return None
