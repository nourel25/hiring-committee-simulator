"""Talking to the local model: one place for the client and the call."""

import time

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

OLLAMA_BASE_URL = "http://localhost:11434/v1"


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
