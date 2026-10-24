"""Which model plays which role, as decided by the Phase 0 experiment (Stage 3).

Change a role's model here, not in the agents. Evidence: the Phase 0 results in the README and the commit history.
"""

from dataclasses import dataclass, field

from openai import OpenAI

from hcs.llm import Provider, make_client


@dataclass(frozen=True)
class ModelChoice:
    """A model and where it runs."""

    provider: Provider
    model: str


# In-loop judge: the only family through the gate (7/7 on checklist v3, 0 failed, 27 s/review).
JUDGE_A = ModelChoice("ollama", "qwen2.5:7b")

# Second in-loop judge from another family: open until Stage 4 (no other family passed the gate).
JUDGE_B: ModelChoice | None = None

# Outside the loop, eval set only: bigger and from another family (7/7 on v3, strict schema).
HELD_OUT_JUDGE = ModelChoice("groq", "openai/gpt-oss-120b")

# The grounding checker's judge layer, in the loop (Step 5.2). Not the refiner's family. On the 12
# held-out claims qwen2.5:7b got 7/12 (1 lenient); this 27B model on Groq got 12/12 (0 lenient) with
# the same prompt. Groq has no strict schema for it, so format slips get 3 repair retries.
GROUNDING_JUDGE = ModelChoice("groq", "qwen/qwen3.8-27b")
GROUNDING_JUDGE_RETRIES = 3

# The devil's advocate, in the loop (2026-10-24). Compared on 36 cases with the rule in
# hcs/run_compare_da.py (written before the Google runs): only this model and gemma-4-31b-it found
# every must-have gap with no false high; gemma had one failed run (a provider 500) and is 5x slower,
# qwen2.5:7b missed 2 gaps and made a false high, gemini-3.8-flash a false high. Same prompt (v3).
DEVILS_ADVOCATE = ModelChoice("groq", "qwen/qwen3.8-27b")
DEVILS_ADVOCATE_RETRIES = 3

# The rubric these choices were measured on.
CHECKLIST = "backend_payments"

# API reasoning models think before the JSON: up to 1,849 tokens in the v3 run, so keep a wide margin.
API_MAX_TOKENS = 4000


@dataclass
class Endpoint:
    """Where one agent's calls go: its own client and the settings that go with it, so an agent can
    run on an API while the rest of the committee runs locally."""

    client: OpenAI
    settings: dict = field(default_factory=dict)


def endpoint(choice: ModelChoice, retries: int) -> Endpoint:
    """A client and settings for a configured model: APIs get room for reasoning tokens, and Gemini
    gets no seed (its endpoint rejects the field)."""
    settings = {"model": choice.model, "max_retries": retries}
    if choice.provider != "ollama":
        settings["max_tokens"] = API_MAX_TOKENS
    if choice.provider == "gemini":
        settings["seed"] = None
    return Endpoint(make_client(choice.provider), settings)


def grounding_endpoint() -> Endpoint:
    return endpoint(GROUNDING_JUDGE, GROUNDING_JUDGE_RETRIES)


def devils_advocate_endpoint() -> Endpoint:
    return endpoint(DEVILS_ADVOCATE, DEVILS_ADVOCATE_RETRIES)
