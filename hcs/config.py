"""Which model plays which role, as decided by the Phase 0 experiment (Stage 3).

Change a role's model here, not in the agents. Evidence: the Phase 0 results in the README and the commit history.
"""

from dataclasses import dataclass

from hcs.llm import Provider


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

# The rubric these choices were measured on.
CHECKLIST = "backend_payments"
