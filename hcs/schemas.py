from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


class Check(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(description="The check's id, copied exactly from the input, e.g. 'TL1'.")
    check: str = Field(description="The check's text, copied exactly from the input.")
    evidence: str | None = Field(
        max_length=200,
        description="An exact quote from the CV relevant to this check, or null if the CV has none.",
    )
    passed: bool = Field(description="True only if the evidence clearly proves the check.")


# Not filled by the model: code computes the decision from the score bands.
Decision = Literal["strong_no", "lean_no", "lean_hire", "strong_hire"]


class Review(BaseModel):
    checks: list[Check] = Field(
        min_length=1,
        description="One entry per check, in the order given.",
    )


def review_model_for(ids: list[str]) -> type[Review]:
    """A Review whose check ids can only be the given ids.

    The schema then lists the ids as an enum, so Ollama's grammar cannot write any other id,
    and validation rejects one if it appears anyway.
    """
    check_model = create_model(
        "Check",
        __base__=Check,
        id=(Literal[tuple(ids)], Field(description=Check.model_fields["id"].description)),
    )
    return create_model(
        "Review",
        __base__=Review,
        checks=(
            list[check_model],
            Field(min_length=1, description=Review.model_fields["checks"].description),
        ),
    )


class ChecklistItem(BaseModel):
    """One requirement a reviewer checks. Importance is used by code only, never shown to the model."""

    id: str = Field(description="Short unique id, e.g. 'TL1'.")
    text: str = Field(description="One observable requirement, phrased positively.")
    importance: Literal["must", "nice"] = Field(
        description="'must' = required by the posting; 'nice' = a bonus."
    )
