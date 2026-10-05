from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
