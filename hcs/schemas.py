from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


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


class DraftCheck(BaseModel):
    """One check the job analyst drafts from a posting. Code gives it an id."""

    source: str = Field(
        max_length=300,
        description="The posting line this check comes from, copied exactly.",
    )
    check: str = Field(
        description="One thing the candidate has done, starting with 'Has', e.g. 'Has built …'."
    )
    does_not_count: str | None = Field(
        description=(
            "A clause naming the closest work that does not prove the check, "
            "e.g. 'scripts, notebooks and reports do not count', or null if there is none."
        )
    )
    importance: Literal["must", "nice"] = Field(
        description="'must' if the posting requires it; 'nice' if it is a bonus."
    )

    @field_validator("does_not_count")
    @classmethod
    def says_what_does_not_count(cls, value: str | None) -> str | None:
        """Code joins this clause to the check, so it must read as an exclusion, never a requirement."""
        if value is None:
            return None
        clause = value.strip().rstrip(".")
        if not clause.casefold().endswith(("does not count", "do not count")):
            raise ValueError(
                "does_not_count must end in 'does not count' or 'do not count', "
                "e.g. 'scripts and reports do not count'"
            )
        return clause


class DroppedLine(BaseModel):
    """A posting line that no CV can prove, left out on purpose."""

    source: str = Field(max_length=300, description="The posting line, copied exactly.")
    reason: str = Field(description="Why a CV cannot prove it, e.g. 'a personality trait'.")


class JobAnalysis(BaseModel):
    """What the job analyst returns: every posting line ends up in checks, lookups or dropped."""

    checks: list[DraftCheck] = Field(
        min_length=1,
        description="Checks a reviewer can judge from a CV, one per thing to prove.",
    )
    lookups: list[str] = Field(
        description="Named tools, year counts and industries, for code to look up, e.g. 'PostgreSQL'."
    )
    dropped: list[DroppedLine] = Field(
        description="Posting lines that are neither a check nor a lookup."
    )
