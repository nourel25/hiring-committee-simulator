from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)


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
    # Used by code only (the devil's advocate's evaluation), never shown to a model.
    sources: list[str] = Field(default=[], description="The posting lines this check comes from.")


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


# The devil's advocate (v2): the model rates every requirement line; code picks the objections.
Shown = Literal["fully", "partly", "not"]
Severity = Literal["low", "medium", "high"]


def null_if_written_out(value: object) -> object:
    """A model sometimes writes null as the word "null"; treat that (and "None" or "") as missing."""
    if isinstance(value, str) and value.strip().casefold() in {"", "null", "none"}:
        return None
    return value


def quote_list(value: object) -> object:
    """Evidence as a list of quotes: null becomes [], a single quote becomes [quote], and quotes
    written as "null" are dropped. Also reads records logged before evidence was a list (v1, v2)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        return [item for item in value if null_if_written_out(item) is not None]
    return value


Quote = Annotated[str, Field(max_length=200)]


class LineRating(BaseModel):
    """How well the CV shows one requirement line. The quotes come before the verdict."""

    id: str = Field(description="The requirement's id, copied exactly from the input, e.g. 'D1'.")
    # v3 (Step 4.3f): a list, because a requirement with several parts (deployment, monitoring and
    # on-call) is shown on several CV lines; v2 allowed one quote and the model glued lines together.
    evidence: list[Quote] = Field(
        max_length=3,
        description="Up to three exact quotes from the CV, each from one line, most relevant first.",
    )
    reason: str = Field(
        max_length=300,
        description="One sentence on what the CV shows or fails to show for this requirement.",
    )
    shown: Shown = Field(description="'fully', 'partly' or 'not', as the instructions define.")

    _evidence_list = field_validator("evidence", mode="before")(quote_list)


class RequirementRatings(BaseModel):
    ratings: list[LineRating] = Field(
        min_length=1, description="One rating per requirement, in the order given."
    )


def ratings_model_for(ids: list[str]) -> type[RequirementRatings]:
    """RequirementRatings whose ids can only be the given ids, each rated exactly once.

    The enum stops the grammar writing other ids; the validator sends a reply that skips or repeats
    a requirement back for repair, so no requirement goes unrated.
    """
    rating_model = create_model(
        "LineRating",
        __base__=LineRating,
        id=(Literal[tuple(ids)], Field(description=LineRating.model_fields["id"].description)),
    )

    def every_line_once(ratings: RequirementRatings) -> RequirementRatings:
        got = [rating.id for rating in ratings.ratings]
        if sorted(got) != sorted(ids):
            raise ValueError(f"rate each of {', '.join(ids)} exactly once; got {', '.join(got)}")
        return ratings

    return create_model(
        "RequirementRatings",
        __base__=RequirementRatings,
        __validators__={"every_line_once": model_validator(mode="after")(every_line_once)},
        ratings=(
            list[rating_model],
            Field(min_length=1, description=RequirementRatings.model_fields["ratings"].description),
        ),
    )


class Objection(BaseModel):
    """One reason not to hire, built by code from a rating: the requirement line, the CV quotes,
    the model's reason, and a severity code sets from the rating."""

    requirement: str | None = Field(max_length=200)
    evidence: list[Quote] = []
    reason: str = Field(max_length=300)
    severity: Severity

    _evidence_list = field_validator("evidence", mode="before")(quote_list)


# The grounding judge (Step 5.2): reads what the rules can't, whether a claim's meaning is supported.
class ClaimJudgement(BaseModel):
    """One claim judged against the facts it cites. The unsupported part comes before the verdict."""

    id: str = Field(description="The claim's id, copied exactly from the input, e.g. 'G1'.")
    unsupported: str | None = Field(
        max_length=300,
        description="The words of the claim that the cited facts do not say, or null if none.",
    )
    supported: bool = Field(
        description="True only if the cited facts say everything the claim says."
    )

    _unsupported_null = field_validator("unsupported", mode="before")(null_if_written_out)


class GroundingJudgement(BaseModel):
    judgements: list[ClaimJudgement] = Field(
        min_length=1, description="One judgement per claim, in the order given."
    )


def judgement_model_for(ids: list[str]) -> type[GroundingJudgement]:
    """GroundingJudgement whose ids can only be the given ids, each judged exactly once."""
    judgement_model = create_model(
        "ClaimJudgement",
        __base__=ClaimJudgement,
        id=(Literal[tuple(ids)], Field(description=ClaimJudgement.model_fields["id"].description)),
    )

    def every_claim_once(judgement: GroundingJudgement) -> GroundingJudgement:
        got = [item.id for item in judgement.judgements]
        if sorted(got) != sorted(ids):
            raise ValueError(f"judge each of {', '.join(ids)} exactly once; got {', '.join(got)}")
        return judgement

    return create_model(
        "GroundingJudgement",
        __base__=GroundingJudgement,
        __validators__={"every_claim_once": model_validator(mode="after")(every_claim_once)},
        judgements=(
            list[judgement_model],
            Field(
                min_length=1, description=GroundingJudgement.model_fields["judgements"].description
            ),
        ),
    )


# The refiner (Step 6): a short summary at the top of the CV, each line citing the facts it uses.
MAX_SUMMARY_LINES = 3
# Wider than the 20 words the prompt asks for: the grammar cuts a string at its limit, mid-word.
MAX_SUMMARY_LINE = 200


class SummaryLine(BaseModel):
    """One line of the new summary. The facts come first, so the model picks them before writing."""

    facts: list[str] = Field(min_length=1, description="Ids of the facts this line uses.")
    text: Annotated[
        str, Field(max_length=MAX_SUMMARY_LINE, description="The line, in plain words.")
    ]


class Refinement(BaseModel):
    summary: list[SummaryLine] = Field(
        min_length=1, max_length=MAX_SUMMARY_LINES, description="One to three summary lines."
    )


def refinement_model_for(fact_ids: list[str]) -> type[Refinement]:
    """A Refinement whose lines can only cite the given fact ids."""
    line_model = create_model(
        "SummaryLine",
        __base__=SummaryLine,
        # v2: exactly one fact per line, so a line can't move a detail from one fact to another.
        facts=(
            list[Literal[tuple(fact_ids)]],
            Field(
                min_length=1,
                max_length=1,
                description=SummaryLine.model_fields["facts"].description,
            ),
        ),
    )
    return create_model(
        "Refinement",
        __base__=Refinement,
        summary=(
            list[line_model],
            Field(
                min_length=1,
                max_length=MAX_SUMMARY_LINES,
                description=Refinement.model_fields["summary"].description,
            ),
        ),
    )
