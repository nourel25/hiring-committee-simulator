"""Rules applied to a reviewer's reply after the model answers. Agent-independent."""

from hcs.schemas import Check, ChecklistItem, Decision, Objections, Review

# Characters models sometimes write instead of the plain ones in the CV.
LOOKALIKES = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote (curly apostrophe)
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
)


def has_evidence(check: Check) -> bool:
    """True only if the check carries a real quote, not None and not blank."""
    return check.evidence is not None and check.evidence.strip() != ""


def apply_evidence_policy(review: Review) -> int:
    """A pass without evidence counts as a fail. Returns how many checks were changed."""
    changed = 0
    for check in review.checks:
        if check.passed and not has_evidence(check):
            check.passed = False
            changed += 1
    return changed


def normalize(text: str) -> str:
    """Make text comparable: plain characters, single spaces, no capitals."""
    return " ".join(text.translate(LOOKALIKES).split()).casefold()


def quote_in_cv(quote: str, cv_text: str) -> bool:
    """True if the quote appears in the CV once both are normalised."""
    return normalize(quote) in normalize(cv_text)


def apply_truth_check(review: Review, cv_text: str) -> tuple[int, list[str]]:
    """A pass whose quote is not in the CV counts as a fail. Returns (changed count, ids of checks whose quote was not found)."""
    changed = 0
    missing: list[str] = []
    for check in review.checks:
        if has_evidence(check) and not quote_in_cv(check.evidence, cv_text):
            missing.append(check.id)
            if check.passed:
                check.passed = False
                changed += 1
    return changed, missing


# Points per check: a must-have counts three times as much as a nice-to-have.
WEIGHTS = {"must": 3, "nice": 1}


def results_by_id(review: Review) -> dict[str, bool]:
    """Map each check id in the reply to whether it passed."""
    return {check.id: check.passed for check in review.checks}


def compute_score(review: Review, checklist: list[ChecklistItem]) -> float:
    """Weighted share of checklist items passed, on a 0-10 scale. Missing items count as failed."""
    results = results_by_id(review)
    total = sum(WEIGHTS[item.importance] for item in checklist)
    passed = sum(WEIGHTS[item.importance] for item in checklist if results.get(item.id, False))
    return passed / total * 10


def failed_must_haves(review: Review, checklist: list[ChecklistItem]) -> list[str]:
    """Ids of must-have items that did not pass, including any the reply left out."""
    results = results_by_id(review)
    return [
        item.id
        for item in checklist
        if item.importance == "must" and not results.get(item.id, False)
    ]


def score_to_decision(score: float) -> Decision:
    """Map a 0-10 score to a decision using the design doc's bands."""
    if score < 4:
        return "strong_no"
    if score < 6:
        return "lean_no"
    if score < 8:
        return "lean_hire"
    return "strong_hire"


def decide(score: float, failed: list[str]) -> Decision:
    """Score bands, but any failed must-have caps the decision at lean_no."""
    decision = score_to_decision(score)
    if failed and decision in ("lean_hire", "strong_hire"):
        return "lean_no"
    return decision


def ungrounded_objections(objections: Objections, posting_text: str, cv_text: str) -> list[int]:
    """Positions of objections that cite nothing, or quote a line missing from the posting or the CV.

    The committee should ignore these: an objection nobody can trace is an opinion, not evidence.
    """
    return [
        position
        for position, objection in enumerate(objections.objections)
        if (objection.requirement is None and objection.evidence is None)
        or (
            objection.requirement is not None
            and not quote_in_cv(objection.requirement, posting_text)
        )
        or (objection.evidence is not None and not quote_in_cv(objection.evidence, cv_text))
    ]
