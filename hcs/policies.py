"""Rules applied to a reviewer's reply after the model answers. Agent-independent."""

from hcs.schemas import Check, Decision, Review

# Characters models sometimes write instead of the plain ones in the CV.
LOOKALIKES = str.maketrans({
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote (curly apostrophe)
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
})


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


def apply_truth_check(review: Review, cv_text: str) -> int:
    """A pass whose quote is not in the CV counts as a fail. Returns how many checks were changed."""
    changed = 0
    for check in review.checks:
        if has_evidence(check) and not quote_in_cv(check.evidence, cv_text):
            print(f"Warning: {check.id} quote not found in the CV: {check.evidence!r}")
            if check.passed:
                check.passed = False
                changed += 1
    return changed


def compute_score(review: Review) -> float:
    """Share of checks passed, on a 0-10 scale (all checks weighted equally)."""
    passed = sum(1 for check in review.checks if check.passed)
    return passed / len(review.checks) * 10


def score_to_decision(score: float) -> Decision:
    """Map a 0-10 score to a decision using the design doc's bands."""
    if score < 4:
        return "strong_no"
    if score < 6:
        return "lean_no"
    if score < 8:
        return "lean_hire"
    return "strong_hire"
