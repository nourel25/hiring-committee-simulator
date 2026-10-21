"""The recruiter's screen done by code: title, dated years, names and line length in the top third.

Each result is returned as a Check, the same shape a reviewer returns, so scoring treats both alike.
For these checks, evidence holds what the code found, not a quote; the evidence policy and the
truth check are for model checks only.
"""

import math
import re

from hcs.cases import RecruiterScreen
from hcs.schemas import Check

# "(2021-present)" or "(2018 – 2021)": the dated roles in a CV.
# A line longer than this is a paragraph, not a bullet a recruiter can read at a glance. The longest
# bullet in the 14 labelled CVs is about 130 characters; a paragraph is several hundred.
MAX_LINE = 200
PERIOD = re.compile(r"\((\d{4})\s*[-–]\s*(\d{4}|present)\)", re.IGNORECASE)


def top_third(cv: str) -> str:
    """The first third of the CV's non-blank lines, rounded up: what a recruiter reads first."""
    lines = [line.strip() for line in cv.splitlines() if line.strip()]
    return "\n".join(lines[: math.ceil(len(lines) / 3)])


def periods(cv: str, this_year: int) -> list[tuple[int, int]]:
    """Every dated role as (start, end) years; 'present' is this year; a reversed range is skipped."""
    found = []
    for start, end in PERIOD.findall(cv):
        end_year = this_year if end.casefold() == "present" else int(end)
        if int(start) <= end_year:
            found.append((int(start), end_year))
    return found


def years_of_experience(cv: str, this_year: int) -> int:
    """Years covered by the dated roles, counting overlapping roles once."""
    merged: list[list[int]] = []
    for start, end in sorted(periods(cv, this_year)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def keywords_found(text: str, keywords: list[str]) -> list[str]:
    """The keywords that appear as whole words, ignoring case: 'SQL' is not found in 'PostgreSQL'."""
    # Lookarounds instead of \b, so names that end in a symbol (C++, Node.js) still match.
    return [
        keyword
        for keyword in keywords
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, re.IGNORECASE)
    ]


def current_title(cv: str) -> str:
    """The first non-blank line up to its first comma or bracket: the most recent job title."""
    first = next((line.strip() for line in cv.splitlines() if line.strip()), "")
    return re.split(r"[,(]", first, maxsplit=1)[0].strip()


def screening_checks(cv: str, screen: RecruiterScreen, this_year: int) -> list[Check]:
    """The recruiter's four code checks, R1 to R4."""
    title = current_title(cv)
    years = years_of_experience(cv, this_year)
    found = keywords_found(top_third(cv), screen.keywords)
    missing = [keyword for keyword in screen.keywords if keyword not in found]
    longest = max((len(line) for line in top_third(cv).splitlines()), default=0)
    return [
        Check(
            id="R1",
            check="The current job title matches the role",
            evidence=title or None,
            passed=any(accepted.casefold() in title.casefold() for accepted in screen.titles),
        ),
        Check(
            id="R2",
            check=f"Has at least {screen.min_years} years of experience in dated roles",
            evidence=f"{years} years from the dated roles",
            passed=years >= screen.min_years,
        ),
        Check(
            id="R3",
            check="Names the posting's must-have tools in the top third of the CV",
            evidence=f"found: {', '.join(found) or 'none'}; missing: {', '.join(missing) or 'none'}",
            passed=not missing,
        ),
        # Was C3, judged by the model (Step 4.3b); no single quote can prove a property of the
        # whole text, so the model failed it on every CV.
        Check(
            id="R4",
            check="The top third is short lines that can each be read at a glance",
            evidence=f"longest line: {longest} characters (limit {MAX_LINE})",
            passed=longest <= MAX_LINE,
        ),
    ]
