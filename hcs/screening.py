"""The recruiter's screen done by code: title, dated years, names and line length in the top third.

Each result is returned as a Check, the same shape a reviewer returns, so scoring treats both alike.
For these checks, evidence holds what the code found, not a quote; the evidence policy and the
truth check are for model checks only.
"""

import math
import re

from hcs.cases import RecruiterScreen
from hcs.schemas import Check

# A line longer than this is a paragraph, not a bullet a recruiter can read at a glance. The longest
# bullet in the 14 labelled CVs is about 130 characters; a paragraph is several hundred.
MAX_LINE = 200
# "(2021-present)", "(2018 – 2021)" or, from an uploaded CV, "(Mar 2026-present)": dated lines.
PERIOD = re.compile(
    r"\((?:([a-z]{3}) )?(\d{4})\s*[-–]\s*(?:([a-z]{3}) )?(\d{4}|present)\)", re.IGNORECASE
)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
# A section heading: a short line of words only ("Work Experience", "Skills & Tools") with a word
# CVs use for sections, so a bullet such as "Built Django sites" or a headline is never one.
HEADING = re.compile(r"[A-Za-z][A-Za-z&/]*(?: [A-Za-z&/]+){0,3}")
SECTION_WORDS = re.compile(
    r"\b(?:experience|employment|work|career|history|education|academic|skills?|technolog\w*"
    r"|technical|projects?|certificat\w*|courses?|training|summary|profile|about|languages?"
    r"|awards?|publications?|volunteer\w*|interests|achievements|qualifications?|tools)\b",
    re.IGNORECASE,
)
EDUCATION = re.compile(r"educat|academic|certific|qualification|course|training", re.IGNORECASE)
# A dated line that is a degree, for CVs without an Education heading.
DEGREE = re.compile(
    r"\b(?:universit\w*|faculty|college|b\.?sc|bachelor\w*|master\w*|m\.?sc|ph\.?d|diploma)\b",
    re.IGNORECASE,
)


def is_heading(line: str) -> bool:
    line = line.strip()
    return bool(HEADING.fullmatch(line) and SECTION_WORDS.search(line))


def role_lines(cv: str) -> list[str]:
    """The dated lines that are jobs: not degrees, not lines under an Education-like heading.
    The 30 labelled CVs have no headings and no dated degrees; uploaded CVs have both."""
    roles, in_education = [], False
    for line in (line.strip() for line in cv.splitlines()):
        if is_heading(line):
            in_education = bool(EDUCATION.search(line))
        elif PERIOD.search(line) and not in_education and not DEGREE.search(line):
            roles.append(line)
    return roles


def top_third(cv: str) -> str:
    """The first third of the CV's non-blank lines, rounded up: what a recruiter reads first."""
    lines = [line.strip() for line in cv.splitlines() if line.strip()]
    return "\n".join(lines[: math.ceil(len(lines) / 3)])


def recruiter_view(cv: str) -> str:
    """The top third without its dated role lines: what the recruiter model sees.

    Title and years are R1's and R2's job. With recruiter checklist v3 the model quoted the title line
    as evidence for C1 in 11 of 19 cases, so code removes those lines and the model cannot pick them.
    """
    return "\n".join(line for line in top_third(cv).splitlines() if not PERIOD.search(line))


def month_index(month: str, year: int, default: int) -> int:
    """Months since year 0; a missing month is `default` (1 = January)."""
    number = MONTHS.index(month.casefold()[:3]) + 1 if month else default
    return year * 12 + number - 1


def periods(cv: str, this_year: int, this_month: int = 12) -> list[tuple[int, int]]:
    """Every job as (start, end) in months. A year without a month runs from January, so
    "(2018-2021)" is 36 months; 'present' is this_month of this year (default December: the year
    counts in full, as before months were read). A reversed range is skipped."""
    found = []
    for line in role_lines(cv):
        start_month, start, end_month, end = PERIOD.search(line).groups()
        first = month_index(start_month, int(start), 1)
        if end.casefold() == "present":
            last = month_index("", this_year, this_month)
        else:
            last = month_index(end_month, int(end), 1)
        if first <= last:
            found.append((first, last))
    return found


def years_of_experience(cv: str, this_year: int, this_month: int = 12) -> int:
    """Whole years covered by the jobs, counting overlapping jobs once."""
    merged: list[list[int]] = []
    for start, end in sorted(periods(cv, this_year, this_month)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) // 12


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
