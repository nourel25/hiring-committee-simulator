"""Tests for the recruiter's code screen. No model calls; the year is fixed, never today's."""

import pytest

from hcs.cases import RecruiterScreen, load_case, load_cases, load_posting
from hcs.screening import (
    current_title,
    keywords_found,
    recruiter_view,
    role_lines,
    screening_checks,
    top_third,
    years_of_experience,
)

THIS_YEAR = 2026


@pytest.mark.parametrize(
    "line_count, kept",
    [(9, 3), (4, 2), (3, 1), (1, 1)],
)
def test_top_third_rounds_up_and_ignores_blank_lines(line_count, kept):
    cv = "\n\n".join(f"line {number}" for number in range(1, line_count + 1))
    assert top_third(cv).splitlines() == [f"line {number}" for number in range(1, kept + 1)]


def test_the_recruiter_view_drops_dated_role_lines_but_keeps_other_numbers():
    cv = (
        "Data Engineer, shop (2021-present)\nAWS Certified (2023)\n- Built pipelines\n- a\n- b\n- c"
    )

    assert recruiter_view(cv) == "AWS Certified (2023)"


@pytest.mark.parametrize(
    "cv, years",
    [
        ("Engineer (2021-present)", 5),
        ("A (2018-2022)\nB (2020-2023)", 5),
        ("A (2018-2021)\nB (2021-present)", 8),
        ("A (2015-2017)\nB (2020-2023)", 5),
        ("A (2018 – 2021)", 3),
        ("A (2022-2019)\nB (2020-2023)", 3),
        ("No dates at all", 0),
    ],
    ids=["present", "overlap", "back to back", "gap", "en dash", "reversed skipped", "none"],
)
def test_years_count_overlapping_roles_once(cv, years):
    assert years_of_experience(cv, THIS_YEAR) == years


@pytest.mark.parametrize(
    "cv, years",
    [
        ("Engineer (Mar 2026-present)", 0),
        ("A (Dec 2024-Jan 2026)", 1),
        ("A (Mar 2023-Feb 2024)", 0),
        ("A (Jan 2020-Jun 2023)\nB (Mar 2023-present)", 6),
    ],
    ids=["months this year", "13 months", "11 months", "overlap with months"],
)
def test_months_are_counted_when_the_cv_gives_them(cv, years):
    assert years_of_experience(cv, THIS_YEAR) == years


def test_a_degree_is_not_work_experience():
    cv = "\n".join(
        [
            "Software Engineer, Northwind (Mar 2026-present)",
            "- Built the rights module",
            "Education",
            "B.Sc. Computer Science, Example University (2021-2025)",
        ]
    )

    assert years_of_experience(cv, THIS_YEAR) == 0
    assert role_lines(cv) == ["Software Engineer, Northwind (Mar 2026-present)"]


def test_a_dated_degree_without_an_education_heading_is_not_work_either():
    cv = "Engineer, shop (2024-present)\nB.Sc. Physics, Example University (2019-2023)"

    assert role_lines(cv) == ["Engineer, shop (2024-present)"]


def test_keywords_match_whole_words_and_ignore_case():
    text = "Built services on PostgreSQL in python, and C++ tools"
    assert keywords_found(text, ["SQL", "Python", "C++", "Go"]) == ["Python", "C++"]


@pytest.mark.parametrize(
    "first_line, title",
    [
        ("Backend Engineer, insurance platform (2021-present)", "Backend Engineer"),
        ("Data Engineer (2020-present)", "Data Engineer"),
        ("\n\n  Data Analyst, logistics company (2020-present)", "Data Analyst"),
        ("Software Engineer—Northwind, Lisbon (2026-present)", "Software Engineer"),
        ("Data Engineer | Acme (2020-present)", "Data Engineer"),
    ],
)
def test_the_current_title_is_the_first_line_before_a_comma_or_bracket(first_line, title):
    assert current_title(first_line + "\n- a bullet") == title


def test_a_strong_cv_passes_the_screen():
    case = load_case("data-good-1")
    screen = load_posting(case.checklist).recruiter

    checks = screening_checks(case.cv, screen, THIS_YEAR)

    assert [(check.id, check.passed) for check in checks] == [
        ("R1", True),
        ("R2", True),
        ("R3", True),
        ("R4", True),
    ]
    assert checks[1].evidence == "8 years from the dated roles"


def test_the_screen_is_shallow_on_purpose():
    # The poor data CV has the years and the words; only its title gives it away. Depth is the
    # tech lead's job, which is why the committee has more than one reviewer.
    case = load_case("data-poor-1")
    screen = load_posting(case.checklist).recruiter

    checks = screening_checks(case.cv, screen, THIS_YEAR)

    assert [(check.id, check.passed) for check in checks] == [
        ("R1", False),
        ("R2", True),
        ("R3", True),
        ("R4", True),
    ]
    assert checks[0].evidence == "Data Analyst"


def test_missing_keywords_are_named_in_the_evidence():
    screen = RecruiterScreen(titles=["engineer"], keywords=["Python", "SQL"], min_years=0)

    # Three lines, so the top third is the first line only.
    [_, _, keywords, _] = screening_checks(
        "Python Engineer (2024-present)\n- Built services\n- Wrote SQL", screen, THIS_YEAR
    )

    assert keywords.passed is False
    assert keywords.evidence == "found: Python; missing: SQL"


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.id)
def test_the_screen_agrees_with_every_recruiter_key(case):
    # The keys say what should happen; the code says what does. A difference is a code bug or
    # missing screen data (as "backend developer" was).
    screen = load_posting(case.checklist).recruiter
    got = {check.id: check.passed for check in screening_checks(case.cv, screen, THIS_YEAR)}

    assert got == {check_id: case.recruiter.expected[check_id].passed for check_id in got}


def test_a_paragraph_in_the_top_third_fails_the_line_length_check():
    case = load_case("present-2")
    screen = load_posting(case.checklist).recruiter

    lines = screening_checks(case.cv, screen, THIS_YEAR)[3]

    assert lines.id == "R4"
    assert lines.passed is False
    assert lines.evidence.startswith("longest line: ")
