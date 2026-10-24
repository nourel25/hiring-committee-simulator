"""Tests for the PDF export: read the PDF back with pypdf and check the CV's words survived."""

from io import BytesIO

import pytest
from pypdf import PdfReader

from hcs.export import Role, cv_to_pdf, find_font, line_kind, safe_markdown, split_role

CV = """Backend Engineer, shop (2020-present)
- Cut checkout latency 5.2× (1.8s → 0.35s)
- Built the payments API in Python

Technologies
Languages:Python, SQL

Education
B.Sc. Computer Science (2016-2020)"""


def text_of(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)


@pytest.mark.parametrize(
    ("line", "next_line", "kind"),
    [
        ("Backend Engineer, shop (2020-present)", "", "role"),
        ("- Built the payments API", "", "bullet"),
        ("Education", "", "heading"),
        ("Infra & Testing", "", "heading"),
        ("Languages:Python, JavaScript, C++", "", "skill"),
        ("Infra & Testing:Docker, Git, GitHub Actions", "", "skill"),
        ("Doctors Hub Application /github", "- Stack: Python, Django", "entry"),
        ("Doctors Hub Application /github", "Education", "text"),
        ("Python Backend Software Engineer with hands-on experience", "", "text"),
    ],
)
def test_each_line_is_set_by_its_kind(line, next_line, kind):
    assert line_kind(line, next_line) == kind


@pytest.mark.parametrize(
    ("line", "role"),
    [
        (
            "Backend Engineer, shop (2020-present)",
            Role("Backend Engineer", "shop", "2020 – Present"),
        ),
        (
            "Software Engineer—Northwind, Lisbon (Remote) (2026-present)",
            Role("Software Engineer", "Northwind, Lisbon (Remote)", "2026 – Present"),
        ),
        ("Data Analyst (2019-2021)", Role("Data Analyst", "", "2019 – 2021")),
    ],
)
def test_a_role_line_splits_into_title_employer_and_dates(line, role):
    assert split_role(line) == role


def test_cv_text_never_switches_on_markdown_styles():
    assert safe_markdown("__init__ and **kwargs -- fast") == "_init_ and *kwargs – fast"


def test_without_a_unicode_font_every_line_survives_with_plain_symbols():
    text = text_of(cv_to_pdf(CV, fonts=[]))

    assert "Backend Engineer" in text
    assert "2020 - Present" in text
    assert "5.2x (1.8s -> 0.35s)" in text
    assert "Built the payments API in Python" in text
    assert "Languages: Python, SQL" in text
    assert "EDUCATION" in text


@pytest.mark.skipif(find_font() is None, reason="no Unicode font on this machine")
def test_with_a_unicode_font_symbols_are_kept():
    text = text_of(cv_to_pdf(CV))

    assert "5.2×" in text
    assert "→" in text


def test_the_name_goes_on_top_and_the_contact_lines_join_under_it():
    text = text_of(cv_to_pdf(CV, header="Jane Doe\njane@example.com\nBerlin", fonts=[]))

    assert "jane@example.com  |  Berlin" in text
    assert text.index("Jane Doe") < text.index("jane@example.com") < text.index("Backend Engineer")


def test_a_long_cv_flows_onto_more_pages():
    long_cv = "\n".join(f"- Bullet number {n} about the work" for n in range(200))

    assert len(PdfReader(BytesIO(cv_to_pdf(long_cv, fonts=[]))).pages) > 1
