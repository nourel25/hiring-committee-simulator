"""Tests for the PDF export: read the PDF back with pypdf and check the CV's words survived."""

from io import BytesIO

import pytest
from pypdf import PdfReader

from hcs.export import cv_to_pdf, find_font, line_kind

CV = """Backend Engineer, shop (2020-present)
- Cut checkout latency 5.2× (1.8s → 0.35s)
- Built the payments API in Python

Education
B.Sc. Computer Science (2016-2020)"""


def text_of(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("Backend Engineer, shop (2020-present)", "role"),
        ("- Built the payments API", "bullet"),
        ("Education", "heading"),
        ("Infra & Testing", "heading"),
        ("Python Backend Software Engineer with hands-on experience", "text"),
        ("Doctors Hub Application /github", "text"),
    ],
)
def test_each_line_is_set_by_its_kind(line, kind):
    assert line_kind(line) == kind


def test_without_a_unicode_font_every_line_survives_with_plain_symbols():
    text = text_of(cv_to_pdf(CV, fonts=[]))

    assert "Backend Engineer, shop (2020-present)" in text
    assert "5.2x (1.8s -> 0.35s)" in text
    assert "Built the payments API in Python" in text
    assert "Education" in text


@pytest.mark.skipif(find_font() is None, reason="no Unicode font on this machine")
def test_with_a_unicode_font_symbols_are_kept():
    text = text_of(cv_to_pdf(CV))

    assert "5.2×" in text
    assert "→" in text


def test_the_header_goes_above_the_cv():
    text = text_of(cv_to_pdf(CV, header="Jane Doe\njane@example.com", fonts=[]))

    assert text.index("Jane Doe") < text.index("jane@example.com") < text.index("Backend Engineer")


def test_a_long_cv_flows_onto_more_pages():
    long_cv = "\n".join(f"- Bullet number {n} about the work" for n in range(200))

    assert len(PdfReader(BytesIO(cv_to_pdf(long_cv, fonts=[]))).pages) > 1
