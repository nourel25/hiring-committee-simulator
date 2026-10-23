"""Tests for reading uploaded CVs into the committee's plain format."""

import io

import pytest
from docx import Document

from hcs.parsing import normalise_cv, normalise_dates, text_from_upload
from hcs.profile import build_facts
from hcs.screening import years_of_experience

RAW = """Jane Example
jane@example.com | +44 7700 900123 | linkedin.com/in/jane
Backend Engineer, PayCo
Jan 2021 – Present
• Built the payouts API in Python, used by
  9,000 merchants
• Deploy my services myself
Junior Developer, Agency (2018 - 2021)
- Built Django sites
"""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Engineer, Shop Jan 2021 – Present", "Engineer, Shop (2021-present)"),
        ("Engineer, Shop (2018 - 2021)", "Engineer, Shop (2018-2021)"),
        ("Engineer, Shop 03/2019 — current", "Engineer, Shop (2019-present)"),
        ("Cut costs by 30% in 2023", "Cut costs by 30% in 2023"),
    ],
)
def test_date_ranges_are_written_the_way_the_checks_read_them(line, expected):
    assert normalise_dates(line) == expected


def test_a_pasted_cv_becomes_role_lines_and_bullets_without_identity():
    cv = normalise_cv(RAW)

    assert cv.splitlines() == [
        "Backend Engineer, PayCo (2021-present)",
        "- Built the payouts API in Python, used by 9,000 merchants",
        "- Deploy my services myself",
        "Junior Developer, Agency (2018-2021)",
        "- Built Django sites",
    ]
    assert years_of_experience(cv, 2026) == 8
    assert [fact.role for fact in build_facts(cv)][1] == "Backend Engineer, PayCo (2021-present)"


def test_a_word_file_is_read():
    document = Document()
    for line in RAW.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)

    assert normalise_cv(text_from_upload("cv.docx", buffer.getvalue())) == normalise_cv(RAW)


def minimal_pdf(lines: list[str]) -> bytes:
    """A one-page PDF with the lines as text, written by hand (no PDF writer needed)."""
    text = " ".join(f"({line}) Tj T*" for line in lines)
    stream = f"BT /F1 11 Tf 14 TL 50 750 Td {text} ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return out


def test_a_pdf_is_read():
    pdf = minimal_pdf(["Backend Engineer, PayCo 2021 - Present", "- Built the payouts API"])

    assert normalise_cv(text_from_upload("cv.pdf", pdf)).splitlines() == [
        "Backend Engineer, PayCo (2021-present)",
        "- Built the payouts API",
    ]


def test_runs_of_spaces_from_a_pdf_layout_become_one():
    assert normalise_cv("Backend Engineer, PayCo    2021 - Present") == (
        "Backend Engineer, PayCo (2021-present)"
    )
