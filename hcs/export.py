"""Turn the final plain-text CV back into a PDF: dated role lines and section headings in bold,
bullets indented. Text only, no layout from the uploaded file: the committee never saw it."""

import re
from pathlib import Path

from fpdf import FPDF

from hcs.screening import PERIOD

# A Unicode font from the system, so symbols such as × and → survive; the first pair found wins.
FONTS = [
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
    (Path("/Library/Fonts/Arial Unicode.ttf"), Path("/Library/Fonts/Arial Unicode.ttf")),
]
# Without one, the built-in Helvetica only knows Latin-1, so common symbols get plain stand-ins.
ASCII = {"→": "->", "×": "x", "—": "-", "–": "-", "•": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}
HEADING = re.compile(r"[A-Z][A-Za-z&]*( [A-Z&][A-Za-z&]*){0,2}")

SIZE = 10.5
LINE = 5.2
INDENT = 4


def find_font(candidates: list[tuple[Path, Path]] = FONTS) -> tuple[Path, Path] | None:
    return next(((r, b) for r, b in candidates if r.exists() and b.exists()), None)


def line_kind(line: str) -> str:
    """role, heading, bullet or text: how one CV line is set."""
    if PERIOD.search(line):
        return "role"
    if line.startswith("- "):
        return "bullet"
    if HEADING.fullmatch(line):
        return "heading"
    return "text"


def cv_to_pdf(cv: str, header: str = "", fonts: list[tuple[Path, Path]] = FONTS) -> bytes:
    """The CV as an A4 PDF. `header` (name, contact line) goes on top: parsing removed them from
    the CV, so the user adds them back here, on this machine only. `fonts=[]` forces Helvetica."""
    font = find_font(fonts)
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    if font:
        pdf.add_font("cv", "", str(font[0]))
        pdf.add_font("cv", "B", str(font[1]))
        family = "cv"
    else:
        family = "helvetica"

    def clean(text: str) -> str:
        if font:
            return text
        for symbol, plain in ASCII.items():
            text = text.replace(symbol, plain)
        return text.encode("latin-1", "replace").decode("latin-1")

    def write(text: str, style: str = "", size: float = SIZE, indent: float = 0) -> None:
        pdf.set_font(family, style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(pdf.epw - indent, LINE, clean(text), new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    header_lines = [line.strip() for line in header.splitlines() if line.strip()]
    for number, line in enumerate(header_lines):
        write(line, "B" if number == 0 else "", 15 if number == 0 else SIZE)
    if header_lines:
        pdf.ln(3)

    for raw in cv.splitlines():
        line = raw.strip()
        if not line:
            pdf.ln(LINE / 2)
            continue
        kind = line_kind(line)
        if kind == "heading":
            pdf.ln(2)
            write(line, "B", SIZE + 1.5)
        elif kind == "role":
            pdf.ln(1)
            write(line, "B")
        elif kind == "bullet":
            write("•  " + line[2:] if font else "-  " + line[2:], indent=INDENT)
        else:
            write(line)
    return bytes(pdf.output())
