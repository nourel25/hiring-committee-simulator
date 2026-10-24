"""Turn the final plain-text CV back into a PDF, in a one-column template recruiters and ATS
parsers read well (the style of the widely used "Jake's Resume"): the name centred with the
contact line under it, section headings in capitals over a rule, each role as bold title, italic
employer and dates on the right, bullets with a hanging indent, skills as bold label + values.
Text only, no layout from the uploaded file: the committee never saw it."""

import re
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF

from hcs.screening import PERIOD, is_heading

# A Unicode font from the system (regular, bold, italic), so symbols such as × and → survive;
# the first complete set found wins.
FONTS = [
    tuple(Path(f"C:/Windows/Fonts/{name}.ttf") for name in ("calibri", "calibrib", "calibrii")),
    tuple(Path(f"C:/Windows/Fonts/{name}.ttf") for name in ("arial", "arialbd", "ariali")),
    tuple(
        Path(f"/usr/share/fonts/truetype/dejavu/{name}.ttf")
        for name in ("DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique")
    ),
]
# Without one, the built-in Helvetica only knows Latin-1, so common symbols get plain stand-ins.
ASCII = {"→": "->", "×": "x", "—": "-", "–": "-", "•": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}
SKILL = re.compile(r"([A-Z][\w &/+-]{0,30}?)\s*:\s*(\S.*)")
EMPLOYER = re.compile(r"\s*(?:—|–| - |, )\s*")

INK = (20, 20, 20)
ACCENT = (24, 52, 92)
RULE = (150, 150, 150)
SIZE = 10
LINE = 4.6
HANG = 4.5


@dataclass
class Role:
    title: str
    employer: str
    dates: str


def find_font(candidates: list[tuple[Path, ...]] = FONTS) -> tuple[Path, ...] | None:
    return next((paths for paths in candidates if all(path.exists() for path in paths)), None)


def line_kind(line: str, next_line: str = "") -> str:
    """How one CV line is set: role, bullet, heading, skill, entry (a project name over its
    bullets) or text."""
    if PERIOD.search(line):
        return "role"
    if line.startswith("- "):
        return "bullet"
    # A section word is needed, so a headline such as "Software Engineer" is not a section.
    if is_heading(line):
        return "heading"
    if SKILL.fullmatch(line):
        return "skill"
    if next_line.startswith("- ") and len(line) <= 80:
        return "entry"
    return "text"


def split_role(line: str) -> Role:
    """'Software Engineer—Northwind, Lisbon (2026-present)' -> title, employer, '2026 – Present'."""
    match = PERIOD.search(line)
    start_month, start, end_month, end = match.groups()
    begin = f"{start_month.title()} {start}" if start_month else start
    finish = "Present" if end.lower() == "present" else end
    if end_month and finish != "Present":
        finish = f"{end_month.title()} {finish}"
    dates = f"{begin} – {finish}"
    rest = (line[: match.start()] + line[match.end() :]).strip(" ,—–-")
    title, employer = (EMPLOYER.split(rest, maxsplit=1) + [""])[:2]
    return Role(title.strip(), employer.strip(), dates)


def safe_markdown(text: str) -> str:
    """fpdf2's markdown reads **, __ and -- as styles; CV text should never switch them on."""
    return text.replace("**", "*").replace("__", "_").replace("--", "–")


def contact_address(item: str) -> str | None:
    """Where a contact item should link: an email, a phone number or a web address."""
    if re.fullmatch(r"\S+@\S+\.\w+", item):
        return f"mailto:{item}"
    digits = re.sub(r"[^\d+]", "", item)
    if re.fullmatch(r"\+?[\d\s().-]+", item) and sum(c.isdigit() for c in digits) >= 9:
        return f"tel:{digits}"
    if re.match(r"(?:https?://)?(?:www\.)?[\w-]+\.[a-z]{2,}(?:/\S*)?$", item):
        return item if item.startswith("http") else f"https://{item}"
    return None


def link(text: str, address: str) -> str:
    """A markdown link fpdf2 draws as clickable text; the text is escaped, the address is not."""
    return f"[{safe_markdown(text).replace('[', '(').replace(']', ')')}]({address})"


def contact_line(lines: list[str]) -> str:
    """The contact items joined under the name, each one clickable when it is an address."""
    parts = []
    for item in lines:
        address = contact_address(item)
        parts.append(link(item, address) if address else safe_markdown(item))
    return "  |  ".join(parts)


def with_links(text: str, pending: list[tuple[str, str]]) -> tuple[str, bool]:
    """The line as markdown with the first pending link whose text it contains made clickable;
    that link is used up, so two "/github" labels get the two repositories in order."""
    for index, (anchor, address) in enumerate(pending):
        if anchor in text:
            before, _, after = text.partition(anchor)
            del pending[index]
            rest, _ = with_links(after, pending)
            return safe_markdown(before) + link(anchor, address) + rest, True
    return safe_markdown(text), False


def cv_to_pdf(
    cv: str,
    header: str = "",
    fonts: list[tuple[Path, ...]] = FONTS,
    links: list[tuple[str, str]] = (),
) -> bytes:
    """The CV as an A4 PDF. `header` is the name, then contact lines: parsing removed them from
    the CV, so the user adds them back here, on this machine only. `links` are (text, address)
    pairs from the uploaded file, made clickable where that text appears. `fonts=[]` forces
    Helvetica."""
    pending = list(links)
    font = find_font(fonts)
    pdf = FPDF(format="A4")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=14)
    if font:
        for style, path in zip(("", "B", "I"), font, strict=True):
            pdf.add_font("cv", style, str(path))
        family = "cv"
    else:
        family = "helvetica"

    def clean(text: str) -> str:
        if font:
            return text
        for symbol, plain in ASCII.items():
            text = text.replace(symbol, plain)
        return text.encode("latin-1", "replace").decode("latin-1")

    def use(style: str = "", size: float = SIZE, color: tuple[int, int, int] = INK) -> None:
        pdf.set_font(family, style, size)
        pdf.set_text_color(*color)

    def paragraph(text: str, indent: float = 0, markdown: bool = False) -> None:
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(
            pdf.epw - indent, LINE, clean(text), new_x="LMARGIN", new_y="NEXT", markdown=markdown
        )

    def heading(text: str) -> None:
        pdf.ln(2.5)
        use("B", SIZE + 1.5, ACCENT)
        pdf.cell(0, LINE + 1, clean(text.upper()), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
        pdf.ln(1.5)

    def role(line: str) -> None:
        parts = split_role(line)
        pdf.ln(1)
        use()
        date_width = pdf.get_string_width(clean(parts.dates)) + 2
        top = pdf.get_y()
        pdf.set_xy(pdf.l_margin + pdf.epw - date_width, top)
        pdf.cell(date_width, LINE, clean(parts.dates), align="R")
        pdf.set_xy(pdf.l_margin, top)
        text = f"**{safe_markdown(parts.title)}**"
        if parts.employer:
            text += f"  __{safe_markdown(parts.employer)}__"
        pdf.multi_cell(
            pdf.epw - date_width - 3,
            LINE,
            clean(text),
            new_x="LMARGIN",
            new_y="NEXT",
            markdown=True,
        )

    def bullet(text: str) -> None:
        use()
        top = pdf.get_y()
        pdf.set_xy(pdf.l_margin + 1, top)
        pdf.cell(HANG, LINE, "•" if font else "-")
        pdf.set_xy(pdf.l_margin + 1 + HANG, top)
        marked, linked = with_links(text, pending)
        pdf.multi_cell(
            pdf.epw - 1 - HANG,
            LINE,
            clean(marked if linked else text),
            new_x="LMARGIN",
            new_y="NEXT",
            markdown=linked,
        )

    def linked_paragraph(text: str) -> None:
        marked, linked = with_links(text, pending)
        paragraph(marked if linked else text, markdown=linked)

    pdf.add_page()
    header_lines = [line.strip() for line in header.splitlines() if line.strip()]
    if header_lines:
        use("B", 20, ACCENT)
        pdf.cell(0, 9, clean(header_lines[0]), align="C", new_x="LMARGIN", new_y="NEXT")
        if header_lines[1:]:
            use(size=SIZE - 0.5)
            pdf.multi_cell(
                0,
                LINE,
                clean(contact_line(header_lines[1:])),
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
                markdown=True,
            )
        pdf.ln(3)

    lines = [line.strip() for line in cv.splitlines()]
    for index, line in enumerate(lines):
        if not line:
            continue
        following = next((later for later in lines[index + 1 :] if later), "")
        kind = line_kind(line, following)
        if kind == "heading":
            heading(line)
        elif kind == "role":
            role(line)
        elif kind == "bullet":
            bullet(line[2:])
        elif kind == "skill":
            label, values = SKILL.fullmatch(line).groups()
            use()
            paragraph(f"**{safe_markdown(label)}:** {safe_markdown(values)}", markdown=True)
        elif kind == "entry":
            pdf.ln(1)
            use("B")
            linked_paragraph(line)
        else:
            use()
            linked_paragraph(line)
    return bytes(pdf.output())
