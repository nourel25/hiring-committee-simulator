"""The profile builder: the knowledge base of facts every rewritten claim must point back to.

v1 is code, not a model (Step 5.1). The CVs here are already a list of discrete lines (a dated role,
then one bullet per piece of work), so each line becomes one fact, numbered F1.., tied to its role,
with its numbers stored as values. A model adds nothing until the input is free text (a PDF, a
GitHub page); then a model drafts facts and the user confirms them, as the design says.
"""

import re

from pydantic import BaseModel, Field

from hcs.screening import PERIOD

# "12,000", "1.5", "300", "40%": thousands separators are dropped so "12,000" and "12000" match.
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


class Fact(BaseModel):
    id: str
    text: str
    # The dated role line this fact sits under; None for lines above the first role.
    role: str | None = None
    metrics: list[str] = Field(default=[], description="Every number in the text, normalised.")
    # In the product the user confirms each fact before use; the labelled CVs are taken as confirmed.
    confirmed: bool = True


def numbers_in(text: str) -> list[str]:
    """Every number in the text, without thousands separators: '12,000 users' -> ['12000']."""
    found = []
    for raw in NUMBER.findall(text):
        # A comma followed by exactly three digits is a thousands separator; anything else stays.
        value = re.sub(r",(?=\d{3}(?:\D|$))", "", raw)
        found.append(value)
    return found


def build_facts(cv: str) -> list[Fact]:
    """One fact per non-blank CV line, bullets stripped, in CV order."""
    facts = []
    role = None
    for line in (line.strip() for line in cv.splitlines()):
        if not line:
            continue
        text = line.lstrip("-•* ").strip()
        if PERIOD.search(text):
            role = text
        facts.append(Fact(id=f"F{len(facts) + 1}", text=text, role=role, metrics=numbers_in(text)))
    return facts


def format_facts(facts: list[Fact]) -> str:
    """The knowledge base as the refiner and the grounding judge see it: one 'F3: text' per line."""
    return "\n".join(f"{fact.id}: {fact.text}" for fact in facts)
