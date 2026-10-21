"""Evaluation data: postings, checklists and labelled cases, loaded from TOML files in data/ and validated."""

import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict

from hcs.schemas import ChecklistItem, Decision

DATA_DIR = Path("data")
POSTINGS_DIR = DATA_DIR / "postings"
CHECKLISTS_DIR = DATA_DIR / "checklists"
CASES_DIR = DATA_DIR / "cases"


class Posting(BaseModel):
    """A job posting as published. The job analyst reads its text."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    text: str


class Checklist(BaseModel):
    """The checks for one posting."""

    name: str
    posting: str
    version: int = 1
    items: list[ChecklistItem]


class ExpectedCheck(BaseModel):
    """The answer key for one check. A missing evidence key in TOML means no evidence."""

    passed: bool
    evidence: str | None = None


class Case(BaseModel):
    """One labelled CV, scored against a named checklist, with its answer key."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    label: Literal["good", "borderline", "poor"]
    split: Literal["dev", "heldout"] = "dev"
    checklist: str
    expected_decision: Decision
    cv: str
    expected: dict[str, ExpectedCheck]


def load_toml(path: Path) -> dict:
    """Read one TOML file into a dict."""
    with path.open("rb") as file:
        return tomllib.load(file)


def load_posting(name: str, directory: Path = POSTINGS_DIR) -> Posting:
    """Load and validate the posting stored as <directory>/<name>.toml."""
    return Posting.model_validate(load_toml(directory / f"{name}.toml"))


def load_checklist(name: str, directory: Path = CHECKLISTS_DIR) -> Checklist:
    """Load and validate the checklist stored as <directory>/<name>.toml."""
    return Checklist.model_validate(load_toml(directory / f"{name}.toml"))


def save_checklist(checklist: Checklist, header: str, directory: Path = CHECKLISTS_DIR) -> Path:
    """Write the checklist as <directory>/<name>.toml with comment lines on top; never overwrite.

    A draft is edited by hand after it is saved, so a second save must not erase those edits.
    """
    path = directory / f"{checklist.name}.toml"
    comments = "".join(f"# {line}\n" for line in header.splitlines())
    with path.open("x", encoding="utf-8") as file:
        file.write(comments + tomli_w.dumps(checklist.model_dump()))
    return path


def load_case(case_id: str, directory: Path = CASES_DIR) -> Case:
    """Load and validate the case stored as <directory>/<case_id>.toml."""
    return Case.model_validate(load_toml(directory / f"{case_id}.toml"))


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    """Load every case in the directory, in file-name order."""
    return [Case.model_validate(load_toml(path)) for path in sorted(directory.glob("*.toml"))]


def key_problems(case: Case, checklist: Checklist) -> list[str]:
    """Differences between a case's answer key and its checklist; an empty list means they match."""
    item_ids = [item.id for item in checklist.items]
    missing = [f"missing: {item_id}" for item_id in item_ids if item_id not in case.expected]
    unknown = [f"unknown: {key}" for key in case.expected if key not in item_ids]
    return missing + unknown
