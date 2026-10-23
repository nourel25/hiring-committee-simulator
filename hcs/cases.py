"""Evaluation data: postings, checklists and labelled cases, loaded from TOML files in data/ and validated."""

import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from hcs.schemas import ChecklistItem, Decision

DATA_DIR = Path("data")
POSTINGS_DIR = DATA_DIR / "postings"
CHECKLISTS_DIR = DATA_DIR / "checklists"
CASES_DIR = DATA_DIR / "cases"


class RecruiterScreen(BaseModel):
    """What the recruiter's code looks for, reviewed by a person like a checklist."""

    titles: list[str] = Field(min_length=1)
    keywords: list[str]
    min_years: int = Field(ge=0)


class HiringManagerTargets(BaseModel):
    """What the hiring manager's checks compare a CV with, taken from the brief and reviewed by a
    person. Code writes them into the check text, so the model never has to infer them."""

    system: str
    goals: list[str] = Field(min_length=1)


class DevilsAdvocateLines(BaseModel):
    """Requirements lines the devil's advocate skips, reviewed by a person: traits no CV can prove."""

    ignore: list[str] = []


class Posting(BaseModel):
    """A job posting as published, plus a hand-written company brief, the recruiter's screen and
    the hiring manager's targets."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    text: str
    brief: str
    recruiter: RecruiterScreen
    hiring_manager: HiringManagerTargets
    devils_advocate: DevilsAdvocateLines = DevilsAdvocateLines()


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


class AgentKey(BaseModel):
    """Another reviewer's answer key for the same CV, checked against that reviewer's checklist."""

    expected_decision: Decision
    expected: dict[str, ExpectedCheck]


class Case(BaseModel):
    """One labelled CV, scored against a named checklist, with its answer key.

    `checklist`, `expected` and `expected_decision` are the tech lead's; other reviewers' keys
    are optional, so cases written before them still load.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    label: Literal["good", "borderline", "poor"]
    split: Literal["dev", "heldout"] = "dev"
    checklist: str
    expected_decision: Decision
    cv: str
    expected: dict[str, ExpectedCheck]
    recruiter: AgentKey | None = None
    hiring_manager: AgentKey | None = None


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


GROUNDING_PATH = DATA_DIR / "grounding" / "claims.toml"


class GroundingClaim(BaseModel):
    """One labelled claim for the grounding judge, citing facts built from a case's CV."""

    case: str
    split: Literal["dev", "heldout"]
    text: str
    facts: list[str]
    supported: bool
    note: str | None = None


def load_claims(path: Path = GROUNDING_PATH) -> list[GroundingClaim]:
    return [GroundingClaim.model_validate(claim) for claim in load_toml(path)["claims"]]
