"""What the UI shows, as plain data: tables of checks, objections and sessions. No Streamlit here,
so every view can be tested without a browser; app.py only lays these out."""

import json
from pathlib import Path

from hcs.cases import POSTINGS_DIR, Case, load_posting
from hcs.chair import ChairVerdict
from hcs.committee import MEMBERS
from hcs.refinement import Session
from hcs.run_refinement import SESSIONS_PATH
from hcs.runlog import RunRecord
from hcs.screening import PERIOD, years_of_experience

TITLES = {
    "recruiter": "Recruiter",
    "hiring_manager": "Hiring manager",
    "tech_lead": "Tech lead",
    "devils_advocate": "Devil's advocate",
}


# The labels in plain words, in the order the picker lists them.
FIT = {"good": "Strong fit", "borderline": "Borderline", "poor": "Weak fit"}


def example_name(case: Case, this_year: int) -> str:
    """A labelled CV as a user reads it: 'Strong fit: Backend Developer, fintech startup, 5 years'.

    The role comes from the first dated line, since some CVs open with a summary instead."""
    role = next((line for line in case.cv.splitlines() if PERIOD.search(line)), case.id)
    role = PERIOD.sub("", role).strip(" ,")
    return f"{FIT[case.label]}: {role}, {years_of_experience(case.cv, this_year)} years"


def examples_for(posting: str, cases: list[Case], this_year: int) -> dict[str, Case]:
    """The posting's labelled CVs by their readable names, strong fits first."""
    ordered = sorted(
        (case for case in cases if case.checklist == posting),
        key=lambda c: list(FIT).index(c.label),
    )
    return {example_name(case, this_year): case for case in ordered}


def postings() -> dict[str, str]:
    """Every posting file as name -> job title, for the picker."""
    return {
        path.stem: load_posting(path.stem).title for path in sorted(POSTINGS_DIR.glob("*.toml"))
    }


def check_rows(record: RunRecord) -> list[dict]:
    """One row per check: passed or not, the check, and the quote it rests on."""
    if record.review is None:
        return []
    return [
        {
            "": "✓" if check.passed else "✗",
            "id": check.id,
            "check": check.check,
            "evidence": check.evidence or "",
        }
        for check in record.review.checks
    ]


def objection_rows(record: RunRecord) -> list[dict]:
    """The devil's advocate's objections, strongest first."""
    return [
        {
            "severity": objection.severity,
            "requirement": objection.requirement or "",
            "reason": objection.reason,
            "evidence": " / ".join(objection.evidence),
        }
        for objection in record.objections or []
    ]


def member_summary(member: str, record: RunRecord) -> str:
    """A one-line headline for a member's card."""
    if record.status != "ok":
        return f"{TITLES[member]}: the review failed"
    if record.review is not None:
        return f"{TITLES[member]}: {record.score:.1f}/10, {record.decision}"
    return f"{TITLES[member]}: {len(record.objections or [])} objection(s)"


def chair_rows(verdict: ChairVerdict) -> list[dict]:
    """The chair's ranked objections."""
    return [
        {
            "severity": item.severity,
            "kind": item.kind,
            "objection": item.text,
            "raised by": ", ".join(TITLES.get(member, member) for member in item.members),
        }
        for item in verdict.objections
    ]


def session_members(session: Session, round_index: int) -> list[tuple[str, RunRecord]]:
    """The members of one round in meeting order."""
    records = session.rounds[round_index].committee.records
    return [(member, records[member]) for member in MEMBERS if member in records]


def load_sessions(path: Path = SESSIONS_PATH) -> list[dict]:
    """Every logged refinement session, newest last; [] when none has been run."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def session_rows(sessions: list[dict]) -> list[dict]:
    """One row per logged session, for the results table."""
    return [
        {
            "case": session["case_id"],
            "label": session["label"],
            "verdicts": " → ".join(session["verdicts"]),
            "recruiter": " → ".join(f"{score:.1f}" for score in session["recruiter_scores"]),
            "lines kept": len(session["kept"]),
            "rejected": len(session["rejected"]),
            "stop": session["stop_reason"],
        }
        for session in sessions
    ]
