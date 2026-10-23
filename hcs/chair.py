"""The chair: turns the four members' records into one verdict, a confidence, ranked objections and
dissent. Code, not a model (Step 4.5): every input is already a computed decision or a code-picked
objection, so a model would only add noise to a rule that can be tested and tuned against labels.

The verdict answers "does the CV show the job's requirements?", which is what the labels measure.
The tech lead judges the requirements; a high devil's-advocate objection (a requirement not shown at
all, 0 false highs in v3) can stop a hire. The recruiter judges presentation, which the refiner can
fix, and the hiring manager judges fit to this team's goals: both feed objections and dissent, and
lower the confidence, but don't change the verdict.
"""

from dataclasses import dataclass, field
from typing import Literal

from hcs.cases import Checklist
from hcs.evaluation import must_haves_named
from hcs.runlog import RunRecord
from hcs.schemas import Decision

Verdict = Literal["hire", "borderline", "no_hire"]
Confidence = Literal["high", "medium", "low"]
Kind = Literal["presentation", "gap"]

HIRE_SIDE: set[Decision] = {"lean_hire", "strong_hire"}
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
MAX_OBJECTIONS = 5


@dataclass
class ChairObjection:
    """One reason against the CV, merged across the members who raised it."""

    text: str
    severity: Literal["high", "medium", "low"]
    # A suggestion only: the reviewers can't see the knowledge base, the refiner sets the final kind.
    kind: Kind
    members: list[str]
    evidence: list[str] = field(default_factory=list)
    # The tech lead's check id, so the devil's advocate's objections can join it (None otherwise).
    check_id: str | None = None


@dataclass
class ChairVerdict:
    verdict: Verdict
    confidence: Confidence
    objections: list[ChairObjection]
    # Members whose own decision is on the other side of the verdict.
    dissent: list[str]
    scores: dict[str, float | None]
    reasons: list[str]


def decide_verdict(tech_lead: Decision | None, devil_highs: int) -> tuple[Verdict, list[str]]:
    """The hire rule: the tech lead's decision, with a high devil's-advocate objection blocking hire.
    A failed tech-lead run gives no evidence of fit, so the case can't be a hire."""
    if tech_lead is None:
        return "borderline", ["the tech lead's review failed"]
    if tech_lead == "strong_no":
        return "no_hire", ["the tech lead found the requirements clearly not met"]
    if tech_lead in HIRE_SIDE and devil_highs == 0:
        return "hire", [f"the tech lead's decision is {tech_lead}"]
    if tech_lead in HIRE_SIDE:
        return "borderline", [
            f"the tech lead's decision is {tech_lead}, but the devil's advocate found "
            f"{devil_highs} requirement(s) not shown at all"
        ]
    return "borderline", ["a must-have requirement is not shown (the tech lead's gate)"]


def side(decision: Decision | None) -> str | None:
    if decision is None:
        return None
    return "hire" if decision in HIRE_SIDE else "no"


def find_dissent(verdict: Verdict, decisions: dict[str, Decision | None]) -> list[str]:
    """Members whose decision disagrees with the verdict: hire-side against a no_hire or
    borderline, no-side against a hire. A borderline verdict counts as the no side."""
    verdict_side = "hire" if verdict == "hire" else "no"
    return [
        member
        for member, decision in decisions.items()
        if side(decision) is not None and side(decision) != verdict_side
    ]


def confidence_from(dissent: list[str], members: int) -> Confidence:
    if not dissent:
        return "high"
    return "medium" if len(dissent) == 1 and members > 2 else "low"


def check_objections(
    record: RunRecord | None, member: str, checklist: Checklist, kind: Kind
) -> list[ChairObjection]:
    """Every failed check of one member: a failed must-have is high for the tech lead and medium for
    the recruiter (fixable wording), a failed nice-to-have is low."""
    if record is None or record.review is None:
        return []
    importance = {item.id: item.importance for item in checklist.items}
    objections = []
    for check in record.review.checks:
        if check.passed or check.id not in importance:
            continue
        if importance[check.id] == "must":
            severity = "high" if kind == "gap" else "medium"
        else:
            severity = "low"
        objections.append(
            ChairObjection(
                text=check.check,
                severity=severity,
                kind=kind,
                members=[member],
                evidence=[check.evidence] if check.evidence else [],
                check_id=check.id,
            )
        )
    return objections


def rank_objections(
    tech_lead: RunRecord | None,
    tech_checklist: Checklist,
    devil: RunRecord | None,
    others: list[ChairObjection],
) -> list[ChairObjection]:
    """The tech lead's failed checks with the devil's advocate's objections merged into the
    must-haves they name, then the other members' objections, ranked by severity and by how many
    members raised them."""
    merged = check_objections(tech_lead, "tech_lead", tech_checklist, "gap")
    by_id = {item.check_id: item for item in merged}
    # Only "not shown at all" (high) counts: in v3 the devil's advocate rated a met requirement
    # "partly" (medium) on 15 of 16 gap-free CVs, so a medium is noise (decision closing Step 4.3).
    highs = [item for item in (devil.objections or []) if item.severity == "high"] if devil else []
    for objection in highs:
        targets = [
            by_id[check_id]
            for check_id in must_haves_named(objection.requirement, tech_checklist)
            if check_id in by_id
        ]
        if targets:
            for target in targets:
                target.members.append("devils_advocate")
                if SEVERITY_RANK[objection.severity] < SEVERITY_RANK[target.severity]:
                    target.severity = objection.severity
        else:
            merged.append(
                ChairObjection(
                    text=objection.requirement or objection.reason,
                    severity=objection.severity,
                    kind="gap",
                    members=["devils_advocate"],
                    evidence=list(objection.evidence),
                )
            )
    merged.extend(others)
    # Stable sort: within the same severity and support, the tech lead's order is kept.
    return sorted(merged, key=lambda item: (SEVERITY_RANK[item.severity], -len(item.members)))


def chair(
    *,
    tech_lead: RunRecord | None,
    tech_checklist: Checklist,
    recruiter: RunRecord | None,
    recruiter_checklist: Checklist,
    hiring_manager: RunRecord | None,
    hiring_manager_checklist: Checklist,
    devil: RunRecord | None,
) -> ChairVerdict:
    """Weigh one committee round. Any member may be None (not run, or its call failed)."""
    members = {"tech_lead": tech_lead, "recruiter": recruiter, "hiring_manager": hiring_manager}
    decisions = {name: record.decision if record else None for name, record in members.items()}
    devil_highs = (
        sum(1 for objection in (devil.objections or []) if objection.severity == "high")
        if devil
        else 0
    )

    verdict, reasons = decide_verdict(decisions["tech_lead"], devil_highs)
    dissent = find_dissent(verdict, decisions)
    others = [
        *check_objections(recruiter, "recruiter", recruiter_checklist, "presentation"),
        *check_objections(hiring_manager, "hiring_manager", hiring_manager_checklist, "gap"),
    ]
    ranked = rank_objections(tech_lead, tech_checklist, devil, others)
    voted = sum(1 for decision in decisions.values() if decision is not None)
    return ChairVerdict(
        verdict=verdict,
        confidence=confidence_from(dissent, voted),
        objections=ranked[:MAX_OBJECTIONS],
        dissent=dissent,
        scores={name: record.score if record else None for name, record in members.items()},
        reasons=reasons,
    )


LABEL_VERDICT: dict[str, Verdict] = {"good": "hire", "borderline": "borderline", "poor": "no_hire"}
