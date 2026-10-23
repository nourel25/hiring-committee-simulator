"""One committee round: the four members review independently, then the chair weighs them."""

import time
from dataclasses import dataclass

from openai import OpenAI

from hcs.cases import Case, Checklist, Posting, load_checklist, load_posting
from hcs.chair import ChairVerdict, chair
from hcs.pipeline import (
    review_case,
    review_devils_advocate,
    review_hiring_manager,
    review_recruiter,
)
from hcs.runlog import RUNS_PATH, RunRecord

MEMBERS = ["recruiter", "hiring_manager", "tech_lead", "devils_advocate"]


@dataclass
class Checklists:
    tech_lead: Checklist
    recruiter: Checklist
    hiring_manager: Checklist

    @classmethod
    def for_posting(cls, name: str) -> "Checklists":
        return cls(
            load_checklist(name), load_checklist("recruiter"), load_checklist("hiring_manager")
        )


@dataclass
class CommitteeRound:
    case_id: str
    records: dict[str, RunRecord]
    verdict: ChairVerdict
    seconds: float


def weigh(records: dict[str, RunRecord], checklists: Checklists) -> ChairVerdict:
    """The chair's verdict on a set of member records; failed runs count as missing."""

    def ok(member: str) -> RunRecord | None:
        record = records.get(member)
        return record if record is not None and record.status == "ok" else None

    return chair(
        tech_lead=ok("tech_lead"),
        tech_checklist=checklists.tech_lead,
        recruiter=ok("recruiter"),
        recruiter_checklist=checklists.recruiter,
        hiring_manager=ok("hiring_manager"),
        hiring_manager_checklist=checklists.hiring_manager,
        devil=ok("devils_advocate"),
    )


def run_committee(
    client: OpenAI,
    case: Case,
    this_year: int,
    posting: Posting | None = None,
    checklists: Checklists | None = None,
    runs_path=RUNS_PATH,
    carried: dict[str, RunRecord] | None = None,
) -> CommitteeRound:
    """Run the four members on one CV (none sees another's output), then the chair. Members in
    `carried` are not run again: their earlier reviews are reused."""
    posting = posting or load_posting(case.checklist)
    checklists = checklists or Checklists.for_posting(case.checklist)
    carried = carried or {}
    calls = {
        "recruiter": lambda: review_recruiter(
            client, case, posting, checklists.recruiter, this_year, runs_path
        ),
        "hiring_manager": lambda: review_hiring_manager(
            client, case, posting, checklists.hiring_manager, runs_path
        ),
        "tech_lead": lambda: review_case(client, case, checklists.tech_lead, runs_path),
        "devils_advocate": lambda: review_devils_advocate(client, case, posting, runs_path),
    }
    start = time.perf_counter()
    records = {
        member: carried[member] if member in carried else call() for member, call in calls.items()
    }
    return CommitteeRound(
        case_id=case.id,
        records=records,
        verdict=weigh(records, checklists),
        seconds=time.perf_counter() - start,
    )


def format_transcript(round_: CommitteeRound) -> str:
    """A readable account of one round: each member's view, then the chair's verdict."""
    lines = [f"Committee on {round_.case_id}", ""]
    for member in MEMBERS:
        record = round_.records.get(member)
        if record is None:
            continue
        title = member.replace("_", " ").title()
        if record.status != "ok":
            lines.append(f"{title}: the review failed ({record.error})")
            continue
        if record.review is not None:
            lines.append(f"{title}: {record.score:.1f}/10, {record.decision}")
            for check in record.review.checks:
                mark = "yes" if check.passed else "no "
                quote = f' "{check.evidence}"' if check.evidence else ""
                lines.append(f"  [{mark}] {check.id} {check.check}{quote}")
        else:
            lines.append(f"{title}: {len(record.objections or [])} objection(s)")
            for objection in record.objections or []:
                lines.append(
                    f"  ({objection.severity}) {objection.requirement}: {objection.reason}"
                )
        lines.append("")
    verdict = round_.verdict
    lines.append(f"Chair: {verdict.verdict} (confidence {verdict.confidence})")
    lines.extend(f"  because {reason}" for reason in verdict.reasons)
    if verdict.dissent:
        lines.append(f"  dissent: {', '.join(verdict.dissent)}")
    for number, objection in enumerate(verdict.objections, start=1):
        members = ", ".join(objection.members)
        lines.append(
            f"  {number}. ({objection.severity}, {objection.kind}) {objection.text} [{members}]"
        )
    return "\n".join(lines)
