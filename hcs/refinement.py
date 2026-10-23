"""The refinement loop: committee → triage → refiner → grounding check → committee again.

Stop conditions (any one ends the session and goes to the user):
- nothing fixable is left: every remaining objection is a gap, or there is none;
- the round limit is reached;
- the refiner wrote no line that passed the grounding check, or the CV did not change.
"Verdict is hire" is not a stop condition on its own here (unlike the design draft): the chair's
verdict reads the requirements only, so a good CV with a weak top would never be refined.
"""

from dataclasses import dataclass, field

from openai import OpenAI

from hcs.agents import refiner
from hcs.cases import Case, Posting, load_posting
from hcs.committee import Checklists, CommitteeRound, run_committee
from hcs.grounding import rule_problems
from hcs.llm import LLMOutputError, get_structured
from hcs.pipeline import judge_claims, run_identity
from hcs.profile import Fact, build_facts
from hcs.runlog import RUNS_PATH, RunRecord, append_run
from hcs.schemas import Refinement, SummaryLine, refinement_model_for
from hcs.screening import PERIOD

MAX_ROUNDS = 3
# The recruiter's checks a summary can fix by rewording real facts. R1 (title) and R2 (years) are
# facts about the career, so a failure there is a gap.
FIXABLE = ("R3", "R4", "C1", "C2")


@dataclass
class GroundedLine:
    line: SummaryLine
    problems: list[str]


@dataclass
class Draft:
    """One refiner pass: the lines kept, the lines rejected and why, and the logged calls."""

    kept: list[SummaryLine]
    rejected: list[GroundedLine]
    records: list[RunRecord] = field(default_factory=list)


@dataclass
class Round:
    number: int
    cv: str
    committee: CommitteeRound
    fixable: list[str]
    draft: Draft | None = None


@dataclass
class Session:
    case_id: str
    rounds: list[Round]
    stop_reason: str
    final_cv: str
    gaps: list[str]


def fixable_objections(committee: CommitteeRound, checklists: Checklists) -> list[str]:
    """The recruiter's failed checks that a new summary could fix, as check texts."""
    record = committee.records.get("recruiter")
    if record is None or record.review is None:
        return []
    texts = {item.id: item.text for item in checklists.recruiter.items}
    return [
        texts[check.id]
        for check in record.review.checks
        if check.id in FIXABLE and not check.passed
    ]


def gap_objections(committee: CommitteeRound) -> list[str]:
    """Objections the refiner must not reword away: everything the chair ranked as a gap."""
    return [item.text for item in committee.verdict.objections if item.kind == "gap"]


def with_summary(cv: str, lines: list[str]) -> str:
    """The CV with the lines above its first dated role replaced by the new summary."""
    body = [line for line in cv.strip().splitlines()]
    first_role = next((index for index, line in enumerate(body) if PERIOD.search(line)), 0)
    return "\n".join([*lines, "", *body[first_role:]])


def ground(
    client: OpenAI,
    lines: list[SummaryLine],
    facts: list[Fact],
    keywords: list[str],
    case_id: str,
    runs_path,
) -> tuple[list[SummaryLine], list[GroundedLine], RunRecord | None]:
    """Rules first, then the judge on the lines the rules passed. A failed judge call keeps nothing:
    an unchecked line never reaches the CV."""
    rejected = [
        GroundedLine(line, problems)
        for line in lines
        if (problems := rule_problems(line.text, line.facts, facts, keywords))
    ]
    passed = [line for line in lines if not rule_problems(line.text, line.facts, facts, keywords)]
    if not passed:
        return [], rejected, None
    record = judge_claims(
        client, [(line.text, line.facts) for line in passed], facts, case_id, runs_path
    )
    if record.status != "ok":
        return (
            [],
            rejected + [GroundedLine(line, ["the grounding judge failed"]) for line in passed],
            record,
        )
    kept = []
    for line, judgement in zip(passed, record.judgements, strict=True):
        if judgement.supported:
            kept.append(line)
        else:
            rejected.append(GroundedLine(line, [f"not supported: {judgement.unsupported}"]))
    return kept, rejected, record


def call_refiner(
    client: OpenAI,
    facts: list[Fact],
    problems: list[str],
    keywords: list[str],
    feedback: list[str],
    case_id: str,
    runs_path,
    model: str,
) -> tuple[RunRecord, list[SummaryLine]]:
    messages = refiner.build_messages(facts, problems, keywords, feedback)
    common = run_identity(
        agent="refiner",
        prompt_version=refiner.PROMPT_VERSION,
        case_id=case_id,
        checklist=None,
        messages=messages,
        model=model,
        temperature=refiner.TEMPERATURE,
        seed=refiner.SEED,
        max_tokens=refiner.MAX_TOKENS,
        max_retries=refiner.MAX_RETRIES,
    )
    try:
        result = get_structured(
            client,
            messages,
            response_model=refinement_model_for([fact.id for fact in facts]),
            model=model,
            temperature=refiner.TEMPERATURE,
            seed=refiner.SEED,
            max_tokens=refiner.MAX_TOKENS,
            max_retries=refiner.MAX_RETRIES,
        )
    except LLMOutputError as error:
        record = RunRecord(
            **common,
            status="error",
            error=str(error),
            attempts=error.attempts,
            raw_reply=error.raw_reply,
        )
        append_run(record, runs_path)
        return record, []
    summary = Refinement.model_validate(result.parsed.model_dump()).summary
    record = RunRecord(
        **common,
        status="ok",
        attempts=result.attempts,
        seconds=result.seconds,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        raw_reply=result.raw_reply,
        summary=summary,
    )
    append_run(record, runs_path)
    return record, summary


def write_summary(
    client: OpenAI,
    facts: list[Fact],
    problems: list[str],
    keywords: list[str],
    case_id: str,
    runs_path=RUNS_PATH,
    model: str = refiner.MODEL,
) -> Draft:
    """One refiner pass with the grounding check; rejected lines go back once with the reasons."""
    record, lines = call_refiner(client, facts, problems, keywords, [], case_id, runs_path, model)
    records = [record]
    kept, rejected, judge = ground(client, lines, facts, keywords, case_id, runs_path)
    records += [judge] if judge else []
    if rejected:
        feedback = [f'"{item.line.text}": {"; ".join(item.problems)}' for item in rejected]
        record, lines = call_refiner(
            client, facts, problems, keywords, feedback, case_id, runs_path, model
        )
        records.append(record)
        kept, rejected, judge = ground(client, lines, facts, keywords, case_id, runs_path)
        records += [judge] if judge else []
    return Draft(kept=kept, rejected=rejected, records=records)


def run_session(
    client: OpenAI,
    case: Case,
    this_year: int,
    posting: Posting | None = None,
    max_rounds: int = MAX_ROUNDS,
    runs_path=RUNS_PATH,
    refiner_model: str = refiner.MODEL,
) -> Session:
    """Committee rounds with refinement in between, until a stop condition fires."""
    posting = posting or load_posting(case.checklist)
    checklists = Checklists.for_posting(case.checklist)
    # The knowledge base is the original CV: a refined summary is never a source of facts.
    facts = build_facts(case.cv)
    cv = case.cv
    rounds: list[Round] = []
    reason = "round limit"

    for number in range(1, max_rounds + 1):
        current = case.model_copy(update={"cv": cv})
        committee = run_committee(client, current, this_year, posting, checklists, runs_path)
        fixable = fixable_objections(committee, checklists)
        rounds.append(Round(number, cv, committee, fixable))
        if not fixable:
            reason = "hire" if committee.verdict.verdict == "hire" else "only gaps remain"
            break
        if number == max_rounds:
            break
        draft = write_summary(
            client, facts, fixable, posting.recruiter.keywords, case.id, runs_path, refiner_model
        )
        rounds[-1].draft = draft
        if not draft.kept:
            reason = "no grounded summary"
            break
        new_cv = with_summary(case.cv, [line.text for line in draft.kept])
        if new_cv == cv:
            reason = "no change"
            break
        cv = new_cv

    return Session(
        case_id=case.id,
        rounds=rounds,
        stop_reason=reason,
        final_cv=cv,
        gaps=gap_objections(rounds[-1].committee),
    )
