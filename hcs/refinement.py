"""The refinement loop: committee → triage → refiner → grounding check → committee again.

Stop conditions (any one ends the session and goes to the user):
- nothing fixable is left: every remaining objection is a gap, or there is none;
- the round limit is reached;
- the refiner wrote no line that passed the grounding check, or the CV did not change.
"Verdict is hire" is not a stop condition on its own here (unlike the design draft): the chair's
verdict reads the requirements only, so a good CV with a weak top would never be refined.

v2 (after the first run): the summary goes under a headline with the current job title, because a
summary on the first line hid the title from R1 (a must-have) in every refined CV; and only the
recruiter reviews again, because the refiner changes only the top of the CV and the other members
judge the experience below it, which never changes. In v1 they picked the shorter summary line as
evidence, lost words like "myself", and two good CVs lost their hire verdict.

v3 (after the second run): code does the triage the design gives the refiner. For each fixable
recruiter check it looks for a fact that can fix it (a bullet with the missing keyword, a bullet
with a number, a bullet saying what the work is); a check no fact can fix is a gap. The model only
shortens the chosen facts, and a fact whose shortened line fails grounding twice is used as written.
"""

import re
from dataclasses import dataclass, field

from openai import OpenAI

from hcs.agents import refiner
from hcs.cases import Case, Posting, load_posting
from hcs.committee import Checklists, CommitteeRound, run_committee
from hcs.config import Endpoint
from hcs.grounding import restates, rule_problems
from hcs.llm import LLMOutputError, get_structured
from hcs.pipeline import judge_claims, run_identity
from hcs.policies import quote_in_cv
from hcs.profile import Fact, build_facts
from hcs.runlog import RUNS_PATH, RunRecord, append_run
from hcs.schemas import MAX_SUMMARY_LINES, Refinement, SummaryLine, refinement_model_for
from hcs.screening import PERIOD, current_title, keywords_found, top_third

MAX_ROUNDS = 3
# The recruiter's checks a summary can fix by rewording real facts. R1 (title) and R2 (years) are
# facts about the career, so a failure there is a gap.
FIXABLE = ("R3", "R4", "C1", "C2")
# Members whose reviews carry over between rounds: they judge the experience, which never changes.
CARRIED = ("hiring_manager", "tech_lead", "devils_advocate")
# A fact used as written when its shortened line fails grounding twice: longer ones are paragraphs.
MAX_VERBATIM_WORDS = 25
YEAR = re.compile(r"(?:19|20)\d{2}")


# The grounding judge's endpoint (its own client: it may be on an API while the committee runs
# locally). Judge(client) keeps it on the committee's client.
Judge = Endpoint


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
    # Facts used as written because every shortened line for them was rejected.
    verbatim: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """The triage: which facts to put in the summary, and which failed checks no fact can fix."""

    facts: list[Fact]
    fixable: list[str]
    gaps: list[str]


@dataclass
class Round:
    number: int
    cv: str
    committee: CommitteeRound
    plan: Plan
    draft: Draft | None = None

    @property
    def fixable(self) -> list[str]:
        return self.plan.fixable


@dataclass
class Session:
    case_id: str
    rounds: list[Round]
    stop_reason: str
    final_cv: str
    gaps: list[str]


def bullets(facts: list[Fact]) -> list[Fact]:
    """The facts that describe work: lines under a dated role, not the role lines themselves."""
    return [fact for fact in facts if fact.role is not None and not PERIOD.search(fact.text)]


def has_result_number(fact: Fact) -> bool:
    """A number that isn't a year: a count, an amount, a time or a share."""
    return any(not YEAR.fullmatch(number) for number in fact.metrics)


def triage(
    cv: str,
    committee: CommitteeRound,
    facts: list[Fact],
    keywords: list[str],
    covered: list[str] = (),
) -> Plan:
    """For each failed fixable recruiter check, find facts that can fix it; the rest are gaps.

    v4: facts already in the summary (`covered`) stay there and are not chosen again; the new
    facts fill only the lines left. In v3 every round rebuilt the summary from its own failures,
    so a keyword line added in round 2 was dropped in round 3 for a number line, back and forth.
    """
    record = committee.records.get("recruiter")
    if record is None or record.review is None:
        return Plan([], [], [])
    failed = [
        check.id for check in record.review.checks if check.id in FIXABLE and not check.passed
    ]
    work = [fact for fact in bullets(facts) if fact.id not in covered]
    chosen: list[Fact] = []
    fixable: list[str] = []
    gaps: list[str] = []

    def choose(fact: Fact | None) -> bool:
        if fact is None:
            return False
        if fact not in chosen:
            chosen.append(fact)
        return True

    for check_id in failed:
        if check_id == "R3":
            missing = [kw for kw in keywords if not keywords_found(top_third(cv), [kw])]
            found = [
                choose(next((f for f in work if keywords_found(f.text, [kw])), None))
                for kw in missing
            ]
            (fixable if any(found) else gaps).append(check_id)
        elif check_id == "C2":
            ok = choose(result_line(committee, work))
            (fixable if ok else gaps).append(check_id)
        elif check_id == "C1":
            ok = choose(work_line(committee, work))
            (fixable if ok else gaps).append(check_id)
        else:  # R4: a long line in the top third; a short summary pushes it down
            (fixable if work else gaps).append(check_id)
    if fixable and not chosen:
        choose(work_line(committee, work))
    room = MAX_SUMMARY_LINES - len(covered)
    return Plan(chosen[: max(room, 0)], fixable, gaps)


# Checks whose passing quote is a measured result of the candidate's own work.
RESULT_CHECKS = ("TL7", "HM3")


def verified_quotes(committee: CommitteeRound) -> list[str]:
    """Quotes the tech lead and the hiring manager used for checks they passed, measured results
    first: lines a reviewer has already judged to be the candidate's own work."""
    passed = [
        check
        for member in ("tech_lead", "hiring_manager")
        if (record := committee.records.get(member)) and record.review
        for check in record.review.checks
        if check.passed and check.evidence
    ]
    passed.sort(key=lambda check: check.id not in RESULT_CHECKS)
    return [check.evidence for check in passed]


def result_line(committee: CommitteeRound, work: list[Fact]) -> Fact | None:
    """A bullet with a number (not a year) that a reviewer quoted for a passed check.

    v4: in v3 code took the first bullet with any number, and in present-4 that was the employer's
    description ("4 million users in 12 countries"), which the refiner then framed as the
    candidate's. A reviewer's passing quote is a line already judged to be the candidate's own
    work; with none, C2 is a gap rather than a guess."""
    for quote in verified_quotes(committee):
        match = next(
            (fact for fact in work if has_result_number(fact) and quote_in_cv(quote, fact.text)),
            None,
        )
        if match:
            return match
    return None


def work_line(committee: CommitteeRound, work: list[Fact]) -> Fact | None:
    """The bullet that best says what the candidate's work is: the one the tech lead quoted for the
    first check it passed. The first bullet can be personal qualities ("Hard-working and reliable
    developer", ref-1 in the v3 run), so it's only the fallback when the tech lead quoted nothing."""
    record = committee.records.get("tech_lead")
    quotes = [
        check.evidence
        for check in (record.review.checks if record and record.review else [])
        if check.passed and check.evidence
    ]
    for quote in quotes:
        match = next((fact for fact in work if quote_in_cv(quote, fact.text)), None)
        if match:
            return match
    return work[0] if work else None


def gap_objections(committee: CommitteeRound) -> list[str]:
    """Objections the refiner must not reword away: everything the chair ranked as a gap."""
    return [item.text for item in committee.verdict.objections if item.kind == "gap"]


def with_summary(cv: str, lines: list[str]) -> str:
    """The CV with the lines above its first dated role replaced by a headline (the current job title,
    so the recruiter's title check still reads it first) and the new summary."""
    body = [line for line in cv.strip().splitlines()]
    first_role = next((index for index, line in enumerate(body) if PERIOD.search(line)), 0)
    headline = current_title(body[first_role])
    return "\n".join([headline, *lines, "", *body[first_role:]])


def ground(
    judge: Judge,
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
    # A line that only reuses one short fact's words needs no judge.
    restated = [line for line in passed if restates(line.text, line.facts, facts)]
    passed = [line for line in passed if line not in restated]
    if not passed:
        return restated, rejected, None
    record = judge_claims(
        judge.client,
        [(line.text, line.facts) for line in passed],
        facts,
        case_id,
        runs_path,
        **judge.settings,
    )
    if record.status != "ok":
        return (
            restated,
            rejected + [GroundedLine(line, ["the grounding judge failed"]) for line in passed],
            record,
        )
    kept = list(restated)
    for line, judgement in zip(passed, record.judgements, strict=True):
        if judgement.supported:
            kept.append(line)
        else:
            rejected.append(GroundedLine(line, [f"not supported: {judgement.unsupported}"]))
    return kept, rejected, record


def call_refiner(
    client: OpenAI,
    chosen: list[Fact],
    keywords: list[str],
    feedback: list[str],
    case_id: str,
    runs_path,
    model: str,
) -> tuple[RunRecord, list[SummaryLine]]:
    messages = refiner.build_messages(chosen, keywords, feedback)
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
            response_model=refinement_model_for([fact.id for fact in chosen]),
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
    chosen: list[Fact],
    keywords: list[str],
    case_id: str,
    runs_path=RUNS_PATH,
    model: str = refiner.MODEL,
    judge: Judge | None = None,
) -> Draft:
    """Shorten the chosen facts and ground each line; rejected lines go back once with the reasons.
    A chosen fact still without a line after that is used as written when it is short."""
    judge = judge or Judge(client)
    record, lines = call_refiner(client, chosen, keywords, [], case_id, runs_path, model)
    records = [record]
    kept, rejected, check = ground(judge, lines, facts, keywords, case_id, runs_path)
    records += [check] if check else []
    if rejected or not lines:
        feedback = [f'"{item.line.text}": {"; ".join(item.problems)}' for item in rejected]
        redo = [fact for fact in chosen if all(fact.id not in line.facts for line in kept)]
        record, lines = call_refiner(client, redo, keywords, feedback, case_id, runs_path, model)
        records.append(record)
        more, rejected, check = ground(judge, lines, facts, keywords, case_id, runs_path)
        kept += more
        records += [check] if check else []
    verbatim = []
    for fact in chosen:
        covered = any(fact.id in line.facts for line in kept)
        if not covered and len(fact.text.split()) <= MAX_VERBATIM_WORDS:
            kept.append(SummaryLine(facts=[fact.id], text=fact.text))
            verbatim.append(fact.id)
    order = [fact.id for fact in chosen]
    kept.sort(key=lambda line: order.index(line.facts[0]))
    return Draft(kept=kept, rejected=rejected, records=records, verbatim=verbatim)


def run_session(
    client: OpenAI,
    case: Case,
    this_year: int,
    posting: Posting | None = None,
    max_rounds: int = MAX_ROUNDS,
    runs_path=RUNS_PATH,
    refiner_model: str = refiner.MODEL,
    judge: Judge | None = None,
    devil: Endpoint | None = None,
) -> Session:
    """Committee rounds with refinement in between, until a stop condition fires."""
    posting = posting or load_posting(case.checklist)
    checklists = Checklists.for_posting(case.checklist)
    # The knowledge base is the original CV: a refined summary is never a source of facts.
    facts = build_facts(case.cv)
    keywords = posting.recruiter.keywords
    cv = case.cv
    rounds: list[Round] = []
    reason = "round limit"
    # The summary so far: each round adds lines and never drops the earlier ones (v4).
    summary: list[SummaryLine] = []

    for number in range(1, max_rounds + 1):
        current = case.model_copy(update={"cv": cv})
        carried = (
            {member: rounds[0].committee.records[member] for member in CARRIED} if rounds else {}
        )
        committee = run_committee(
            client, current, this_year, posting, checklists, runs_path, carried, devil
        )
        covered = [line.facts[0] for line in summary]
        plan = triage(cv, committee, facts, keywords, covered)
        rounds.append(Round(number, cv, committee, plan))
        if not plan.facts:
            if plan.fixable and len(summary) >= MAX_SUMMARY_LINES:
                reason = "summary full"
            else:
                reason = "hire" if committee.verdict.verdict == "hire" else "only gaps remain"
            break
        if number == max_rounds:
            break
        draft = write_summary(
            client, facts, plan.facts, keywords, case.id, runs_path, refiner_model, judge
        )
        rounds[-1].draft = draft
        if not draft.kept:
            reason = "no grounded summary"
            break
        summary += draft.kept
        new_cv = with_summary(case.cv, [line.text for line in summary])
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
