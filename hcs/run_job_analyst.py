"""Draft a checklist from a posting with the job analyst and print it next to the hand-written one.

With --save, the draft is written to data/checklists/ for a person to review and edit.
With --eval, the tech lead (judge A, local) then scores every labelled CV with the drafted checklist,
to see whether the decisions still come out right.
"""

import argparse
import re

from hcs.agents import job_analyst
from hcs.cases import Checklist, load_cases, load_checklist, load_posting, save_checklist
from hcs.config import API_MAX_TOKENS
from hcs.evaluation import CASE_WIDTH, format_or_dash
from hcs.llm import LLMOutputError, get_structured, make_client
from hcs.pipeline import review_case

# A posting's reviewed checklist and its labelled cases use the posting's name as the checklist name.
DEFAULT_POSTING = "backend_payments"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read which posting to analyse, with which model, where, and what to do with the draft."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default=job_analyst.MODEL,
        help=f"the job analyst's model (default: {job_analyst.MODEL})",
    )
    parser.add_argument(
        "--posting",
        default=DEFAULT_POSTING,
        help=f"the posting's file name in data/postings/, without .toml (default: {DEFAULT_POSTING})",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        default="ollama",
        help="where the job analyst runs (groq: strict schema, larger max_tokens)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="write the draft to data/checklists/ for review (never overwrites a file)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="score the posting's labelled CVs with the drafted checklist and compare the decisions",
    )
    return parser.parse_args(argv)


def draft_name(posting_name: str, model: str) -> str:
    """A checklist name that is also a safe file name, e.g. 'backend_payments-draft-qwen2.5-7b'."""
    return f"{posting_name}-draft-{re.sub(r'[^A-Za-z0-9_.-]', '-', model)}"


def draft_checklist(posting_name: str, model: str, provider: str) -> Checklist | None:
    """Ask the job analyst for a checklist, print it with the answer key, and return it."""
    posting = load_posting(posting_name)
    api = provider != "ollama"
    try:
        result = get_structured(
            make_client(provider),
            job_analyst.build_messages(posting.text),
            job_analyst.analysis_model_for(posting.text),
            model=model,
            temperature=job_analyst.TEMPERATURE,
            seed=job_analyst.SEED,
            max_tokens=API_MAX_TOKENS if api else job_analyst.MAX_TOKENS,
            max_retries=job_analyst.MAX_RETRIES,
            strict=api,
        )
    except LLMOutputError as error:
        print(f"Gave up after {error.attempts} attempt(s): {error}")
        print(f"Last reply: {error.raw_reply!r}")
        return None

    analysis = result.parsed
    drafted = job_analyst.to_checklist(analysis, draft_name(posting_name, model), posting.title)
    print(f"Done in {result.attempts} attempt(s), {result.seconds:.1f}s")
    print(f"Tokens: {result.prompt_tokens} in, {result.completion_tokens} out")

    print("\n== Drafted checks")
    for item, draft in zip(drafted.items, analysis.checks):
        print(f"{item.id} [{item.importance}] {item.text}")
        print(f"    from: {draft.source}")
    print(f"\n== Lookups: {analysis.lookups}")
    print("== Dropped")
    for line in analysis.dropped:
        print(f"    {line.source} -> {line.reason}")

    print("\n== Warnings")
    for warning in job_analyst.draft_warnings(analysis) or ["none"]:
        print(f"    {warning}")

    print("\n== Hand-written answer key")
    try:
        answer_key = load_checklist(posting_name)
    except FileNotFoundError:
        print(f"    none yet: no data/checklists/{posting_name}.toml")
        return drafted
    for item in answer_key.items:
        print(f"{item.id} [{item.importance}] {item.text}")
    return drafted


def save_draft(checklist: Checklist, model: str) -> None:
    """Write the draft for review, with where it came from on top. Refuses to overwrite."""
    header = (
        f"Draft by the job analyst ({job_analyst.PROMPT_VERSION}, {model}). Not reviewed yet.\n"
        "Review every check and its importance, run the downstream eval, then rename the file "
        "and its name field together."
    )
    try:
        path = save_checklist(checklist, header)
    except FileExistsError:
        print(f"\nNot saved: {checklist.name}.toml already exists and may hold your edits.")
        return
    print(f"\nSaved the draft to {path}")


def score_cases(checklist: Checklist, posting_name: str) -> None:
    """Downstream check on the posting's cases: decisions only, since the keys use the reviewed ids."""
    cases = [case for case in load_cases() if case.checklist == posting_name]
    if not cases:
        print(f"\nNo labelled cases for {posting_name} yet, so nothing to score.")
        return
    client = make_client()
    right = 0
    print(f"\n== Tech lead with {checklist.name}")
    for case in cases:
        record = review_case(client, case, checklist)
        ok = record.decision == case.expected_decision
        right += ok
        print(
            f"{case.id:<{CASE_WIDTH}} {case.split:<8} expected {case.expected_decision:<12} "
            f"got {str(record.decision):<12} {format_or_dash(record.score, '.1f'):>4}  "
            f"{'ok' if ok else 'WRONG'}  failed must-haves: {record.failed_must_haves}"
        )
    print(f"Decisions: {right}/{len(cases)}")


def main() -> None:
    args = parse_args()
    drafted = draft_checklist(args.posting, args.model, args.provider)
    if drafted is None:
        return
    if args.save:
        save_draft(drafted, args.model)
    if args.eval:
        score_cases(drafted, args.posting)


if __name__ == "__main__":
    main()
