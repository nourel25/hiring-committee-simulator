"""Run refinement sessions on labelled cases and measure what the design asks of the loop:
poor CVs stay no_hire, borderline CVs are not talked into a hire, every kept line is grounded,
and the recruiter's score rises where there was something to fix."""

import argparse
import json
from datetime import date
from pathlib import Path

from hcs.cases import CASES_DIR, load_case, load_cases
from hcs.config import devils_advocate_endpoint, grounding_endpoint
from hcs.evaluation import CASE_WIDTH
from hcs.llm import make_client
from hcs.refinement import Judge, Session, run_session

SESSIONS_PATH = Path("runs") / "sessions.jsonl"

HEADER = (
    f"{'case':<{CASE_WIDTH}}{'label':<12}{'verdict':<24}{'recruiter':<12}"
    f"{'rounds':>6}  {'kept':>4} {'rejected':>8}  stop"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case ids (default: every case)")
    parser.add_argument("--refiner", help="the refiner's Ollama model (default: refiner.MODEL)")
    parser.add_argument(
        "--local-judge",
        action="store_true",
        help="run the grounding judge locally (judge A) instead of config.GROUNDING_JUDGE",
    )
    parser.add_argument(
        "--local-devil",
        action="store_true",
        help="run the devil's advocate locally (judge A) instead of config.DEVILS_ADVOCATE",
    )
    parser.add_argument(
        "--dir", type=Path, default=CASES_DIR, help="where the cases are (default: data/cases)"
    )
    return parser.parse_args(argv)


def recruiter_score(session: Session, index: int) -> float | None:
    return session.rounds[index].committee.records["recruiter"].score


def save(session: Session, label: str, path: Path = SESSIONS_PATH) -> None:
    """One line per session: what the user would review (both CVs, verdicts, every line and why)."""
    drafts = [round_.draft for round_ in session.rounds if round_.draft]
    line = {
        "case_id": session.case_id,
        "label": label,
        "stop_reason": session.stop_reason,
        "verdicts": [round_.committee.verdict.verdict for round_ in session.rounds],
        "recruiter_scores": [recruiter_score(session, i) for i in range(len(session.rounds))],
        "first_cv": session.rounds[0].cv,
        "final_cv": session.final_cv,
        "kept": [line.model_dump() for draft in drafts for line in draft.kept],
        "rejected": [
            {"line": item.line.model_dump(), "problems": item.problems}
            for draft in drafts
            for item in draft.rejected
        ],
        "verbatim": [fact_id for draft in drafts for fact_id in draft.verbatim],
        "recruiter_gaps": session.rounds[-1].plan.gaps,
        "gaps": session.gaps,
        "suggestions": session.suggestions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(line) + "\n")


def main() -> None:
    args = parse_args()
    client = make_client()
    cases = (
        [load_case(case_id, args.dir) for case_id in args.cases]
        if args.cases
        else load_cases(args.dir)
    )
    settings = {"refiner_model": args.refiner} if args.refiner else {}
    settings["judge"] = Judge(client) if args.local_judge else grounding_endpoint()
    settings["devil"] = None if args.local_devil else devils_advocate_endpoint()
    results = []

    print(HEADER)
    for case in cases:
        session = run_session(client, case, date.today().year, **settings)
        save(session, case.label)
        first, last = session.rounds[0].committee.verdict, session.rounds[-1].committee.verdict
        drafts = [round_.draft for round_ in session.rounds if round_.draft]
        kept = sum(len(draft.kept) for draft in drafts)
        rejected = sum(len(draft.rejected) for draft in drafts)
        before, after = recruiter_score(session, 0), recruiter_score(session, -1)
        scores = f"{before or 0:.1f}->{after or 0:.1f}"
        results.append((case, first.verdict, last.verdict, before, after, session))
        print(
            f"{case.id:<{CASE_WIDTH}}{case.label:<12}{first.verdict + '->' + last.verdict:<24}"
            f"{scores:<12}{len(session.rounds):>6}  {kept:>4} {rejected:>8}  {session.stop_reason}"
        )
        for draft in drafts:
            for line in draft.kept:
                print(f"    kept     {line.text}  {line.facts}")
            for item in draft.rejected:
                print(f"    rejected {item.line.text}  ({'; '.join(item.problems)})")

    poor = [r for r in results if r[0].label == "poor"]
    borderline = [r for r in results if r[0].label == "borderline"]
    refined = [r for r in results if r[5].final_cv != r[5].rounds[0].cv]
    print()
    print(
        f"Poor CVs still no_hire after refinement: {sum(1 for r in poor if r[2] == 'no_hire')}/{len(poor)}"
    )
    print(
        f"Borderline CVs turned into a hire: {sum(1 for r in borderline if r[1] != 'hire' and r[2] == 'hire')}/{len(borderline)}"
    )
    print(f"Verdict changes: {sum(1 for r in results if r[1] != r[2])}/{len(results)}")
    print(f"CVs refined: {len(refined)}/{len(results)}")
    if refined:
        gains = [(r[4] or 0) - (r[3] or 0) for r in refined]
        print(
            f"Recruiter score on refined CVs: mean change {sum(gains) / len(gains):+.1f}, "
            f"rose in {sum(1 for g in gains if g > 0)}/{len(gains)}"
        )


if __name__ == "__main__":
    main()
