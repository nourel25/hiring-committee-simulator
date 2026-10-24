"""One committee session on your own CV: the meeting transcript for every round, the before/after
diff, the gaps report, and an approval step. Nothing is ever submitted: approving only saves the CV.

    python -m hcs.session path/to/cv.txt --posting backend_payments
"""

import argparse
import difflib
from datetime import date
from pathlib import Path

from hcs.cases import Case, load_posting
from hcs.committee import format_transcript
from hcs.config import devils_advocate_endpoint, grounding_endpoint
from hcs.llm import make_client
from hcs.refinement import MAX_ROUNDS, Session, run_session

APPROVED_DIR = Path("runs") / "approved"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "cv", type=Path, help="a plain-text CV: dated role lines, then one bullet per line"
    )
    parser.add_argument("--posting", default="backend_payments", help="a posting in data/postings")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="the round limit")
    parser.add_argument("--yes", action="store_true", help="approve without asking (for scripts)")
    return parser.parse_args(argv)


def diff(before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.strip().splitlines(), after.strip().splitlines(), "before", "after", lineterm=""
    )
    return "\n".join(lines) or "(no change)"


def report(session: Session) -> str:
    """Everything the user reads before approving."""
    parts = []
    for round_ in session.rounds:
        parts.append(f"===== Round {round_.number}")
        parts.append(format_transcript(round_.committee))
        if round_.draft:
            parts.append("\nRefiner:")
            parts += [
                f"  kept     {line.text}  (facts {', '.join(line.facts)})"
                for line in round_.draft.kept
            ]
            parts += [
                f"  rejected {item.line.text}  ({'; '.join(item.problems)})"
                for item in round_.draft.rejected
            ]
        parts.append("")
    parts.append(f"Stopped: {session.stop_reason}")
    parts.append("\nGaps (not reworded; only new experience closes them):")
    parts += [f"  - {gap}" for gap in session.gaps] or ["  none"]
    parts.append("\nBefore / after:")
    parts.append(diff(session.rounds[0].cv, session.final_cv))
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    posting = load_posting(args.posting)
    case = Case(
        id=args.cv.stem,
        label="borderline",  # unknown for a real CV; only the evaluation reads labels
        checklist=args.posting,
        expected_decision="lean_no",
        cv=args.cv.read_text(encoding="utf-8"),
        expected={},
    )
    session = run_session(
        make_client(),
        case,
        date.today().year,
        posting,
        args.rounds,
        judge=grounding_endpoint(),
        devil=devils_advocate_endpoint(),
    )
    print(report(session))

    if session.final_cv == session.rounds[0].cv:
        return
    answer = "y" if args.yes else input("\nSave the refined CV? You can still edit it. [y/N] ")
    if answer.strip().casefold() == "y":
        APPROVED_DIR.mkdir(parents=True, exist_ok=True)
        path = APPROVED_DIR / f"{case.id}.txt"
        path.write_text(session.final_cv + "\n", encoding="utf-8")
        print(f"Saved {path}. Review it before you apply: nothing is sent anywhere.")


if __name__ == "__main__":
    main()
