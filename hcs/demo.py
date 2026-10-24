"""The demo: run one session on a labelled (invented) CV and save everything the UI shows, so a
static page can replay it with no models and no server.

    python -m hcs.demo record present-1      # runs the committee and refinement, writes the JSON
    python -m hcs.demo render                # builds docs/index.html from the saved JSON

The JSON keeps each member's checks with their quotes, the devil's advocate's objections, the
chair's verdict, the refiner's kept and rejected lines with the reasons, the gaps and the diff.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from hcs.cases import load_posting
from hcs.config import devils_advocate_endpoint, grounding_endpoint
from hcs.llm import make_client
from hcs.profile import build_facts
from hcs.refinement import Round, Session, run_session
from hcs.run_held_out import load_any_case
from hcs.runlog import RunRecord
from hcs.session import diff
from hcs.views import TITLES

DOCS = Path("docs")
SESSION_JSON = DOCS / "demo-session.json"
TEMPLATE = Path(__file__).with_name("demo_template.html")
PAGE = DOCS / "index.html"


def member_view(record: RunRecord) -> dict:
    """One member's review as plain data: decision and score, checks or objections."""
    view = {
        "model": record.model,
        "status": record.status,
        "decision": record.decision,
        "score": None if record.score is None else round(record.score, 1),
        "seconds": None if record.seconds is None else round(record.seconds, 1),
    }
    if record.review is not None:
        view["checks"] = [
            {"id": c.id, "check": c.check, "passed": c.passed, "evidence": c.evidence}
            for c in record.review.checks
        ]
    if record.objections is not None:
        view["objections"] = [
            {
                "requirement": o.requirement,
                "severity": o.severity,
                "reason": o.reason,
                "evidence": list(o.evidence),
            }
            for o in record.objections
        ]
    return view


def round_view(round_: Round) -> dict:
    verdict = round_.committee.verdict
    view = {
        "number": round_.number,
        "cv": round_.cv,
        "members": {
            member: {"title": TITLES[member], **member_view(record)}
            for member, record in round_.committee.records.items()
        },
        "chair": {
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "reasons": verdict.reasons,
            "dissent": verdict.dissent,
            "objections": [
                {
                    "text": o.text,
                    "severity": o.severity,
                    "kind": o.kind,
                    "members": [TITLES.get(m, m) for m in o.members],
                }
                for o in verdict.objections
            ],
        },
        "plan": {
            "facts": [fact.id for fact in round_.plan.facts],
            "fixable": round_.plan.fixable,
            "gaps": round_.plan.gaps,
        },
    }
    if round_.draft is not None:
        view["draft"] = {
            "kept": [line.model_dump() for line in round_.draft.kept],
            "rejected": [
                {"line": item.line.model_dump(), "problems": item.problems}
                for item in round_.draft.rejected
            ],
        }
    return view


def session_view(session: Session, case_id: str) -> dict:
    case = load_any_case(case_id)
    posting = load_posting(case.checklist)
    return {
        "case": {"id": case.id, "label": case.label, "expected": case.expected_decision},
        "posting": {"title": posting.title, "text": posting.text},
        "facts": [{"id": f.id, "text": f.text} for f in build_facts(case.cv)],
        "rounds": [round_view(round_) for round_ in session.rounds],
        "stop_reason": session.stop_reason,
        "gaps": session.gaps,
        "final_cv": session.final_cv,
        "diff": diff(session.rounds[0].cv, session.final_cv),
        "recorded": date.today().isoformat(),
    }


def record(case_id: str, path: Path = SESSION_JSON) -> dict:
    """Run one session with the configured models (synthetic CV, so the API judges are used)."""
    case = load_any_case(case_id)
    session = run_session(
        make_client(),
        case,
        date.today().year,
        load_posting(case.checklist),
        judge=grounding_endpoint(),
        devil=devils_advocate_endpoint(),
    )
    view = session_view(session, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(view, indent=2, ensure_ascii=False), encoding="utf-8")
    return view


def render(path: Path = SESSION_JSON, page: Path = PAGE) -> None:
    """The static page: the template with the session JSON inlined, so it works from a file too."""
    data = path.read_text(encoding="utf-8").replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*SESSION*/null", data)
    page.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("case_id")
    sub.add_parser("render")
    args = parser.parse_args()
    if args.command == "record":
        view = record(args.case_id)
        print(f"Saved {SESSION_JSON}: {len(view['rounds'])} round(s), {view['stop_reason']}")
    else:
        render()
        print(f"Built {PAGE}")


if __name__ == "__main__":
    main()
