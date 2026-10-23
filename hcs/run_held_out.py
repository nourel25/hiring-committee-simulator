"""The held-out gap: the held-out judge (Groq, another family, never in the loop) scores each refined
CV before and after refinement, as the recruiter and as the tech lead. If the in-loop scores rise
while the held-out ones stay flat, the refiner is gaming the committee.

Reads runs/sessions.jsonl (written by hcs.run_refinement); synthetic CVs only.
"""

import json
from datetime import date

from hcs.cases import load_case, load_checklist, load_posting
from hcs.config import API_MAX_TOKENS, HELD_OUT_JUDGE
from hcs.evaluation import CASE_WIDTH
from hcs.llm import make_client
from hcs.pipeline import review_case, review_recruiter
from hcs.run_refinement import SESSIONS_PATH


def latest_refined_sessions() -> list[dict]:
    """The last session logged for each case, keeping only those whose CV changed."""
    sessions = {}
    with SESSIONS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                session = json.loads(line)
                sessions[session["case_id"]] = session
    return [s for s in sessions.values() if s["final_cv"] != s["first_cv"]]


def main() -> None:
    client = make_client(HELD_OUT_JUDGE.provider)
    settings = {"model": HELD_OUT_JUDGE.model, "max_tokens": API_MAX_TOKENS, "strict": True}
    recruiter_checklist = load_checklist("recruiter")
    rows = []

    print(
        f"{'case':<{CASE_WIDTH}}{'label':<12}{'in-loop R':<12}{'held-out R':<12}tech lead (held-out)"
    )
    for session in latest_refined_sessions():
        case = load_case(session["case_id"])
        posting = load_posting(case.checklist)
        tech_checklist = load_checklist(case.checklist)
        results = {}
        for when in ("first_cv", "final_cv"):
            version = case.model_copy(update={"cv": session[when]})
            recruiter = review_recruiter(
                client, version, posting, recruiter_checklist, date.today().year, **settings
            )
            tech_lead = review_case(client, version, tech_checklist, **settings)
            results[when] = (recruiter.score, tech_lead.decision)
        in_loop = session["recruiter_scores"]
        before, after = results["first_cv"], results["final_cv"]
        rows.append((case, in_loop, before, after))
        print(
            f"{case.id:<{CASE_WIDTH}}{case.label:<12}"
            f"{f'{in_loop[0]:.1f}->{in_loop[-1]:.1f}':<12}"
            f"{f'{before[0] or 0:.1f}->{after[0] or 0:.1f}':<12}{before[1]}->{after[1]}"
        )

    rose = [row for row in rows if row[1][-1] > row[1][0]]
    held_rose = sum(1 for row in rose if (row[3][0] or 0) > (row[2][0] or 0))
    at_risk = [row for row in rows if row[0].label in ("borderline", "poor")]
    unchanged = sum(1 for row in at_risk if row[2][1] == row[3][1])
    print()
    print(f"Held-out recruiter score rose where the in-loop one rose: {held_rose}/{len(rose)}")
    print(
        f"Held-out tech-lead decision unchanged on borderline and poor CVs: {unchanged}/{len(at_risk)}"
    )


if __name__ == "__main__":
    main()
