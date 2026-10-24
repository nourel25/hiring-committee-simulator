"""The Hiring Committee Simulator in the browser.

    streamlit run app.py

Needs Ollama running locally (ollama serve). The grounding judge runs on Groq unless you tick
"keep everything on this machine"; only synthetic CVs should go to the API.
"""

from datetime import date

import streamlit as st

from hcs.cases import Case, load_cases, load_posting
from hcs.config import devils_advocate_endpoint, grounding_endpoint
from hcs.export import cv_to_pdf
from hcs.llm import make_client
from hcs.parsing import (
    Link,
    body_links,
    identity_header,
    links_from_upload,
    short_url,
    split_identity,
    text_from_upload,
)
from hcs.refinement import MAX_ROUNDS, Judge, Session, run_session
from hcs.session import diff
from hcs.tracing import build, summary
from hcs.views import (
    chair_rows,
    check_rows,
    examples_for,
    known_invented,
    load_sessions,
    member_summary,
    objection_rows,
    postings,
    session_members,
    session_rows,
)

st.set_page_config(page_title="Hiring Committee Simulator", layout="wide")


def ollama_ready() -> bool:
    try:
        make_client().models.list()
        return True
    except Exception:  # any connection failure means the same thing to the user
        return False


def show_round(session: Session, index: int) -> None:
    round_ = session.rounds[index]
    verdict = round_.committee.verdict
    left, middle, right = st.columns(3)
    left.metric("Chair's verdict", verdict.verdict)
    middle.metric("Confidence", verdict.confidence)
    right.metric("Dissent", ", ".join(verdict.dissent) or "none")
    for reason in verdict.reasons:
        st.caption(f"Because {reason}.")

    for member, record in session_members(session, index):
        carried = index > 0 and record is session.rounds[0].committee.records.get(member)
        label = member_summary(member, record) + (" (carried from round 1)" if carried else "")
        with st.expander(label, expanded=member == "recruiter"):
            rows = check_rows(record) or objection_rows(record)
            if rows:
                st.dataframe(rows, hide_index=True)
            else:
                st.write("No objections.")

    if verdict.objections:
        st.markdown("**Ranked objections**")
        st.dataframe(chair_rows(verdict), hide_index=True)

    if round_.draft:
        st.markdown("**Refiner**")
        for line in round_.draft.kept:
            st.success(f"{line.text}  — from {', '.join(line.facts)}")
        for item in round_.draft.rejected:
            st.error(f"{item.line.text}  — {'; '.join(item.problems)}")


def new_session() -> None:
    st.header("New session")
    options = postings()
    posting_name = st.selectbox("Job posting", list(options), format_func=options.get)
    with st.expander("The posting"):
        st.text(load_posting(posting_name).text)

    examples = examples_for(posting_name, load_cases(), date.today().year)
    example = st.selectbox(
        "Try an example CV (invented for testing; the label is the verdict we expect)",
        ["(none)", *examples],
    )
    uploaded = st.file_uploader("…or upload your CV", type=["pdf", "docx", "txt"])
    default = ""
    removed: list[str] = []
    links: list[Link] = []
    if uploaded is not None:
        default, removed = split_identity(text_from_upload(uploaded.name, uploaded.getvalue()))
        links = links_from_upload(uploaded.name, uploaded.getvalue())
        st.info(
            "Check the text below before you run: PDFs lose layout. Scanned PDFs have no text to "
            f"read. {len(removed)} line(s) with your name, contact details or links were removed, "
            "so the committee judges the work only; they come back on the PDF at the end."
        )
    elif example != "(none)":
        default = examples[example].cv
    cv = st.text_area(
        "CV: one dated role line, e.g. 'Backend Engineer, shop (2020-present)', then one bullet per line",
        value=default,
        height=260,
    )
    rounds = st.slider("Round limit", 1, MAX_ROUNDS, MAX_ROUNDS)
    picked = examples[example] if example != "(none)" else None
    synthetic = known_invented(cv, picked, uploaded is not None)
    local = st.checkbox(
        "Keep everything on this machine (local grounding judge and devil's advocate, weaker)",
        value=not synthetic,
    )
    consent = True
    if not local:
        st.caption(
            "The devil's advocate sends the CV, and the grounding judge each summary line with its "
            "facts, to Groq."
        )
        if not synthetic:
            st.warning("This isn't an unedited example CV, so it may be a real person's.")
            consent = st.checkbox("Send it to Groq anyway: it is invented, or I accept that")

    if st.button("Run the committee", type="primary", disabled=not cv.strip() or not consent):
        if not ollama_ready():
            st.error("Ollama isn't reachable on localhost:11434. Start it with `ollama serve`.")
            return
        case = Case(
            id=picked.id if picked and uploaded is None else "my-cv",
            label="borderline",  # unknown for a new CV; only the evaluation reads labels
            checklist=posting_name,
            expected_decision="lean_no",
            cv=cv,
            expected={},
        )
        client = make_client()
        # Local: every agent on this machine. Otherwise the grounding judge and the devil's
        # advocate run on their API models (synthetic CVs only).
        judge = Judge(client) if local else grounding_endpoint()
        devil = None if local else devils_advocate_endpoint()
        # Kept beside the session, never in the case: no model reads these lines.
        st.session_state.identity = identity_header(removed, links)
        st.session_state.links = links
        with st.spinner("The committee is meeting: about a minute per round on a laptop GPU…"):
            st.session_state.session = run_session(
                client,
                case,
                date.today().year,
                load_posting(posting_name),
                rounds,
                judge=judge,
                devil=devil,
            )

    session: Session | None = st.session_state.get("session")
    if session is None:
        return
    st.divider()
    st.subheader(f"Stopped: {session.stop_reason}")
    tabs = st.tabs([f"Round {round_.number}" for round_ in session.rounds])
    for index, tab in enumerate(tabs):
        with tab:
            show_round(session, index)

    st.subheader("Gaps")
    st.caption("Not reworded: only new experience closes these.")
    for gap in session.gaps or ["none"]:
        st.write(f"- {gap}")

    st.subheader("Before / after")
    st.code(diff(session.rounds[0].cv, session.final_cv), language="diff")
    edited = st.text_area("Your CV: edit freely before you save it", session.final_cv, height=260)
    header = st.text_area(
        "Name and contact details for the PDF",
        st.session_state.get("identity", ""),
        height=130,
        placeholder="Jane Doe\nBerlin, Germany\njane@example.com\n+49 151 2345 6789\n"
        "linkedin.com/in/janedoe",
    )
    st.caption(
        "Line 1: your full name, set large and centred. Every next line: one contact item, such "
        "as city, email, phone, or a web address like linkedin.com/in/you; they are joined with | "
        "under your name, and emails, phone numbers and addresses become clickable. Filled in "
        "from what was removed from your upload, with link targets read from the file: check them."
    )
    placed = body_links(edited, st.session_state.get("links", []), header)
    if placed:
        st.caption(
            "Links kept from your file: "
            + "; ".join(f"{text} → {short_url(address)}" for text, address in placed)
        )
    text_column, pdf_column = st.columns(2)
    text_column.download_button("Download as text", edited, file_name="cv.txt")
    pdf_column.download_button(
        "Download as PDF",
        cv_to_pdf(edited, header, links=placed),
        file_name="cv.pdf",
        mime="application/pdf",
    )
    st.caption("Nothing is sent anywhere: you review and apply yourself.")


def past_sessions() -> None:
    st.header("Past refinement sessions")
    sessions = load_sessions()
    if not sessions:
        st.info("No sessions yet: run `python -m hcs.run_refinement`.")
        return
    st.dataframe(session_rows(sessions), hide_index=True)
    picked = st.selectbox(
        "Open a session", range(len(sessions)), format_func=lambda i: sessions[i]["case_id"]
    )
    session = sessions[picked]
    st.code(diff(session["first_cv"], session["final_cv"]), language="diff")
    for line in session["kept"]:
        st.success(f"{line['text']}  — from {', '.join(line['facts'])}")
    for item in session["rejected"]:
        st.error(f"{item['line']['text']}  — {'; '.join(item['problems'])}")


def cost_and_latency() -> None:
    st.header("Cost and latency per agent")
    try:
        count = build()
    except FileNotFoundError:
        st.info("No runs logged yet.")
        return
    st.caption(f"{count} logged model calls, from runs/runs.jsonl.")
    rows = [
        {
            "agent": agent,
            "model": model,
            "runs": runs,
            "failed": failed,
            "seconds": round(seconds or 0, 1),
            "tokens in": round(tokens_in or 0),
            "tokens out": round(tokens_out or 0),
        }
        for agent, model, runs, failed, seconds, tokens_in, tokens_out in summary()
    ]
    st.dataframe(rows, hide_index=True)


st.title("Hiring Committee Simulator")
st.caption(
    "A recruiter, a hiring manager, a tech lead and a devil's advocate review a CV; a chair decides; "
    "a refiner improves the top of the CV from the candidate's own facts."
)
pages = {
    "New session": new_session,
    "Past sessions": past_sessions,
    "Cost and latency": cost_and_latency,
}
pages[st.sidebar.radio("Page", list(pages))]()
