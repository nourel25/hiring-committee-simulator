# Hiring Committee Simulator

A panel of LLM agents reviews a CV against a real job posting the way a hiring team would.
The panel is a recruiter, a hiring manager, a tech lead and a devil's advocate, and a chair decides.
A refiner then improves the CV's presentation using only the candidate's own facts. Every rewritten
line has to pass a grounding check before it reaches the CV.

**[See a recorded session →](https://nourel25.github.io/hiring-committee-simulator/)** It's a static
replay of one run on an invented CV: every reviewer's checks with their quotes, the chair's verdict,
and the refiner's kept and rejected lines with the reasons. Nothing runs in your browser.

Most of it runs on a laptop GPU (RTX 3060, 6 GB) through Ollama. Three judges run on an API (Groq):
the grounding judge and the devil's advocate, which a 27B model does measurably better, and a
held-out judge used only for evaluation. A checkbox keeps everything on the machine.

The interesting part is not the agents but how they were measured. Every prompt and checklist
change was tested on labelled CVs. The answer keys and success criteria were committed **before**
each run, so the git history shows nothing was tuned after the fact.

```mermaid
flowchart LR
    CV[CV] --> F[Facts<br/>code]
    P[Posting] --> R
    CV --> R[Recruiter<br/>code + model]
    CV --> H[Hiring manager<br/>model]
    CV --> T[Tech lead<br/>model]
    CV --> D[Devil's advocate<br/>model rates, code picks]
    R & H & T & D --> C[Chair<br/>code]
    C -->|fixable objections| RF[Refiner<br/>family C]
    F --> RF
    RF --> G[Grounding<br/>rules + judge]
    G -->|grounded lines| CV2[Revised CV] --> R
    C -->|hire / only gaps / 3 rounds| U[User reviews diff<br/>and approves]
```

## Design in one paragraph

**Models judge, code decides.** Each reviewer answers a checklist of yes/no checks. For every check
it must quote the CV before giving its verdict. Code then does the rest:
- it fails any pass whose quote isn't really in the CV;
- it computes the score (must-haves weigh 3, nice-to-haves 1);
- it caps the decision when a must-have fails;
- the chair turns the members' decisions into a verdict.

What code can compute, code computes: years of experience, keywords, title, line length, the facts,
and the numbers and names in a claim. No model ever gives a gut score.

## Results

The 30 labelled CVs, 2 postings and all answer keys are synthetic: written for this project, never
real people. Small sets give rough signals: at 30 cases, ±16 points.

| What | Result | Target (pre-registered) |
|---|---|---|
| Tech lead (qwen2.5:7b), decisions on 30 CVs | 28/30 (11/11 on the cases it was tuned on, 17/19 on the rest) | no regression |
| Second judge family (llama3.1:8b) | 14/19 on unseen CVs | ≥ 17/19 — **not met**, judge B open |
| Committee verdict vs labels, all local | **29/30**, 0 failed calls, 63 s per session | ≥ 23/30 |
| Devil's advocate, four models over 36 cases (rule: 0 failed, every gap found, 0 false highs, then most explained mediums) | qwen3.8-27b on Groq the only eligible model: 16/16 gaps, 0 false highs, 26/36 mediums explained (qwen2.5:7b local: 14/16 gaps, 1 false high) | an eligible model, else it stays local |
| Committee with the devil's advocate on Groq | 29/30 (a different miss: one false high, not repeated at temperature 0), 0/120 failed calls, 59 s per session | ≥ 29/30, 0 poor hired, ≤ 1 failed: met, so it runs in the loop |
| Leniency (hire on a poor CV) | 0/3 | 0 |
| Grounding judge, qwen2.5:7b local, held-out claims | 7/12, 1 lenient | ≥ 10/12, ≤ 1 lenient: **not met** |
| Grounding judge, qwen3.8-27b on Groq, same prompt | **12/12**, 0 lenient (with 3 format retries) | met, so it runs in the loop |
| Refinement v4: poor CVs still no_hire / borderline turned into hire | 5/5 / 0/14 | all / none |
| Refinement v4: good CVs that lost their hire | 0 | none |
| Refinement v4: recruiter score rose on refined CVs | **16/21 (76%)**; held-out CVs 3/4 | ≥ 70% |
| Refinement v4: kept lines a manual read finds unsupported | 0 of 38 | 0 |
| Held-out gap: the API judge's recruiter score rose where the loop's did | 13/16 | ≥ 70% |
| Held-out gap: the API tech lead's decision unchanged on refined borderline and poor CVs | 5/6 (one moved strong_no → lean_no, towards its key, not to a hire) | all: **not met** by the letter |

Refinement took four versions, each with criteria committed before its run:
- **v1** put the summary above the job title, which hid the title from the title check.
- **v2** let the 7B refiner choose the facts. It picked titles and vague bullets.
- **v3** moved fact choice into code. Each round rebuilt the summary from that round's failures,
  so fixes were lost between rounds. Code also once chose the employer's numbers ("4 million
  users") as the candidate's result.
- **v4** keeps earlier lines, and takes numbers only from lines a reviewer has already passed as
  the candidate's own work. It met every criterion.

The other members (hiring manager, tech lead, devil's advocate) judge the experience section, which
the refiner never touches, so refinement cannot move the verdict. That is by design: a rewrite that
looks better should not make a candidate look more qualified.

Phase 0 compared four local 7–8B families as the tech lead. Only qwen2.5:7b passed the gate,
getting every decision right with checklist v3. A local gemma3:12b judged well but took 206 s per
review, so the held-out judge moved to `openai/gpt-oss-120b` on Groq.

### What went wrong, and what it taught

- **Form over meaning.** Small models pass a check when the CV has *something that looks like it*:
  - Python scripts passed as backend services;
  - the employer's user count passed as the candidate's result;
  - buzzwords passed as a description of the work.

  Checks now say what doesn't count.
- **Change what the model sees before what it's told.** Telling the recruiter "a job title alone
  doesn't count" made it quote job titles more often. Removing the dated title lines from its input
  fixed it.
- **Position bias.** Asked for "the three strongest objections", the devil's advocate returned the
  posting's first three requirements in order, 24 times out of 24. Now it rates every requirement
  and code picks.
- **The strict–lenient swing.** Rewording the 7B grounding judge to stop it rejecting true
  paraphrases made it accept inflated claims (lenient errors 1 → 3 of 12). Wording that only moves
  errors from side to side points at the model's limit. The same prompt on a 27B model fixed it.
- **Measure, then move only what needs it.** Only the grounding judge and the devil's advocate went
  to an API, each after a pre-registered comparison. The refiner's problems were design problems,
  which a bigger model wouldn't have fixed, and a bigger job analyst made every line a must-have.
- **Check that a metric can fail.** One devil's-advocate metric was met by almost every CV, good
  ones included, so it measured nothing.

The full journal of decisions is in the commit messages. Each experiment's criteria sit in the
commit before its run.

## Run it

You need Python 3.12 and [Ollama](https://ollama.com). Pull the models first:
- `qwen2.5:7b` for the judges;
- `mistral:7b` for the refiner.

The grounding judge, the devil's advocate and the held-out judge run on Groq: put a key in `.env`
(see `.env.example`). Only synthetic CVs go there. The `--local-devil` and `--local-judge` flags keep a
run on the machine.

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
python -m pip install -r requirements-dev.txt
pytest                                              # no model needed
```

The evaluation scripts need Ollama running (`ollama serve`):

```bash
python -m hcs.run_eval --agent tech_lead          # one reviewer on every labelled CV
python -m hcs.run_committee --transcript          # the whole committee, with meeting transcripts
python -m hcs.run_grounding                       # the grounding judge on labelled claims
python -m hcs.run_refinement                      # refinement sessions on every labelled CV
python -m hcs.session my_cv.txt --posting backend_payments   # your own plain-text CV
python -m hcs.tracing                             # cost and latency per agent, from the run log
```

A browser UI (Streamlit):

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

It has three pages:
- **New session.** Paste or upload a CV (PDF, Word or text), or start from a labelled example, and
  pick a posting. An upload is read into the plain format and shown for you to correct before
  anything runs:
  - bullets become `- ` and date ranges become `(2021-present)`;
  - wrapped lines are joined;
  - your name, email, phone and links are removed, so the committee judges the work only.

  Scanned PDFs have no text to read, and two-column layouts come out in the wrong order.
  For every round you see:
  - each member's checks with their quotes, and the devil's advocate's objections;
  - the chair's verdict, confidence, dissent and ranked objections;
  - the refiner's kept and rejected lines, with the reasons.

  At the end you get the gaps, a before/after diff, and an editable CV to download.
- **Past sessions.** The logged refinement runs.
- **Cost and latency.** Calls, failures, seconds and tokens per agent and model, from the run log.

A checkbox keeps the grounding judge and the devil's advocate local, so nothing leaves the machine.
It is ticked for you when you upload a file.

The held-out judge needs a Groq key in `.env` (see `.env.example`):

```bash
python -m hcs.run_held_out
```

## Layout

| Path | What |
|---|---|
| `hcs/agents/` | One file per agent: prompt, model settings, message builder |
| `hcs/pipeline.py` | One review: call, policies, score, decision, log |
| `hcs/policies.py` | Evidence policy, truth check, scoring, the must-have gate |
| `hcs/screening.py` | The recruiter's code checks |
| `hcs/chair.py`, `hcs/committee.py` | The verdict rule and one committee round |
| `hcs/profile.py`, `hcs/grounding.py` | Facts and the grounding rules |
| `hcs/refinement.py` | The refinement loop |
| `hcs/evaluation.py` | Comparing runs with answer keys |
| `data/postings/`, `data/checklists/`, `data/cases/` | Postings, reviewed checklists, labelled CVs with keys |
| `hcs/demo.py`, `docs/` | Record one session and build the static replay page (GitHub Pages) |
| `runs/` | The run log (every call, replayable) and traces; not committed |

## Limitations

- The answer keys were written by the same author as the prompts, and most of the later ones have
  not had a second reader.
- The sets are small: 42 CVs and 34 claims.
- Only one model family passed as a judge. The members differ in what they see and check, not in
  model family.
- Refinement results across runs are not exactly repeatable. Groq doesn't guarantee seeds, and one
  local committee verdict changed between two runs at temperature 0.
- The refiner only writes a summary at the top of the CV. Real CVs need more, but a whole-CV rewrite
  gives a 7B model room to invent.
- Every labelled CV is invented and has one claim per line. The first real CVs showed shapes the
  set doesn't cover yet: compound bullets, skills blocks, dated degrees and an existing summary.
  Parsing and year counting were fixed for them; the tech lead's REST check still reads them too
  literally.
- These parts of the design are not built yet:
  - the company researcher (each posting has a hand-written brief);
  - logging real outcomes.

## License

MIT. See [LICENSE](LICENSE).
