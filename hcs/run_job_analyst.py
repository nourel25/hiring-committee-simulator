"""Turn one posting into a checklist with the job analyst and print it next to the hand-written one."""

from hcs.agents import job_analyst
from hcs.cases import load_checklist, load_posting
from hcs.llm import LLMOutputError, get_structured, make_client

POSTING_NAME = "backend_payments"


def main() -> None:
    posting = load_posting(POSTING_NAME)
    try:
        result = get_structured(
            make_client(),
            job_analyst.build_messages(posting.text),
            job_analyst.analysis_model_for(posting.text),
            model=job_analyst.MODEL,
            temperature=job_analyst.TEMPERATURE,
            seed=job_analyst.SEED,
            max_tokens=job_analyst.MAX_TOKENS,
            max_retries=job_analyst.MAX_RETRIES,
        )
    except LLMOutputError as error:
        print(f"Gave up after {error.attempts} attempt(s): {error}")
        print(f"Last reply: {error.raw_reply!r}")
        return

    analysis = result.parsed
    drafted = job_analyst.to_checklist(analysis, f"{POSTING_NAME}-draft", posting.title)
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

    print("\n== Hand-written answer key")
    answer_key = load_checklist(POSTING_NAME)
    for item in answer_key.items:
        print(f"{item.id} [{item.importance}] {item.text}")


if __name__ == "__main__":
    main()
