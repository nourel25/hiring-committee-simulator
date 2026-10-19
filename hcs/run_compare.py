"""Compare local models on the labelled cases: decisions, validity, leniency, agreement, noise, speed."""

import argparse

from hcs.agents import tech_lead
from hcs.cases import load_cases, load_checklist
from hcs.evaluation import MODEL_HEADER, compare, format_model_row, format_row, summarize_model
from hcs.llm import gpu_share, make_client
from hcs.pipeline import review_case

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "granite4.1:8b"]
TEMPERATURE = 0.7
DEFAULT_SEEDS = 5
# API reasoning models think before the JSON; the probe used up to 915 of 1000 tokens.
API_MAX_TOKENS = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read which models to compare and how many noisy repeats to run per case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=MODELS,
        help="Ollama model names (default: the four Phase 0 families)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"repeats per case at temperature {TEMPERATURE}, seeds 1..N (0 = temperature 0 only)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        default="ollama",
        help="where the models run (groq: strict schema, larger max_tokens, no GPU share)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    seeds = list(range(1, args.seeds + 1))
    api = args.provider != "ollama"
    settings = {
        "max_tokens": API_MAX_TOKENS if api else tech_lead.MAX_TOKENS,
        "strict": api,
    }
    client = make_client(args.provider)
    cases = [(case, load_checklist(case.checklist)) for case in load_cases()]
    summaries = []

    for model in args.models:
        print(f"== {model}")
        zero_reports = []
        noisy_reports = []
        for case, checklist in cases:
            record = review_case(client, case, checklist, model=model, **settings)
            zero_reports.append(compare(case, record))
            print(f"temp 0  {format_row(zero_reports[-1])}")
            for seed in seeds:
                record = review_case(
                    client,
                    case,
                    checklist,
                    model=model,
                    temperature=TEMPERATURE,
                    seed=seed,
                    **settings,
                )
                noisy_reports.append(compare(case, record))
                print(f"seed {seed}  {format_row(noisy_reports[-1])}")
        gpu = None if api else gpu_share(model)
        summaries.append(summarize_model(model, zero_reports, noisy_reports, gpu))
        print()
    print("dec@0: decisions right at temperature 0. failed, retried, sec: all runs.")
    print(f"lenient, agree, gap, range: {len(seeds)} runs per case at temperature {TEMPERATURE}.")
    print(MODEL_HEADER)
    for summary in summaries:
        print(format_model_row(summary))


if __name__ == "__main__":
    main()
