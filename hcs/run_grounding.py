"""Run the grounding judge on the labelled claims, one call per case, and compare with the labels."""

import argparse
from itertools import groupby

from hcs.cases import load_case, load_claims
from hcs.config import API_MAX_TOKENS
from hcs.llm import make_client
from hcs.pipeline import judge_claims
from hcs.profile import build_facts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="the judge's model (default: grounding_judge.MODEL)")
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        default="ollama",
        help="where the model runs (groq: a larger max_tokens for reasoning models)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="ask the provider to enforce the schema"
    )
    parser.add_argument(
        "--retries", type=int, help="repair retries per call (default: grounding_judge.MAX_RETRIES)"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    client = make_client(args.provider)
    settings = {"model": args.model} if args.model else {}
    if args.provider != "ollama":
        settings |= {"max_tokens": API_MAX_TOKENS, "strict": args.strict}
    if args.retries is not None:
        settings["max_retries"] = args.retries
    results = {"dev": [], "heldout": []}
    failed = 0
    seconds = 0.0

    for case_id, group in groupby(load_claims(), key=lambda claim: claim.case):
        claims = list(group)
        facts = build_facts(load_case(case_id).cv)
        record = judge_claims(
            client, [(claim.text, claim.facts) for claim in claims], facts, case_id, **settings
        )
        seconds += record.seconds or 0
        if record.status != "ok":
            failed += 1
            print(f"{case_id}: failed ({record.error})")
            continue
        for claim, judgement in zip(claims, record.judgements, strict=True):
            ok = judgement.supported == claim.supported
            results[claim.split].append((claim, judgement.supported))
            mark = "yes" if ok else "NO "
            print(
                f"{mark} {claim.split:<8}{case_id:<18}expected {claim.supported!s:<6}{claim.text}"
            )
            if judgement.unsupported:
                print(f"      unsupported: {judgement.unsupported}")

    print()
    for split, pairs in results.items():
        right = sum(1 for claim, got in pairs if got == claim.supported)
        lenient = sum(1 for claim, got in pairs if got and not claim.supported)
        strict = sum(1 for claim, got in pairs if claim.supported and not got)
        print(f"{split}: {right}/{len(pairs)} right, {lenient} lenient, {strict} strict")
    print(f"Failed calls: {failed}. Model time: {seconds:.0f}s")


if __name__ == "__main__":
    main()
