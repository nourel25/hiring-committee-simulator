"""The grounding checker's rules layer: code checks that a claim cites real facts and that every number
and name in it comes from those facts. The judge layer (a model) then reads what code can't: whether
the meaning is supported. See hcs/agents/grounding_judge.py.
"""

import re

from hcs.profile import Fact, numbers_in
from hcs.screening import keywords_found

# Words a name check would flag only because they start a sentence or are a pronoun.
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#]*(?:[.\-][A-Za-z0-9+#]+)*")
SENTENCE_END = re.compile(r"[.!?:;]\s*$")


def names_in(text: str) -> list[str]:
    """Words that look like names (tools, companies, acronyms): a capital letter after the first
    character (PostgreSQL, REST), or a capital first letter anywhere but the start of a sentence."""
    names = []
    for match in TOKEN.finditer(text):
        word = match.group()
        before = text[: match.start()]
        starts_sentence = not before.strip() or bool(SENTENCE_END.search(before))
        inner_capital = any(char.isupper() for char in word[1:])
        if word == "I":
            continue
        if inner_capital or (word[0].isupper() and not starts_sentence):
            names.append(word)
    return list(dict.fromkeys(names))


def rule_problems(
    claim: str, fact_ids: list[str], facts: list[Fact], keywords: list[str] = ()
) -> list[str]:
    """Why code rejects a claim, or [] when the rules pass. keywords: the posting's, so a keyword
    the cited facts don't contain is reported as padding even when written in lower case."""
    by_id = {fact.id: fact for fact in facts}
    if not fact_ids:
        return ["cites no fact"]
    unknown = [fact_id for fact_id in fact_ids if fact_id not in by_id]
    if unknown:
        return [f"cites unknown facts: {', '.join(unknown)}"]

    cited = [by_id[fact_id] for fact_id in fact_ids]
    cited_text = "\n".join(fact.text for fact in cited)
    cited_numbers = {number for fact in cited for number in fact.metrics}
    problems = [
        f"number {number} is not in the cited facts"
        for number in dict.fromkeys(numbers_in(claim))
        if number not in cited_numbers
    ]
    problems += [
        f"name '{name}' is not in the cited facts"
        for name in names_in(claim)
        if not keywords_found(cited_text, [name])
    ]
    flagged = {name.casefold() for name in names_in(claim)}
    problems += [
        f"posting keyword '{keyword}' is not in the cited facts"
        for keyword in keywords_found(claim, list(keywords))
        if keyword.casefold() not in flagged and not keywords_found(cited_text, [keyword])
    ]
    return problems
