"""Tests for the knowledge base built from a CV and the grounding checker's rules layer."""

import json

import pytest
from pydantic import ValidationError

from hcs.agents.grounding_judge import build_user_message
from hcs.cases import load_case, load_claims
from hcs.grounding import names_in, rule_problems
from hcs.pipeline import judge_claims
from hcs.profile import build_facts, format_facts, numbers_in
from hcs.runlog import read_runs
from hcs.schemas import judgement_model_for
from tests.fakes import FakeClient, make_response

CV = """
Backend engineer who builds payment APIs.

Software Engineer, logistics platform (2021-present)
- Develop the Python REST API used by 12,000 merchants
- Cut checkout latency by 40% in 2023

Python Developer, agency (2019-2021)
- Built Django sites
"""
FACTS = build_facts(CV)


def test_every_line_is_a_fact_tied_to_its_role():
    assert [(fact.id, fact.role) for fact in FACTS] == [
        ("F1", None),
        ("F2", "Software Engineer, logistics platform (2021-present)"),
        ("F3", "Software Engineer, logistics platform (2021-present)"),
        ("F4", "Software Engineer, logistics platform (2021-present)"),
        ("F5", "Python Developer, agency (2019-2021)"),
        ("F6", "Python Developer, agency (2019-2021)"),
    ]
    assert FACTS[2].text == "Develop the Python REST API used by 12,000 merchants"
    assert format_facts(FACTS[:2]).splitlines()[1].startswith("F2: Software Engineer")


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("used by 12,000 merchants", ["12000"]),
        ("from 1.5 seconds to 300 ms", ["1.5", "300"]),
        ("by 40% in 2023", ["40", "2023"]),
        ("Python, SQL, 3 tools", ["3"]),
    ],
)
def test_numbers_are_stored_as_values(text, numbers):
    assert numbers_in(text) == numbers


def test_names_are_words_with_capitals_not_sentence_starts():
    assert names_in("Built APIs. Deployed them on Kubernetes with PostgreSQL") == [
        "APIs",
        "Kubernetes",
        "PostgreSQL",
    ]


def test_a_claim_built_from_its_facts_passes():
    assert rule_problems("Built a Python REST API used by 12,000 merchants", ["F3"], FACTS) == []


@pytest.mark.parametrize(
    ("claim", "fact_ids", "problem"),
    [
        ("Built APIs", [], "cites no fact"),
        ("Built APIs", ["F9"], "cites unknown facts: F9"),
        ("API used by 20,000 merchants", ["F3"], "number 20000 is not in the cited facts"),
        ("Python REST API on Kubernetes", ["F3"], "name 'Kubernetes' is not in the cited facts"),
        # A real number from another fact is still not supported by the fact the claim cites.
        ("Python REST API, 40% faster", ["F3"], "number 40 is not in the cited facts"),
    ],
)
def test_the_rules_reject_what_the_cited_facts_dont_contain(claim, fact_ids, problem):
    assert problem in rule_problems(claim, fact_ids, FACTS)


def test_a_posting_keyword_in_lower_case_is_padding_too():
    problems = rule_problems("python API with schema design", ["F3"], FACTS, ["schema design"])

    assert problems == ["posting keyword 'schema design' is not in the cited facts"]


@pytest.mark.parametrize("claim", load_claims(), ids=lambda claim: claim.text[:40])
def test_every_labelled_claim_passes_the_rules_so_the_set_measures_the_judge(claim):
    facts = build_facts(load_case(claim.case).cv)

    assert rule_problems(claim.text, claim.facts, facts) == []


def test_the_judge_sees_each_claim_with_only_its_cited_facts():
    message = build_user_message([("API for merchants", ["F3"])], FACTS)

    assert message == (
        "<claims>\nG1: API for merchants\n  F3: Develop the Python REST API used by 12,000 merchants"
        "\n</claims>"
    )


def test_an_unsupported_part_overrules_a_supported_verdict(tmp_path):
    reply = {
        "judgements": [
            {"id": "G2", "support": [], "unsupported": None, "supported": True},
            {"id": "G1", "support": [], "unsupported": "from scratch", "supported": True},
        ]
    }
    client = FakeClient([make_response(json.dumps(reply))])

    record = judge_claims(
        client,
        [("Built it from scratch", ["F3"]), ("API", ["F3"])],
        FACTS,
        "x",
        tmp_path / "r.jsonl",
    )

    assert [(item.id, item.supported) for item in record.judgements] == [
        ("G1", False),
        ("G2", True),
    ]
    assert record.evidence_policy_changes == 1
    assert read_runs(tmp_path / "r.jsonl") == [record]


def test_a_judgement_must_cover_every_claim_once():
    model = judgement_model_for(["G1", "G2"])
    one = {"id": "G1", "support": [], "unsupported": None, "supported": True}

    with pytest.raises(ValidationError):
        model.model_validate({"judgements": [one]})


def test_the_judge_must_quote_its_support_first():
    model = judgement_model_for(["G1"])

    assert list(model.model_json_schema()["$defs"]["ClaimJudgement"]["properties"])[:3] == [
        "id",
        "support",
        "unsupported",
    ]
    assert "support" in model.model_json_schema()["$defs"]["ClaimJudgement"]["required"]
