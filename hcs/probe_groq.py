"""Probe Groq before trusting it as the held-out judge: does it accept our schema and follow it?

Sends one real review request (case poor-1) twice: once as our code sends it today
(json_schema, not strict), once in strict mode. Prints what happened; logs nothing.
Needs GROQ_API_KEY in the environment or in .env.
"""

import copy
import os

from dotenv import load_dotenv
from openai import APIError, OpenAI
from pydantic import ValidationError

from hcs.agents import tech_lead
from hcs.cases import load_case, load_checklist
from hcs.schemas import review_model_for

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"


def forbid_extra_keys(schema: dict) -> dict:
    """A copy of the schema where every object says additionalProperties: false (strict mode needs it)."""
    schema = copy.deepcopy(schema)
    for part in [schema, *schema.get("$defs", {}).values()]:
        if part.get("type") == "object":
            part["additionalProperties"] = False
    return schema


def probe(client: OpenAI, messages: list[dict[str, str]], response_model, strict: bool) -> None:
    """Send one request and report the outcome: error, cut-off, invalid reply, or valid reply."""
    schema = response_model.model_json_schema()
    json_schema = {"name": response_model.__name__, "schema": schema}
    if strict:
        json_schema = {**json_schema, "schema": forbid_extra_keys(schema), "strict": True}

    print(f"--- strict={strict}")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=tech_lead.TEMPERATURE,
            seed=tech_lead.SEED,
            max_tokens=tech_lead.MAX_TOKENS,
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
    except APIError as error:
        print(f"Request refused: {type(error).__name__}: {error}")
        return

    choice = response.choices[0]
    print(f"finish_reason={choice.finish_reason}, tokens out={response.usage.completion_tokens}")
    try:
        review = response_model.model_validate_json(choice.message.content or "")
    except ValidationError as error:
        print(f"Reply does not match the schema: {error.error_count()} error(s)")
        print((choice.message.content or "")[:300])
        return
    print("Valid reply. ids:", [check.id for check in review.checks])
    print("passed:", [check.id for check in review.checks if check.passed])


def main() -> None:
    load_dotenv()
    client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
    case = load_case("poor-1")
    checklist = load_checklist(case.checklist)
    messages = tech_lead.build_messages(case.cv, checklist.items)
    response_model = review_model_for([item.id for item in checklist.items])

    probe(client, messages, response_model, strict=False)
    probe(client, messages, response_model, strict=True)


if __name__ == "__main__":
    main()
