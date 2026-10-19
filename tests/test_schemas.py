"""Tests for hcs.schemas.review_model_for: check ids locked to the checklist. No model calls."""

import pytest
from pydantic import ValidationError

from hcs.schemas import Review, review_model_for

REPLY = '{"checks": [{"id": "%s", "check": "c", "evidence": null, "passed": false}]}'


def test_the_schema_lists_only_the_checklist_ids():
    schema = review_model_for(["TL1", "TL2"]).model_json_schema()

    assert schema["$defs"]["Check"]["properties"]["id"]["enum"] == ["TL1", "TL2"]


def test_the_field_order_is_unchanged():
    schema = review_model_for(["TL1"]).model_json_schema()

    assert list(schema["$defs"]["Check"]["properties"]) == ["id", "check", "evidence", "passed"]


def test_a_known_id_is_accepted_and_is_still_a_review():
    review = review_model_for(["TL1"]).model_validate_json(REPLY % "TL1")

    assert isinstance(review, Review)


def test_an_id_with_extra_text_is_rejected():
    with pytest.raises(ValidationError):
        review_model_for(["TL1"]).model_validate_json(REPLY % "TL1: Has built services")
