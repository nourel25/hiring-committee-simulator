"""Tests for hcs.config: the model choices from Phase 0 are the ones the code actually uses."""

import dataclasses

import pytest

from hcs import config
from hcs.agents import refiner, tech_lead


def test_the_tech_lead_uses_judge_a():
    assert tech_lead.MODEL == config.JUDGE_A.model
    assert config.JUDGE_A.provider == "ollama"


def test_the_held_out_judge_is_another_family_outside_the_local_loop():
    assert config.HELD_OUT_JUDGE.provider != "ollama"
    assert config.HELD_OUT_JUDGE.model.split(":")[0] != config.JUDGE_A.model.split(":")[0]
    assert "qwen" not in config.HELD_OUT_JUDGE.model.lower()


def test_a_model_choice_cannot_be_changed_by_accident():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.JUDGE_A.model = "llama3.1:8b"


def test_the_grounding_judge_is_neither_the_refiner_nor_the_held_out_judge():
    # The refiner must not grade its own family's writing, and the held-out judge stays independent.
    assert refiner.MODEL.split(":")[0] not in config.GROUNDING_JUDGE.model
    assert config.GROUNDING_JUDGE.model != config.HELD_OUT_JUDGE.model
    assert "gpt-oss" not in config.GROUNDING_JUDGE.model
