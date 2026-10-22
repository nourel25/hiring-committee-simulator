"""Tests for the command-line options of hcs.run_compare. No model calls."""

import pytest

from hcs.run_compare import DEFAULT_SEEDS, MODELS, parse_args


def test_no_arguments_means_all_phase_0_models_and_default_seeds():
    args = parse_args([])

    assert args.models == MODELS
    assert args.seeds == DEFAULT_SEEDS


def test_models_and_seeds_can_be_given_on_the_command_line():
    args = parse_args(["gemma3:12b", "--seeds", "0"])

    assert args.models == ["gemma3:12b"]
    assert args.seeds == 0


def test_the_provider_defaults_to_ollama_and_accepts_groq():
    assert parse_args([]).provider == "ollama"
    assert parse_args(["--provider", "groq"]).provider == "groq"


def test_an_unknown_provider_is_refused(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--provider", "openrouter"])


def test_run_eval_defaults_to_the_tech_lead_and_accepts_the_recruiter():
    from hcs.run_eval import parse_args as eval_args

    assert eval_args([]).agent == "tech_lead"
    assert eval_args(["--agent", "recruiter"]).agent == "recruiter"
    assert eval_args(["--agent", "hiring_manager"]).agent == "hiring_manager"
