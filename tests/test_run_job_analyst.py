"""Tests for the command-line options of hcs.run_job_analyst. No model calls."""

from hcs.agents import job_analyst
from hcs.run_job_analyst import parse_args


def test_no_arguments_means_the_local_job_analyst_without_eval():
    args = parse_args([])

    assert args.model == job_analyst.MODEL
    assert args.provider == "ollama"
    assert args.eval is False


def test_model_provider_and_eval_can_be_given_on_the_command_line():
    args = parse_args(["openai/gpt-oss-120b", "--provider", "groq", "--eval"])

    assert args.model == "openai/gpt-oss-120b"
    assert args.provider == "groq"
    assert args.eval is True
