"""Tests for the command-line options of hcs.run_job_analyst. No model calls."""

from hcs.agents import job_analyst
from hcs.run_job_analyst import DEFAULT_POSTING, draft_name, parse_args


def test_no_arguments_means_the_local_job_analyst_without_eval():
    args = parse_args([])

    assert args.model == job_analyst.MODEL
    assert args.posting == DEFAULT_POSTING
    assert args.provider == "ollama"
    assert args.eval is False
    assert args.save is False


def test_model_provider_and_eval_can_be_given_on_the_command_line():
    args = parse_args(
        [
            "openai/gpt-oss-120b",
            "--posting",
            "data_logistics",
            "--provider",
            "groq",
            "--save",
            "--eval",
        ]
    )

    assert args.model == "openai/gpt-oss-120b"
    assert args.posting == "data_logistics"
    assert args.provider == "groq"
    assert args.eval is True
    assert args.save is True


def test_a_draft_name_is_a_safe_file_name():
    assert draft_name("backend_payments", "qwen2.5:7b") == "backend_payments-draft-qwen2.5-7b"
    assert draft_name("x", "openai/gpt-oss-120b") == "x-draft-openai-gpt-oss-120b"
