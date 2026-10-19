"""Tests for the command-line options of hcs.run_compare. No model calls."""

from hcs.run_compare import DEFAULT_SEEDS, MODELS, parse_args


def test_no_arguments_means_all_phase_0_models_and_default_seeds():
    args = parse_args([])

    assert args.models == MODELS
    assert args.seeds == DEFAULT_SEEDS


def test_models_and_seeds_can_be_given_on_the_command_line():
    args = parse_args(["gemma3:12b", "--seeds", "0"])

    assert args.models == ["gemma3:12b"]
    assert args.seeds == 0
