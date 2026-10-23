"""Tests for the session command: the before/after diff and the report the user reads."""

from hcs.session import diff, parse_args


def test_the_diff_shows_the_new_summary_and_the_removed_lines():
    before = "Passionate team player\n\nEngineer, shop (2020-present)\n- Built APIs"
    after = "Builds APIs for a shop\n\nEngineer, shop (2020-present)\n- Built APIs"

    lines = diff(before, after).splitlines()

    assert "-Passionate team player" in lines and "+Builds APIs for a shop" in lines
    assert diff(before, before) == "(no change)"


def test_the_session_defaults_to_the_design_round_limit():
    args = parse_args(["cv.txt"])

    assert (args.posting, args.rounds, args.yes) == ("backend_payments", 3, False)
