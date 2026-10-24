"""Tests for the rule that picks the devil's advocate's model."""

from hcs.run_compare_da import ModelResult, winner


def result(model, failed=0, gaps_found=2, false_alarms=0, explained=3, unexplained=1, seconds=10.0):
    return ModelResult(
        model, 36, failed, gaps_found, 2, false_alarms, explained, unexplained, seconds
    )


def test_only_models_that_read_the_gaps_right_are_eligible():
    assert not result("a", failed=1).eligible
    assert not result("b", gaps_found=1).eligible
    assert not result("c", false_alarms=1).eligible
    assert result("d").eligible


def test_the_best_explained_share_wins_then_fewer_unexplained_then_speed():
    best = result("best", explained=9, unexplained=1)
    assert winner([result("x", explained=3, unexplained=1), best]) is best
    tie_fewer = result("fewer", explained=3, unexplained=1, seconds=50)
    assert winner([result("more", explained=6, unexplained=2), tie_fewer]) is tie_fewer
    fast = result("fast", seconds=5)
    assert winner([result("slow", seconds=40), fast]) is fast


def test_no_eligible_model_means_no_winner():
    assert winner([result("a", false_alarms=1), result("b", failed=2)]) is None
