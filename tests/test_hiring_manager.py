"""Tests for the hiring manager's checks. No model calls."""

import pytest

from hcs.agents.hiring_manager import fill_checks
from hcs.cases import HiringManagerTargets, load_checklist, load_posting
from hcs.schemas import ChecklistItem


def test_the_targets_are_written_into_the_checks():
    items = [
        ChecklistItem(id="HM1", text="Has worked on {system}; X does not count", importance="nice"),
        ChecklistItem(id="HM2", text="Has worked towards {goals}", importance="nice"),
    ]
    targets = HiringManagerTargets(system="a payouts service", goals=["fewer failures", "speed"])

    filled = fill_checks(items, targets)

    assert [item.text for item in filled] == [
        "Has worked on a payouts service; X does not count",
        "Has worked towards fewer failures or speed",
    ]
    # The checklist itself is unchanged: the same file serves every posting.
    assert items[0].text == "Has worked on {system}; X does not count"


@pytest.mark.parametrize("posting_name", ["backend_payments", "data_logistics"])
def test_every_placeholder_is_filled_for_the_real_postings(posting_name):
    checklist = load_checklist("hiring_manager")

    filled = fill_checks(checklist.items, load_posting(posting_name).hiring_manager)

    assert all("{" not in item.text and "}" not in item.text for item in filled)
    assert [item.id for item in filled] == [item.id for item in checklist.items]
