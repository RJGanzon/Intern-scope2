"""GradePortalPlugin fills one student's row and nothing else.

The interesting failure this guards is silent and total: address a cell by a
label that does not carry the row, and the plugin encodes fifty students into
row 1. So the assertions are mostly about *which* cell was targeted, not just
that something was typed.

The page here is a real snapshot of v0_base taken through WebObserver, so the
labels under test are the ones aria-labelledby actually produces.

Run:  python -m pytest tests/scope2/test_grade_portal_plugin.py -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))

from agent.task_plugins.grade_portal_plugin import (  # noqa: E402
    COLUMN_MAP, GradePortalPlugin,
)
from data_sources.grade_sheet_source import GradeSheetSource  # noqa: E402
from executor.scanner import variant_url  # noqa: E402

pytest.importorskip("playwright.sync_api")

SHEET = REPO / "components" / "scope2" / "data" / "sheets" / "grade_sheet.xlsx"


class RecordingExecutor:
    """Stands in for ActionExecutor: remembers actions, performs none."""

    def __init__(self):
        self.actions = []

    def execute(self, prediction):
        self.actions.append(prediction)
        return None

    def clicks(self):
        return [a for a in self.actions if a["action_type"] == "click"]

    def typed(self):
        return [a["text"] for a in self.actions if a["action_type"] == "keyboard"]


@pytest.fixture(scope="module")
def state():
    """One real snapshot of the base variant."""
    from observers.web_observer import WebObserver

    obs = WebObserver(headless=True, max_elements=2000)
    assert obs.connect(variant_url("v0_base")), "playwright could not launch"
    try:
        obs._page.wait_for_selector("#records-body input", timeout=15_000)
        return obs.snapshot()
    finally:
        obs.disconnect()


@pytest.fixture
def source():
    return GradeSheetSource(SHEET)


def drive(plugin, state, max_steps=20):
    """Run the plugin to completion, returning the number of steps taken."""
    for step in range(max_steps):
        handled, keep_going = plugin.handle_step(state, step)
        if not keep_going:
            return step + 1
    pytest.fail(f"plugin did not finish within {max_steps} steps")


def test_fills_only_the_target_students_row(state, source):
    """Every click lands on a cell whose label names this student."""
    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=0)
    drive(plugin, state)

    targeted = [e["label"] for e in state["elements"]
                for c in ex.clicks()
                if tuple(c["click_position"]) == GradePortalPlugin._centre(e)]
    assert targeted, "no click matched any element on the page"
    for label in targeted:
        assert "Abad, Andrea A." in label, f"clicked {label!r} - wrong student's row"


def test_types_the_sheet_values_for_that_student(state, source):
    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=0)
    drive(plugin, state)

    source.refresh(0)
    assert sorted(ex.typed()) == sorted([
        source.lookup("PROGRAM"),
        source.lookup("YEAR LEVEL"),
        source.lookup("FINAL GRADE"),
    ])


def test_a_different_record_targets_a_different_row(state, source):
    """Record 49 is Zamora; nothing of Abad's row may be touched."""
    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=49)
    drive(plugin, state)

    targeted = [e["label"] for e in state["elements"]
                for c in ex.clicks()
                if tuple(c["click_position"]) == GradePortalPlugin._centre(e)]
    assert targeted
    for label in targeted:
        assert "Zamora, Zoe T." in label


def test_unmapped_columns_are_left_blank(state, source):
    """Remarks and Recommendations have no source column, so they are never
    typed into - a partially filled row is the correct outcome."""
    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=0)
    drive(plugin, state)

    targeted = [e["label"] for e in state["elements"]
                for c in ex.clicks()
                if tuple(c["click_position"]) == GradePortalPlugin._centre(e)]
    for label in targeted:
        assert not label.startswith("Remarks")
        assert not label.startswith("Recommendations")
    assert len(targeted) == len(COLUMN_MAP)


def test_reports_done_rather_than_looping(state, source):
    """Once the mapped cells are filled the plugin breaks the loop instead of
    refilling them - the (True, False) half of the TaskPlugin contract."""
    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=0)
    steps = drive(plugin, state)

    assert steps == len(COLUMN_MAP) + 1, "expected one step per cell, then done"
    handled, keep_going = plugin.handle_step(state, steps)
    assert (handled, keep_going) == (True, False)


def test_already_correct_cells_are_not_retyped(state, source):
    """A cell whose value already matches the sheet is skipped, so re-running
    over a filled portal is a no-op rather than fifty redundant pastes."""
    source.refresh(0)
    filled = {
        "Course":      source.lookup("PROGRAM"),
        "Year 1-5":    source.lookup("YEAR LEVEL"),
        "Grade 0-100": source.lookup("FINAL GRADE"),
    }
    prefilled = dict(state)
    prefilled["elements"] = [
        dict(e, value=next((v for c, v in filled.items()
                            if (e.get("label") or "").startswith(c)
                            and "Abad, Andrea A." in (e.get("label") or "")),
                           e.get("value")))
        for e in state["elements"]
    ]

    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=0)
    handled, keep_going = plugin.handle_step(prefilled, 0)

    assert (handled, keep_going) == (True, False)
    assert ex.actions == []


def test_row_key_uses_the_split_name_columns(source):
    """The merged NAME OF STUDENT header leaves the given name and initial in
    positional columns; all three are needed to name a portal row."""
    plugin = GradePortalPlugin(RecordingExecutor(), source, record_num=0)
    plugin._load_record()
    assert plugin._row_key == "abad andrea a"


def test_missing_row_is_reported_not_scrolled_forever(source):
    """A student who is not on the roster must end in a fall-through, not an
    endless scroll."""
    empty = {"elements": [], "screen_resolution": [1920, 1080]}
    ex = RecordingExecutor()
    plugin = GradePortalPlugin(ex, source, record_num=0)

    for step in range(plugin._MAX_SCROLLS):
        assert plugin.handle_step(empty, step) == (True, True)
    assert plugin.handle_step(empty, plugin._MAX_SCROLLS) == (False, False)
    assert all(a["action_type"] == "scroll" for a in ex.actions)
