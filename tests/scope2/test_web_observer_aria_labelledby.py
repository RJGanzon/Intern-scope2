"""The web observer names a sheet cell by its aria-labelledby references.

The grade portal is a sheet: no <label for> is possible, so every input cell
carries aria-labelledby pointing at its column header *and* its row's student
name cell (mocksite/shared/portal.js). An observer that reads aria-label and
stops falls through to the name attribute, which portal.js deliberately shares
down a whole column - so all fifty rows of Grade arrive called "grade" and the
agent cannot tell row 1 from row 50. Every row looks like a duplicate of the
first, and there is nothing in the state to disambiguate them.

This is rule 3 of the cascade in labeling/resolve.py ("aria-label /
aria-labelledby"), whose reference reading is executor/extract_context.js.

Run:  python -m pytest tests/scope2/test_web_observer_aria_labelledby.py -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))

from executor.scanner import variant_url  # noqa: E402
from observers.web_observer import WebObserver  # noqa: E402

pytest.importorskip("playwright.sync_api")

ROSTER_ROWS = 50


@pytest.fixture(scope="module")
def v0_elements():
    """One snapshot of the base variant, straight off the observer."""
    obs = WebObserver(headless=True, max_elements=2000)
    assert obs.connect(variant_url("v0_base")), "playwright could not launch"
    try:
        obs._page.wait_for_selector("#records-body input", timeout=15_000)
        return obs.snapshot()["elements"]
    finally:
        obs.disconnect()


def labels_of(elements, control_type):
    return [e["label"] for e in elements if e["type"] == control_type]


def test_every_input_cell_gets_its_own_name(v0_elements):
    """One distinct label per cell, not per column."""
    labels = labels_of(v0_elements, "editcontrol")
    assert labels, "no editcontrols captured - check the observer's type mapping"
    assert len(set(labels)) == len(labels), (
        f"{len(labels)} input cells share only {len(set(labels))} labels; "
        "aria-labelledby is not being resolved"
    )


def test_a_column_is_named_once_per_row(v0_elements):
    """Each input column contributes one label per roster row."""
    labels = labels_of(v0_elements, "editcontrol")
    for column in ("Course", "Year 1-5", "Grade 0-100"):
        in_column = [l for l in labels if l.startswith(column + " ")]
        assert len(set(in_column)) == ROSTER_ROWS, (
            f"column {column!r}: {len(set(in_column))} distinct labels, "
            f"expected {ROSTER_ROWS}"
        )


def test_the_name_is_the_header_then_the_row(v0_elements):
    """Referenced ids are concatenated in the order aria-labelledby gives them:
    column header first, then the row's student-name cell."""
    grades = sorted(
        l for l in labels_of(v0_elements, "editcontrol")
        if l.startswith("Grade 0-100 ")
    )
    assert grades[0] == "Grade 0-100 Abad, Andrea A."


def test_selects_are_named_per_row_too(v0_elements):
    """The Remarks column is a <select>, so it lands in comboboxcontrol - the
    same aria-labelledby applies and must be read there as well."""
    labels = labels_of(v0_elements, "comboboxcontrol")
    assert len(set(labels)) == ROSTER_ROWS, (
        f"{len(labels)} selects share {len(set(labels))} labels"
    )


def test_aria_label_still_wins_over_labelledby(v0_elements):
    """Rule 3 tries aria-label first. The row checkboxes carry one, and must
    keep it rather than picking up a reference-built name."""
    labels = labels_of(v0_elements, "checkboxcontrol")
    assert "Select all rows" in labels
    assert "Select Abad, Andrea A." in labels


def test_v4_has_no_labelledby_and_must_not_crash():
    """v4_unassociated drops aria-labelledby entirely. The helper returns "" and
    the cascade falls through - no exception, and elements still arrive."""
    obs = WebObserver(headless=True, max_elements=2000)
    assert obs.connect(variant_url("v4_unassociated"))
    try:
        obs._page.wait_for_selector("#records-body input", timeout=15_000)
        elements = obs.snapshot()["elements"]
    finally:
        obs.disconnect()

    assert labels_of(elements, "editcontrol"), "v4 produced no input elements"
