"""Remarks follows the grade, and stops the moment a person disagrees.

Requested directly: "make the remarks automatic based on the grade input unless
changed", with the row left unsaved so it can be double-checked first.

This is a change to what the TASK is, not only to the portal. Remarks was the
one column with no answer anywhere in the source data - the field the agent had
to decide rather than copy. With the portal deriving it, Scope #2's demonstrated
work is three columns of transfer, and the pass/fail rule lives in the
application instead of in anything learned. Recorded here so the trade is
visible later, not rediscovered.

Driven through a real browser rather than asserted against the source, because
the behaviour that matters is what the page does on a keystroke.

Run:  python -m pytest tests/scope2/test_portal_derived_remarks.py -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))

pytest.importorskip("playwright.sync_api")

from executor.scanner import CHROMIUM, variant_url  # noqa: E402


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    if not CHROMIUM.exists():
        pytest.skip(f"no chromium at {CHROMIUM}")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROMIUM), headless=True,
                              args=["--headless=new"])
        try:
            yield b
        finally:
            b.close()


def open_variant(browser, name="v0_base"):
    page = browser.new_page()
    page.goto(variant_url(name))
    page.wait_for_selector("#records-body tr")
    return page


def cell(page, row, key):
    """The control, not its <td> - both carry data-key, the way the portal's own
    CONTROLS selector already accounts for."""
    row_sel = f"#records-body tr[data-row='{row}']"
    return page.locator(
        f"{row_sel} input[data-key='{key}'], "
        f"{row_sel} select[data-key='{key}'], "
        f"{row_sel} textarea[data-key='{key}']")


def type_grade(page, row, value):
    box = cell(page, row, "grade")
    box.fill("")
    box.type(str(value))
    return cell(page, row, "remarks").input_value()


# ── the derivation ───────────────────────────────────────────────────────────

def test_a_passing_grade_fills_remarks(browser):
    page = open_variant(browser)
    try:
        assert type_grade(page, 0, 85) == "Passed"
    finally:
        page.close()


def test_a_failing_grade_fills_the_other_way(browser):
    page = open_variant(browser)
    try:
        assert type_grade(page, 0, 60) == "Failed"
    finally:
        page.close()


def test_the_boundary_is_the_passing_mark_itself(browser):
    page = open_variant(browser)
    try:
        assert type_grade(page, 0, 75) == "Passed"
        assert type_grade(page, 0, 74) == "Failed"
    finally:
        page.close()


def test_clearing_the_grade_clears_the_remark(browser):
    """A row with no grade has no verdict. Leaving a stale Passed behind would
    read as an encoded row that nobody encoded."""
    page = open_variant(browser)
    try:
        assert type_grade(page, 0, 85) == "Passed"
        assert type_grade(page, 0, "") == ""
    finally:
        page.close()


def test_each_row_derives_from_its_own_grade(browser):
    page = open_variant(browser)
    try:
        type_grade(page, 0, 90)
        type_grade(page, 1, 50)
        assert cell(page, 0, "remarks").input_value() == "Passed"
        assert cell(page, 1, "remarks").input_value() == "Failed"
    finally:
        page.close()


# ── "unless changed" ─────────────────────────────────────────────────────────

def test_a_manual_choice_survives_a_later_grade_edit(browser):
    """The override, which is the reason this is not just a computed column."""
    page = open_variant(browser)
    try:
        type_grade(page, 0, 85)
        cell(page, 0, "remarks").select_option("Failed")
        type_grade(page, 0, 95)
        assert cell(page, 0, "remarks").input_value() == "Failed"
    finally:
        page.close()


def test_an_override_is_confined_to_its_own_row(browser):
    page = open_variant(browser)
    try:
        type_grade(page, 0, 85)
        cell(page, 0, "remarks").select_option("Failed")
        assert type_grade(page, 1, 88) == "Passed"
    finally:
        page.close()


def test_clearing_an_override_hands_the_row_back_to_the_grade(browser):
    """The natural way to undo, and it has to actually work or an accidental
    override is permanent for the life of the page."""
    page = open_variant(browser)
    try:
        type_grade(page, 0, 85)
        cell(page, 0, "remarks").select_option("Failed")
        cell(page, 0, "remarks").select_option("")
        assert type_grade(page, 0, 91) == "Passed"
    finally:
        page.close()


# ── the variants ─────────────────────────────────────────────────────────────

def test_the_inverted_scale_is_read_from_the_table_not_assumed(browser):
    """v6b runs 1.00-5.00 where LOWER is better and 3.00 passes. A hardcoded
    >= 75 would mark every row Failed and still look like it was working."""
    page = open_variant(browser, "v6b_scale")
    try:
        assert type_grade(page, 0, "1.75") == "Passed"
        assert type_grade(page, 0, "3.00") == "Passed"
        assert type_grade(page, 0, "3.25") == "Failed"
    finally:
        page.close()


def test_the_wording_comes_from_the_column_that_owns_it(browser):
    """v6a spells them PASSED/FAILED. Writing "Passed" into that select would
    set nothing at all, silently."""
    page = open_variant(browser, "v6a_options")
    try:
        assert type_grade(page, 0, 85) == "PASSED"
        assert type_grade(page, 0, 40) == "FAILED"
    finally:
        page.close()


def test_a_relabelled_column_still_derives(browser):
    """v2 renames what people see and keeps its keys - the derivation joins on
    keys precisely so a cosmetic rename does not break it."""
    page = open_variant(browser, "v2_relabeled")
    try:
        assert type_grade(page, 0, 85) != ""
    finally:
        page.close()


# ── nothing is saved ─────────────────────────────────────────────────────────

def test_deriving_a_remark_does_not_commit_the_row(browser):
    """The point of the request: everything stays reviewable until a person
    presses Save."""
    page = open_variant(browser)
    try:
        type_grade(page, 0, 85)
        committed = page.evaluate("() => window.__portal.row(0).remarks")
        staged = page.evaluate("() => window.__portal.read(0).remarks")
        assert staged == "Passed"
        assert committed in ("", None), "the row was written before anyone saved"
    finally:
        page.close()
