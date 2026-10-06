"""The grade sheets open at the top.

Direct report: "Launch Test Tools" on Sheet-to-Portal Matcher opened Excel at
the bottom of the sheet. A workbook stores its scroll position and selection,
and make_sheets.py copies a template whose view came along: SUMMARY (the
active tab) opened with row 65 at the top, past the last student.

Fixed twice over: make_sheets.reset_views() runs before every save, and the
two committed workbooks had only their <sheetView> XML reset, cell values
verified unchanged.

Run:  python -m pytest tests/scope2/test_sheet_views.py -q
"""

import sys
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

REPO = Path(__file__).resolve().parent.parent.parent
SHEETS = REPO / "components" / "scope2" / "data" / "sheets"
sys.path.insert(0, str(SHEETS))

from make_sheets import reset_views  # noqa: E402

WORKBOOKS = ["grade_sheet.xlsx", "grade_sheet_status.xlsx"]


@pytest.mark.parametrize("name", WORKBOOKS)
def test_every_tab_opens_at_the_top_left(name):
    wb = openpyxl.load_workbook(SHEETS / name)
    for ws in wb.worksheets:
        view = ws.sheet_view
        assert view.topLeftCell in (None, "A1"), f"{ws.title} scrolled to {view.topLeftCell}"
        assert all(s.activeCell in (None, "A1") for s in view.selection), ws.title


@pytest.mark.parametrize("name", WORKBOOKS)
def test_the_tab_excel_opens_on_is_still_summary(name):
    """Only the scroll position was reset - not which tab is in front."""
    assert openpyxl.load_workbook(SHEETS / name).active.title == "SUMMARY"


def test_reset_views_undoes_a_scrolled_template(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.sheet_view.topLeftCell = "A65"
    ws.sheet_view.selection[0].activeCell = "I8"
    ws.sheet_view.selection[0].sqref = "I8"
    reset_views(wb)
    path = tmp_path / "out.xlsx"
    wb.save(path)
    view = openpyxl.load_workbook(path).active.sheet_view
    assert view.topLeftCell == "A1"
    assert view.selection[0].activeCell == "A1"


def test_reset_views_keeps_a_frozen_header(tmp_path):
    wb = openpyxl.Workbook()
    wb.active.freeze_panes = "A13"
    reset_views(wb)
    path = tmp_path / "out.xlsx"
    wb.save(path)
    assert openpyxl.load_workbook(path).active.freeze_panes == "A13"
