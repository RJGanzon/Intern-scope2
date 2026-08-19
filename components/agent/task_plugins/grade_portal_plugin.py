"""
components/agent/task_plugins/grade_portal_plugin.py
=====================================================
GradePortalPlugin - Scope #2: encode one student's grade sheet row into the
web grade portal.

Where FormFillerPlugin walks a desktop form tab by tab, this walks a *sheet*:
one HTML table, fifty rows, one row per student. That difference drives the
whole design. There is no "focus the first empty field and Tab through it" -
tabbing across a fifty-row grid would wander into other students' rows. The
plugin instead addresses each cell directly by its accessible name, which
WebObserver now resolves from aria-labelledby:

    "Grade 0-100 Abad, Andrea A."
     ^^^^^^^^^^^ column          ^^^^^^^^^^^^^^^^ row

Both halves come from the DOM: the column header cell and the row's student
name cell. That is the only thing making row 1 distinguishable from row 50, so
this plugin does not work against a WebObserver that stops at aria-label.

Division of labour with the transformer
---------------------------------------
The model in intelligence/model/transformer.py predicts an *action* - a verb,
a click point, a key count. It never predicts a value; predict() has no text
output at all. Values come from the data source, keyed by the label of the
field being filled (see FormFillerPlugin._lookup_field, which does exactly this
against NotepadDataSource).

That works upstream because a Notepad record is written in the form's own
vocabulary, so the form's label *is* the record key. Scope #2 is the first case
where it is not: the grade book says PROGRAM and FINAL GRADE, the portal says
Course and Grade 0-100. GradeSheetSource deliberately refuses to bridge that -
its docstring calls a synonym "the matcher's decision to make, not a lookup's"
and returns None rather than guess.

So the bridge lives here, in COLUMN_MAP, because TaskPlugin is the documented
home for task-specific logic and this keeps both neighbours honest: the data
source stays a dumb reader, and the transformer keeps predicting only actions.

    COLUMN_MAP is scaffolding, not the destination.

It is hand-written, and hand-writing it is the very thing Scope #2 exists to
learn. It is also brittle by construction: v2_relabeled renames Grade 0-100 to
"Final Rating 0-100" and Course to "Degree Program", and this map goes blind on
that variant. Recorded demonstrations are what replace it - once they exist,
the mapping is a learned association rather than a literal in this file.

Usage
-----
    plugin = GradePortalPlugin(executor, GradeSheetSource(path), record_num=0)
    handled, keep_going = plugin.handle_step(state, step_idx)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base_plugin import TaskPlugin

logger = logging.getLogger(__name__)


# The sheet->portal bridge. Keys are the portal's column label exactly as the
# header cell reads (WebObserver hands back header + row, and the header half
# is matched as a prefix). Values are the grade-book column, resolved through
# GradeSheetSource._resolve_column, so spelling drift on the sheet side is
# still tolerated - only the *meaning* is pinned here.
#
# Remarks and Recommendations are absent on purpose: neither has a source
# column in the grade book. Remarks (Passed/Failed) is derivable from the
# grade, but deriving it would mean inventing data the sheet never stated, so
# the plugin leaves both blank and lets a demonstration show how a human fills
# them.
COLUMN_MAP: Dict[str, str] = {
    "Course":      "PROGRAM",
    "Year 1-5":    "YEAR LEVEL",
    "Grade 0-100": "FINAL GRADE",
}

# The grade book's merged "NAME OF STUDENT" header spans three columns, so
# pandas names only the first and leaves the rest as positional placeholders.
# The surname alone cannot pick a portal row ("Abad, Andrea A."), so the given
# name and middle initial have to be read out of those placeholders.
_SURNAME_COLUMN = "NAME OF STUDENT"
_GIVEN_NAME_COLUMNS = ("Unnamed: 3", "Unnamed: 4")

# Cells this plugin will type into. A row checkbox is a control, not a field.
_FILLABLE_TYPES = {"editcontrol", "comboboxcontrol"}


def _fold(text: Any) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Comparing names across a spreadsheet and a DOM means comparing "ABAD" with
    "Abad," - the comma, the case and the middle-initial period are all noise.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


class GradePortalPlugin(TaskPlugin):
    """Fill one student's row on the grade portal.

    Parameters
    ----------
    executor    : ActionExecutor.
    data_source : GradeSheetSource (or anything with refresh/lookup).
    record_num  : Which student, **0-based** - this indexes the data source
                  directly, and GradeSheetSource.refresh() is 0-based.
                  FormFillerPlugin's record_num is 1-based; they are different
                  numbers for the same idea, so do not copy one into the other.
    step_delay  : Seconds to settle after an action.
    observe_fn  : () -> state dict. Set by the agent; used to re-read the page
                  after scrolling, when the target row was off-screen.
    """

    def __init__(
        self,
        executor,
        data_source,
        record_num: int = 0,
        step_delay: float = 0.6,
        observe_fn=None,
    ) -> None:
        self._executor    = executor
        self._data_source = data_source
        self._record_num  = record_num
        self.step_delay   = step_delay
        self._observe_fn  = observe_fn

        self._record_loaded = False
        self._row_key: str  = ""
        # Cell labels already typed into this run. The portal echoes a typed
        # value straight back into the DOM, so `value` is normally enough to
        # tell done from pending - but a value the portal reformats (85 -> 85.00)
        # would otherwise look permanently wrong and loop forever.
        self._filled: set   = set()

        # Scrolling is capped: a row that never appears means the roster does
        # not contain this student, and scrolling forever hides that.
        self._scrolls        = 0
        self._MAX_SCROLLS    = 12

    # ── TaskPlugin contract ───────────────────────────────────────────────

    def handle_step(self, state: Dict[str, Any], step_idx: int) -> Tuple[bool, bool]:
        """Fill one cell per step.

        Returns (True, True) after acting, (True, False) when the row is
        complete, and (False, False) only when the plugin cannot tell what to
        do - which hands the step to the transformer rather than guessing.
        """
        if not self._record_loaded:
            self._load_record()

        if not self._row_key:
            logger.warning("GradePortal: no student name for record %d - "
                           "cannot identify a portal row.", self._record_num)
            return (False, False)

        pending = self._pending_cells(state)

        if pending is None:
            # The row is not on screen at all. Scroll toward it and re-look.
            if self._scrolls >= self._MAX_SCROLLS:
                logger.warning("GradePortal: row %r never appeared after %d scrolls "
                               "- is this student on the roster?",
                               self._row_key, self._scrolls)
                return (False, False)
            self._scrolls += 1
            self._scroll_down(state)
            return (True, True)

        if not pending:
            logger.info("GradePortal: record %d (%s) complete.",
                        self._record_num, self._row_key)
            return (True, False)

        element, value = pending[0]
        self._fill_cell(element, value)
        return (True, True)

    def on_record_start(self, record_num: int, state: Dict[str, Any]) -> None:
        """Move to another student. Clears every per-row cache."""
        self._record_num    = record_num
        self._record_loaded = False
        self._row_key       = ""
        self._filled        = set()
        self._scrolls       = 0

    # ── record ────────────────────────────────────────────────────────────

    def _load_record(self) -> None:
        self._data_source.refresh(self._record_num)
        self._row_key = self._student_row_key()
        self._record_loaded = True
        logger.info("GradePortal: record %d -> row %r",
                    self._record_num, self._row_key)

    def _student_row_key(self) -> str:
        """The folded name used to recognise this student's row.

        Built from the three columns the merged header covers, in sheet order:
        surname, given name, middle initial -> "abad andrea a". The portal
        writes the same three parts as "Abad, Andrea A.", which folds to the
        same string.
        """
        record = self._data_source.get_all()
        parts: List[str] = [record.get(_SURNAME_COLUMN, "")]
        parts.extend(record.get(c, "") for c in _GIVEN_NAME_COLUMNS)
        return _fold(" ".join(p for p in parts if p))

    # ── the portal ────────────────────────────────────────────────────────

    def _pending_cells(self, state: Dict[str, Any]
                       ) -> Optional[List[Tuple[Dict[str, Any], str]]]:
        """Cells of this student's row that still need a value.

        Returns None - distinct from an empty list - when the row is not in the
        snapshot at all, because "not on screen" and "already done" call for
        opposite responses.
        """
        row_cells = self._cells_in_row(state)
        if not row_cells:
            return None

        pending: List[Tuple[Dict[str, Any], str]] = []
        for column, element in row_cells:
            label = element.get("label") or ""
            if label in self._filled:
                continue
            value = self._value_for(column)
            if not value:
                continue  # no source column (Remarks, Recommendations) - leave blank
            if _fold(element.get("value")) == _fold(value):
                continue  # already correct
            pending.append((element, value))
        return pending

    def _cells_in_row(self, state: Dict[str, Any]
                      ) -> List[Tuple[str, Dict[str, Any]]]:
        """(portal column, element) for every mapped cell belonging to this row.

        A cell's label is "<column> <student>", so a cell is this student's when
        the label starts with a mapped column and the remainder folds to the row
        key. Both halves must match: matching on the name alone would sweep in
        the row checkbox and the Remarks select.
        """
        found = []
        for element in state.get("elements", []):
            if element.get("type") not in _FILLABLE_TYPES:
                continue
            label = element.get("label") or ""
            for column in COLUMN_MAP:
                if not label.startswith(column):
                    continue
                if _fold(label[len(column):]) == self._row_key:
                    found.append((column, element))
                break
        return found

    def _value_for(self, column: str) -> str:
        """The sheet value for a portal column, via COLUMN_MAP.

        Unmapped columns return "" - the plugin fills what the grade book
        actually contains and nothing else.
        """
        sheet_column = COLUMN_MAP.get(column)
        if not sheet_column:
            return ""
        return self._data_source.lookup(sheet_column) or ""

    # ── actions ───────────────────────────────────────────────────────────

    def _fill_cell(self, element: Dict[str, Any], value: str) -> None:
        """Click the cell, then paste. The executor's text path selects all
        before pasting, so a retried step overwrites rather than appends."""
        label = element.get("label") or ""
        centre = self._centre(element)
        if centre is None:
            logger.warning("GradePortal: %r has no usable bbox - skipping.", label)
            self._filled.add(label)
            return

        logger.info("GradePortal: %r <- %r", label, value)
        self._executor.execute({"action_type": "click", "click_position": list(centre)})
        time.sleep(self.step_delay)
        self._executor.execute({
            "action_type": "keyboard",
            "text":        value,
            "key_count":   len(value),
        })
        self._filled.add(label)
        time.sleep(self.step_delay)

    def _scroll_down(self, state: Dict[str, Any]) -> None:
        """Scroll the sheet toward the target row, from the middle of the page
        so the wheel event lands on the table rather than the page margin."""
        width, height = (state.get("screen_resolution") or [1920, 1080])[:2]
        self._executor.execute({
            "action_type":    "scroll",
            "click_position": [width // 2, height // 2],
            "direction":      "down",
            "clicks":         3,
        })
        time.sleep(self.step_delay)

    @staticmethod
    def _centre(element: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        bbox = element.get("bbox") or []
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        return ((x1 + x2) // 2, (y1 + y2) // 2)
