"""The vocabulary bridge, learned rather than hand-written.

Scope #2's blocker in one sentence: the agent asks the data source using the
name on the screen, and the sheet's columns are named something else. A Notepad
record's keys ARE the form's labels, so scope #1 never met this.

Three answers were tried, and the choice was made on measurements taken here,
not on preference:

  1. GradePortalPlugin.COLUMN_MAP - hand-written, and its own docstring calls
     itself scaffolding. It is precisely what scope #2 exists to learn, and it
     goes blind when a variant renames a column.

  2. The LLM. google/gemma-3-4b, the model on this machine, failed three
     framings: asked to copy a value it answered 85 for the Course, the Year and
     the Grade of one row; told the row half of a grid label must be ignored, it
     answered 85 for all three again; asked to pick from a numbered list it
     picked the student's NAME every time - matching on the half it had been
     told to ignore.

  3. The ported matcher. 1,121 parameters, 17 features, Hungarian assignment,
     abstention thresholds. Course -> PROGRAM, Year -> YEAR LEVEL,
     Grade -> FINAL GRADE, all at 1.000, and it declines the two fields that
     have no column at all.

Run:  python -m pytest tests/scope2/test_matched_source.py -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))

pytest.importorskip("torch")

from data_sources.grade_sheet_source import GradeSheetSource  # noqa: E402
from data_sources.matched_source import MatchedFieldSource, infer_type  # noqa: E402

SHEET = REPO / "components" / "scope2" / "data" / "sheets" / "grade_sheet.xlsx"

# Exactly what the portal's own headers declare, and the reason it is spelled
# out rather than simplified: the same checkpoint mapped Course to "No." at
# 0.833 when every field arrived as bare "text" with no placeholder.
PORTAL = [
    dict(label="Course", input_type="text", placeholder="BS Computer Science", maxlength=60),
    dict(label="Year", input_type="number", placeholder="1-5", min="1", max="5", step="1"),
    dict(label="Grade", input_type="number", placeholder="0-100", min="0", max="100",
         step="0.01", required=True),
    dict(label="Remarks", input_type="select", options=["Passed", "Failed"]),
    dict(label="Recommendations", input_type="textarea",
         placeholder="Optional notes for the adviser", maxlength=200),
]


@pytest.fixture(scope="module")
def matched():
    if not SHEET.exists():
        pytest.skip(f"no grade sheet at {SHEET}")
    src = MatchedFieldSource(GradeSheetSource(str(SHEET)), PORTAL)
    src.refresh(0)
    src.mapping()          # one sentence-transformer pass, shared by the module
    return src


# ── the mapping the matcher finds ────────────────────────────────────────────

def test_it_maps_every_column_that_has_a_source(matched):
    mapping = matched.mapping()
    assert mapping.get("Course") == "PROGRAM"
    assert mapping.get("Year") == "YEAR LEVEL"
    assert mapping.get("Grade") == "FINAL GRADE"


def test_it_abstains_on_the_fields_that_have_none(matched):
    """Remarks is derived by the portal from the grade, and Recommendations is
    optional and left blank. Neither has a column, and a confident wrong guess
    would be worse than no answer - abstaining is the correct behaviour, not a
    shortfall."""
    mapping = matched.mapping()
    assert "Remarks" not in mapping
    assert "Recommendations" not in mapping


# ── what the agent actually asks ─────────────────────────────────────────────

def test_it_answers_a_question_phrased_in_the_portals_words(matched):
    """The whole point. This exact string is what the agent asks with, and the
    unwrapped source answers None to all three."""
    assert matched.lookup("Course Abad, Andrea A.") == "BS Information Systems"
    assert matched.lookup("Year 1-5 Abad, Andrea A.") == "2"
    assert matched.lookup("Grade 0-100 Abad, Andrea A.") == "85"


def test_the_unwrapped_source_cannot(matched):
    """States the problem this class exists for, so the test file still explains
    itself if the wrapper is ever removed."""
    plain = GradeSheetSource(str(SHEET))
    plain.refresh(0)
    assert plain.lookup("Course Abad, Andrea A.") is None


def test_it_still_answers_the_sources_own_words(matched):
    """A caller asking in the sheet's vocabulary is not broken by the wrapper."""
    assert matched.lookup("PROGRAM") == "BS Information Systems"


def test_an_abstained_field_is_answered_with_nothing(matched):
    """An abstention IS an answer: the matcher looked at every column and found
    none that feeds this field. Remarks is derived by the portal from the grade;
    Recommendations is optional.

    Reporting them as unanswerable - the first version of this - sent them to
    the LLM instead, which is the one place they must not go. Asked what
    belonged in Remarks, gemma-3-4b replied 85, the grade, into a field the page
    fills for itself. Claiming to answer, with lookup() returning None, is what
    makes the agent treat them as confirmed blank and leave them alone."""
    assert matched.can_answer("Remarks Abad, Andrea A.")
    assert matched.lookup("Remarks Abad, Andrea A.") is None
    assert matched.can_answer("Course Abad, Andrea A.")


def test_a_field_belonging_to_neither_system_is_still_unanswerable(matched):
    """The distinction the above must not erase: a name that is not one of the
    portal's columns at all is genuinely unknown, and should reach the LLM."""
    assert not matched.can_answer("Some Field From Another Application")


def test_the_row_half_selects_the_row_not_the_column(matched):
    """Same column, different students: the mapping is per column and the value
    comes from whichever record is currently loaded."""
    matched.refresh(1)
    assert matched.lookup("Grade 0-100 Aguilar, Benjamin L.") == "96"
    matched.refresh(0)
    assert matched.lookup("Grade 0-100 Abad, Andrea A.") == "85"


def test_the_longest_matching_column_wins():
    """"Year" and "Year 1-5" would both prefix a cell named "Year 1-5 Abad...",
    and picking by iteration order would make the answer depend on dict order."""
    src = MatchedFieldSource(object(), [])
    src._mapping = {"Year": "YEAR LEVEL", "Year 1-5": "YEAR ONE TO FIVE"}
    assert src.column_for("Year 1-5 Abad, Andrea A.") == "YEAR ONE TO FIVE"


def test_a_cell_belonging_to_no_known_column_maps_to_nothing():
    src = MatchedFieldSource(object(), [])
    src._mapping = {"Course": "PROGRAM"}
    assert src.column_for("Something Else Entirely") is None
    assert src.column_for("") is None


# ── the type inference the matcher relies on ─────────────────────────────────

def test_numeric_columns_are_recognised():
    """value-shape features compare a column's samples against a field's type; a
    column declared "text" when it holds numbers weakens every pair it is in."""
    assert infer_type(["85", "96", "74"]) == "number"
    assert infer_type(["1.75", "3.00"]) == "number"
    assert infer_type(["BS Information Systems", "BS Computer Science"]) == "text"
    assert infer_type([]) == "text"


# ── it stays a DataSource ────────────────────────────────────────────────────

def test_it_passes_through_what_it_does_not_override(matched):
    """The agent reaches for read_full_text, headers, samples and record_count;
    wrapping must not hide them."""
    assert matched.record_count() == 50
    assert "PROGRAM" in matched.headers()
    assert "PROGRAM: BS Information Systems" in matched.read_full_text()
