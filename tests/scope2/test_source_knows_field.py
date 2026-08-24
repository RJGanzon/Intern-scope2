"""A source that cannot identify a field is not saying the field is blank.

From the first live run without the plugin. The log said the same thing 109
times:

    [OPT2] batch fast-fill 'Course Abad, Andrea A.' -> confirmed blank, Tab past
           (no transformer, no LLM, no click)

The agent asked the grade book for a field named after the portal's cell, the
grade book's columns are PROGRAM and FINAL GRADE, so every lookup missed - and a
miss was read as the record asserting the field is empty. It tabbed through the
entire sheet without filling a single cell, and neither the transformer nor the
LLM was consulted once.

The collapse is invisible in scope #1, where a Notepad record's keys ARE the
form's labels, so a miss really does mean blank. It is fatal the moment the two
vocabularies differ, which is the whole of scope #2.

can_answer() separates them. Bridging vocabularies stays the LLM's job - it can
see the screen and the record together - and GradeSheetSource.lookup() still
refuses to guess a synonym, which is what it was written to refuse.

Run:  python -m pytest tests/scope2/test_source_knows_field.py -q
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))

from data_sources.base import DataSource  # noqa: E402
from data_sources.grade_sheet_source import GradeSheetSource  # noqa: E402

SHEET = REPO / "components" / "scope2" / "data" / "sheets" / "grade_sheet.xlsx"


def source():
    src = GradeSheetSource(str(SHEET))
    src.refresh(0)
    return src


# ── the distinction ──────────────────────────────────────────────────────────

def test_a_column_of_the_sheet_is_answerable():
    assert source().can_answer("PROGRAM")
    assert source().can_answer("FINAL GRADE")


def test_a_portal_cell_name_is_not():
    """The name the agent actually asks with, and the reason 109 fields were
    ruled blank."""
    src = source()
    assert not src.can_answer("Course Abad, Andrea A.")
    assert not src.can_answer("Grade 0-100 Abad, Andrea A.")


def test_lookup_still_refuses_to_guess_a_synonym():
    """can_answer reports; it does not quietly start translating. Course ->
    PROGRAM is the matcher's decision, and putting it in a lookup would hide an
    unaudited mapping under everything."""
    src = source()
    assert src.lookup("Course") is None
    assert src.lookup("PROGRAM") == "BS Information Systems"


def test_an_empty_cell_is_answerable_but_has_no_value():
    """The case the whole distinction exists to preserve: a field the source DOES
    know and which is genuinely blank still reports can_answer, so a real
    confirmed-blank is still reachable."""
    src = source()
    known_but_empty = [h for h in src.headers() if src.lookup(h) is None]
    for header in known_but_empty[:3]:
        assert src.can_answer(header), f"{header!r} is a real column"


# ── the default keeps every other source unchanged ───────────────────────────

def test_the_contract_defaults_to_claiming_it_can_answer():
    """Scope #1's sources must behave exactly as before. A source that cannot
    tell the two cases apart should keep claiming it can answer, because that is
    what the agent already assumed of it."""

    class Old(DataSource):
        def lookup(self, field_name, section=""):
            return None

        def get_all(self):
            return {}

    assert Old().can_answer("anything at all")


# ── what the agent does with it ──────────────────────────────────────────────

def test_the_agent_treats_a_source_without_the_method_as_answering():
    from agent.agent import LLMAgent

    agent = LLMAgent.__new__(LLMAgent)
    agent._source = object()                       # no can_answer at all
    assert agent._source_knows("First Name")


def test_the_agent_asks_the_source_when_it_can():
    from agent.agent import LLMAgent

    class Picky:
        def can_answer(self, field):
            return field == "PROGRAM"

    agent = LLMAgent.__new__(LLMAgent)
    agent._source = Picky()
    assert agent._source_knows("PROGRAM")
    assert not agent._source_knows("Course Abad, Andrea A.")


def test_a_source_that_raises_is_treated_as_answering():
    """Never let a broken adapter turn into 'skip every field'. The old
    behaviour is the safe direction here."""
    from agent.agent import LLMAgent

    class Broken:
        def can_answer(self, field):
            raise RuntimeError("boom")

    agent = LLMAgent.__new__(LLMAgent)
    agent._source = Broken()
    assert agent._source_knows("First Name")


def test_both_fast_fill_branches_consult_it():
    """The edit and combobox branches each decide confirmed-blank separately;
    guarding only one would leave half the grid still being tabbed past."""
    source_text = (REPO / "components" / "agent" / "agent.py").read_text(encoding="utf-8")
    assert source_text.count("if not self._source_knows(_bf_label):") == 2


# ── the LLM can read the record ──────────────────────────────────────────────

def test_the_record_can_be_read_as_text():
    """The seam the agent reads a source through. Without it the plugin-free
    path called a Notepad-only method, got nothing, and logged
    'no Notepad text available' against a web page."""
    text = source().read_full_text()
    assert "PROGRAM: BS Information Systems" in text
    assert "FINAL GRADE: 85" in text


def test_the_merged_header_placeholders_are_left_out():
    """pandas names the columns under a merged header "Unnamed: 3". They carry
    the rest of the student's name and are meaningless as labels, so they would
    read as fields that do not exist."""
    assert "Unnamed" not in source().read_full_text()


def test_reading_does_not_need_a_state():
    """Notepad needs one to find its window; a spreadsheet on disk does not, and
    the agent passes one regardless."""
    assert source().read_full_text({"elements": []}) == source().read_full_text()
