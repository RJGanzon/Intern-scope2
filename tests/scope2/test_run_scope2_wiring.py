"""Scope #2's entry point is Scope #1's pipeline with different parts in the slots.

The claim this file guards is architectural, not behavioural: swapping the
application costs configuration, not agent edits. If run_scope2.py ever starts
reaching around LLMAgent - constructing its own executor, driving the browser
itself, special-casing the portal inside the agent - that claim is gone, and the
failure would be invisible because the run would still work.

So these tests assert what run_scope2.py hands LLMAgent, not what happens next.
The run itself is live and belongs to the operator.

Run:  python -m pytest tests/scope2/test_run_scope2_wiring.py -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))

import run_scope2  # noqa: E402


# ── --records ────────────────────────────────────────────────────────────────

def test_a_single_student():
    assert list(run_scope2.parse_records("3")) == [3]


def test_a_range_is_inclusive_at_both_ends():
    assert list(run_scope2.parse_records("0-4")) == [0, 1, 2, 3, 4]


def test_records_are_zero_based_like_the_sheet():
    """GradeSheetSource.refresh() is 0-based and FormFillerPlugin's record_num is
    1-based. They are different numbers for the same idea; "0" must mean the
    first student, not the one before it."""
    assert list(run_scope2.parse_records("0")) == [0]


# ── the countdown contract ───────────────────────────────────────────────────

def test_countdown_speaks_the_sentinel_format_the_play_panel_parses():
    lines = []
    run_scope2.print_countdown(3, sleep_fn=lambda _s: None, print_fn=lines.append)
    assert lines[0] == "COUNTDOWN_BEGIN"
    assert lines[-1] == "COUNTDOWN_END"
    assert [l for l in lines if l.startswith("COUNTDOWN ")] == \
        ["COUNTDOWN 3", "COUNTDOWN 2", "COUNTDOWN 1"]


def test_the_hint_line_tells_the_truth_about_this_workflow():
    """The Play panel shows the line after COUNTDOWN_BEGIN as its hint. Scope #1
    says "click the target window"; here the browser is driven over CDP and
    clicking into it mid-run is the thing that breaks it."""
    lines = []
    run_scope2.print_countdown(1, sleep_fn=lambda _s: None, print_fn=lines.append)
    hint = lines[1]
    assert not hint.startswith("COUNTDOWN")
    assert "keep your hands off" in hint.lower()
    assert "click on the target window" not in hint.lower()


def test_countdown_lines_are_plain_ascii():
    """A real em-dash came out as a replacement character through the piped
    subprocess once already; the fix was ASCII, not an encoding change."""
    lines = []
    run_scope2.print_countdown(2, sleep_fn=lambda _s: None, print_fn=lines.append)
    for line in lines:
        line.encode("ascii")


# ── what LLMAgent actually receives ──────────────────────────────────────────

class _Recorder:
    """Stands in for LLMAgent and remembers its kwargs."""

    calls = []

    def __init__(self, **kwargs):
        _Recorder.calls.append(kwargs)

    def run(self, **_kw):
        return []


@pytest.fixture
def captured(monkeypatch, tmp_path):
    _Recorder.calls = []

    class FakeObserver:
        available = True
        disconnected = False

        def connect(self):
            return True

        def snapshot(self):
            return {"window_title": "Grade Encoding Portal", "elements": []}

        def disconnect(self):
            FakeObserver.disconnected = True

    monkeypatch.setattr(run_scope2, "build_observer", lambda args: FakeObserver())
    monkeypatch.setattr(run_scope2, "_report", lambda *a, **k: None)

    import agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "LLMAgent", _Recorder)

    def run(*extra):
        run_scope2.main(["--records", "0-1", *extra])
        return _Recorder.calls

    return run


def test_every_scope_specific_part_goes_through_a_seam(captured):
    """The four slots, and nothing else. Each is a constructor parameter
    LLMAgent already had for scope #1."""
    from agent.scope import GRADE_PORTAL_SCOPE
    from data_sources.grade_sheet_source import GradeSheetSource
    from agent.task_plugins.grade_portal_plugin import GradePortalPlugin

    call = captured()[0]
    assert call["scope"] is GRADE_PORTAL_SCOPE
    assert isinstance(call["data_source"], GradeSheetSource)
    assert isinstance(call["task_plugin"], GradePortalPlugin)
    assert call["observer"] is not None


def test_the_plugin_does_not_bring_its_own_executor(captured):
    """LLMAgent wires the executor in. A plugin that built its own would be
    driving the screen behind the agent's back."""
    assert captured()[0]["task_plugin"]._executor is None


def test_each_student_gets_a_fresh_source(captured):
    """A source shared across students is one cache away from filling every row
    with the first student's grades."""
    calls = captured()
    assert len(calls) == 2
    assert calls[0]["data_source"] is not calls[1]["data_source"]
    assert [c["record_num"] for c in calls] == [0, 1]


def test_the_capsule_router_cannot_override_an_explicit_checkpoint(captured):
    assert captured()[0]["route_capsule"] is False


def test_no_plugin_runs_the_pure_transformer_shape(captured):
    """The shape run_task.py itself uses, once a trained checkpoint exists."""
    call = captured("--no-plugin")[0]
    assert call["task_plugin"] is None
    assert call["pure_transformer"] is False
    assert call["disable_auto_handlers"] is True


def test_the_browser_is_released_even_when_the_run_raises(monkeypatch, capsys):
    seen = {"disconnected": False}

    class FakeObserver:
        def disconnect(self):
            seen["disconnected"] = True

    class Exploding:
        def __init__(self, **kwargs):
            pass

        def run(self, **_kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(run_scope2, "build_observer", lambda args: FakeObserver())
    monkeypatch.setattr(run_scope2, "_report", lambda *a, **k: None)
    import agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "LLMAgent", Exploding)

    run_scope2.main(["--records", "0"])
    assert seen["disconnected"], "a crashed run left the CDP session open"


def test_a_missing_sheet_stops_before_touching_the_browser(monkeypatch):
    """Failing here costs nothing; failing after attaching leaves a browser
    session open and a countdown already spent."""
    monkeypatch.setattr(run_scope2, "build_observer",
                        lambda args: pytest.fail("attached before checking the sheet"))
    with pytest.raises(SystemExit, match="grade sheet not found"):
        run_scope2.main(["--sheet", "no/such/sheet.xlsx"])
