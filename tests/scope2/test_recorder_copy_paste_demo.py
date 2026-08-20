"""A demo filled by copy-paste records the demonstration, not the shopping trip.

Scope #2's source is a spreadsheet, so a human demonstrating it does this per
value: click the Excel cell, Ctrl+C, Alt+Tab, click the portal cell, Ctrl+V.
Only the last two are the demonstration. The first three were all recorded as
real steps, and two of them as *fills*:

  - the Excel click, because WebObserver reports the page it is attached to no
    matter which window is in front, so the click was matched against the
    portal's element list and landed on whatever input sat underneath;
  - Ctrl+C, because a hotkey step becomes ACTION_KEYBOARD - the type class -
    in transformer.py's action derivation;
  - Alt+Tab, because nothing tracked alt, so the Tab half arrived as a plain
    "tab" hotkey and became a fill too.

Three corrupt steps for every two good ones. None of it raises anything.

Run:  python -m pytest tests/scope2/test_recorder_copy_paste_demo.py -q
"""

import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))

from recorder import recorder as rec  # noqa: E402

PORTAL = "Grade Encoding Portal - V0 Base"
CHROME = f"{PORTAL} - Google Chrome"
EXCEL = "grade_sheet.xlsx - Excel"


class PortalObserver:
    """WebObserver's defining behaviour: it reports its page, always."""

    def snapshot(self):
        return {
            "application": "browser",
            "window_title": PORTAL,
            "elements": [{
                "element_id": "e1", "type": "editcontrol",
                "label": "Grade 0-100 Abad, Andrea A.", "window_role": "active",
                "bbox": [500, 300, 600, 320],
            }],
        }


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    r = rec.DemoRecorder(output_dir=str(tmp_path), observer_factory=PortalObserver)
    deadline = time.time() + 5
    while r._observer is None and time.time() < deadline:
        time.sleep(0.02)
    assert r._observer is not None, "worker never built the observer"
    r._recording = True
    yield r
    r._quit_event.set()


def steps(recorder):
    with recorder._lock:
        return list(recorder._steps)


def process(recorder, event):
    """Run one queued event through the worker's own handler, synchronously."""
    recorder._process_event(event)


# ── the three source-side actions ────────────────────────────────────────────

def test_a_click_in_the_source_window_is_not_a_demo_step(recorder, capsys):
    """The dangerous one: the observer cannot see Excel, so the click would be
    attributed to the portal input underneath it."""
    process(recorder, {"action_type": "click", "click_pos": [550, 310],
                       "fg_title": EXCEL})
    assert steps(recorder) == []
    assert "source-side" in capsys.readouterr().out


def test_a_click_in_the_observed_window_is_kept(recorder):
    process(recorder, {"action_type": "click", "click_pos": [550, 310],
                       "fg_title": CHROME})
    assert len(steps(recorder)) == 1


def test_copying_from_the_source_is_not_a_fill(recorder):
    """Ctrl+C in Excel. transformer.py folds any hotkey into the type class, so
    this used to teach 'fill something here' once per value copied."""
    process(recorder, {"action_type": "hotkey", "hotkey": "ctrl+c",
                       "fg_title": EXCEL})
    assert steps(recorder) == []


def test_alt_tab_never_reaches_the_queue_as_a_tab_press(recorder):
    """Alt is now tracked, so the Tab half of an app switch is not a form Tab."""
    recorder._on_key_press(_key("alt_l"))
    recorder._on_key_press(_key("tab"))
    recorder._on_key_release(_key("alt_l"))
    assert recorder._action_queue.qsize() == 0

    # and a real Tab, alt released, still records
    recorder._on_key_press(_key("tab"))
    assert recorder._action_queue.qsize() == 1


def test_tab_still_works_when_alt_was_never_pressed(recorder):
    recorder._on_key_press(_key("tab"))
    queued = recorder._action_queue.get_nowait()
    assert queued["action_type"] == "hotkey"
    assert queued["hotkey"] == "tab"


def test_typing_in_another_window_is_not_a_fill(recorder):
    """Found from a real question: "can I still use my second monitor?" Yes -
    but typed text carried no window at all, so a command typed into a terminal
    over there recorded as a fill on the portal. Clicks were already covered;
    keystrokes were not."""
    process(recorder, {"action_type": "keyboard", "text": "git status",
                       "fg_title": "Windows PowerShell"})
    assert steps(recorder) == []


def test_typing_into_the_portal_still_records(recorder):
    process(recorder, {"action_type": "keyboard", "text": "85", "fg_title": CHROME})
    assert len(steps(recorder)) == 1


def test_the_window_is_captured_where_the_typing_began(recorder, monkeypatch):
    """Text is flushed by the NEXT click, which may be in a different window
    than the one typed into. Capturing at flush time would blame the wrong one."""
    monkeypatch.setattr(rec, "_foreground_title", lambda: EXCEL)
    recorder._on_key_press(_char("8"))
    recorder._on_key_press(_char("5"))

    monkeypatch.setattr(rec, "_foreground_title", lambda: CHROME)
    recorder._flush_text_to_queue()

    queued = recorder._action_queue.get_nowait()
    assert queued["text"] == "85"
    assert queued["fg_title"] == EXCEL


# ── the two real steps ───────────────────────────────────────────────────────

def test_the_pasted_value_reaches_the_trace(recorder, monkeypatch):
    """The clipboard was already being read and then dropped on the floor, so a
    copy-paste demo recorded every value it entered as ""."""
    monkeypatch.setitem(sys.modules, "pyperclip", _FakeClipboard("87"))
    process(recorder, {"action_type": "hotkey", "hotkey": "ctrl+v",
                       "fg_title": CHROME})

    written = steps(recorder)
    assert len(written) == 1
    strokes = written[0]["keyboard"]["actions"][0]["strokes"]
    assert strokes[0]["pasted_text"] == "87"


def test_a_copy_does_not_claim_to_have_pasted_anything(recorder, monkeypatch):
    """ctrl+c reads the same clipboard; only a paste entered a value."""
    monkeypatch.setitem(sys.modules, "pyperclip", _FakeClipboard("87"))
    process(recorder, {"action_type": "hotkey", "hotkey": "ctrl+c",
                       "fg_title": CHROME})

    written = steps(recorder)
    assert len(written) == 1
    assert written[0]["keyboard"]["actions"][0]["strokes"][0]["pasted_text"] == ""


# ── the whole workflow, end to end ───────────────────────────────────────────

def test_one_cell_of_a_copy_paste_demo_records_exactly_two_steps(recorder, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyperclip", _FakeClipboard("87"))

    process(recorder, {"action_type": "click", "click_pos": [100, 100], "fg_title": EXCEL})
    process(recorder, {"action_type": "hotkey", "hotkey": "ctrl+c", "fg_title": EXCEL})
    process(recorder, {"action_type": "click", "click_pos": [550, 310], "fg_title": CHROME})
    process(recorder, {"action_type": "hotkey", "hotkey": "ctrl+v", "fg_title": CHROME})

    written = steps(recorder)
    assert len(written) == 2, [s["keyboard"] or s["mouse"] for s in written]
    assert written[0]["mouse"]["actions"][0]["position"] == [550.0, 310.0]
    assert written[1]["keyboard"]["actions"][0]["hotkey"] == "ctrl+v"


# ── scope #1 is unaffected ───────────────────────────────────────────────────

def test_a_uia_recording_is_not_filtered_by_this(tmp_path, monkeypatch):
    """UIA snapshots the foreground window, so the observed title IS the
    foreground title and the source-side rule can never fire on it."""
    monkeypatch.setattr(rec, "_PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)

    class UIALike:
        def snapshot(self):
            return {"application": "car_insurance_form_wx.exe",
                    "window_title": "Car Insurance Entry",
                    "elements": [{"element_id": "e1", "type": "editcontrol",
                                  "label": "First Name", "window_role": "active",
                                  "bbox": [10, 10, 100, 30]}]}

    r = rec.DemoRecorder(output_dir=str(tmp_path), observer_factory=UIALike)
    try:
        deadline = time.time() + 5
        while r._observer is None and time.time() < deadline:
            time.sleep(0.02)
        r._recording = True
        r._process_event({"action_type": "click", "click_pos": [50, 20],
                          "fg_title": "Car Insurance Entry"})
        with r._lock:
            assert len(r._steps) == 1
    finally:
        r._quit_event.set()


def test_an_unknown_foreground_window_does_not_drop_the_step(recorder):
    """_foreground_title returns "" when Win32 is unavailable. Silently
    discarding a whole recording on a machine that cannot answer the question
    would be far worse than keeping a few source-side clicks."""
    process(recorder, {"action_type": "click", "click_pos": [550, 310], "fg_title": ""})
    assert len(steps(recorder)) == 1


# ── helpers ──────────────────────────────────────────────────────────────────

class _FakeClipboard:
    def __init__(self, text):
        self._text = text

    def paste(self):
        return self._text


class _char:
    """pynput reports printable keys as objects carrying .char."""

    def __init__(self, ch):
        self.char = ch

    def __str__(self):
        return self.char


class _key:
    """pynput reports named keys as objects with no .char."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Key.{self.name}"
