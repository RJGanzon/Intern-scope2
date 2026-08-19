"""DemoRecorder can record a browser demo, not only a desktop one.

DemoRecorder is the recorder that matters for behavioral cloning: it fires on
each click and keystroke, so every training step is one action with the state
immediately before and after it. ScreenObserver samples on a timer instead, and
a timer catches frames mid-action and frames with no action at all.

It constructed UIAutomationObserver unconditionally and raised ImportError
without it, so scope #2's browser demos had to be recorded with the weaker
recorder - or with the stronger one reading Chrome's UIA tree, which cannot
resolve aria-labelledby and therefore cannot tell row 1 from row 50.

The load-bearing detail is WHICH THREAD builds the observer. Playwright's sync
API is thread-affine, and _request_snapshot swallows every exception into {} -
so an observer built on the wrong thread does not fail, it records a full
session of empty states that looks exactly like a successful recording.

Run:  python -m pytest tests/scope2/test_demo_recorder_perception.py -q
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


@pytest.fixture(autouse=True)
def _pynput_present(monkeypatch):
    """pynput gates construction and has nothing to do with observer choice."""
    monkeypatch.setattr(rec, "_PYNPUT_AVAILABLE", True)


class FakeObserver:
    """Records the thread that built it and the thread that snapshots it."""

    def __init__(self):
        self.built_on = threading.get_ident()
        self.snapshot_threads = set()
        self.disconnected = False

    def snapshot(self):
        self.snapshot_threads.add(threading.get_ident())
        return {"window_title": "Grade Encoding Portal", "elements": []}

    def disconnect(self):
        self.disconnected = True


def build(tmp_path, **kwargs):
    rec_obj = rec.DemoRecorder(output_dir=str(tmp_path), **kwargs)
    return rec_obj


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ── injection ────────────────────────────────────────────────────────────────

def test_an_injected_observer_replaces_uia_entirely(tmp_path, monkeypatch):
    """The point of the seam: no UIAutomation needed at all."""
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    built = []

    def factory():
        obs = FakeObserver()
        built.append(obs)
        return obs

    recorder = build(tmp_path, observer_factory=factory)
    try:
        assert wait_for(lambda: built), "worker thread never built the observer"
        assert recorder._observer is built[0]
    finally:
        recorder._quit_event.set()


def test_uia_is_still_required_when_nothing_is_injected(tmp_path, monkeypatch):
    """Scope #1's contract is unchanged: no observer, no recording, said loudly."""
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    with pytest.raises(ImportError, match="UIAutomationObserver"):
        build(tmp_path)


def test_default_still_builds_uia_on_the_constructing_thread(tmp_path, monkeypatch):
    """UIA keeps its original eager construction - moving it would change the
    scope #1 recording path for no reason."""
    made = []

    class FakeUIA:
        def __init__(self, *a, **k):
            made.append(threading.get_ident())

        def snapshot(self):
            return {}

    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", True)
    monkeypatch.setattr(rec, "_UIAObserver", FakeUIA)

    recorder = build(tmp_path)
    try:
        assert made == [threading.get_ident()]
    finally:
        recorder._quit_event.set()


# ── the thread-affinity contract ─────────────────────────────────────────────

def test_the_worker_thread_builds_and_snapshots_the_same_observer(tmp_path, monkeypatch):
    """Playwright's sync API cannot cross threads, and _request_snapshot turns a
    thread violation into {} - a whole session of empty states. So whichever
    thread builds the observer must be the one that snapshots it."""
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    built = []
    recorder = build(tmp_path, observer_factory=lambda: built.append(FakeObserver()) or built[-1])
    try:
        assert wait_for(lambda: built and built[0].snapshot_threads)
        obs = built[0]
        assert obs.built_on != threading.get_ident(), "built on the caller's thread"
        assert obs.snapshot_threads == {obs.built_on}
    finally:
        recorder._quit_event.set()


def test_the_worker_releases_the_observer_when_recording_ends(tmp_path, monkeypatch):
    """A CDP session holds a node driver subprocess; ending the recording must
    let it go. The operator's browser stays open - they were driving it."""
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    built = []
    recorder = build(tmp_path, observer_factory=lambda: built.append(FakeObserver()) or built[-1])
    assert wait_for(lambda: built)

    recorder._quit_event.set()
    assert wait_for(lambda: built[0].disconnected), "observer never disconnected"


def test_a_failed_observer_stops_the_recording_instead_of_recording_nothing(
        tmp_path, monkeypatch, capsys):
    """Well-formed files full of empty states are worse than no files: they look
    like data. Fail where the operator can still see it."""
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)

    def factory():
        raise RuntimeError("WebObserver failed to attach at http://localhost:9222")

    recorder = build(tmp_path, observer_factory=factory)
    assert wait_for(lambda: recorder._quit_event.is_set())
    assert "no state" in capsys.readouterr().out


def test_snapshot_is_empty_rather_than_raising_before_the_observer_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    recorder = build(tmp_path, observer_factory=FakeObserver)
    try:
        recorder._observer = None
        assert recorder._request_snapshot() == {}
    finally:
        recorder._quit_event.set()


# ── perception="web" ─────────────────────────────────────────────────────────

def test_web_perception_falls_back_when_no_browser_is_listening(tmp_path, monkeypatch, capsys):
    """Decided up front over plain HTTP, so the operator learns before recording
    an hour of row-blind demos rather than after."""
    made = []

    class FakeUIA:
        def __init__(self, *a, **k):
            made.append(1)

        def snapshot(self):
            return {}

    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", True)
    monkeypatch.setattr(rec, "_UIAObserver", FakeUIA)
    monkeypatch.setattr(rec, "_WEB_OBSERVER_AVAILABLE", True)
    monkeypatch.setattr(rec, "_cdp_reachable", lambda url, timeout=2.0: False)

    recorder = build(tmp_path, perception="web")
    try:
        out = capsys.readouterr().out
        assert "No browser answering" in out
        assert "--remote-debugging-port" in out
        assert made, "did not fall back to UIA"
        assert recorder._observer_factory is None
    finally:
        recorder._quit_event.set()


def test_web_perception_builds_a_web_observer_when_a_browser_is_there(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    monkeypatch.setattr(rec, "_WEB_OBSERVER_AVAILABLE", True)
    monkeypatch.setattr(rec, "_cdp_reachable", lambda url, timeout=2.0: True)

    seen = {}

    class FakeWeb(FakeObserver):
        available = True

        def __init__(self, browser_url=None, max_elements=None):
            super().__init__()
            seen["browser_url"] = browser_url
            seen["max_elements"] = max_elements

        def connect(self):
            return True

    monkeypatch.setattr(rec, "_WebObserver", FakeWeb)

    recorder = build(tmp_path, perception="web",
                     browser_url="http://localhost:9999", max_elements=303)
    try:
        assert wait_for(lambda: recorder._observer is not None)
        assert seen == {"browser_url": "http://localhost:9999", "max_elements": 303}
    finally:
        recorder._quit_event.set()


def test_web_perception_without_playwright_says_so_and_falls_back(tmp_path, monkeypatch, capsys):
    class FakeUIA:
        def __init__(self, *a, **k):
            pass

        def snapshot(self):
            return {}

    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", True)
    monkeypatch.setattr(rec, "_UIAObserver", FakeUIA)
    monkeypatch.setattr(rec, "_WEB_OBSERVER_AVAILABLE", False)

    recorder = build(tmp_path, perception="web")
    try:
        assert "playwright" in capsys.readouterr().out
        assert recorder._observer_factory is None
    finally:
        recorder._quit_event.set()


# ── replay ───────────────────────────────────────────────────────────────────

def test_replay_refuses_an_observer_it_cannot_use_on_its_own_thread(tmp_path, monkeypatch):
    """replay() snapshots from the calling thread. Silently producing a replay
    session of empty states would be indistinguishable from real demo data."""
    monkeypatch.setattr(rec, "_UIA_OBSERVER_AVAILABLE", False)
    recorder = build(tmp_path, observer_factory=FakeObserver)
    try:
        with pytest.raises(RuntimeError, match="worker thread"):
            recorder.replay(str(tmp_path))
    finally:
        recorder._quit_event.set()
