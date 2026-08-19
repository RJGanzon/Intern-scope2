"""trace_type="web" records through the DOM, not through UIAutomation.

Before this, "web" was only a string stamped into the trace file - perception
still fell through to UIAutomation. That is the worst kind of wrong: Chrome
exposes a UIA tree, so recording *worked*, it just described a sheet-style page
with one name per column instead of one per cell. Every demonstration of "fill
row 37" would be indistinguishable from "fill row 1", and the defect would only
surface much later as a model that cannot target rows.

The recorder's own runtime dependencies (mss, pynput, pyperclip, PIL) are not
needed to choose an observer, so only screen capture is stubbed here. The
browser and the DOM read are real.

Run:  python -m pytest tests/scope2/test_recorder_web_perception.py -q
"""

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))

pytest.importorskip("playwright.sync_api")

from executor.scanner import variant_url  # noqa: E402
from recorder import recorder as rec  # noqa: E402

CDP_PORT = 9223


@pytest.fixture(scope="module")
def cdp_browser(tmp_path_factory):
    """A browser with a debugging port, started as its own process.

    Deliberately not launched through the sync Playwright API: WebObserver
    starts its own sync Playwright session, and two of those in one thread
    raise "Sync API inside the asyncio loop". A separate process is also what
    the real thing looks like - the operator's browser, already running.
    """
    import json
    import subprocess
    import urllib.request

    from executor.scanner import CHROMIUM

    if not CHROMIUM.exists():
        pytest.skip(f"no chromium at {CHROMIUM}")

    profile = tmp_path_factory.mktemp("cdp-profile")
    proc = subprocess.Popen([
        str(CHROMIUM),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={profile}",
        "--headless=new", "--no-first-run", "--no-default-browser-check",
        variant_url("v0_base"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    url = f"http://localhost:{CDP_PORT}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/json/version", timeout=1) as r:
                json.load(r)
                break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.skip("chromium did not open a debugging port")

    try:
        yield url
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture
def capture_stubbed(monkeypatch):
    """Screen capture is irrelevant to observer choice; the rest is real."""
    monkeypatch.setattr(rec, "_MSS_AVAILABLE", True)


def build(trace_type, browser_url, tmp_path):
    return rec.ScreenObserver(
        output_dir=str(tmp_path), trace_type=trace_type, browser_url=browser_url
    )


def test_web_trace_type_selects_the_dom_observer(capture_stubbed, cdp_browser, tmp_path):
    obs = build("web", cdp_browser, tmp_path)
    assert obs._web_wanted, "trace_type='web' did not select WebObserver"
    assert obs._uia_observer is None, "UIAutomation would shadow the DOM observer"


def test_gui_trace_type_is_unaffected(capture_stubbed, cdp_browser, tmp_path):
    """The new branch must not capture the default desktop path."""
    obs = build("gui", cdp_browser, tmp_path)
    assert not obs._web_wanted


def test_unreachable_browser_degrades_instead_of_raising(capture_stubbed, tmp_path, capsys):
    """No browser on the port: recording still starts, and says what was lost."""
    obs = build("web", "http://localhost:9", tmp_path)
    assert not obs._web_wanted
    assert "remote-debugging-port" in capsys.readouterr().out


@pytest.mark.skipif(not rec._MSS_AVAILABLE, reason="recording needs mss")
def test_a_real_recording_captures_the_dom(cdp_browser, tmp_path):
    """End to end, because every part of this failed silently once.

    The session is built on the capture thread: Playwright's sync API is
    thread-affine, and a session created on the constructing thread raised
    greenlet "cannot switch to a different thread" inside snapshot(), which was
    swallowed into an empty state. The recorder still announced "Semantic mode
    (web)" and saved well-formed traces containing zero elements. Asserting on
    frame count alone would not have caught it, so this asserts on labels.
    """
    obs = rec.ScreenObserver(output_dir=str(tmp_path), trace_type="web",
                             browser_url=cdp_browser)
    obs.start(interval_sec=1.0)
    time.sleep(4)
    traces = obs.stop()

    assert traces, "no traces recorded"
    for trace in traces:
        elements = trace["state"]["elements"]
        assert elements, "trace recorded an empty state"
        labels = [e["label"] for e in elements if e["type"] == "editcontrol"]
        assert len(set(labels)) == len(labels) > 1, "rows are not distinguishable"
        assert any("Abad, Andrea A." in l for l in labels)


@pytest.mark.skipif(not rec._MSS_AVAILABLE, reason="recording needs mss")
def test_snapshot_keeps_up_with_the_capture_interval(cdp_browser):
    """A snapshot has to be quicker than the interval it is captured at.

    Reading the page property-by-property took ~10 CDP round trips per element,
    so the portal's 303 controls took longer than one second and a demo
    recorded no frames at all.
    """
    from observers.web_observer import WebObserver

    observer = WebObserver(browser_url=cdp_browser, max_elements=2000)
    assert observer.connect()
    try:
        observer.snapshot()  # warm up
        started = time.perf_counter()
        state = observer.snapshot()
        elapsed = time.perf_counter() - started
    finally:
        observer.disconnect()

    assert len(state["elements"]) > 300
    assert elapsed < 1.0, f"snapshot took {elapsed:.2f}s, slower than a 1s interval"
