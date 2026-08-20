"""Element rectangles are asked of Windows, not reconstructed from the DOM.

Found in the first real recording session, and it had already produced 60
perfectly well-formed, completely unusable steps before anyone could tell.

The operator's laptop is the SECONDARY display, at X=1920 beside a 1920-wide
primary, running at 125%. `_screen_origin` scaled the whole of `window.screenX`
by devicePixelRatio - but that value carries the offset of every monitor to the
left, and that offset is not in this window's scale factor. Result: every bbox
sat 1920 x 0.25 = ~480 px right of the truth (measured: 465, the difference
being the border term). Y was exact, because the two monitors share a top edge
and 0 x 0.25 is 0.

Nothing raised. The states were full, the window title was right, the source-side
filter worked, the rows were even correct - only the columns were wrong. Solving
for the transform that would fix the recording gave a pure horizontal shift of
+465 px, which put 28 of 28 clicks back on an element.

Chrome_RenderWidgetHostHWND is the OS window that IS the viewport, and
GetWindowRect answers in the coordinate space of the calling process - the same
space pynput records clicks in, because it is the same process. Nothing to
reconstruct, so nothing to get wrong.

Run:  python -m pytest tests/scope2/test_web_observer_multi_monitor.py -q
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))

from observers.web_observer import WebObserver  # noqa: E402


class FakePage:
    """The geometry a browser on a secondary monitor really reports."""

    def __init__(self, **overrides):
        self.geometry = {
            "screenX": 1920, "screenY": 0,
            "outerWidth": 1100, "outerHeight": 800,
            "innerWidth": 1084, "innerHeight": 705,
            "dpr": 1.25, "screenWidth": 1536, "screenHeight": 864,
        }
        self.geometry.update(overrides)

    def evaluate(self, _script):
        return self.geometry


def observer(rect=None, virtual=None, title="Grade Encoding Portal"):
    obs = WebObserver(headless=True)
    obs._page_title_cache = title
    obs._content_rect_win32 = lambda _t: rect
    obs._virtual_screen = staticmethod(lambda: virtual)
    return obs


# ── the fix ──────────────────────────────────────────────────────────────────

def test_the_origin_is_the_content_rect_windows_reports():
    """No arithmetic on screenX at all when Windows can answer."""
    rect = {"left": 2689.0, "top": 86.0, "right": 3456.0, "bottom": 816.0}
    origin = observer(rect).\
        _screen_origin(FakePage())

    assert origin["source"] == "win32"
    assert (origin["dx"], origin["dy"]) == (2689.0, 86.0)


def test_the_scale_is_measured_rather_than_taken_from_the_dom():
    """width / innerWidth includes page zoom for free, which devicePixelRatio
    alone does not describe once the user has zoomed."""
    rect = {"left": 1000.0, "top": 100.0, "right": 2084.0, "bottom": 900.0}
    origin = observer(rect)._screen_origin(FakePage(innerWidth=542))
    assert origin["dpr"] == 2.0          # 1084 measured / 542 reported


def test_screen_resolution_covers_every_monitor():
    """It normalises click coordinates. A browser on a second monitor has
    coordinates past the primary's width, and dividing those by one monitor's
    size puts them outside 0..1."""
    rect = {"left": 2689.0, "top": 86.0, "right": 3456.0, "bottom": 816.0}
    origin = observer(rect, virtual={"width": 3456, "height": 1080})._screen_origin(FakePage())
    assert (origin["screen_width"], origin["screen_height"]) == (3456, 1080)


# ── the stub windows ─────────────────────────────────────────────────────────

def test_a_stub_sized_rect_is_refused():
    """Chrome keeps several Chrome_RenderWidgetHostHWND windows, most of them
    2x2 pixels. Taking one collapsed every bbox onto a 2-pixel square - and,
    again, raised nothing."""
    stub = {"left": 3090.0, "top": 85.0, "right": 3092.0, "bottom": 87.0}
    origin = observer(stub)._screen_origin(FakePage())
    assert origin["source"] == "dom", "a 2x2 rect was accepted as a viewport"


# ── the fallback, and its known limit ────────────────────────────────────────

def test_the_dom_path_still_runs_when_windows_cannot_answer():
    """Another OS, no pywin32, or no matching window: better a reconstructed
    origin than viewport coordinates pretending to be screen ones."""
    origin = observer(rect=None)._screen_origin(FakePage(screenX=150, screenY=120, dpr=1))
    assert origin["source"] == "dom"
    assert (round(origin["dx"]), round(origin["dy"])) == (158, 207)


def test_the_dom_path_inflates_every_monitor_offset_to_its_left():
    """The defect itself, as a property of the formula rather than a remembered
    number. Moving the window from the primary onto a second monitor 1920 px to
    the right should shift the origin by 1920. The DOM path shifts it by
    1920 x dpr instead, because it scales the whole of screenX - so at 125% it
    over-shoots by 480 px, which is what put every recorded click a column or
    two off while looking entirely plausible."""
    at_primary = observer(rect=None)._screen_origin(FakePage(screenX=0))
    at_second  = observer(rect=None)._screen_origin(FakePage(screenX=1920))

    moved = at_second["dx"] - at_primary["dx"]
    assert moved == 1920 * 1.25 == 2400
    assert moved - 1920 == 480          # pure invention, per 1920 of offset


def test_the_vertical_error_is_zero_when_the_monitors_share_a_top_edge():
    """Why this hid: screenY was 0 for both monitors, and 0 x anything is 0. The
    rows were right, only the columns were wrong."""
    aligned = observer(rect=None)._screen_origin(FakePage(screenX=1920, screenY=0))
    alone   = observer(rect=None)._screen_origin(FakePage(screenX=0, screenY=0))
    assert aligned["dy"] == alone["dy"]


# ── the recorder's live guard ────────────────────────────────────────────────

def test_the_recorder_can_tell_a_click_landed_on_nothing():
    """The signal that would have caught this in the first five seconds instead
    of after sixty steps."""
    from recorder.recorder import _elem_under

    state = {"elements": [
        {"label": "Grade 0-100 Abad, Andrea A.", "bbox": [3200, 303, 3293, 337]},
    ]}
    assert _elem_under(state, [3250, 320])["label"].startswith("Grade")
    assert _elem_under(state, [3250 + 465, 320]) is None       # the real failure
    assert _elem_under(state, None) is None
    assert _elem_under({}, [10, 10]) is None


def test_the_smallest_containing_element_wins():
    """A cell inside a row inside a table: the innermost one is the target."""
    from recorder.recorder import _elem_under

    state = {"elements": [
        {"label": "table", "bbox": [0, 0, 4000, 3000]},
        {"label": "cell", "bbox": [3200, 303, 3293, 337]},
    ]}
    assert _elem_under(state, [3250, 320])["label"] == "cell"
