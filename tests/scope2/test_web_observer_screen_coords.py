"""Web element bboxes are screen coordinates, like every other observer's.

recorder.py's trace schema says "bbox: [x1,y1,x2,y2] - screen coordinates",
and ActionExecutor._click drives pyautogui in screen pixels. Playwright reports
viewport CSS pixels. Left untranslated the two differ by the window origin plus
the browser UI, which does not fail loudly - it clicks a few hundred pixels
above the intended cell, on whatever happens to be there.

The geometry here was calibrated against Chrome_RenderWidgetHostHWND, the OS
window that is exactly the web content area: with these formulas the computed
origin matched it exactly at three different window positions.

Run:  python -m pytest tests/scope2/test_web_observer_screen_coords.py -q
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "components"))

from observers.web_observer import WebObserver  # noqa: E402


class FakePage:
    """Reports the geometry a real browser window reports."""

    def __init__(self, **overrides):
        self.geometry = {
            "screenX": 150, "screenY": 120,
            "outerWidth": 1100, "outerHeight": 800,
            "innerWidth": 1084, "innerHeight": 705,
            "dpr": 1, "screenWidth": 1920, "screenHeight": 1080,
        }
        self.geometry.update(overrides)

    def evaluate(self, _script):
        return self.geometry


def test_origin_matches_the_measured_content_area():
    """The case calibrated against the OS: a 1100x800 window at (150,120) has
    its content area at (158,207) - 8 px of border in, and the browser UI down.
    """
    origin = WebObserver(headless=True)._screen_origin(FakePage())
    assert (round(origin["dx"]), round(origin["dy"])) == (158, 207)


def test_bottom_border_is_not_counted_as_browser_ui():
    """outerHeight - innerHeight covers the UI *and* the bottom border, because
    a window has no top border. Counting it put every click 8 px low."""
    page = FakePage()
    border = (page.geometry["outerWidth"] - page.geometry["innerWidth"]) / 2
    naive = page.geometry["screenY"] + (
        page.geometry["outerHeight"] - page.geometry["innerHeight"])

    origin = WebObserver(headless=True)._screen_origin(page)
    assert origin["dy"] == naive - border


def test_device_pixel_ratio_scales_into_physical_pixels():
    """pyautogui clicks physical pixels; the DOM reports CSS pixels."""
    origin = WebObserver(headless=True)._screen_origin(FakePage(dpr=2))
    assert (origin["dx"], origin["dy"]) == (316, 414)
    assert (origin["screen_width"], origin["screen_height"]) == (3840, 2160)


def test_bbox_is_offset_and_scaled():
    origin = WebObserver(headless=True)._screen_origin(FakePage())
    box = {"x": 100, "y": 50, "width": 60, "height": 20}
    assert WebObserver._to_screen(box, origin) == [258, 257, 318, 277]


def test_translation_can_be_turned_off():
    """screen_coords=False leaves raw viewport geometry, for tests that compare
    against the DOM directly."""
    obs = WebObserver(headless=True, screen_coords=False)
    assert obs._screen_origin(FakePage()) is None
    box = {"x": 100, "y": 50, "width": 60, "height": 20}
    assert WebObserver._to_screen(box, None) == [100, 50, 160, 70]


def test_unavailable_geometry_does_not_half_apply_an_offset():
    """If the page cannot be evaluated, bboxes stay in viewport space rather
    than getting an offset with no matching screen_resolution."""

    class Broken:
        def evaluate(self, _):
            raise RuntimeError("page closed")

    assert WebObserver(headless=True)._screen_origin(Broken()) is None
