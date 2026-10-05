"""Cosmetic portal changes must not move anything the agent can see.

Requested directly: make Scope #2 "more visually appealing and interesting for
presentation" without touching the backend. The risk in that is quiet: the
WebObserver reports every control's screen box, and the grade-portal model was
trained on demonstrations recorded against this exact layout. A restyle that
shifts the sheet down by one toolbar's worth of pixels, or widens a column,
hands the model a page it has never seen while looking identical to a person.

So the layout is frozen as data. portal_geometry_baseline.json holds the box,
name and type of every element the observer's selector matches, captured from
every variant BEFORE the restyle. This test re-captures them and requires an
exact match. Colors, fonts weights, glows and anything the observer does not
select are free to change; geometry and the observed element list are not.

Regenerate the baseline only for a deliberate layout change, and then re-record
or re-validate the demonstrations it invalidates:
    python tests/scope2/test_portal_geometry_frozen.py --write

Run:  python -m pytest tests/scope2/test_portal_geometry_frozen.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))

from executor.scanner import CHROMIUM, VARIANTS, variant_url  # noqa: E402

BASELINE = Path(__file__).with_name("portal_geometry_baseline.json")
VIEWPORT = {"width": 1440, "height": 900}

# The observer's own selector (web_observer.py), so "what the agent sees" is
# the thing being frozen, not a guess at it.
_OBSERVED = (
    "input, select, textarea, button, a[href], "
    "[role='button'], [role='textbox'], [role='combobox'], "
    "[role='checkbox'], [role='radio'], [role='tab'], "
    "[role='menuitem'], [role='link']"
)

_CAPTURE_JS = """
(selector) => Array.from(document.querySelectorAll(selector)).map((el) => {
  const r = el.getBoundingClientRect();
  return {
    tag: el.tagName.toLowerCase(),
    type: el.getAttribute("type") || "",
    id: el.id || "",
    box: [r.x, r.y, r.width, r.height].map((v) => Math.round(v * 100) / 100),
  };
})
"""


def capture(page, name):
    page.goto(variant_url(name))
    page.wait_for_selector("#records-body tr")
    return page.evaluate(_CAPTURE_JS, _OBSERVED)


def capture_all():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROMIUM), headless=True,
                              args=["--headless=new"])
        try:
            page = b.new_page(viewport=VIEWPORT)
            return {name: capture(page, name) for name in VARIANTS}
        finally:
            b.close()


@pytest.fixture(scope="module")
def current():
    pytest.importorskip("playwright.sync_api")
    if not CHROMIUM.exists():
        pytest.skip(f"no chromium at {CHROMIUM}")
    return capture_all()


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", VARIANTS)
def test_observed_elements_unchanged(current, baseline, name):
    """Same elements, same order: no new button or link slipped into view."""
    now = [(e["tag"], e["type"], e["id"]) for e in current[name]]
    then = [(e["tag"], e["type"], e["id"]) for e in baseline[name]]
    assert now == then


@pytest.mark.parametrize("name", VARIANTS)
def test_every_box_where_it_was(current, baseline, name):
    moved = [(b["id"] or b["tag"], b["box"], c["box"])
             for b, c in zip(baseline[name], current[name])
             if b["box"] != c["box"]]
    assert not moved, f"{len(moved)} element(s) moved, first: {moved[:3]}"


if __name__ == "__main__" and "--write" in sys.argv:
    BASELINE.write_text(json.dumps(capture_all(), indent=1), encoding="utf-8")
    print(f"wrote {BASELINE}")
