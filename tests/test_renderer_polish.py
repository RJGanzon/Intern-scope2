"""The main window's 2026-10-06 polish pass: CSS only, but two parts of it are
behaviour a person would notice if they regressed.

  * A hidden button must not be on screen. .btn sets display:inline-flex, an
    author rule, which beat the browser's [hidden]{display:none}: renderer.js
    set btnStartLmStudioServer.hidden = true once the server was running, and
    "Start server" still sat next to "Server running". Fixed with
    .btn[hidden] { display: none; }.
  * The loaded task in the Play panel must not crush its own box. The slot is
    a child of an overflow-hidden flex column; with readable description text
    it was the slot that shrank, and the text spilled out of its border. It is
    now flex:none and the description is clamped to three lines.

Also pins the themed browser surfaces (select chevron, scrollbars), since the
standard scrollbar-width property silently reverts Chromium to the OS bar and
would undo them.

Same harness as test_renderer_run_views.py: the real index.html in Chromium,
every preload API faked.

Run:  python -m pytest tests/test_renderer_polish.py -q
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "app_electron" / "renderer" / "index.html"
STYLE = REPO / "app_electron" / "renderer" / "style.css"

FAKE_APIS = """
(() => {
  const fn = () => Promise.resolve([]);
  const api = extra => new Proxy(extra || {}, {
    get: (t, k) => (k in t ? t[k] : fn),
  });
  window.recorderAPI = api({ onEvent: () => {}, setActiveSection: () => {} });
  window.capsulesAPI = api();
  window.workflowsAPI = api();
  window.settingsAPI = api();
  window.inboxAPI = api({ onEvent: () => {} });
  window.agentHud = api({ onDecision() {}, onScope2() {}, onReset() {}, onFinish() {} });
})();
"""

LONG_DESCRIPTION = " ".join(["Fill the car insurance form from the intake packet."] * 8)


@pytest.fixture()
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        # The app's own default window size (main.js).
        pg = browser.new_page(viewport={"width": 1040, "height": 720})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.add_init_script(FAKE_APIS)
        pg.goto(INDEX.as_uri())
        pg.wait_for_load_state("load")
        pg.errors = errors
        yield pg
        browser.close()


# ── hidden buttons ──────────────────────────────────────────────────────────

def test_start_server_button_hidden_when_renderer_hides_it(page):
    page.evaluate("document.getElementById('btnStartLmStudioServer').hidden = true")
    assert page.evaluate(
        "getComputedStyle(document.getElementById('btnStartLmStudioServer')).display"
    ) == "none"


def test_a_hidden_button_comes_back_when_unhidden(page):
    """The fix must not make hidden sticky. (Computed display reads "flex",
    not "inline-flex": the button is an item in .settings-row's flex row,
    which blockifies it. Visible is what matters.)"""
    page.evaluate("""() => {
        const b = document.getElementById('btnStartLmStudioServer');
        b.hidden = true; b.hidden = false;
    }""")
    assert page.evaluate(
        "getComputedStyle(document.getElementById('btnStartLmStudioServer')).display"
    ) != "none"


def test_every_hidden_btn_in_markup_is_not_displayed(page):
    shown = page.evaluate("""() => Array.from(document.querySelectorAll('.btn[hidden]'))
        .filter(b => getComputedStyle(b).display !== 'none').map(b => b.id)""")
    assert shown == []


# ── Play panel slot ─────────────────────────────────────────────────────────

def _load_long_capsule(page):
    page.evaluate("""(desc) => {
        document.getElementById('settingsWrap').hidden = true;
        document.getElementById('recorderPanel').hidden = true;
        document.getElementById('playPanel').hidden = false;
        document.getElementById('ppSlot').classList.add('filled');
        document.getElementById('ppSlotHint').hidden = true;
        document.getElementById('ppCapsule').hidden = false;
        document.getElementById('ppCapsuleName').textContent = 'form_filling';
        document.getElementById('ppCapsuleMeta').textContent = desc;
    }""", LONG_DESCRIPTION)


def test_long_description_stays_inside_its_slot(page):
    _load_long_capsule(page)
    slot, text = page.evaluate("""() => {
        const s = document.getElementById('ppSlot').getBoundingClientRect();
        const t = document.getElementById('ppCapsule').getBoundingClientRect();
        return [[s.top, s.bottom], [t.top, t.bottom]];
    }""")
    assert text[0] >= slot[0] and text[1] <= slot[1], (slot, text)


def test_description_is_clamped_to_three_lines(page):
    _load_long_capsule(page)
    lines = page.evaluate("""() => {
        const m = document.getElementById('ppCapsuleMeta');
        const cs = getComputedStyle(m);
        return Math.round(m.getBoundingClientRect().height / parseFloat(cs.lineHeight));
    }""")
    assert lines == 3


# ── themed browser surfaces ─────────────────────────────────────────────────

def test_selects_draw_their_own_chevron(page):
    appearance, image = page.evaluate("""() => {
        const s = getComputedStyle(document.getElementById('llmProviderSelect'));
        return [s.appearance, s.backgroundImage];
    }""")
    assert appearance == "none"
    assert image.startswith('url("data:image/svg+xml')


def test_standard_scrollbar_properties_not_used():
    """scrollbar-width / scrollbar-color switch Chromium back to the OS-drawn
    bar and make every ::-webkit-scrollbar rule a no-op."""
    css = STYLE.read_text(encoding="utf-8")
    rules = [line for line in css.splitlines()
             if not line.strip().startswith(("/*", "*", "//"))]
    assert not any("scrollbar-width:" in l or "scrollbar-color:" in l for l in rules)


def test_no_page_errors(page):
    assert page.errors == []
