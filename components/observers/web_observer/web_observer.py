"""
observers/web_observer.py
==========================
Browser automation observer via Playwright (Chrome DevTools Protocol).

Used when the target app is a web application running in a browser.
Returns elements in the same dict format as UIAutomationObserver so
the rest of the pipeline needs no changes.

Advantages over UIA for web:
  - Full DOM tree access (labels, roles, attributes, positions)
  - Works with iframes and shadow DOM
  - Can interact with elements that are off-screen
  - No pixel coordinate guessing needed

Dependencies
------------
    pip install playwright
    playwright install chromium

Usage
-----
    obs = WebObserver()
    ok  = obs.connect()       # attach to open browser or launch new one
    state = obs.snapshot()    # returns trace-compatible state dict
    obs.disconnect()
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── path setup ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))   # observers/
_COMP = os.path.dirname(_HERE)                        # components/
_ROOT = os.path.dirname(_COMP)                        # Intern/
for _p in (_ROOT, _COMP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, Page, Browser
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

# Interactive ARIA roles to always capture
_INTERACTIVE_ROLES = {
    "button", "textbox", "combobox", "checkbox", "radio",
    "link", "menuitem", "tab", "listbox", "option",
    "searchbox", "spinbutton", "slider", "switch",
}


class WebObserver:
    """
    Observes browser state via Playwright.

    Parameters
    ----------
    headless    : Run browser headlessly (False = visible browser window).
    browser_url : Connect to an already-running browser via CDP URL
                  e.g. "http://localhost:9222". None = launch a new browser.
    """

    def __init__(
        self,
        headless:     bool          = False,
        browser_url:  Optional[str] = None,
        max_elements: int           = 1000,
        screen_coords: bool         = True,
    ):
        self.headless     = headless
        self._page_title_cache: str = ""
        self.browser_url  = browser_url
        # recorder.py documents the trace element schema as
        # "bbox: [x1,y1,x2,y2] - screen coordinates", and ActionExecutor._click
        # drives pyautogui in screen pixels. Playwright's bounding_box() is in
        # viewport CSS pixels, so the two disagree by the window origin plus the
        # browser chrome - a page that looks perfectly perceived produces clicks
        # that land a few hundred pixels high. Off only for tests that compare
        # against raw DOM geometry.
        self.screen_coords = screen_coords
        # A data-entry grid is routinely larger than a form: 50 rows x 5 inputs
        # is 250 controls before any chrome. The old fixed cap of 200 silently
        # hid everything past it.
        self.max_elements = max_elements
        self._pw:      Any = None
        self._browser: Any = None
        self._page:    Any = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return _PLAYWRIGHT_AVAILABLE

    @property
    def connected(self) -> bool:
        return self._page is not None

    def connect(self, url: Optional[str] = None) -> bool:
        """
        Connect to a browser. Launches a new one if browser_url is not set.

        Parameters
        ----------
        url : Optional page URL to navigate to after connecting.
        """
        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning("WebObserver: playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
        try:
            self._pw = sync_playwright().start()
            if self.browser_url:
                self._browser = self._pw.chromium.connect_over_cdp(self.browser_url)
                self._page = self._browser.contexts[0].pages[0]
            else:
                self._browser = self._pw.chromium.launch(headless=self.headless)
                context = self._browser.new_context()
                self._page = context.new_page()

            if url:
                self._page.goto(url)

            logger.info("WebObserver: connected  url=%s", self._page.url)
            return True
        except Exception as exc:
            logger.warning("WebObserver: connect failed — %s", exc)
            # Release the driver before giving up. sync_playwright().start()
            # installs an asyncio loop in this thread, and a failed attach used
            # to leave it there: the *next* component to use Playwright - in a
            # test run, every later browser test - then died with "Sync API
            # inside the asyncio loop", pointing at code that had done nothing
            # wrong. A caller that retries connect() also needs the slate clean.
            self.disconnect()
            return False

    def disconnect(self):
        """Close browser and release Playwright resources."""
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._page = self._browser = self._pw = None

    def snapshot(self) -> Dict[str, Any]:
        """
        Capture the current browser page state.
        Returns a trace-compatible state dict matching UIAutomationObserver output.
        """
        if not self.connected:
            return _empty_state()
        try:
            return self._capture()
        except Exception as exc:
            logger.warning("WebObserver: snapshot failed — %s", exc)
            return _empty_state()

    def navigate(self, url: str) -> bool:
        """Navigate to a URL."""
        if not self.connected:
            return False
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            return True
        except Exception as exc:
            logger.warning("WebObserver: navigate failed — %s", exc)
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture(self) -> Dict[str, Any]:
        page  = self._page
        title = page.title()
        # _screen_origin finds the OS window by this title; it runs a few
        # lines below and there is no reason to ask the page twice.
        self._page_title_cache = title
        url   = page.url
        vp    = page.viewport_size or {"width": 1920, "height": 1080}
        W, H  = vp["width"], vp["height"]

        # Screen geometry, so bboxes and screen_resolution describe the same
        # space. The transformer normalises click_xy by screen_resolution, so
        # reporting the viewport size here while emitting screen bboxes would
        # scale every coordinate wrongly on top of offsetting it.
        origin = self._screen_origin(page)
        if origin:
            W, H = origin["screen_width"], origin["screen_height"]

        elements = self._extract_elements(page, W, H, origin)

        # document.activeElement, now that one pass reads it for free. The
        # agent's auto-skip and auto-fill both start from the focused element,
        # so leaving this None made every web state look like nothing had focus.
        focused = next((e["element_id"] for e in elements if e["focused"]), None)

        return {
            "application":        "browser",
            "window_title":       title,
            "process_id":         None,
            "screen_resolution":  [W, H],
            "focused_element_id": focused,
            "elements":           elements,
            "source":             "web",
            "web_context": {
                "url":   url,
                "title": title,
            },
        }

    def _content_rect_win32(self, title: str) -> Optional[Dict[str, float]]:
        """The content area's rectangle, asked of Windows rather than computed.

        Chrome puts the web content in its own child window class,
        Chrome_RenderWidgetHostHWND, whose rectangle IS the viewport - no
        borders, no toolbar, nothing to subtract. GetWindowRect answers in the
        same coordinate space the calling process sees, which is also the space
        pynput reports clicks in and pyautogui clicks in, because they are all
        this process. That shared space is the point: it holds whatever the
        DPI-awareness of this process is, on whichever monitor, with no
        reconstruction to get wrong.

        Returns None when Windows cannot be asked (another OS, no pywin32, no
        matching window), leaving the DOM-geometry path to try instead.
        """
        try:
            import win32gui
        except Exception:
            return None

        needle = (title or "").strip().lower()
        if not needle:
            return None

        found: List[int] = []

        def _visit(hwnd, _ctx):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            # Chrome's own window title is "<page title> - Google Chrome".
            if needle not in (win32gui.GetWindowText(hwnd) or "").lower():
                return True
            child: List[int] = []

            def _visit_child(ch, _c):
                # Chrome keeps several windows of this class, most of them
                # 2x2-pixel stubs. The viewport is the largest one; taking the
                # first found a stub, and every bbox collapsed onto a 2-pixel
                # square without erroring.
                if win32gui.GetClassName(ch) == "Chrome_RenderWidgetHostHWND":
                    try:
                        l, t, r, b = win32gui.GetWindowRect(ch)
                    except Exception:
                        return True
                    child.append(((r - l) * (b - t), ch))
                return True

            try:
                win32gui.EnumChildWindows(hwnd, _visit_child, None)
            except Exception:
                return True
            if child:
                found.append(max(child)[1])
            return True

        try:
            win32gui.EnumWindows(_visit, None)
        except Exception:
            return None
        if not found:
            return None

        try:
            left, top, right, bottom = win32gui.GetWindowRect(found[0])
        except Exception:
            return None
        # A viewport is not a few pixels across. Anything this small is a stub
        # or a collapsed window, and using it would silently squash every bbox.
        if right - left < 200 or bottom - top < 200:
            return None
        return {"left": float(left), "top": float(top),
                "right": float(right), "bottom": float(bottom)}

    @staticmethod
    def _virtual_screen() -> Optional[Dict[str, int]]:
        """Size of the whole desktop, every monitor included.

        screen_resolution normalises click coordinates, and a browser on a
        secondary monitor has coordinates past the primary's width - dividing
        those by one monitor's size puts them outside 0..1.
        """
        try:
            import win32api
            import win32con
            return {
                "width":  int(win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)),
                "height": int(win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)),
            }
        except Exception:
            return None

    def _screen_origin(self, page: Any) -> Optional[Dict[str, float]]:
        """Where the viewport's top-left sits on the physical screen.

        Windows is asked first (see _content_rect_win32). The DOM-geometry path
        below is the fallback, and it has a real limit worth knowing: it scales
        the whole of window.screenX by devicePixelRatio, but on a multi-monitor
        desktop that value carries the offset of every monitor to the left, and
        that offset is not in this window's scale factor. Measured on a real
        two-monitor setup - a 1920-wide primary with the laptop beside it at
        125% - every bbox came out 465 px too far right (1920 x 0.25, less the
        border term), which is enough to land every click on the wrong column
        while looking entirely plausible. Y was exact, because the monitors
        shared a top edge and 0 x 0.25 is 0.

        `window.screenX/Y` is the browser window; the content box starts inside
        the border and below the browser's own UI. The border is the leftover
        outer width split between the two sides.

        The vertical term is not simply `outerHeight - innerHeight`: a window
        has a border along the bottom but none along the top, so that
        difference is the UI *plus* one border. Measured against Chrome's
        Chrome_RenderWidgetHostHWND (the OS window that is exactly the content
        area) the naive formula sits 8 px low at every window position, 8 being
        the border - hence the subtraction.

        Everything the DOM reports is in CSS pixels, so a display running at a
        scale factor needs devicePixelRatio applied to reach the pixels
        pyautogui clicks.

        Returns None when translation is off or unavailable, which leaves
        viewport coordinates in place rather than emitting a half-applied
        offset.
        """
        if not self.screen_coords:
            return None
        try:
            geometry = page.evaluate(
                """() => ({
                    screenX: window.screenX, screenY: window.screenY,
                    outerWidth: window.outerWidth, outerHeight: window.outerHeight,
                    innerWidth: window.innerWidth, innerHeight: window.innerHeight,
                    dpr: window.devicePixelRatio || 1,
                    screenWidth: window.screen.width, screenHeight: window.screen.height,
                })"""
            )
        except Exception as exc:
            logger.warning("WebObserver: screen origin unavailable (%s) - "
                           "bboxes stay in viewport coordinates.", exc)
            return None

        inner_w = float(geometry.get("innerWidth") or 0)
        rect = self._content_rect_win32(self._page_title_cache) if inner_w else None
        # Checked here as well as inside the lookup: this is the point of use,
        # and a rect a few pixels across would collapse every bbox onto a dot
        # rather than fail, whichever source produced it.
        if rect and (rect["right"] - rect["left"] < 200
                     or rect["bottom"] - rect["top"] < 200):
            logger.warning("WebObserver: content rect %s is too small to be a "
                           "viewport - falling back to DOM geometry.", rect)
            rect = None

        if rect:
            # The scale comes from the measurement itself rather than
            # devicePixelRatio, so page zoom is included for free.
            scale = (rect["right"] - rect["left"]) / inner_w
            virtual = self._virtual_screen()
            return {
                "dx":  rect["left"],
                "dy":  rect["top"],
                "dpr": scale,
                "screen_width":  (virtual or {}).get(
                    "width", int(geometry["screenWidth"] * scale)),
                "screen_height": (virtual or {}).get(
                    "height", int(geometry["screenHeight"] * scale)),
                "source": "win32",
            }

        border = max(0.0, (geometry["outerWidth"] - geometry["innerWidth"]) / 2.0)
        chrome = max(0.0, geometry["outerHeight"] - geometry["innerHeight"] - border)
        dpr    = geometry["dpr"] or 1.0
        return {
            "dx":  (geometry["screenX"] + border) * dpr,
            "dy":  (geometry["screenY"] + chrome) * dpr,
            "dpr": dpr,
            "screen_width":  int(geometry["screenWidth"] * dpr),
            "screen_height": int(geometry["screenHeight"] * dpr),
            "source": "dom",
        }

    @staticmethod
    def _to_screen(box: Dict[str, float],
                   origin: Optional[Dict[str, float]]) -> List[int]:
        """A Playwright bounding box as a screen-coordinate bbox."""
        if origin:
            dpr, dx, dy = origin["dpr"], origin["dx"], origin["dy"]
            x1 = box["x"] * dpr + dx
            y1 = box["y"] * dpr + dy
            return [int(x1), int(y1),
                    int(x1 + box["width"] * dpr), int(y1 + box["height"] * dpr)]
        return [int(box["x"]), int(box["y"]),
                int(box["x"] + box["width"]), int(box["y"] + box["height"])]

    def _extract_elements(self, page: Any, W: int, H: int,
                          origin: Optional[Dict[str, float]] = None
                          ) -> List[Dict[str, Any]]:
        """Extract interactive and labelled elements from the page DOM."""
        elements: List[Dict[str, Any]] = []

        # Query all potentially interactive elements
        selector = (
            "input, select, textarea, button, a[href], "
            "[role='button'], [role='textbox'], [role='combobox'], "
            "[role='checkbox'], [role='radio'], [role='tab'], "
            "[role='menuitem'], [role='link']"
        )

        try:
            raw = page.evaluate(_EXTRACT_JS, {"selector": selector,
                                              "limit": self.max_elements})
        except Exception as exc:
            logger.warning("WebObserver: element extraction failed — %s", exc)
            return elements

        # Truncating silently is worse than truncating: a sheet-style page (the
        # scope #2 grade portal is 50 rows x 5 inputs) loses its last rows with
        # no sign, and the agent reports success having never seen them. Say so.
        if raw["total"] > self.max_elements:
            logger.warning(
                "WebObserver: page has %d interactive elements, capturing %d - "
                "the rest are invisible to the agent. Raise max_elements.",
                raw["total"], self.max_elements,
            )

        title = raw["title"]
        for i, item in enumerate(raw["elements"]):
            elements.append({
                "element_id":   f"web_{i}",
                "type":         _map_type(item["tag"], item["role"], item["inputType"]),
                "control_type": item["tag"],
                "bbox":         self._to_screen(item["box"], origin),
                "text":         item["name"],
                "value":        item["value"],
                "label":        item["name"],
                "enabled":      item["enabled"],
                "visible":      True,
                "focused":      item["focused"],
                "confidence":   1.0,
                "source":       "web",
                "window_role":  "active",
                "window_title": title,
            })

        return elements


# ── helpers ───────────────────────────────────────────────────────────────────

# The whole page in one round trip.
#
# This used to be a Playwright call per property per element - bounding_box,
# get_attribute x5, inner_text, input_value, is_enabled, is_visible - about ten
# CDP round trips each. On the grade portal's 303 controls that is ~3000 round
# trips, and a single snapshot took longer than the recorder's one-second
# capture interval, so a browser demo recorded zero frames. The scanner already
# reads a page this way (scope2/executor/extract_context.js); this mirrors it.
#
# The name cascade below is rule 3 of scope2/labeling/resolve.py, "aria-label /
# aria-labelledby", ahead of placeholder. A sheet-style page names its cells
# only by reference, so stopping at aria-label yields one label per *column*
# where there is really one per *cell*: on the grade portal that is a handful of
# names for 250 inputs, and the agent cannot tell row 1 from row 50.
_EXTRACT_JS = """
({selector, limit}) => {
  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();

  const ariaLabelledBy = (el) => {
    const ids = (el.getAttribute("aria-labelledby") || "").split(/\\s+/).filter(Boolean);
    if (!ids.length) return "";
    const root = el.getRootNode();
    const byId = (id) => (root && root.getElementById ? root.getElementById(id)
                                                      : document.getElementById(id));
    return ids.map((id) => { const n = byId(id); return n ? clean(n.textContent) : ""; })
              .filter(Boolean).join(" ");
  };

  const all = Array.from(document.querySelectorAll(selector));
  const out = [];
  for (const el of all) {
    if (out.length >= limit) break;

    const rect = el.getBoundingClientRect();
    if (rect.width < 4 || rect.height < 4) continue;
    const style = getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none") continue;

    const tag = el.tagName.toLowerCase();
    const name = clean(
      el.getAttribute("aria-label")
      || ariaLabelledBy(el)
      || el.getAttribute("placeholder")
      || el.getAttribute("title")
      || el.getAttribute("name")
      || el.innerText
      || ""
    ).slice(0, 120);

    const isField = tag === "input" || tag === "textarea" || tag === "select";
    out.push({
      tag: tag,
      role: el.getAttribute("role") || tag,
      inputType: tag === "input" ? (el.getAttribute("type") || "") : "",
      name: name,
      value: isField ? (el.value || "") : "",
      enabled: !el.disabled,
      focused: el === document.activeElement,
      box: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
    });
  }
  return {elements: out, total: all.length, title: document.title};
}
"""


def _map_type(tag: str, role: str, input_type: str = "") -> str:
    """Map a DOM tag/role onto the canonical vocabulary in observers/schema.py.

    These names are not cosmetic. The agent filters elements by `type`, and
    schema.CONTROL_TYPES is the only vocabulary it recognises - so emitting
    "input" where the contract says "editcontrol" does not degrade perception,
    it removes it: every element is silently dropped and the agent sees a blank
    page. That is the exact failure schema.py's own docstring warns about, and
    validate_state() flags it.

    `input_type` is the <input type="..."> attribute. Without it a checkbox and
    a text box both arrive as a tag of "input", and the agent would try to type
    into a checkbox.
    """
    _INPUT_TYPE_MAP = {
        "checkbox":  "checkboxcontrol",
        "radio":     "radiobuttoncontrol",
        "button":    "buttoncontrol",
        "submit":    "buttoncontrol",
        "reset":     "buttoncontrol",
        "image":     "buttoncontrol",
        "range":     "slidercontrol",
        "number":    "editcontrol",
    }
    _TAG_MAP = {
        "input":    "editcontrol",
        "textarea": "editcontrol",
        "select":   "comboboxcontrol",
        "button":   "buttoncontrol",
        "a":        "hyperlinkcontrol",
    }
    _ROLE_MAP = {
        "button":   "buttoncontrol",
        "textbox":  "editcontrol",
        "combobox": "comboboxcontrol",
        "checkbox": "checkboxcontrol",
        "radio":    "radiobuttoncontrol",
        "tab":      "tabitemcontrol",
        "menuitem": "menuitemcontrol",
        "link":     "hyperlinkcontrol",
    }

    if tag == "input" and input_type:
        mapped = _INPUT_TYPE_MAP.get(input_type.lower())
        if mapped:
            return mapped
    return _TAG_MAP.get(tag) or _ROLE_MAP.get(role, "customcontrol")


def _empty_state() -> Dict[str, Any]:
    return {
        "application":        "browser",
        "window_title":       "",
        "process_id":         None,
        "screen_resolution":  [1920, 1080],
        "focused_element_id": None,
        "elements":           [],
        "source":             "web",
        "web_context":        {},
    }
