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
    ):
        self.headless     = headless
        self.browser_url  = browser_url
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
        url   = page.url
        vp    = page.viewport_size or {"width": 1920, "height": 1080}
        W, H  = vp["width"], vp["height"]

        elements = self._extract_elements(page, W, H)

        return {
            "application":        "browser",
            "window_title":       title,
            "process_id":         None,
            "screen_resolution":  [W, H],
            "focused_element_id": None,
            "elements":           elements,
            "source":             "web",
            "web_context": {
                "url":   url,
                "title": title,
            },
        }

    def _extract_elements(self, page: Any, W: int, H: int) -> List[Dict[str, Any]]:
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
            handles = page.query_selector_all(selector)
        except Exception:
            return elements

        # Truncating silently is worse than truncating: a sheet-style page (the
        # scope #2 grade portal is 50 rows x 5 inputs) loses its last rows with
        # no sign, and the agent reports success having never seen them. Say so.
        if len(handles) > self.max_elements:
            logger.warning(
                "WebObserver: page has %d interactive elements, capturing %d - "
                "the rest are invisible to the agent. Raise max_elements.",
                len(handles), self.max_elements,
            )

        for i, handle in enumerate(handles[:self.max_elements]):
            try:
                box = handle.bounding_box()
                if not box or box["width"] < 4 or box["height"] < 4:
                    continue

                tag      = handle.evaluate("el => el.tagName.toLowerCase()")
                role     = handle.get_attribute("role") or tag
                name     = (
                    handle.get_attribute("aria-label")
                    or handle.get_attribute("placeholder")
                    or handle.get_attribute("title")
                    or handle.get_attribute("name")
                    or handle.inner_text()
                    or ""
                ).strip()[:120]

                value    = handle.input_value() if tag in ("input", "textarea", "select") else ""
                enabled  = handle.is_enabled()
                visible  = handle.is_visible()

                if not visible:
                    continue

                input_type = (handle.get_attribute("type") or "") if tag == "input" else ""
                elem_type = _map_type(tag, role, input_type)

                elements.append({
                    "element_id":   f"web_{i}",
                    "type":         elem_type,
                    "control_type": tag,
                    "bbox":         [
                        int(box["x"]), int(box["y"]),
                        int(box["x"] + box["width"]),
                        int(box["y"] + box["height"]),
                    ],
                    "text":         name,
                    "value":        value,
                    "label":        name,
                    "enabled":      enabled,
                    "visible":      True,
                    "focused":      False,
                    "confidence":   1.0,
                    "source":       "web",
                    "window_role":  "active",
                    "window_title": self._page.title() if self._page else "",
                })
            except Exception:
                continue

        return elements


# ── helpers ───────────────────────────────────────────────────────────────────

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
