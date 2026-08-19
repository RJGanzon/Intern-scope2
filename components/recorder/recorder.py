"""
ScreenObserver - Screen Recording + Input Tracking + Trace Pipeline

Records the screen at a fixed interval while simultaneously tracking all mouse
and keyboard events using passive pynput listeners. The built-in pipeline
translates the captured frames and inputs into trace JSON files via
TraceTranslator.

CLASSES:
    MouseInput        - Records mouse clicks/drags passively
    KeyboardInput     - Records keyboard strokes passively
    ScreenObserver    - Owns inputs + screen capture, runs the pipeline

QUICK START (run record_trace.py instead of importing directly):
    python record_trace.py

PROGRAMMATIC USAGE:
    observer = ScreenObserver(output_dir="data/output/traces/live")
    observer.start(interval_sec=1.0)
    # ... interact with your screen ...
    traces = observer.stop()          # stops recording, translates, saves JSONs
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    _PILImage = None

# ── resolve project root ────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))   # components/recorder/
_COMP_DIR   = os.path.dirname(_THIS_DIR)                   # components/
_INTERN_DIR = os.path.dirname(_COMP_DIR)                   # Intern/
for _p in (_INTERN_DIR, _COMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── optional dependencies ─────────────────────────────────────────────────────

try:
    from pynput import mouse as _pynput_mouse, keyboard as _pynput_keyboard
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

try:
    import mss
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False

try:
    import pyperclip
    _PYPERCLIP_AVAILABLE = True
except ImportError:
    _PYPERCLIP_AVAILABLE = False

try:
    from observers.excel_observer import ExcelObserver as _ExcelObserver
    _EXCEL_OBSERVER_AVAILABLE = True
except ImportError:
    try:
        from components.observers.excel_observer import ExcelObserver as _ExcelObserver
        _EXCEL_OBSERVER_AVAILABLE = True
    except ImportError:
        _EXCEL_OBSERVER_AVAILABLE = False

try:
    from observers.ui_observer import UIAutomationObserver as _UIAObserver
    _UIA_OBSERVER_AVAILABLE = True
except ImportError:
    try:
        from components.observers.ui_observer import UIAutomationObserver as _UIAObserver
        _UIA_OBSERVER_AVAILABLE = True
    except ImportError:
        _UIA_OBSERVER_AVAILABLE = False

def _cdp_reachable(browser_url: str, timeout: float = 2.0) -> bool:
    """Is a browser listening on this CDP endpoint?

    Asked over plain HTTP rather than by attaching, because the decision has to
    be made on the constructing thread while the Playwright session belongs to
    the capture thread. /json/version is CDP's cheapest endpoint and needs no
    driver.
    """
    import json as _json
    import urllib.request as _urlreq

    try:
        with _urlreq.urlopen(f"{browser_url.rstrip('/')}/json/version",
                             timeout=timeout) as response:
            _json.load(response)
        return True
    except Exception:
        return False


try:
    from observers.web_observer import WebObserver as _WebObserver
    _WEB_OBSERVER_AVAILABLE = True
except ImportError:
    try:
        from components.observers.web_observer import WebObserver as _WebObserver
        _WEB_OBSERVER_AVAILABLE = True
    except ImportError:
        _WEB_OBSERVER_AVAILABLE = False

try:
    from observers.vlm.vision_observer.cv_vision_observer import CVVisionObserver as _CVVisionObserver
    _VISION_OBSERVER_AVAILABLE = True
except ImportError:
    try:
        from components.observers.vlm.vision_observer.cv_vision_observer import CVVisionObserver as _CVVisionObserver
        _VISION_OBSERVER_AVAILABLE = True
    except ImportError:
        _VISION_OBSERVER_AVAILABLE = False


# =============================================================================
# MOUSE INPUT
# =============================================================================

class MouseInput:
    """
    Passively listens to mouse events and records them as structured actions.

    Each action:
        {
            "id":        "mouse_action_NNNN",
            "position":  [x, y],
            "type":      "click" | "double_click" | "drag" | "highlight",
            "timestamp": "<ISO-8601>"
        }

    Usage:
        inp = MouseInput()
        inp.start()
        inp.stop()
        actions = inp.get_actions()
    """

    DOUBLE_CLICK_THRESHOLD = 0.35   # seconds between two clicks = double-click
    DRAG_THRESHOLD = 5              # pixel distance before press+move = drag

    def __init__(self):
        self._actions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._listener: Optional[Any] = None
        self._action_counter = 0
        self._press_pos: Optional[Tuple[int, int]] = None
        self._press_time: Optional[float] = None
        self._last_click_time: float = 0.0
        self._last_click_pos: Optional[Tuple[int, int]] = None
        self._dragging: bool = False

    def start(self):
        if not _PYNPUT_AVAILABLE:
            print("Warning: pynput not installed — mouse events will not be recorded.")
            return
        if self._listener is not None:
            return
        self._listener = _pynput_mouse.Listener(
            on_click=self._on_click, on_move=self._on_move)
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None

    def get_actions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._actions)

    def clear(self):
        with self._lock:
            self._actions.clear()
            self._action_counter = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_move(self, x: int, y: int):
        if self._press_pos is not None and not self._dragging:
            px, py = self._press_pos
            if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= self.DRAG_THRESHOLD:
                self._dragging = True

    def _on_click(self, x: int, y: int, button, pressed: bool):
        if pressed:
            self._press_pos = (x, y)
            self._dragging = False
        else:
            if self._press_pos is None:
                return
            now = time.time()
            if self._dragging:
                action_type = "drag"
            elif (
                self._last_click_pos is not None
                and (now - self._last_click_time) <= self.DOUBLE_CLICK_THRESHOLD
                and abs(x - self._last_click_pos[0]) <= 5
                and abs(y - self._last_click_pos[1]) <= 5
            ):
                action_type = "double_click"
                with self._lock:
                    if self._actions and self._actions[-1]["type"] == "click":
                        self._actions.pop()
            else:
                action_type = "click"

            self._record(action_type, x, y)
            self._last_click_time = now
            self._last_click_pos = (x, y)
            self._press_pos = None
            self._dragging = False

    def _record(self, action_type: str, x: int, y: int):
        with self._lock:
            self._actions.append({
                "id": f"mouse_action_{self._action_counter:04d}",
                "position": [x, y],
                "type": action_type,
                "timestamp": datetime.now().isoformat(),
            })
            self._action_counter += 1


# =============================================================================
# KEYBOARD INPUT
# =============================================================================

class KeyboardInput:
    """
    Passively listens to keyboard events and groups them into stroke sequences.

    Each action:
        { "strokes": ["a", "b", "Key.enter", ...] }

    A new group is started after GROUP_TIMEOUT seconds of inactivity.

    Usage:
        inp = KeyboardInput()
        inp.start()
        inp.stop()
        actions = inp.get_actions()
    """

    GROUP_TIMEOUT = 1.0  # seconds of inactivity before opening a new group

    def __init__(self, clipboard: Optional["ClipboardMonitor"] = None):
        self._actions: List[Dict[str, Any]] = []
        self._current_strokes: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._listener: Optional[Any] = None
        self._last_key_time: float = 0.0
        self._clipboard = clipboard

    def start(self):
        if not _PYNPUT_AVAILABLE:
            print("Warning: pynput not installed — keyboard events will not be recorded.")
            return
        if self._listener is not None:
            return
        self._listener = _pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._flush()

    def get_actions(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._flush_locked()
            return list(self._actions)

    def clear(self):
        with self._lock:
            self._actions.clear()
            self._current_strokes.clear()

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_press(self, key):
        now = time.time()
        key_name = self._key_name(key)
        with self._lock:
            if self._current_strokes and (now - self._last_key_time) > self.GROUP_TIMEOUT:
                self._flush_locked()

            stroke: Dict[str, Any] = {
                "key":       key_name,
                "timestamp": datetime.now().isoformat(),
            }

            # Ctrl+C (\x03) — snapshot clipboard in background so we don't block
            if key_name == "\x03" and self._clipboard:
                threading.Thread(
                    target=self._clipboard.record_copy, daemon=True
                ).start()

            # Ctrl+V (\x16) — attach pasted text to this stroke record
            if key_name == "\x16" and self._clipboard:
                pasted = self._clipboard.record_paste()
                if pasted:
                    stroke["pasted_text"] = pasted

            self._current_strokes.append(stroke)
            self._last_key_time = now

    def _flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if self._current_strokes:
            self._actions.append({"strokes": list(self._current_strokes)})
            self._current_strokes.clear()

    @staticmethod
    def _key_name(key) -> str:
        try:
            return key.char
        except AttributeError:
            return str(key)


# =============================================================================
# CLIPBOARD MONITOR
# =============================================================================

class ClipboardMonitor:
    """
    Captures clipboard text content at copy and paste events.

    Integrates with KeyboardInput — when Ctrl+C (\x03) is detected the
    clipboard is snapshotted in a background thread; when Ctrl+V (\x16) is
    detected the last known content is attached to the stroke record so the
    trace carries the full pasted text.

    Each event:
        {
            "event":     "copy" | "paste",
            "content":   "<clipboard text>",
            "timestamp": "<ISO-8601>"
        }
    """

    def __init__(self):
        self._last_content: str = ""
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def snapshot(self) -> str:
        """Read the current clipboard text. Returns empty string on failure."""
        if _PYPERCLIP_AVAILABLE:
            try:
                return pyperclip.paste() or ""
            except Exception:
                pass
        try:
            import tkinter as tk
            r = tk.Tk()
            r.withdraw()
            content = r.clipboard_get()
            r.destroy()
            return content
        except Exception:
            return ""

    def record_copy(self) -> str:
        """
        Called after Ctrl+C. Waits briefly for the OS to finish copying,
        then snapshots and stores the clipboard content.
        """
        time.sleep(0.08)   # let the OS complete the copy operation
        content = self.snapshot()
        with self._lock:
            self._last_content = content
            self._events.append({
                "event":     "copy",
                "content":   content,
                "timestamp": datetime.now().isoformat(),
            })
        return content

    def record_paste(self) -> str:
        """
        Called on Ctrl+V. Records a paste event using the last known
        clipboard content and returns it so it can be attached to the stroke.
        """
        with self._lock:
            content = self._last_content
            self._events.append({
                "event":     "paste",
                "content":   content,
                "timestamp": datetime.now().isoformat(),
            })
        return content

    def get_last(self) -> str:
        with self._lock:
            return self._last_content

    def get_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def clear(self):
        with self._lock:
            self._events.clear()
            self._last_content = ""

    def prime(self):
        """Snapshot clipboard at recording start to catch pre-existing content."""
        content = self.snapshot()
        with self._lock:
            self._last_content = content


# =============================================================================
# SCREEN OBSERVER  (+ built-in pipeline)
# =============================================================================

class ScreenObserver:
    """
    Orchestrates screen capture, input tracking, and trace translation.

    Captures full-screen frames at a fixed interval using ``mss`` while running
    ``MouseInput`` and ``KeyboardInput`` listeners in the background.  When
    stopped, it automatically feeds the captured frames into ``TraceTranslator``
    (CV/OCR) and writes one trace JSON per consecutive frame pair.

    Args:
        output_dir:   Where trace JSONs are saved (created if needed).
        trace_type:   "web" | "excel" | "gui"  (written into every trace, and
                      selects the observer: "web" attaches to a browser over
                      CDP, "excel" uses COM, anything else uses UIAutomation).
        application:  Optional app-name tag passed to TraceTranslator.
        monitor:      mss monitor index (1 = primary screen).
        browser_url:  CDP endpoint for trace_type="web". The browser must
                      already be running with --remote-debugging-port.
        max_elements: Cap on elements captured per web snapshot. The grade
                      portal is 303, so the default 200 of older builds
                      silently truncated it.

    Usage:
        observer = ScreenObserver(output_dir="data/output/traces/live")
        observer.start(interval_sec=1.0)
        # ... interact with your screen ...
        traces = observer.stop()        # blocks briefly while translating

    Recording a browser demo:
        chrome.exe --remote-debugging-port=9222
        observer = ScreenObserver(trace_type="web")   # attaches to that Chrome
    """

    def __init__(
        self,
        output_dir: str = "data/output/traces/live",
        trace_type: str = "gui",
        application: Optional[str] = None,
        monitor: int = 1,
        continual_learner: Optional[Any] = None,
        perception: str = "auto",
        browser_url: str = "http://localhost:9222",
        max_elements: int = 1000,
    ):
        if not _MSS_AVAILABLE:
            raise ImportError(
                "mss is required for screen capture. Install with: pip install mss"
            )

        # Each recording session gets its own timestamped subfolder
        # e.g. data/output/traces/live/session_20260321_143012
        from datetime import datetime as _dt
        _session_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"session_{_session_ts}")
        self.base_traces_dir = output_dir   # root dir, used for gitignore placement
        self.trace_type = trace_type
        self.application = application
        self.monitor_index = monitor

        self.clipboard = ClipboardMonitor()
        self.mouse    = MouseInput()
        self.keyboard = KeyboardInput(clipboard=self.clipboard)

        # Priority 0: VISION (screenshot + CV/OCR) when perception="vision".
        # Records demos as the agent will SEE them at run time — pixels, not the
        # accessibility tree. Run on the full-screen frame the capture loop already
        # grabs, so element bboxes are absolute screen coords matching the recorded
        # (also absolute) mouse clicks — which is exactly what training needs to map
        # a click to the element it landed on. Skips Excel/UIA when active.
        self._vision_observer: Optional[Any] = None
        if perception == "vision":
            if not _VISION_OBSERVER_AVAILABLE:
                raise ImportError("perception='vision' needs the CV vision observer "
                                  "(opencv-python, pytesseract, mss, Pillow).")
            self._vision_observer = _CVVisionObserver()   # image set per frame
            print("[ScreenObserver] VISION perception active — recording demos from pixels (CV+OCR).")

        # Priority 1: WebObserver (DOM state) when trace_type="web".
        #
        # Without this a browser demo is recorded through UIAutomation, which
        # sees Chrome's accessibility tree rather than the DOM. On a sheet-style
        # page that loses the one thing the trace needs: UIA does not resolve
        # aria-labelledby, so all fifty rows of a column arrive under one name
        # and the trace cannot say which row the human was filling. The demo
        # looks fine and trains a model that cannot tell rows apart.
        #
        # This attaches over CDP to a browser the human is already driving, so
        # it needs one started with --remote-debugging-port=9222. Attaching (not
        # launching) is the point: the demonstration has to happen in the
        # operator's own browser.
        # The Playwright session is created by the capture thread, not here.
        # Playwright's sync API is thread-affine: a session built on this thread
        # and used from the capture thread raises greenlet "cannot switch to a
        # different thread", which snapshot() catches and turns into an empty
        # state - so every frame records zero elements while the recorder still
        # prints "Semantic mode (web)" and saves perfectly well-formed, empty
        # traces. Whether a browser is *there* is decided now, over plain HTTP,
        # so the fallback to UIA can still be chosen before recording starts.
        self._web_observer: Optional[Any] = None
        self._web_wanted:   bool          = False
        self._browser_url                 = browser_url
        self._max_elements                = max_elements
        if self._vision_observer is None and trace_type == "web":
            if not _WEB_OBSERVER_AVAILABLE:
                print("[ScreenObserver] trace_type='web' but WebObserver is unavailable "
                      "(pip install playwright) — falling back to UIAutomation.")
            elif not _cdp_reachable(browser_url):
                print(f"[ScreenObserver] No browser answering at {browser_url}. "
                      "Start Chrome with --remote-debugging-port=9222, or recording "
                      "will fall back to UIAutomation and lose per-row labels.")
            else:
                self._web_wanted = True
                print(f"[ScreenObserver] WebObserver will attach at {browser_url} — "
                      "DOM state active (aria-labelledby resolved).")

        # Priority 2: ExcelObserver (semantic COM state) when trace_type="excel"
        self._excel_observer: Optional[Any] = None
        if (self._vision_observer is None and not self._web_wanted
                and trace_type == "excel" and _EXCEL_OBSERVER_AVAILABLE):
            self._excel_observer = _ExcelObserver()
            if self._excel_observer.connect():
                print("[ScreenObserver] ExcelObserver connected — Excel semantic mode active.")
            else:
                print("[ScreenObserver] ExcelObserver could not connect — trying UIAutomation.")
                self._excel_observer = None

        # Priority 3: UIAutomationObserver — works for all apps, no OCR needed
        self._uia_observer: Optional[Any] = None
        if (self._vision_observer is None and self._excel_observer is None
                and not self._web_wanted and _UIA_OBSERVER_AVAILABLE):
            obs = _UIAObserver()
            if obs.available:
                self._uia_observer = obs
                print("[ScreenObserver] UIAutomationObserver active — semantic state enabled for all apps.")
            else:
                print("[ScreenObserver] UIAutomation unavailable — falling back to OCR.")

        # Priority 4: OCR via TraceTranslator (fallback only)
        if (self._excel_observer is None and self._uia_observer is None
                and not self._web_wanted):
            print("[ScreenObserver] Using OCR fallback (TraceTranslator).")

        self._continual_learner = continual_learner

        self._frames: List[Any] = []   # (ts, img) or (ts, img, semantic_state)
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._interval_sec: float = 1.0  # default; overwritten by start()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, interval_sec: float = 2.0):
        """Begin recording (non-blocking)."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            print("ScreenObserver is already running.")
            return

        self._interval_sec = interval_sec
        self._frames.clear()
        self._stop_event.clear()
        self.mouse.clear()
        self.keyboard.clear()
        self.clipboard.clear()
        self.clipboard.prime()   # snapshot any pre-existing clipboard content

        self.mouse.start()
        self.keyboard.start()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="ScreenObserver-capture")
        self._capture_thread.start()
        print(f"ScreenObserver started  [interval={interval_sec}s | output={self.output_dir}]")
        print("Press Ctrl+C (or call observer.stop()) to finish recording.\n")

    def stop(self) -> List[Dict[str, Any]]:
        """Stop recording, translate frames to traces, save JSONs, return traces."""
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None

        self.mouse.stop()
        self.keyboard.stop()

        # No WebObserver teardown here: the capture thread created the session
        # and tears it down itself, because disconnect() is as thread-affine as
        # snapshot() is. The join above has already waited for that.

        mouse_actions      = self.mouse.get_actions()
        keyboard_actions   = self.keyboard.get_actions()
        clipboard_events   = self.clipboard.get_events()
        frames             = list(self._frames)   # list of (timestamp_str, PIL_image)

        print(
            f"\nRecording stopped — {len(frames)} frames | "
            f"{len(mouse_actions)} mouse actions | "
            f"{len(keyboard_actions)} keyboard groups | "
            f"{len(clipboard_events)} clipboard events"
        )

        return self._translate_and_save(
            frames=frames,
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            clipboard_events=clipboard_events,
        )

    def record(self, duration_sec: float, interval_sec: float = 1.0) -> List[Dict[str, Any]]:
        """Blocking convenience: start → wait → stop."""
        self.start(interval_sec=interval_sec)
        time.sleep(duration_sec)
        return self.stop()

    # ── pipeline (private) ────────────────────────────────────────────────────

    @staticmethod
    def _derive_action_from(
        step_mouse: list,
        step_strokes: list,
        step_clipboard: list,
    ) -> dict:
        """
        Convert raw mouse/keyboard/clipboard events for one trace step into
        a structured action dict that BCTrainer can learn from.

        Priority:
          1. Clipboard paste  → action_type="paste"
          2. Keyboard strokes → action_type="keyboard"
          3. Mouse click/drag → action_type="click" / "double_click" / "drag"
          4. Nothing          → action_type="noop"
        """
        # 1. Clipboard paste
        if step_clipboard:
            text = step_clipboard[-1].get("text", "")
            click_pos = None
            if step_mouse:
                pos = step_mouse[-1].get("position", [])
                if len(pos) == 2:
                    click_pos = [int(pos[0]), int(pos[1])]
            return {
                "action_type":    "paste",
                "text":           text,
                "click_position": click_pos,
            }

        # 2. Keyboard text
        if step_strokes:
            # Keys with zero standalone effect -- meaningless without a
            # companion key (a lone Shift/Ctrl/Alt/Win/Caps Lock press does
            # nothing to any GUI by itself). Deliberately NARROWER than the
            # old single _IGNORE set used below: navigation keys (Tab,
            # arrows, Escape, Page Up/Down, Home/End, Insert/Delete, F-keys)
            # are excluded from this set on purpose -- those DO have real,
            # independent effects (Tab moves focus, Escape closes a
            # dropdown) that validate_transitions.py's check_transition()
            # already correctly validates via its own focus/value-changed
            # check for empty-text keyboard actions. Confirmed live 2026-08-11
            # before shipping: an earlier draft of this fix folded navigation
            # keys into the same suppression and turned a real, legitimate
            # lone Tab press into action_type="noop" -- destroying real
            # training signal (Tab correctly moving focus is a GOOD example
            # to train on), not fixing anything. Caught by testing
            # _derive_action_from({'key': 'tab'}) directly before committing.
            _PURE_MODIFIERS = {"shift", "ctrl", "alt", "win", "caps lock"}
            _IGNORE = _PURE_MODIFIERS | {
                       "tab", "esc", "escape", "up", "down", "left", "right",
                       "page up", "page down", "home", "end", "insert", "delete",
                       "f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12"}
            chars = []
            saw_backspace = False
            saw_meaningful_key = False   # any key besides a bare modifier / blank
            for stroke in step_strokes:
                key = stroke.get("key", "")
                low = key.lower()
                if low and low not in _PURE_MODIFIERS:
                    saw_meaningful_key = True
                if low == "backspace":
                    saw_backspace = True
                    if chars:
                        chars.pop()
                elif low == "space":
                    chars.append(" ")
                elif low == "enter" or low == "return":
                    chars.append("\n")
                elif low not in _IGNORE and len(key) == 1:
                    chars.append(key)
            typed_text = "".join(chars)
            # Suppress the keyboard-action classification ONLY when every
            # stroke was either a blank/malformed key value or a bare
            # modifier with nothing else in the group -- genuinely nothing
            # for the model to learn from either way. Found 2026-08-11
            # tracing a real failure: session_20260808_144216 step 0 had
            # keystrokes=[""] (an empty/malformed key value), text="", and
            # zero observable effect -- exactly the "mistrains the model on
            # a transition that didn't really happen" case
            # validate_transitions.py's own docstring warns about. Falling
            # through to the mouse/noop checks below treats this the same
            # as if step_strokes had been empty in the first place --
            # semantically correct, and noop steps are already excluded
            # from validate_transitions.py's actionable-transitions
            # denominator. Every navigation key (Tab/arrows/Escape/etc.)
            # still produces a real keyboard action even with empty text,
            # same as before this fix.
            if saw_meaningful_key:
                click_pos = None
                if step_mouse:
                    pos = step_mouse[0].get("position", [])
                    if len(pos) == 2:
                        click_pos = [int(pos[0]), int(pos[1])]
                return {
                    "action_type":    "keyboard",
                    "text":           typed_text,
                    "keystrokes":     [s.get("key", "") for s in step_strokes],
                    "click_position": click_pos,
                }

        # 3. Mouse action
        if step_mouse:
            evt = step_mouse[-1]
            evt_type = evt.get("type", "click")
            pos  = evt.get("position", [])
            pos2 = evt.get("end_position", [])
            click_pos = [int(pos[0]), int(pos[1])] if len(pos) == 2 else None
            if evt_type == "drag" and len(pos2) == 2:
                return {
                    "action_type":    "drag",
                    "click_position": click_pos,
                    "end_position":   [int(pos2[0]), int(pos2[1])],
                }
            return {
                "action_type":    evt_type,   # "click" or "double_click"
                "click_position": click_pos,
            }

        # 4. Nothing happened
        return {"action_type": "noop"}

    def _translate_and_save(
        self,
        frames: List[Any],
        mouse_actions: List[Dict],
        keyboard_actions: List[Dict],
        clipboard_events: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert frame pairs + inputs into trace JSONs via TraceTranslator.

        Each trace covers exactly one State1 -> State2 step.  Mouse and
        keyboard actions are filtered to only those that occurred during the
        time window between the two frames, so every trace gets its own
        independent input snapshot rather than the full-session dump.
        """
        if len(frames) < 2:
            print("Need at least 2 frames to build a trace — nothing saved.")
            return []

        # Frames are always 3-tuples (ts, img, semantic_state|None)
        frame_ts        = [f[0] for f in frames]
        frame_imgs      = [f[1] for f in frames]
        raw_states      = [f[2] for f in frames]
        _semantic_mode  = any(s is not None for s in raw_states)
        semantic_states = [s if s is not None else {} for s in raw_states]

        if _semantic_mode:
            source_label = semantic_states[0].get("source", "semantic") if semantic_states else "semantic"
            print(f"\nSemantic mode ({source_label}) — {len(frame_imgs)} frames (no OCR needed).")
            states = semantic_states
        else:
            # OCR fallback via TraceTranslator
            from trace_translator.trace_translator import TraceTranslator
            translator = TraceTranslator(use_cv=True, use_html=False)

            print(f"\nOCR fallback — translating {len(frame_imgs)} frames ...")
            states: List[Dict[str, Any]] = []
            for i, img in enumerate(frame_imgs):
                print(f"  [{i+1}/{len(frame_imgs)}] ", end="", flush=True)
                state = translator._state_from_pil(
                    img,
                    source_label=f"frame_{i:04d}",
                    application=self.application,
                )
                states.append(state)
                print(f"{len(state['elements'])} elements")

        os.makedirs(self.output_dir, exist_ok=True)
        traces: List[Dict[str, Any]] = []

        # Each session folder starts from step 0 (folder name provides uniqueness)
        _step_offset = 0

        _clipboard_events = clipboard_events or []

        # ── session manifest (background text, written once per session) ───────
        # Background elements (e.g. Notepad) contain the full source document in
        # their `text` field (up to 815 KB). Storing this in every step bloats
        # traces to ~4 MB each (22 GB total across sessions). We extract the full
        # text once, write it to session_manifest.json, and strip it from step
        # files. Training code reads the manifest to restore the text for
        # _find_source_elem_idx without losing any model training signal.
        _manifest_path = os.path.join(self.output_dir, "session_manifest.json")
        if not os.path.exists(_manifest_path):
            _bg_store: Dict[str, Any] = {}
            for _s in states:
                for _e in _s.get("elements", []):
                    if _e.get("window_role") != "background":
                        continue
                    _t = (_e.get("text")  or "")
                    _v = (_e.get("value") or "")
                    if len(_t) <= 500 and len(_v) <= 500:
                        continue  # not a large blob; no need to externalise
                    _key = ((_e.get("window_title") or "") + "|"
                            + (_e.get("app") or ""))
                    if _key not in _bg_store:
                        _bg_store[_key] = {
                            "window_title": _e.get("window_title", ""),
                            "app":          _e.get("app", ""),
                            "text":         _t,
                            "value":        _v,
                        }
            if _bg_store:
                with open(_manifest_path, "w", encoding="utf-8") as _mf:
                    json.dump({"background": _bg_store}, _mf, ensure_ascii=False)

        def _strip_bg(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Return state with background element text stripped (stored in manifest)."""
            elems = state_dict.get("elements", [])
            stripped = []
            for _e in elems:
                if (_e.get("window_role") == "background"
                        and (len(_e.get("text", "") or "") > 500
                             or len(_e.get("value", "") or "") > 500)):
                    _e = dict(_e)
                    _e["text"]  = ""
                    _e["value"] = (_e.get("value") or "")[:500]
                stripped.append(_e)
            out = dict(state_dict)
            out["elements"] = stripped
            return out

        for i in range(len(states) - 1):
            t_start = frame_ts[i]
            t_end   = frame_ts[i + 1]

            # Filter mouse actions that occurred in this frame's window
            step_mouse = [
                a for a in mouse_actions
                if t_start <= a["timestamp"] < t_end
            ]

            # Filter keyboard strokes that occurred in this frame's window
            step_strokes = [
                stroke
                for group in keyboard_actions
                for stroke in group["strokes"]
                if t_start <= stroke["timestamp"] < t_end
            ]
            step_kb = [{"strokes": step_strokes}] if step_strokes else []

            # Filter clipboard events that occurred in this frame's window
            step_clipboard = [
                ev for ev in _clipboard_events
                if t_start <= ev["timestamp"] < t_end
            ]

            # Compute real duration from frame timestamps
            try:
                from datetime import datetime as _dt
                duration = (
                    _dt.fromisoformat(t_end) - _dt.fromisoformat(t_start)
                ).total_seconds()
            except Exception:
                duration = self._interval_sec

            step_idx = _step_offset + i
            diff = {}
            if not _semantic_mode:
                raw = translator.states_to_trace(states[i], states[i + 1],
                                                 trace_id=f"live_step_{step_idx:04d}")
                diff = raw.get("diff", {})
            trace = {
                "trace_id":  f"live_step_{step_idx:04d}",
                "timestamp": t_start,
                "duration":  duration,
                "type":      self.trace_type,
                "state":     _strip_bg(_fmt_state(states[i])),
                # next_state removed — it equals the next step's state.
                # Training code never reads it; eval reads consecutive files.
                # Full background text externalised to session_manifest.json.
                "mouse":     {"actions": step_mouse},
                "keyboard":  {"actions": step_kb},
                "clipboard": {"events": step_clipboard},
                "diff":      diff,
                "action":    ScreenObserver._derive_action_from(step_mouse, step_strokes, step_clipboard),
            }
            out_path = os.path.join(self.output_dir, f"live_step_{step_idx:04d}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
            traces.append(trace)

            # Notify ContinualLearner so it can trigger background retraining
            if self._continual_learner is not None:
                try:
                    self._continual_learner.add_trace(out_path)
                except Exception:
                    pass

        print(f"\nSaved {len(traces)} trace(s) -> {self.output_dir}")
        return traces

    # ── capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self):
        # Own the browser session for the lifetime of this thread. Playwright's
        # sync API cannot be handed across threads, so it is built here, used
        # here, and released here.
        if self._web_wanted:
            obs = _WebObserver(browser_url=self._browser_url,
                               max_elements=self._max_elements)
            if obs.available and obs.connect():
                self._web_observer = obs
            else:
                print(f"[ScreenObserver] WebObserver failed to attach at "
                      f"{self._browser_url} — this recording has no DOM state.")
        try:
            self._capture_frames()
        finally:
            if self._web_observer is not None:
                try:
                    # Detaching over CDP leaves the operator's browser open -
                    # they were driving it. But Playwright holds a node driver
                    # subprocess, so the session itself must be released.
                    self._web_observer.disconnect()
                except Exception as exc:
                    print(f"[ScreenObserver] WebObserver disconnect failed: {exc}")
                self._web_observer = None

    def _capture_frames(self):
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            while not self._stop_event.is_set():
                ts   = datetime.now().isoformat()
                shot = sct.grab(monitor)
                img  = _PILImage.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

                if self._vision_observer is not None:
                    # Run CV/OCR on the frame we just grabbed (full primary monitor
                    # → bboxes are absolute screen coords, matching mouse clicks).
                    self._vision_observer._image = img
                    semantic_state = self._vision_observer.snapshot()
                    self._frames.append((ts, img, semantic_state))
                elif self._web_observer is not None:
                    # WebObserver emits screen-coordinate bboxes, the same space
                    # as the recorded mouse clicks, so a click maps onto the DOM
                    # element it actually landed on.
                    semantic_state = self._web_observer.snapshot()
                    self._frames.append((ts, img, semantic_state))
                elif self._excel_observer is not None:
                    semantic_state = self._excel_observer.snapshot()
                    self._frames.append((ts, img, semantic_state))
                elif self._uia_observer is not None:
                    semantic_state = self._uia_observer.snapshot()
                    # If UIA returned very few active elements (≤8) the
                    # foreground app is likely a Tkinter / non-UIA window.
                    # Drop the UIA state so the frame falls through to OCR.
                    active_count = sum(
                        1 for e in semantic_state.get("elements", [])
                        if e.get("window_role") == "active"
                        and e.get("type") not in ("windowcontrol", "titlebarcontrol",
                                                   "menubarcontrol", "menuitemcontrol",
                                                   "panecontrol", "buttoncontrol")
                    )
                    if active_count >= 1:
                        self._frames.append((ts, img, semantic_state))
                    else:
                        # Tkinter or other UIA-opaque app — use OCR for this frame
                        self._frames.append((ts, img, None))
                else:
                    self._frames.append((ts, img, None))

                self._stop_event.wait(timeout=self._interval_sec)


# =============================================================================
# HELPERS
# =============================================================================

def _fmt_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return trace-format fields from a raw state dict.

    Updated trace format fields
    ---------------------------
    application        : str   — process name (e.g. "chrome.exe", "EXCEL.EXE")
    window_title       : str   — foreground window title
    process_id         : int|None — OS process ID
    screen_resolution  : [w, h]
    focused_element_id : str|None — element_id of the focused element
    source             : str   — "uia" | "excel_com" | "ocr" | "uia_unavailable"
    elements           : list  — see element schema below

    Element schema
    --------------
    element_id   : str            — unique id within this state
    type         : str            — simplified control type (button, input, label…)
    control_type : str            — raw UIA ControlTypeName (Button, Edit, Text…)
    bbox         : [x1,y1,x2,y2] — screen coordinates
    text         : str            — visible text / name
    value        : str            — current value (edit controls)
    label        : str            — same as text (kept for model compat)
    automation_id: str            — UIA AutomationId
    class_name   : str            — Windows class name
    enabled      : bool
    visible      : bool
    focused      : bool           — True for the currently focused element
    confidence   : float          — 1.0 for UIA/COM; 0–1 for OCR
    source       : str            — "uia" | "excel_com" | "ocr"
    metadata     : dict           — extra per-source data
    """
    out = {
        "application":        state.get("application", "Unknown"),
        "window_title":       state.get("window_title", ""),
        "process_id":         state.get("process_id"),
        "screen_resolution":  state.get("screen_resolution", [0, 0]),
        "focused_element_id": state.get("focused_element_id"),
        "source":             state.get("source", "ocr"),
        "elements":           state.get("elements", []),
    }
    # Carry through Excel-specific semantic context when present
    if "excel_context" in state:
        out["excel_context"] = state["excel_context"]
    return out


def _notepad_uia_text() -> tuple:
    """Return (full_text, selected_text) from Notepad via UIA TextPattern."""
    try:
        import uiautomation as _auto
        root = _auto.GetRootControl()
        np_ctrl = None
        for child in root.GetChildren():
            name = (child.Name or "").lower()
            if "notepad" in name or ".txt" in name:
                np_ctrl = child
                break
        if not np_ctrl:
            return ("", "")
        # Win11 Notepad uses DocumentControl; classic uses EditControl
        doc = None
        for ctrl_type in (_auto.ControlType.DocumentControl,
                          _auto.ControlType.EditControl):
            try:
                doc = np_ctrl.Control(ControlType=ctrl_type, searchDepth=8)
                if doc.Exists(0):
                    break
                doc = None
            except Exception:
                doc = None
        if not doc:
            return ("", "")
        try:
            tp = doc.GetPattern(_auto.PatternId.TextPattern)
            full_range = tp.DocumentRange
            full_text  = full_range.GetText(-1)
            sels = tp.GetSelection()
            sel_text = sels[0].GetText(-1) if sels else ""
            return (full_text, sel_text)
        except Exception:
            return ("", "")
    except Exception:
        return ("", "")


def _get_notepad_line_at(x: int, y: int) -> str:
    """Return visible Notepad lines near click position."""
    return _get_notepad_visible_lines(2)


def _get_notepad_selection() -> str:
    """Return currently selected text in Notepad via UIA TextPattern."""
    try:
        _, sel = _notepad_uia_text()
        return sel.strip()[:80] if sel else ""
    except Exception:
        return ""


def _is_separator(line: str) -> bool:
    """True if line is purely decorative (separators, box-drawing, etc.)."""
    if not line:
        return True
    import unicodedata
    clean = line.strip()
    if not clean:
        return True
    unique_cats = {unicodedata.category(c) for c in clean}
    # Po = punctuation other, Pd = dash, So = symbol other (box drawing), Sm = math symbol
    if unique_cats <= {"Po", "Pd", "So", "Sm", "Zs", "Cc", "Cf"}:
        return True
    # all same character repeated
    if len(set(clean)) <= 2 and len(clean) > 3:
        return True
    # contains a colon = likely a field line like "First Name: John"
    return False


def _timeout_call(fn, timeout_sec: float = 1.0, default=""):
    """Run fn() in a daemon thread; return its result or `default` if it hangs.
    Guards against UWP-Notepad UIA calls that can block indefinitely."""
    import threading as _t
    result = [default]
    def _run():
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0x2)
        except Exception:
            pass
        try:
            result[0] = fn()
        except Exception:
            pass
    th = _t.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout_sec)
    return result[0]


def _get_notepad_visible_lines(max_lines: int = 30) -> str:
    """Return all currently visible lines in Notepad using UIA GetVisibleRanges."""
    try:
        import uiautomation as _auto
        root = _auto.GetRootControl()
        np_ctrl = None
        for child in root.GetChildren():
            name = (child.Name or "").lower()
            if "notepad" in name or ".txt" in name:
                np_ctrl = child
                break
        if not np_ctrl:
            return ""

        doc = None
        for ctrl_type in (_auto.ControlType.DocumentControl,
                          _auto.ControlType.EditControl):
            try:
                candidate = np_ctrl.Control(ControlType=ctrl_type, searchDepth=8)
                if candidate.Exists(0):
                    doc = candidate
                    break
            except Exception:
                pass
        if not doc:
            return ""

        tp = doc.GetPattern(_auto.PatternId.TextPattern)

        # GetVisibleRanges returns only what's actually on screen
        try:
            ranges = tp.GetVisibleRanges()
            if ranges:
                visible_text = "".join(r.GetText(-1) for r in ranges)
            else:
                visible_text = tp.DocumentRange.GetText(-1)
        except Exception:
            visible_text = tp.DocumentRange.GetText(-1)

        lines = []
        for line in visible_text.splitlines():
            line = line.strip()
            if line and not _is_separator(line):
                lines.append(line)
            if len(lines) >= max_lines:
                break
        return "\n".join(lines)
    except Exception:
        return ""


def _first_meaningful_line(elems) -> str:
    # Try UIA first (works on Win11 Notepad)
    result = _get_notepad_visible_lines(3)
    if result:
        return result
    # fallback: scan element text
    for e in elems:
        t = (e.get("value") or e.get("text") or "").strip()
        if not t:
            continue
        for line in t.splitlines():
            line = line.strip()
            if line and not _is_separator(line) and len(line) > 3 and ":" in line:
                return line[:60]
    return ""


def _semantic_desc(action_type, click_pos, text, keystrokes, state,
                   hotkey="", scroll_dy=0.0, drag_src=None, drag_dst=None,
                   notepad_line="", notepad_select="", clipboard=""):
    """Return a human-readable description of an action for console output."""
    elems  = state.get("elements", []) if state else []
    app    = (state.get("application") or "?").replace(".exe", "")

    # Active tab label
    tab_lbl = ""
    for e in elems:
        if e.get("type") in ("tabitem", "tabitemcontrol") and e.get("window_role") == "active":
            t = (e.get("text") or e.get("label") or "").strip()
            if t and len(t) < 30:
                tab_lbl = t
                break

    tab_part = f" [{tab_lbl}]" if tab_lbl else ""

    def _elem_at(px, py, role=None):
        best, best_area = None, float("inf")
        for e in elems:
            if role and e.get("window_role") != role:
                continue
            b = e.get("bbox", [])
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best = e
        # fallback: search all roles if active-only found nothing
        if best is None and role == "active":
            return _elem_at(px, py, role=None)
        return best

    def _lbl(e):
        return ((e.get("label") or e.get("text") or "").strip()[:30] or e.get("type", "?")) if e else "?"

    if action_type == "click" and click_pos:
        e = _elem_at(*click_pos, role="active")
        etype = e.get("type", "") if e else "?"
        lbl = _lbl(e)
        # notepad pane click — show line at cursor (pre-computed)
        if etype in ("panecontrol", "textcontrol", "documentcontrol", "document"):
            ctx = f"  →  {notepad_line!r}" if notepad_line and not _is_separator(notepad_line) else ""
            return f"click       [{app}] [cursor position]{ctx}"
        # combobox OPEN click — value is still the pre-selection default at this
        # instant, so don't show it (misleading). The actual pick is the next
        # listitem click, which shows the chosen option correctly.
        if etype in ("combobox", "comboboxcontrol"):
            return f"click       [{app}]{tab_part} [{lbl}] (combobox — opening)"
        # dropdown item pick — this is where the real selection shows
        if etype in ("listitem", "listitemcontrol"):
            return f"select      [{app}]{tab_part} [{lbl}]"
        return f"click       [{app}]{tab_part} [{lbl}] ({etype})"

    if action_type == "double_click" and click_pos:
        e = _elem_at(*click_pos, role="active")
        etype = e.get("type", "") if e else "?"
        # notepad double-click = word selection (from subprocess-attached state)
        if etype in ("panecontrol", "textcontrol", "documentcontrol", "document"):
            sel = (state.get("_np_selection") or "").strip() if state else ""
            ctx = f"  →  {sel!r}" if sel and not _is_separator(sel) else ""
            return f"double-click [{app}] [word select]{ctx}"
        return f"double-click [{app}]{tab_part} [{_lbl(e)}]"

    if action_type == "drag" and drag_src and drag_dst:
        src_e = _elem_at(*drag_src, role="active")
        etype = src_e.get("type", "") if src_e else ""
        if etype in ("textcontrol", "panecontrol", "documentcontrol", "document"):
            sel = (state.get("_np_selection") or "").strip() if state else ""
            ctx = f"  →  {sel!r}" if sel and not _is_separator(sel) else "  (next Ctrl+C shows exact value)"
            return f"drag        [{app}] [text selection]{ctx}"
        dst_e = _elem_at(*drag_dst)
        return f"drag        [{app}]{tab_part} [{_lbl(src_e)}] → [{_lbl(dst_e)}]"

    if action_type == "scroll":
        direction = "↓ down" if scroll_dy < 0 else "↑ up"
        ticks = abs(int(scroll_dy))
        ctx = ""
        if "notepad" in app.lower():
            visible = (state.get("_np_visible") or "") if state else ""
            ctx = f"\n{visible}" if visible else ""
        else:
            # show visible form fields with values
            parts = []
            for e in elems:
                if e.get("window_role") != "active":
                    continue
                if e.get("type") not in ("input", "combobox", "editcontrol", "comboboxcontrol"):
                    continue
                lbl = (e.get("label") or e.get("text") or "").strip()
                val = (e.get("value") or "").strip()
                if not lbl:
                    continue
                parts.append(f"{lbl}: {val!r}" if val else lbl)
                if len(parts) >= 4:
                    break
            ctx = f"  →  {' | '.join(parts)}" if parts else ""
        return f"scroll      [{app}] {direction} {ticks} tick(s){ctx}"

    if action_type == "hotkey":
        extra = ""
        if clipboard:
            if hotkey == "ctrl+c":
                extra = f"  →  {clipboard[:40]!r}"
            elif hotkey == "ctrl+v":
                extra = f"  pasting {clipboard[:40]!r}"
        return f"hotkey      [{app}]{tab_part} [{hotkey}]{extra}"

    if action_type == "keyboard":
        val = (text or " ".join(keystrokes))[:40]
        focused = next((e for e in elems
                        if e.get("window_role") == "active" and e.get("focused")), None)
        field = _lbl(focused)
        return f"type        [{app}]{tab_part} [{field}]  →  {val!r}"

    return action_type


# =============================================================================
# SNAPSHOT SUBPROCESS — runs UIA in a separate process so the ~280ms snapshot
# never holds the main process GIL (which would block pynput's input hook and
# make Windows drop events). Continuously snapshots, pushes latest state via a
# multiprocessing Queue. Main process just reads the latest — zero UIA on the
# main GIL, so input recording stays instant and lossless.
# =============================================================================

def _snapshot_proc(req_q, res_q, stop_flag, bg_apps):
    """Child-process: ON-DEMAND snapshots. Waits for a request token, takes ONE
    UIA snapshot, returns the slimmed state. ZERO snapshots between requests, so
    the form is never flooded with UIA queries while the user is interacting."""
    import sys as _sys, os as _os, time as _time
    import queue as _q
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _comp = _os.path.dirname(_here)
    _root = _os.path.dirname(_comp)
    for _p in (_root, _comp):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    try:
        from observers.ui_observer import UIAutomationObserver as _Obs
    except Exception:
        try:
            from components.observers.ui_observer import UIAutomationObserver as _Obs
        except Exception:
            return
    try:
        obs = _Obs(background_apps=set(bg_apps)) if bg_apps else _Obs()
    except TypeError:
        obs = _Obs()
    # Keep only the element fields the transformer + recorder actually use.
    # Drops metadata/automation_id/class_name/enabled/visible/pid/control_type/
    # source — the structural bulk that made each state ~138KB. Shrinks state to
    # ~15-20KB → IPC unpickle + per-step json save stay <2ms → main GIL free.
    _KEEP = ("element_id", "type", "label", "text", "value", "bbox",
             "window_role", "window_title", "app", "focused", "confidence")

    def _slim(state):
        slim_elems = []
        for e in state.get("elements", []):
            ne = {k: e[k] for k in _KEEP if k in e}
            # label is display-only → cap hard (was the 149KB bloat: Notepad
            # elements stuff the whole document into label)
            lb = ne.get("label")
            if lb and len(lb) > 200:
                ne["label"] = lb[:200]
            v = ne.get("value")
            if v and len(v) > 6000:
                ne["value"] = v[:6000]
            t = ne.get("text")
            if t and len(t) > 6000:
                ne["text"] = t[:6000]
            slim_elems.append(ne)
        state["elements"] = slim_elems
        return state

    while not stop_flag.value:
        try:
            req = req_q.get(timeout=0.3)   # block until a snapshot is requested
        except _q.Empty:
            continue
        action_type = req if isinstance(req, str) else ""
        try:
            state = _slim(obs.snapshot())
        except Exception:
            state = {}
        # Attach Notepad context HERE (subprocess = off main GIL) so the console
        # can show highlighted/copied text without lagging the user's input.
        try:
            if action_type in ("drag", "double_click"):
                state["_np_selection"] = _get_notepad_selection()
            elif action_type == "scroll":
                state["_np_visible"] = _get_notepad_visible_lines(20)
        except Exception:
            pass
        try:
            res_q.put(state)
        except Exception:
            pass


# =============================================================================
# DEMO RECORDER  (action-triggered BC data collection)
# =============================================================================

class DemoRecorder:
    """
    Action-triggered recorder for Behavioral Cloning demo collection.

    Unlike ScreenObserver (time-based frame capture), DemoRecorder fires on
    every human mouse click or keyboard group and immediately snapshots UI
    state before and after the action.  Each (state, action, next_state) triple
    is one BC training step — no time-window ambiguity, no empty frames.

    Controls (hotkeys while the form is focused):
        F9  — toggle recording on / off
        F10 — save session and quit

    Output: data/demos/human/session_<timestamp>/live_step_NNNN.json
    Format: identical to ScreenObserver + _export_run_traces output —
            train.py reads it without any changes.

    Perception is swappable, the same way ScreenObserver's is. The default walks
    the UIA tree, which is right for a desktop form and wrong for a browser: UIA
    does not resolve aria-labelledby, so all fifty rows of a grid column arrive
    under one name and the demo cannot say which row was filled. perception="web"
    records DOM state instead, from a browser already running with
    --remote-debugging-port. Anything else with a .snapshot() can be passed as
    observer_factory — a factory, not an instance, because the worker thread has
    to own it (see __init__).

    Usage:
        recorder = DemoRecorder()
        recorder.run()          # blocks until F10

        DemoRecorder(perception="web").run()        # browser demo
    """

    _FLUSH_KEYS = {"Key.tab", "Key.enter", "Key.esc", "Key.escape",
                   "tab", "enter", "return", "escape", "esc"}
    _CAPTURE_DELAY = 0.30   # seconds to wait after action before snapshotting

    def __init__(
        self,
        output_dir: str = "data/demos/human",
        trace_type: str = "form_filling",
        observer_factory: Optional[Callable[[], Any]] = None,
        perception: str = "uia",
        browser_url: str = "http://localhost:9222",
        max_elements: int = 1000,
    ):
        if not _PYNPUT_AVAILABLE:
            raise ImportError("pynput is required. Install with: pip install pynput")

        from datetime import datetime as _dt
        _session_ts   = _dt.now().strftime("%Y%m%d_%H%M%S")
        _intern_dir   = _INTERN_DIR
        self.output_dir = os.path.join(_intern_dir, output_dir, f"session_{_session_ts}")
        self.trace_type = trace_type

        # ── which eye records the state ──────────────────────────────────────
        # A factory, not an observer instance, because Playwright's sync API is
        # thread-affine: a WebObserver built here and used from the worker
        # thread raises greenlet "cannot switch to a different thread", which
        # _request_snapshot catches and turns into {} — every step recorded with
        # empty state while the recorder happily reports success. So an injected
        # observer is always constructed BY the worker thread, the one thread
        # that snapshots during a live recording.
        #
        # UIA keeps its original eager construction: it has been used across
        # these threads all along (each calls _init_com), and moving it would
        # change scope #1's recording path for no reason.
        self._observer: Optional[Any] = None
        self._observer_factory: Optional[Callable[[], Any]] = observer_factory

        if observer_factory is None and perception == "web":
            # Decided here, over plain HTTP, so the fallback is chosen BEFORE
            # recording starts rather than discovered in the saved traces.
            if not _WEB_OBSERVER_AVAILABLE:
                print("[DemoRecorder] perception='web' but WebObserver is unavailable "
                      "(pip install playwright) — falling back to UIAutomation.")
            elif not _cdp_reachable(browser_url):
                print(f"[DemoRecorder] No browser answering at {browser_url}. "
                      "Start Chrome with --remote-debugging-port=9222, or recording "
                      "will fall back to UIAutomation and lose per-row labels.")
            else:
                def _make_web_observer():
                    obs = _WebObserver(browser_url=browser_url,
                                       max_elements=max_elements)
                    if not (obs.available and obs.connect()):
                        raise RuntimeError(
                            f"WebObserver failed to attach at {browser_url}")
                    return obs
                self._observer_factory = _make_web_observer
                print(f"[DemoRecorder] WebObserver will attach at {browser_url} — "
                      "DOM state active (aria-labelledby resolved).")

        if self._observer_factory is None:
            if not _UIA_OBSERVER_AVAILABLE:
                raise ImportError("UIAutomationObserver not found in components/observers/.")
            # Option A: only walk the foreground form + Notepad source window.
            # ~10x faster snapshots than walking every visible window.
            try:
                self._observer = _UIAObserver(background_apps={"notepad", ".txt"})
            except TypeError:
                self._observer = _UIAObserver()  # older signature fallback

        self._steps: list = []
        self._lock   = threading.Lock()

        self._recording      = False
        self._pending_text   = ""
        self._pending_keys: list = []
        self._pending_hotkey: str = ""
        self._pre_key_state: dict | None = None
        self._quit_event     = threading.Event()
        # drag detection
        self._mouse_down_pos: list | None = None
        self._mouse_down_time: float = 0.0
        self._last_click_time: float = 0.0
        self._last_click_pos: list | None = None
        # ctrl key state
        self._ctrl_held: bool = False
        # scroll debounce
        self._scroll_accum: float = 0.0
        self._scroll_pos: list = [0, 0]
        self._scroll_timer = None
        self._last_state: dict = {}
        self._cached_state: dict = {}   # legacy, unused
        self._state_lock = threading.Lock()
        import queue as _queue
        self._action_queue = _queue.Queue()

        # ── snapshot subprocess: ON-DEMAND UIA, off the main GIL ─────────────
        # No continuous snapshotting → form is never flooded → no input lag.
        #
        # Disabled by default 2026-08-10, live: under this specific hosting
        # context (Electron -> Node child_process.spawn -> Python bridge ->
        # multiprocessing.Process snapshot subprocess), the subprocess
        # consistently failed to return real data -- not intermittently, but
        # from the very first request of every session, before any user
        # action at all ([SNAP-DIAG] get() timed out: Empty()). Once the
        # bounded request queue filled, it stayed wedged (put() Full()) for
        # the rest of the session -- every subsequent step recorded empty
        # state. CPU-delta sampling confirmed the subprocess process itself
        # was genuinely frozen (0% CPU), not just slow. The same code path
        # tested standalone, outside this process tree, worked fine -- the
        # problem is specific to this multi-layer spawn chain, not the
        # subprocess mechanism itself, and wasn't fully root-caused before
        # switching it off. self._observer (constructed unconditionally
        # above) is documented as "~10x faster" than the general-purpose
        # snapshot anyway (foreground-window + Notepad only, not every
        # visible window) -- using it directly here trades "off the GIL,
        # but frequently returns nothing" for "on the GIL, but works," which
        # is the correct trade until the subprocess issue is actually
        # understood. Re-enable by setting self._use_subprocess = True
        # after construction if the underlying issue gets fixed.
        self._snap_proc = None
        self._use_subprocess = False
        print("  [recorder] MODE: in-process snapshots (subprocess disabled)")

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _request_snapshot(self, action_type: str = "", timeout: float = 4.0) -> dict:
        """Ask the subprocess for ONE fresh snapshot (+ Notepad context for the
        given action_type). Blocks the worker (not the input listener).

        Timeout raised 2.0 -> 4.0s, 2026-08-10: profiled a single real
        UIAutomationObserver.snapshot() at 0.9-1.2s baseline, no other load
        (cProfile: ~13 separate UIA/COM property calls per element, inherent
        to the uiautomation library's per-property-call design, not a bug
        here -- a real fix means adopting UIA's cache-request batching API,
        a substantial rewrite not attempted tonight without the ability to
        live-test it). A 2.0s timeout left almost no margin above that
        baseline -- any real load (this machine syncs the whole repo via
        OneDrive in real time; a concurrent test run; anything) tips a
        marginal snapshot over the timeout, and the request comes back as
        `{}` -- recorded as a real, empty-state step. Every step in a full
        live session showed exactly this. Raising the timeout doesn't make
        snapshots faster, but it stops turning "slightly slow" into
        "silently wrong, empty data saved" -- worse for training data than
        the extra wait.

        Found 2026-08-10, live: if the snapshot subprocess stops responding
        (for any reason -- it died, it's stuck, anything), _req_q (bounded,
        maxsize=4) fills up after 4 unanswered requests. put() had no
        timeout, so every request after that BLOCKED FOREVER -- not this one
        call, the entire worker thread, permanently. Once that happens
        nothing ever recovers: no more steps commit, and Stop can never
        finish either (its own drain loop calls this same method). Matches
        exactly what was observed: 3 real steps recorded with empty state
        (the subprocess was already not answering from step 0), then a
        total, permanent hang once the queue filled. Giving put() the same
        timeout get() already had makes this method's own documented
        contract -- return {} on any failure -- actually hold in every
        case, instead of silently assuming put() could never be the one
        that blocks.
        """
        if not self._use_subprocess:
            if self._observer is None:
                return {}
            try:
                return self._observer.snapshot()
            except Exception:
                return {}
        try:
            try:
                while not self._res_q.empty():
                    self._res_q.get_nowait()
            except Exception:
                pass
            try:
                self._req_q.put(action_type or 1, timeout=timeout)
            except Exception as exc:
                alive = self._snap_proc.is_alive() if self._snap_proc is not None else None
                print(f"       [SNAP-DIAG] put() failed: {exc!r}  subprocess_alive={alive}")
                return {}
            try:
                result = self._res_q.get(timeout=timeout)
            except Exception as exc:
                alive = self._snap_proc.is_alive() if self._snap_proc is not None else None
                print(f"       [SNAP-DIAG] get() timed out: {exc!r}  subprocess_alive={alive}")
                return {}
            if not result:
                print(f"       [SNAP-DIAG] got a real response but it was empty/falsy: {result!r}")
            return result or {}
        except Exception as exc:
            print(f"       [SNAP-DIAG] unexpected: {exc!r}")
            return {}

    # ── public ─────────────────────────────────────────────────────────────────

    def run(self, auto_start: bool = True):
        """Block until F10 is pressed or stop() is called."""
        print("\n" + "=" * 60)
        print("  DEMO RECORDER  (BC data collection)")
        print("  F10 — save session and quit")
        print("=" * 60)
        print("\nRecording started. Fill the form, then press F10 or click Stop.\n")

        if auto_start:
            self._recording = True

        def _make_mouse():
            # NO on_move — drag is detected from press/release positions. A move
            # callback fires hundreds of times during a highlight-drag and each
            # guard call adds GIL load → lag while highlighting. Omit it entirely.
            return _pynput_mouse.Listener(
                on_click=self._guard(self._on_click),
                on_scroll=self._guard(self._on_scroll),
            )

        def _make_keyboard():
            return _pynput_keyboard.Listener(
                on_press=self._guard(self._on_key_press),
                on_release=self._guard(self._on_key_release))

        listeners = {"m": _make_mouse(), "k": _make_keyboard()}
        listeners["m"].start()
        listeners["k"].start()

        # Watchdog: pynput listener threads can die upstream of our guards
        # (e.g. NotImplementedError in pynput's own key conversion). Restart any
        # dead listener so recording never permanently stops mid-session.
        def _report_death(name: str, dead_listener) -> None:
            # .join() on an already-dead pynput listener re-raises whatever
            # exception killed its thread -- without this, "died -- restarting"
            # gave no way to know WHY, or how often, or whether it's the same
            # cause every time. Found 2026-08-08: a live session reported
            # unusually frequent restarts (2 deaths in 3 clicks); needed the
            # real cause before guessing at a fix.
            try:
                dead_listener.join(timeout=0)
            except Exception as exc:
                print(f"  [watchdog] {name} listener died — restarting (cause: {exc!r})")
            else:
                print(f"  [watchdog] {name} listener died — restarting (no exception captured)")

        def _watchdog():
            while not self._quit_event.is_set():
                time.sleep(0.5)
                try:
                    if not listeners["m"].is_alive():
                        _report_death("mouse", listeners["m"])
                        listeners["m"] = _make_mouse(); listeners["m"].start()
                    if not listeners["k"].is_alive():
                        _report_death("keyboard", listeners["k"])
                        listeners["k"] = _make_keyboard(); listeners["k"].start()
                except Exception:
                    pass
        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            self._quit_event.wait()
        except Exception:
            pass
        finally:
            self._recording = False
            try: listeners["m"].stop()
            except Exception: pass
            try: listeners["k"].stop()
            except Exception: pass
            # PROCESS (don't discard) remaining queued actions so nothing is lost.
            import queue as _q
            print(f"\n  Flushing {self._action_queue.qsize()} pending action(s)…")
            deadline = time.time() + 15.0
            while time.time() < deadline:
                try:
                    event = self._action_queue.get_nowait()
                except _q.Empty:
                    break
                try:
                    self._process_event(event)
                except Exception:
                    self._log_crash("drain._process_event")
            self._flush_pending(self._capture())
            if self._steps:
                out = self._save()
                print(f"\n  [SAVED] {len(self._steps)} steps → {out}")
            # stop the snapshot subprocess
            try:
                if self._snap_proc is not None:
                    self._mp_stop.value = True
                    self._snap_proc.join(timeout=2.0)
                    if self._snap_proc.is_alive():
                        self._snap_proc.terminate()
            except Exception:
                pass

    def save(self) -> str:
        return self._save()

    # ── replay ───────────────────────────────────────────────────────────────
    def replay(self, source_session: str, count: int = 1,
               submit_between: bool = True, progress=None) -> int:
        """Re-execute the actions of a recorded session on the LIVE form, capturing
        fresh state each step → saves as NEW session(s). Hands-free data generation.
        Hit Submit & New between runs (submit_between) → each replay fills a
        DIFFERENT record → same navigation, real state variety.

        source_session : path to a saved session folder (live_step_*.json)
        count          : how many replay sessions to produce
        progress       : optional callback(msg) for UI updates
        Returns total steps written.
        """
        if self._observer_factory is not None:
            # replay() snapshots from the calling thread, but an injected
            # observer belongs to the worker thread (see __init__). Refusing is
            # the honest outcome: the alternative is a full replay session whose
            # every state is empty, saved as if it were real training data.
            raise RuntimeError(
                "replay() needs an observer it can use on this thread; the "
                "injected one is owned by the recorder's worker thread. Replay "
                "with the default UIA perception, or replay from a saved session.")
        import pyautogui, glob as _glob
        pyautogui.FAILSAFE = False
        src = source_session
        if not os.path.isabs(src):
            src = os.path.join(_INTERN_DIR, src)
        step_files = sorted(_glob.glob(os.path.join(src, "live_step_*.json")))
        if not step_files:
            if progress: progress(f"No steps in {src}")
            return 0

        def _elem_at(state, pos, role="active"):
            # Smallest element containing pos. Filter to the ACTIVE window's
            # elements first (skips background panes that overlap the same
            # coords — same logic the recorder uses to label the click), then
            # fall back to all roles if nothing matched.
            if not state or not pos:
                return None
            px, py = pos
            best, best_area = None, 1e18
            for e in state.get("elements", []):
                if role and e.get("window_role") != role:
                    continue
                b = e.get("bbox")
                if not b or len(b) != 4:
                    continue
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    if area < best_area:
                        best, best_area = e, area
            if best is None and role == "active":
                return _elem_at(state, pos, role=None)
            return best

        # parse the source actions once
        actions = []
        for f in step_files:
            try:
                t = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            m = t.get("mouse", {}).get("actions", [])
            k = t.get("keyboard", {}).get("actions", [])
            if m:
                a = m[0]
                pos = a.get("position", [0, 0])
                # identify the clicked field from the post-click state (matches
                # the recorder's own label), fall back to the pre-click state
                tgt = _elem_at(t.get("next_state", {}), pos) or _elem_at(t.get("state", {}), pos)
                actions.append({"kind": a.get("type", "click"),
                                "pos": pos,
                                "dst": a.get("dst_position"),
                                "dy":  a.get("dy", 0),
                                "lbl": (tgt or {}).get("label") or (tgt or {}).get("text") or "",
                                "etype": (tgt or {}).get("type") or "",
                                "win": (tgt or {}).get("window_title") or ""})
            elif k:
                g = k[0]
                hk = g.get("hotkey", "")
                if hk:
                    actions.append({"kind": "hotkey", "hotkey": hk})
                else:
                    txt = "".join(s.get("pasted_text") or s.get("key", "")
                                  for s in g.get("strokes", []))
                    actions.append({"kind": "type", "text": txt})

        # JUNK FILTER: the form is the window the user clicked most. Any click
        # whose target landed on a DIFFERENT window (e.g. the recorder GUI, the
        # terminal) is a stray click — drop it. General: no app names hardcoded.
        from collections import Counter as _Counter
        _wins = _Counter(a["win"] for a in actions
                         if a["kind"] in ("click", "double_click") and a.get("win"))
        form_win = _wins.most_common(1)[0][0] if _wins else ""
        if form_win:
            _before = len(actions)
            actions = [a for a in actions
                       if a["kind"] not in ("click", "double_click")
                       or not a.get("win") or a["win"] == form_win]
            _dropped = _before - len(actions)
            if _dropped and progress:
                progress(f"  filtered {_dropped} stray click(s) off form window "
                         f"({form_win!r})")

        def _foreground_form():
            # Bring the wx form window to the front so replayed clicks land on
            # its fields (not the recorder window / panel background).
            try:
                import win32gui
                target = [0]
                def _cb(h, _):
                    if not win32gui.IsWindowVisible(h):
                        return
                    t = (win32gui.GetWindowText(h) or "").lower()
                    if "insurance" in t or "data entry" in t or "car" in t:
                        target[0] = h
                win32gui.EnumWindows(_cb, None)
                if target[0]:
                    win32gui.SetForegroundWindow(target[0])
                    time.sleep(0.4)
            except Exception:
                pass

        total = 0
        from datetime import datetime as _dt
        for run_i in range(count):
            self._steps = []
            ts = _dt.now().strftime("%Y%m%d_%H%M%S_%f")
            self.output_dir = os.path.join(_INTERN_DIR, "data", "demos", "human",
                                           f"session_replay_{ts}")
            if progress: progress(f"Replay {run_i+1}/{count} → {os.path.basename(self.output_dir)}")
            _foreground_form()   # focus the form before each replay pass
            for ai, act in enumerate(actions):
                pre = self._request_snapshot()
                # SEMANTIC REPLAY: re-find the recorded field by its identity
                # (label + type) in the CURRENT UI and click its live center.
                # Pixel-independent — survives the form moving / scrolling.
                # Recorded coords are only the fallback when no match is found.
                resolved = ""
                if act["kind"] in ("click", "double_click") and act.get("lbl"):
                    lbl, et = act["lbl"], act.get("etype", "")
                    match = next((e for e in pre.get("elements", [])
                                  if ((e.get("label") or e.get("text") or "").strip() == lbl)
                                  and (not et or e.get("type") == et)
                                  and e.get("window_role") == "active"
                                  and e.get("bbox") and len(e["bbox"]) == 4), None)
                    if match:
                        b = match["bbox"]
                        act = dict(act, pos=[(b[0]+b[2])/2.0, (b[1]+b[3])/2.0])
                        resolved = f" → {lbl}"
                    else:
                        resolved = f" → (raw coords; '{lbl}' not found)"
                if progress:
                    _tag = act.get("lbl") or act.get("text") or act.get("hotkey") or act.get("pos")
                    progress(f"  [{ai+1}/{len(actions)}] {act['kind']} {_tag}{resolved}")
                try:
                    if act["kind"] in ("click", "double_click"):
                        x, y = act["pos"]
                        pyautogui.moveTo(x, y, duration=0.08)
                        if act["kind"] == "double_click":
                            pyautogui.doubleClick(x, y)
                        else:
                            pyautogui.click(x, y)
                    elif act["kind"] == "drag" and act.get("dst"):
                        x, y = act["pos"]; dx, dy = act["dst"]
                        pyautogui.moveTo(x, y, duration=0.08)
                        pyautogui.dragTo(dx, dy, duration=0.2, button="left")
                    elif act["kind"] == "scroll":
                        x, y = act["pos"]
                        pyautogui.scroll(int(act.get("dy", 0)) or -3, x=int(x), y=int(y))
                    elif act["kind"] == "type":
                        if act.get("text"):
                            pyautogui.typewrite(act["text"], interval=0.02)
                    elif act["kind"] == "hotkey":
                        hk = act["hotkey"]
                        keys = hk.split("+")
                        pyautogui.hotkey(*keys) if len(keys) > 1 else pyautogui.press(hk)
                except Exception as _re:
                    if progress: progress(f"  step {ai} exec error: {_re}")
                time.sleep(0.3)
                post = self._request_snapshot()
                # build the new step directly (same format as live capture)
                _ev = {"action_type": {"click":"click","double_click":"double_click",
                                       "drag":"drag","scroll":"scroll","type":"keyboard",
                                       "hotkey":"hotkey"}.get(act["kind"], "click")}
                self._append_step(
                    state_before=pre, state_after=post,
                    action_type=_ev["action_type"],
                    click_pos=act.get("pos") if act["kind"] in ("click","double_click") else None,
                    text=act.get("text",""),
                    hotkey=act.get("hotkey",""),
                    scroll_pos=act.get("pos") if act["kind"]=="scroll" else None,
                    scroll_dy=act.get("dy",0),
                    drag_src=act.get("pos") if act["kind"]=="drag" else None,
                    drag_dst=act.get("dst") if act["kind"]=="drag" else None,
                )
            out = self._save()
            total += len(self._steps)
            if progress: progress(f"  saved {len(self._steps)} steps → {out}")
            if submit_between and run_i < count - 1:
                # click Submit & New to advance to a fresh record
                try:
                    st = self._request_snapshot()
                    btn = next((e for e in st.get("elements", [])
                                if e.get("type") in ("buttoncontrol","button")
                                and "new" in (e.get("text") or e.get("label") or "").lower()
                                and e.get("bbox")), None)
                    if btn:
                        b = btn["bbox"]; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
                        pyautogui.click(cx, cy); time.sleep(0.6)
                except Exception:
                    pass
        return total

    # ── pynput callbacks ───────────────────────────────────────────────────────

    def _on_hotkey(self, key):
        k = self._key_name(key)
        if k in ("Key.f9", "f9"):
            self._recording = not self._recording
            state_str = "RECORDING" if self._recording else "PAUSED"
            with self._lock:
                count = len(self._steps)
            print(f"\n  [{state_str}]  {count} steps so far\n")
        elif k in ("Key.f10", "f10"):
            self._recording = False
            print("\n  Saving and quitting …")
            self._flush_pending(self._capture())
            out = self._save()
            print(f"\n  {len(self._steps)} steps saved → {out}")
            print(f"\n  Train:  python train.py --trace_dir data/demos/human --epochs 50\n")
            return False  # stop listener

    def _on_move(self, x, y, *args):
        pass  # used implicitly for drag detection via _mouse_down_pos

    def _on_scroll(self, x, y, dx, dy, *args):
        if not self._recording:
            return
        delta = float(dy) if dy != 0 else float(dx)
        if delta == 0:
            return
        self._flush_text_to_queue()
        self._scroll_accum += delta
        self._scroll_pos = [x, y]
        if self._scroll_timer:
            self._scroll_timer.cancel()
        self._scroll_timer = threading.Timer(0.4, self._flush_scroll)
        self._scroll_timer.start()

    def _flush_scroll(self):
        accum = self._scroll_accum
        pos   = self._scroll_pos
        self._scroll_accum = 0.0
        if accum == 0:
            return
        self._action_queue.put({
            "action_type": "scroll",
            "scroll_pos":  pos,
            "scroll_dy":   accum,
        })

    def _on_click(self, x, y, button, pressed, *args):
        if not self._recording:
            return
        if button != _pynput_mouse.Button.left:
            return

        # LISTENER IS PURE EVENT-QUEUEING — never reads UIA/state (that would
        # hold the GIL and lag input). Worker snapshots on-demand and filters.
        if pressed:
            self._mouse_down_pos  = [x, y]
            self._mouse_down_time = time.time()
            return

        if self._mouse_down_pos is None:
            return

        # commit any typed text before this mouse action (preserve order)
        self._flush_text_to_queue()

        dx = abs(x - self._mouse_down_pos[0])
        dy = abs(y - self._mouse_down_pos[1])
        now = time.time()

        # drag: moved far enough that it can't be click jitter. 8px was too low —
        # normal clicks have minor hand movement and were misclassified as drags
        # (231 fake drags polluted the click demos). 40px = real drag/selection.
        if dx > 40 or dy > 40:
            src = list(self._mouse_down_pos)
            self._mouse_down_pos  = None
            self._last_click_time = 0.0
            self._action_queue.put({"action_type": "drag",
                                    "drag_src": src, "drag_dst": [x, y]})
            return

        # double-click: second click within 400ms at same spot
        if (self._last_click_pos
                and abs(x - self._last_click_pos[0]) < 10
                and abs(y - self._last_click_pos[1]) < 10
                and (now - self._last_click_time) < 0.4):
            self._mouse_down_pos  = None
            self._last_click_time = 0.0
            self._last_click_pos  = None
            self._action_queue.put({"action_type": "double_click", "click_pos": [x, y]})
            return

        # single click
        self._mouse_down_pos  = None
        self._last_click_time = now
        self._last_click_pos  = [x, y]
        self._action_queue.put({"action_type": "click", "click_pos": [x, y]})

    def _on_key_press(self, key, *args):
        k = self._key_name(key)
        if k == "f9":
            self._recording = not self._recording
            state_str = "RECORDING" if self._recording else "PAUSED"
            with self._lock:
                count = len(self._steps)
            print(f"\n  [{state_str}]  {count} steps so far\n")
            return
        if k == "f10":
            self._recording = False
            print("\n  Saving and quitting …")
            self._flush_pending(self._capture())
            out = self._save()
            print(f"\n  {len(self._steps)} steps saved → {out}")
            print(f"\n  Train:  python train.py --trace_dir data/demos/human --epochs 50\n")
            self._quit_event.set()
            return
        # Ignore function keys (F1-F12) entirely — never form values. F8 is the
        # GUI replay hotkey; without this it leaks into recorded text as "f8 f8…".
        if len(k) in (2, 3) and k[0] in ("f", "F") and k[1:].isdigit():
            return
        if not self._recording:
            return

        # track ctrl state
        if k in ("ctrl_l", "ctrl_r", "Key.ctrl_l", "Key.ctrl_r", "ctrl"):
            self._ctrl_held = True
            return

        # detect hotkeys: ctrl+letter, tab, enter, escape, arrows
        # control chars: \x01=a \x03=c \x06=f \x16=v \x18=x \x19=y \x1a=z
        _CTRL_CHARS = {'\x01':'a','\x02':'b','\x03':'c','\x04':'d','\x05':'e',
                       '\x06':'f','\x16':'v','\x18':'x','\x19':'y','\x1a':'z'}
        hotkey = None
        if self._ctrl_held or (len(k) == 1 and k in _CTRL_CHARS):
            letter = _CTRL_CHARS.get(k, k.lower() if len(k) == 1 else "")
            if letter in ("a", "c", "v", "x", "z", "y"):
                hotkey = f"ctrl+{letter}"
            elif k in ("tab", "Key.tab"):
                hotkey = "ctrl+tab"
        elif k in ("tab", "Key.tab"):
            hotkey = "tab"
        elif k in ("enter", "return", "Key.enter", "Key.return"):
            hotkey = "enter"
        elif k in ("escape", "esc", "Key.esc", "Key.escape"):
            hotkey = "escape"
        elif k in ("backspace", "Key.backspace"):
            hotkey = "backspace"
        elif k in ("delete", "Key.delete"):
            hotkey = "delete"
        elif k in ("home", "Key.home"):
            hotkey = "home"
        elif k in ("end", "Key.end"):
            hotkey = "end"
        elif k in ("page_up", "Key.page_up"):
            hotkey = "page_up"
        elif k in ("page_down", "Key.page_down"):
            hotkey = "page_down"
        elif k in ("up", "Key.up"):
            hotkey = "arrow_up"
        elif k in ("down", "Key.down"):
            hotkey = "arrow_down"
        elif k in ("left", "Key.left"):
            hotkey = "arrow_left"
        elif k in ("right", "Key.right"):
            hotkey = "arrow_right"

        if hotkey:
            # flush pending text first (worker assigns states)
            if self._pending_text or self._pending_keys:
                self._action_queue.put({
                    "action_type": "keyboard",
                    "text":        self._pending_text,
                    "keystrokes":  list(self._pending_keys),
                })
                self._pending_text  = ""
                self._pending_keys  = []
            self._action_queue.put({"action_type": "hotkey", "hotkey": hotkey})
            return

        # regular character — accumulate, no state read
        try:
            ch = key.char
            if ch and ch.isprintable() and not self._ctrl_held:
                self._pending_text += ch
        except AttributeError:
            self._pending_keys.append(k)

    def _on_key_release(self, key, *args):
        k = self._key_name(key)
        if k in ("ctrl_l", "ctrl_r", "Key.ctrl_l", "Key.ctrl_r", "ctrl"):
            self._ctrl_held = False

    # ── internal ───────────────────────────────────────────────────────────────

    _NOISE_APPS = {"windowsterminal", "code", "devenv", "cursor",
                   "firefox", "chrome", "msedge", "iexplore", "opera", "brave"}

    def _is_noise_app(self, state: dict) -> bool:
        app = (state.get("application") or "").lower().replace(".exe", "")
        return any(n in app for n in self._NOISE_APPS)

    @staticmethod
    def _init_com():
        """Initialize COM for the current thread — required for UIA calls.
        Each thread that touches uiautomation must call this or it crashes."""
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
        except Exception:
            pass

    def _log_crash(self, where: str):
        """Append a full traceback to the crash log so silent listener-thread
        deaths become visible."""
        import traceback, datetime as _dt
        try:
            path = os.path.join(_INTERN_DIR, "data", "output", "recorder_crash.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n[{_dt.datetime.now().isoformat()}] {where}\n")
                f.write(traceback.format_exc())
            print(f"  [CRASH in {where}] {traceback.format_exc().splitlines()[-1]}")
        except Exception:
            pass

    def _guard(self, fn):
        """Wrap a pynput callback so an exception is logged instead of silently
        killing the listener thread.

        Was `except Exception` -- doesn't catch SystemExit/KeyboardInterrupt/
        other BaseException subclasses, so anything in that gap still killed
        the listener with NOTHING logged (found 2026-08-08: a live session
        watchdog-restarted both listeners multiple times with zero exception
        captured via listener.join(), which only makes sense if something
        outside Exception's hierarchy was the cause). BaseException closes
        that gap; this callback's only job is keeping input capture alive,
        so there's no legitimate reason to let anything propagate and kill it.
        """
        def _wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except BaseException:
                self._log_crash(fn.__name__)
                return True   # keep listener alive
        return _wrapped

    def _bg_capture(self):
        """Continuously refresh cached UI state in background.
        snapshot() self-initializes COM via UIAutomationInitializerInThread —
        do NOT call _init_com here or it conflicts and snapshot returns empty."""
        while not self._quit_event.is_set():
            t0 = time.time()
            try:
                state = self._observer.snapshot()
                if state.get("elements"):
                    self._cached_state = state   # atomic ref swap, no lock needed
            except Exception:
                pass
            # The UIA walk holds the GIL the whole time, starving the input
            # listeners. Sleep 3x the snapshot duration so the GIL is free ~75%
            # of the time and clicks/keys stay responsive.
            elapsed = time.time() - t0
            time.sleep(max(0.3, elapsed * 3))

    def _worker(self):
        """Process queued actions. UIA snapshots come from the subprocess via
        _last_state (no UIA on this process's GIL → input never lags/drops).
        In fallback mode (no subprocess) snapshots happen here in-process."""
        import queue as _queue
        # Found 2026-08-10, live: switching the default to in-process
        # snapshots (_use_subprocess = False) put real uiautomation calls on
        # THIS thread for the first time -- and this codebase's own
        # _init_com() docstring already warned "each thread that touches
        # uiautomation must call this or it crashes." It wasn't called
        # anywhere (orphaned, likely written for an earlier threading model
        # before the now-removed subprocess redesign) -- confirmed live by
        # a full test-suite segfault (Windows access violation) inside
        # uiautomation's GetFocusedControl, called from this exact thread
        # via _request_snapshot. The subprocess mode never hit this because
        # each subprocess is its own fresh process with its own COM state;
        # taking that isolation away means this thread needs its own COM
        # init, same as every other thread in this file that touches UIA.
        self._init_com()
        # An injected observer is built here, on the thread that will snapshot
        # it — see __init__. Failing to build is loud: a recording with no
        # observer produces well-formed files full of empty states, which is
        # worse than no recording at all because it looks like data.
        if self._observer_factory is not None:
            try:
                self._observer = self._observer_factory()
            except Exception as exc:
                print(f"[DemoRecorder] observer unavailable — this recording would "
                      f"have no state: {exc}")
                self._quit_event.set()
                return
        try:
            # initial baseline snapshot (on-demand)
            self._last_state = self._request_snapshot() or {}
            while not self._quit_event.is_set():
                try:
                    event = self._action_queue.get(timeout=0.5)
                except _queue.Empty:
                    continue
                try:
                    self._process_event(event)
                except Exception:
                    self._log_crash("worker._process_event")
        finally:
            # Only what this thread built. Detaching over CDP leaves the
            # operator's own browser open — they were driving it — but the
            # Playwright session holds a node subprocess that must be released.
            if self._observer_factory is not None and self._observer is not None:
                disconnect = getattr(self._observer, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect()
                    except Exception as exc:
                        print(f"[DemoRecorder] observer disconnect failed: {exc}")
                self._observer = None

    def _process_event(self, event: dict):
        """pre = snapshot before this action (= prior action's post); take ONE
        fresh on-demand snapshot for post; it becomes the next action's pre.
        UIA fires only here — a few times per demo, never continuously."""
        action_type = event["action_type"]
        pre = self._last_state or {}

        # Coalesce during bursts: if more actions are already queued, DON'T pay
        # the ~200ms snapshot — reuse last state and drain fast. Snapshot only
        # when caught up (queue empty), so the worker keeps pace with the user
        # and the console stops firing in delayed bursts.
        # EXCEPTION: clicks ALWAYS get a fresh snapshot. A click moves keyboard
        # focus; coalescing it reuses a stale state where the focused field never
        # updates — which destroys the focus signal the navigation model needs
        # (focus was stuck on one field for 11 consecutive clicks). Clicks are
        # infrequent vs scroll/move bursts, so this won't reintroduce burst lag.
        backlog = self._action_queue.qsize()
        _is_click = action_type in ("click", "double_click")
        if backlog > 0 and not _is_click:
            time.sleep(0.02)
            post = self._last_state or pre
        else:
            time.sleep(0.12)
            post = self._request_snapshot(action_type)
            if not post.get("elements"):
                post = pre
            self._last_state = post

        # filter noise-app actions (terminal/browser) using fresh post-state
        if self._is_noise_app(post) and self._is_noise_app(pre):
            return
        # drop clicks on the recorder's OWN GUI window (Start/Stop/Replay buttons)
        # — these are control clicks, never part of the demo. Without this they
        # land as "[cursor position]" junk steps at the end of every recording.
        if action_type in ("click", "double_click"):
            _wt = (post.get("window_title") or "") + " " + (pre.get("window_title") or "")
            if "bc recorder" in _wt.lower() or "intern" in _wt.lower():
                return
        # drop bare Notepad single clicks (cursor-positioning noise)
        if action_type == "click" and "notepad" in (post.get("application") or "").lower():
            return
        # DROPDOWN-SELECTION filter: if a combobox dropdown was OPEN when this
        # click happened (list items present in the pre-state), the click is a
        # value-selection, not navigation. Its position lands on the option,
        # which visually sits OVER a lower field — so it would be mis-recorded as
        # a click on that field (phantom "Expiration Date" between Type and Term).
        # The combobox-OPEN click is kept (pre has no list items); only the
        # selection click that follows is dropped. Value-picking is the LLM's job.
        if action_type in ("click", "double_click"):
            _n_li = sum(1 for e in (pre.get("elements") or [])
                        if "listitem" in (e.get("type") or "").lower())
            if _n_li > 0:
                return

        clipboard = ""
        if action_type == "hotkey" and event.get("hotkey") in ("ctrl+c", "ctrl+v"):
            try:
                import pyperclip as _pc
                clipboard = _pc.paste() or ""
            except Exception:
                pass

        self._append_step(
            state_before    = pre,
            state_after     = post,
            action_type     = action_type,
            click_pos       = event.get("click_pos"),
            text            = event.get("text", ""),
            keystrokes      = event.get("keystrokes", []),
            hotkey          = event.get("hotkey", ""),
            scroll_pos      = event.get("scroll_pos"),
            scroll_dy       = event.get("scroll_dy", 0.0),
            drag_src        = event.get("drag_src"),
            drag_dst        = event.get("drag_dst"),
            clipboard       = clipboard,
        )

    def _capture(self) -> dict:
        """Return last worker snapshot (used by F10 flush path)."""
        return getattr(self, "_last_state", None) or {}

    def _flush_text_to_queue(self):
        """Queue accumulated typed text as a keyboard event (no state read)."""
        if self._pending_text or self._pending_keys:
            self._action_queue.put({
                "action_type": "keyboard",
                "text":        self._pending_text,
                "keystrokes":  list(self._pending_keys),
            })
            self._pending_text = ""
            self._pending_keys = []

    def _flush_pending(self, *args, **kwargs):
        self._flush_text_to_queue()

    def _append_step(
        self,
        state_before:   dict,
        state_after:    dict,
        action_type:    str,
        click_pos:      list | None = None,
        text:           str         = "",
        keystrokes:     list        = [],
        hotkey:         str         = "",
        scroll_pos:     list | None = None,
        scroll_dy:      float       = 0.0,
        drag_src:       list | None = None,
        drag_dst:       list | None = None,
        notepad_line:   str         = "",
        notepad_select: str         = "",
        clipboard:      str         = "",
    ):
        ts = datetime.now().isoformat()
        if action_type == "click" and click_pos:
            mouse_entry = {"actions": [{"position": [float(click_pos[0]), float(click_pos[1])],
                                        "type": "click", "timestamp": ts}]}
            kb_entry = {"actions": []}
        elif action_type == "double_click" and click_pos:
            mouse_entry = {"actions": [{"position": [float(click_pos[0]), float(click_pos[1])],
                                        "type": "double_click", "timestamp": ts}]}
            kb_entry = {"actions": []}
        elif action_type == "drag" and drag_src and drag_dst:
            mouse_entry = {"actions": [{"position": [float(drag_src[0]), float(drag_src[1])],
                                        "dst_position": [float(drag_dst[0]), float(drag_dst[1])],
                                        "type": "drag", "timestamp": ts}]}
            kb_entry = {"actions": []}
        elif action_type == "scroll" and scroll_pos:
            mouse_entry = {"actions": [{"position": [float(scroll_pos[0]), float(scroll_pos[1])],
                                        "type": "scroll", "dy": float(scroll_dy), "timestamp": ts}]}
            kb_entry = {"actions": []}
        elif action_type == "hotkey" and hotkey:
            mouse_entry = {"actions": []}
            kb_entry = {"actions": [{"hotkey": hotkey, "strokes": [{"key": hotkey, "pasted_text": ""}]}]}
        elif action_type == "keyboard":
            mouse_entry = {"actions": []}
            if text:
                strokes = [{"pasted_text": text, "key": ""}]
            else:
                strokes = [{"key": k, "pasted_text": ""} for k in keystrokes]
            kb_entry = {"actions": [{"strokes": strokes}]}
        else:
            return

        with self._lock:
            idx = len(self._steps)
            step = {
                "trace_id":   f"live_step_{idx:04d}",
                "timestamp":  datetime.now().isoformat(),
                "duration":   1.0,
                "type":       self.trace_type,
                "state":      _fmt_state(state_before),
                "mouse":      mouse_entry,
                "keyboard":   kb_entry,
                "next_state": _fmt_state(state_after),
            }
            self._steps.append(step)
            desc = _semantic_desc(action_type, click_pos, text, keystrokes, state_after,
                                  hotkey=hotkey, scroll_dy=scroll_dy,
                                  drag_src=drag_src, drag_dst=drag_dst,
                                  notepad_line=notepad_line,
                                  notepad_select=notepad_select,
                                  clipboard=clipboard)
            warn = " [!empty state]" if not state_after.get("elements") else ""
            if warn:
                # DIAGNOSTIC, added 2026-08-10: direct user report ("still so
                # fucking slow" / "still so much delay"), every single step
                # in a live Electron session showed empty state. snapshot()
                # has no concept of a locked target window -- it captures
                # whatever win32gui.GetForegroundWindow() returns at that
                # instant. Need to see WHAT it's actually grabbing (the form?
                # the Electron recorder's own window? something else?) to
                # tell apart "wrong window" from "right window, genuinely
                # nothing there."
                print(f"       [EMPTY-DIAG] app={state_after.get('application')!r} "
                      f"window_title={state_after.get('window_title')!r}")
            print(f"  [{idx:04d}] {desc}{warn}")

        # write this step to disk immediately — never lose data on crash/kill
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"live_step_{idx:04d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(step, f, indent=2, ensure_ascii=False)
        except Exception:
            self._log_crash("incremental_save")

    def _save(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        with self._lock:
            steps = list(self._steps)
        if not steps:
            print("  No steps recorded — nothing saved.")
            return self.output_dir
        for i, step in enumerate(steps):
            path = os.path.join(self.output_dir, f"live_step_{i:04d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(step, f, indent=2, ensure_ascii=False)
        return self.output_dir

    @staticmethod
    def _key_name(key) -> str:
        try:
            if key.char:
                return key.char
        except AttributeError:
            pass
        try:
            return key.name  # "f9", "f10", "tab", "enter", etc.
        except AttributeError:
            return str(key)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["ScreenObserver", "MouseInput", "KeyboardInput", "ClipboardMonitor", "DemoRecorder"]
