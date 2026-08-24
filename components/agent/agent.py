"""
components/agent/agent.py
=========================
LLMAgent — goal-driven agentic loop with multi-provider LLM support.

Supported providers
-------------------
  anthropic  — Claude (claude-sonnet-4-6)         paid, best quality
  groq       — Llama 3.3 70B via Groq API         free tier, very fast
  gemini     — Google Gemini Flash                 free tier
  lmstudio   — Local LM Studio (OpenAI-compatible) completely free, offline
  none       — Transformer-only fallback           no LLM, no API needed

Architecture
------------
  User goal (natural language)
        ↓
  LLMAgent.run()
        ↓  ← UIAutomationObserver.snapshot()  (live screen every step)
  LLM provider  (anthropic | groq | gemini | lmstudio)
    • Understands the task goal
    • Reads current screen state as plain text
    • Decides: continue | done | error | wait
        ↓
  TransformerAgentNetwork.predict()
    • Predicts exact action: click(x,y) | keyboard | no_op
        ↓
  _TextResolver  (Options 1 + 2 + 3)
    • Finds text to type from background window elements
        ↓
  ActionExecutor
    • Fires real OS input via pyautogui + clipboard paste

Usage
-----
  agent = LLMAgent(
      goal="Fill the form using data from Notepad",
      provider="groq",
      api_key="gsk_...",
  )
  results = agent.run(max_steps=30)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMP_DIR  = os.path.dirname(_THIS_DIR)
_ROOT      = os.path.dirname(_COMP_DIR)
for _p in (_ROOT, _COMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── provider SDK imports (all optional) ───────────────────────────────────────
try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    from groq import Groq as _Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

try:
    from google import genai as _genai
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

try:
    from openai import OpenAI as _OpenAI   # used for LM Studio (OpenAI-compatible)
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── default model IDs per provider ────────────────────────────────────────────
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "groq":      "llama-3.3-70b-versatile",
    "gemini":    "gemini-2.0-flash",
    "lmstudio":  "local-model",   # LM Studio uses whatever model is loaded
}

# ── system prompt (shared across all providers) ───────────────────────────────
_SYSTEM_PROMPT = """\
You are Intern, a desktop automation agent. You observe any GUI and control the
mouse and keyboard to accomplish goals — forms, email, Excel, web, file management,
or anything else visible on screen.

You receive three sections each step:
  GOAL             — what the user wants accomplished.
  SCREEN           — every visible UI element (type, label, current value).
  DATA SOURCES     — text visible in background windows (Notepad, files, etc.)
                     Use these values when you need to type something.
  LAST ACTIONS     — your recent actions and whether they changed the screen.

Respond with ONE JSON action only — no other text:

{
  "action_type": "click" | "type" | "hotkey" | "scroll" | "done" | "wait",
  "target":      "<EXACT label or text from SCREEN — never paraphrase>",
  "text":        "<exact value from DATA SOURCES — never invent>",
  "keys":        ["tab"],
  "direction":   "down" | "up",
  "clicks":      3,
  "reason":      "<one sentence>"
}

Action rules:
- "click"  → any element: field, button, checkbox, dropdown, tab, menu item.
             target = EXACT string from SCREEN. Never guess or paraphrase.
- "type"   → types into the focused field. text = value from DATA SOURCES only.
- "hotkey" → keyboard shortcut. keys: ["tab"] advances focus, ["return"] confirms,
             ["escape"] dismisses a dialog. Only ["tab"], ["return"], ["escape"],
             ["ctrl+a"], ["ctrl+c"], ["ctrl+v"] are ever executed — anything else
             is refused. NEVER use a window-closing combo like "alt+f4" or "ctrl+w"
             for ANY reason, including a stuck or confusing field — use ["tab"]
             instead.
- "scroll" → scroll in any direction. target = element to scroll near.
- "done"   → goal is fully achieved. Only use when certain.
- "wait"   → UI is loading or animating. Use sparingly.

General rules:
1. CRITICAL: "text" must be copied VERBATIM from DATA SOURCES. If the exact value is not visible in DATA SOURCES, do NOT type — skip with hotkey ["tab"] instead.
2. Never complete, guess, reformat, or paraphrase values. Copy the exact string.
3. If a field already has the correct value → hotkey ["tab"] to move on.
4. Interact with whatever app is on screen — do not assume it is a form.
5. Output JSON only. No explanation outside the object.
6. ALWAYS act on the CURRENTLY FOCUSED FIELD first. Do not click other fields to skip ahead.
7. Fill fields in order — top to bottom, left to right. Never jump over a field.
8. UNKNOWN VALUE RULE: If you do not see the value for the focused field in DATA SOURCES, do NOT guess, invent, or reuse a value from another field. Output exactly: {"action_type": "hotkey", "keys": ["tab"], "reason": "value not found in data sources — skipping"}
9. UNCERTAINTY RULE: If you are unsure what action to take next, skip with tab rather than guess. A skipped field is recoverable. A wrong value is not.
10. All required data is already provided in DATA SOURCES. Do not click buttons or controls that would fetch, import, or load additional data — doing so will open dialogs that derail the task.
11. DIALOG ESCAPE: If an unexpected dialog or popup appears that you did not intend to open, press Escape immediately: {"action_type": "hotkey", "keys": ["escape"], "reason": "dismissing unexpected dialog"}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  State / history → text helpers
# ══════════════════════════════════════════════════════════════════════════════

# Canonical record parser + default data source live in data_sources.notepad_source.
# This re-export keeps existing call sites working.
from data_sources.notepad_source import _parse_records, NotepadDataSource  # noqa: F401


def _state_to_text(state: Dict[str, Any], record_num: int = 1, visual_cache: Optional[Dict[str, str]] = None, filled_labels: Optional[set] = None) -> str:
    lines: List[str] = []
    lines.append(f"Active window: {state.get('application','?')} — {state.get('window_title','')!r}")

    for w in state.get("windows", []):
        lines.append(
            f"  [{w.get('role','?')}] {w.get('app','')} — "
            f"{w.get('title','')!r}  ({w.get('element_count',0)} elements)"
        )

    lines.append("")
    # ── CURRENT FOCUS banner ─────────────────────────────────────────────────
    focused_id = state.get("focused_element_id")
    all_elems  = state.get("elements", [])
    focused_el = next((e for e in all_elems if e.get("element_id") == focused_id), None)
    if focused_el:
        _fn  = (focused_el.get("label") or focused_el.get("text") or "?").strip()[:200]
        _fv  = (focused_el.get("value") or "").strip()[:200]
        _ft  = focused_el.get("type", "?")
        lines.append(f">>> CURRENTLY FOCUSED: [{_ft}] \"{_fn}\"" +
                     (f"  current value: {_fv!r}" if _fv else "  (empty)") + " <<<")
        lines.append("")

    active = [e for e in state.get("elements", []) if e.get("window_role") in ("active", None)]
    bg     = [e for e in state.get("elements", []) if e.get("window_role") == "background"]

    _INTERACTIVE = {
        "input", "button", "checkbox", "radio", "combobox",
        "listitem", "tabitem", "splitbutton", "link", "list",
        # UIA ControlTypeName variants (suffix "Control")
        "editcontrol", "buttoncontrol", "checkboxcontrol", "radiobuttoncontrol",
        "comboboxcontrol", "listitemcontrol", "tabitemcontrol", "listcontrol",
        "hyperlinkcontrol", "splitbuttoncontrol",
    }
    _SKIP_TYPES  = {
        "window", "titlebar", "pane", "toolbar", "statusbar",
        "menubar", "menu", "group", "separator", "scrollbar",
        "progressbar", "tooltip", "document",
        "windowcontrol", "titlebarcontrol", "panecontrol", "toolbarcontrol",
        "menubarcontrol", "menuitemcontrol", "scrollbarcontrol",
    }

    all_elems = state.get("elements", [])

    # Form-specific control types — only these identify a window as a form.
    # Excludes buttons/scrollbars that appear in ANY window (e.g. Notepad toolbar).
    _FORM_CONTROLS = {
        "input", "combobox", "checkbox", "radio",
        "editcontrol", "comboboxcontrol", "checkboxcontrol", "radiobuttoncontrol",
    }

    # Form elements = windows that contain actual form-fill inputs.
    # This works even when the terminal has OS focus (form is "background").
    form_windows = {
        e.get("window_title") for e in all_elems
        if e.get("type") in _FORM_CONTROLS
    }

    # Fallback: if no form-fill controls detected anywhere, treat the active
    # window as the form (handles wx apps where UIA maps controls to custom types)
    if not form_windows:
        active_title = state.get("window_title", "")
        if active_title:
            form_windows = {active_title}

    form_elems = [e for e in all_elems if e.get("window_title") in form_windows]

    # Data-source elements = windows with no interactive inputs (e.g. Notepad)
    data_elems = [e for e in all_elems if e.get("window_title") not in form_windows]

    labels      = [e for e in form_elems
                   if e.get("type") not in _INTERACTIVE
                   and e.get("type") not in _SKIP_TYPES
                   and (e.get("text") or "").strip()]
    interactive = [e for e in form_elems if e.get("type") in _INTERACTIVE]

    # Debug: log all unique types in the form so we can see what UIA reports
    all_types = sorted({e.get("type", "?") for e in form_elems})


    _filled = filled_labels or set()
    _filled_lower = {s.lower() for s in _filled}

    lines.append("SCREEN (use EXACT strings as 'target' when clicking):")
    for e in labels[:15]:
        focused = " [FOCUSED]" if e.get("focused") else ""
        txt = (e.get('text') or '').strip()
        if len(txt) > 200:
            continue
        lines.append(f"  \"{txt}\"{focused}")

    # Only show unfilled interactive elements (+ always show focused element)
    _focused_id_st = state.get("focused_element_id")
    _shown = 0
    lines.append(f"\nINTERACTIVE ELEMENTS (showing unfilled):")
    for e in interactive:
        if _shown >= 15:
            break
        val   = (e.get("value") or "").strip()
        text  = (e.get("text") or "").strip()
        if len(val) > 300 or len(text) > 300:
            continue
        label = text or val or "(empty)"
        # Skip already-filled non-focused elements to save tokens
        if (label.lower() in _filled_lower
                and e.get("element_id") != _focused_id_st
                and e.get("type") not in ("tabitem", "tabitemcontrol", "buttoncontrol")):
            continue
        focused = " [FOCUSED]" if e.get("focused") else ""
        lines.append(f"  [{e.get('type','?')}] \"{label}\"{focused}"
                     + (f"  current value: {val!r}" if val else ""))
        _shown += 1

    if visual_cache:
        # Only show unfilled fields — already-filled ones waste tokens
        # Always include the focused field's entry even if it's past the cap
        _focused_field_name = ""
        if focused_el:
            _focused_field_name = (focused_el.get("label") or focused_el.get("text") or "").strip().lower()
        lines.append(f"\nDATA SOURCES (Record {record_num}, unfilled fields):")
        _shown_ds = 0
        for field, value in visual_cache.items():
            _is_focused_field = field.lower() == _focused_field_name
            if _shown_ds >= 20 and not _is_focused_field:
                continue
            if field.lower() in _filled_lower and not _is_focused_field:
                continue
            lines.append(f"  {field} : {value}")
            _shown_ds += 1
    elif data_elems:
        # Collect all background text blobs
        bg_blobs = []
        for e in data_elems:
            val = (e.get("value") or "").strip()
            if val:
                bg_blobs.append(val)

        logger.debug("Background elements: %d total, %d with text (blob sizes: %s)",
                    len(data_elems), len(bg_blobs),
                    [len(b) for b in bg_blobs])

        if bg_blobs:
            # Try each blob for record structure; prefer the one that parses
            records  = {}
            raw_text = ""
            for blob in sorted(bg_blobs, key=len, reverse=True):
                r = _parse_records(blob)
                if r:
                    records  = r
                    raw_text = blob
                    break
            if not raw_text:
                raw_text = max(bg_blobs, key=len)
            logger.debug("_parse_records → %d record(s) found (blob size=%d)", len(records), len(raw_text))

            if records:
                rec = records.get(record_num, records.get(min(records), {}))
                lines.append(f"\nDATA SOURCES (Record {record_num}):")
                for field, value in rec.items():
                    lines.append(f"  {field} : {value}")
            else:
                # Plain text — dump up to 3 000 chars to keep prompt small
                lines.append(f"\nDATA SOURCES:")
                lines.append(f"  {raw_text[:3000]}")

    return "\n".join(lines)


def _is_llm_unavailable(llm_action: Dict[str, Any]) -> bool:
    """
    True iff this llm_action is _ask_llm()'s infra-failure sentinel (connection
    error / non-JSON response / timeout — see _ask_llm's except block), NOT a
    genuine "the value should be blank" decision from a reachable LLM.

    The two used to collapse to the same thing downstream (both end up as an
    empty prediction["text"]), which made a dead LLM connection silently look
    like every field was legitimately blank — the agent would Tab-skip an
    entire form instead of surfacing that the provider was unreachable.
    """
    return llm_action.get("action_type") == "wait" and llm_action.get("reason") == "llm unavailable"


# Matches the gate already established for the non-Option-B merge path
# (agent.py's _merge(), ~L4045) — same meaning there: below this, the
# transformer's own pointer isn't trustworthy enough to act on.
_CLICK_CONF_FLOOR = 0.30

# A wrong click on a tab-strip element is far more costly than a wrong click
# on a same-tab field: it skips an entire tab's worth of fields, not just one
# step. Found live 2026-08-07 (twice, same session): the model clicked away
# to the Vehicle tab with confidence 0.38-0.39 while Policyholder still had
# unfilled fields — comfortably above the general 0.30 floor, so nothing
# blocked it. Section changes get a stricter bar than in-tab navigation.
# Not tuned to this form's tabs specifically — applies to any tabitem/
# tabitemcontrol element on any form.
_TAB_CLICK_CONF_FLOOR = 0.50


def _is_tab_strip_click(pos: Optional[List[float]], elements: Optional[List[Dict[str, Any]]]) -> bool:
    """True if `pos` falls inside any tab-strip element's bbox."""
    if not pos or not elements:
        return False
    px, py = pos[0], pos[1]
    for e in elements:
        if (e.get("type") or "").lower() not in ("tabitem", "tabitemcontrol"):
            continue
        if e.get("window_role") == "background":
            continue
        b = e.get("bbox")
        if not b or len(b) != 4:
            continue
        if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
            return True
    return False


def _round_click_pos(pos: List[float]) -> Tuple[float, float]:
    """10px-bucketed position key, matching the rounding _merge() already
    uses for its own no_change blacklist — small pointer jitter between two
    predictions of "the same" click shouldn't dodge the blacklist."""
    return (round(pos[0] / 10) * 10, round(pos[1] / 10) * 10)


def _gate_low_confidence_click(
    pos: Optional[List[float]],
    click_conf: float,
    elements: Optional[List[Dict[str, Any]]] = None,
    blacklist: Optional[Set[Tuple[float, float]]] = None,
) -> Optional[List[float]]:
    """
    Returns `pos` unchanged if it's structurally invalid anyway (nothing to
    gate) or `click_conf` clears the applicable floor and the position isn't
    blacklisted; returns None if it's a plausible click the model itself
    flagged as unreliable, or one already proven unproductive — the caller
    then falls back to the generic Tab-advance already used for a
    structurally invalid pointer, same as this function's own null case.

    Found 2026-08-07, live: a ptr_conf=0.20 guess landed on the tab strip
    and ended a tab pass 9 fields early (Policy: 4/13 filled) — the
    confidence was already being logged, never acted on. This does NOT hand
    WHERE to the LLM (forbidden in pure/Option-B mode, see _merge()) — it
    only declines to act on an unreliable guess. Tab is a universal
    navigation primitive, not task-specific, so this generalizes to any
    form the agent is pointed at.

    A click landing on a tab-strip element uses the stricter
    _TAB_CLICK_CONF_FLOOR instead — see that constant's comment. `elements`
    is optional (defaults to the general floor) so existing call sites that
    can't easily supply it keep working unchanged.

    `blacklist` catches a DIFFERENT failure mode confidence alone can't:
    found live 2026-08-07, a HIGH-confidence (0.91) click landed on the same
    empty, optional combobox 30+ times in a row — correctly recognized each
    time as "nothing to fill here," but the code path handling that (Escape
    + Tab + mark-attempted) `continue`d immediately, bypassing the
    repeat-action guard that would otherwise have broken the loop. A caller
    that tracks proven-unproductive positions (e.g. self._nochange_click_pos)
    can pass them here so a confidently-wrong pointer gets rejected too.
    """
    if not pos or not (pos[0] > 1 or pos[1] > 1):
        return pos
    if blacklist and _round_click_pos(pos) in blacklist:
        return None
    floor = _TAB_CLICK_CONF_FLOOR if _is_tab_strip_click(pos, elements) else _CLICK_CONF_FLOOR
    if click_conf < floor:
        return None
    return pos


_LISTITEM_TYPES = {"listitem", "listitemcontrol"}

# Generic submit-button keywords — any button whose visible text/label contains
# one of these is treated as "the way to finish this record", on any form this
# agent is pointed at. No form-specific button text (e.g. "Submit") is assumed
# anywhere else in the file; this is the one shared, reusable set.
_SUBMIT_KW = {"submit", "save", "ok", "accept", "done", "finish", "new"}

# Generic destructive-button keywords — any button whose visible text/label
# contains one of these must NEVER be clicked as an incidental side effect of
# navigation. Found 2026-08-08, live: the transformer's own low-confidence
# navigate-pointer (ptr_conf=0.53, no different from dozens of other
# navigation clicks) happened to land on a 'Clear All' button, which popped a
# real Windows confirmation dialog (confirmed via its class name, '#32770' —
# a genuine dialog box, not a misidentified sibling window) that then got
# dismissed with Escape every subsequent step for the rest of the run,
# blocking all further progress. The existing _SUBMIT_KW guard (below) only
# covers accidentally FINISHING early, not accidentally DESTROYING data, and
# was gated off entirely in this run's mode anyway (self._no_autohandlers) --
# this set/check applies unconditionally, in every mode, since clicking a
# destructive button is never an acceptable side effect of the model being
# unsure where to click next.
_DESTRUCTIVE_KW = {"clear", "reset", "delete", "erase", "discard", "cancel"}


def _find_submit_button(elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """First button-type element whose text/label matches _SUBMIT_KW, or None.
    Deliberately does NOT filter by window_role — on this form (and likely
    others), the submit button lives on the main window, tagged "background"
    by the state scanner, so filtering it out here would just miss it."""
    return next(
        (e for e in elements
         if e.get("type") in ("buttoncontrol", "button")
         and e.get("bbox")
         and any(kw in (e.get("text") or e.get("label") or "").lower() for kw in _SUBMIT_KW)),
        None,
    )


def _next_record_or_finished(current: int, total: int) -> Optional[int]:
    """Given the record number just submitted and the total record count
    found in the intake text, return the next record number to advance to,
    or None if `current` was the last one (the run is genuinely finished).

    Pure and total-agnostic on purpose: `total` comes from actually parsing
    the intake text's record blocks (see _total_records_in), not a hardcoded
    "5 records" assumption — a 1-record or 20-record intake file both just
    work, no agent-side change needed either way.
    """
    return current + 1 if current < total else None


def _total_records_in(records: Dict[int, Any]) -> int:
    """How many records a _parse_records()-shaped {record_num: {...}} dict
    represents. 1 (not 0) when empty — the un-multi-record default, so a
    source with no 'RECORD N OF M' structure at all still behaves exactly
    as a single-record run always has (record_num never exceeds total, so
    _next_record_or_finished always signals 'finished' after one submit)."""
    return max(records.keys()) if records else 1


def _find_destructive_button_at(elements: List[Dict[str, Any]], pos) -> Optional[Dict[str, Any]]:
    """Any button-type element at `pos`, or None.

    Found 2026-08-08, live, in the SAME conversation this was first written:
    the initial version only matched _DESTRUCTIVE_KW ("clear", "reset", ...)
    — caught the 'Clear All' button that had just caused a real, confirmed
    incident, but direct user pushback ("It's not just the Clear All") was
    right. Reading car_insurance_form_wx.py's own footer button list
    directly (not guessing) found FIVE buttons sharing one row: Submit,
    Clear All, Print Preview, Load Record, Save Record — and 'Load Record'
    (_on_load) opens a file picker and, if a file is chosen, silently
    OVERWRITES every field with a different record's data; 'Print Preview'
    (_on_print_preview) opens its own separate modal dialog. Both are just
    as capable of derailing a run as 'Clear All' was, and neither matches
    any destructive keyword — a keyword list is exactly the whack-a-mole
    trap that would need a new entry every time a form added one more
    button with an unanticipated label.

    The actually-generalizing fix: this function is called from any branch
    that clicks a raw, transformer-derived screen position rather than a
    position already known (by construction) to belong to a fillable field
    — originally just the NAVIGATE branch (moving the pointer between
    FIELDS to fill them), extended 2026-08-09 to the combobox-open-dropdown
    branch once live evidence showed it shares the exact same risky input
    (_snap2) without the same check. Neither branch has a legitimate reason
    to ever click a button-type element at all, regardless of its label.
    The one place a button SHOULD get clicked deliberately (Submit, once a
    record is genuinely finished) already goes through its own separate,
    dedicated path (_find_submit_button, called directly by the
    stuck-guard's "last tab exhausted" branch) — it never flows through the
    uncertain click_position this function checks. So instead of trying to
    name every button that must not be clicked, this simply refuses ALL of
    them here; the one that legitimately should get clicked already
    bypasses this check entirely.

    _DESTRUCTIVE_KW is kept (unused by this function now) since it still
    documents which specific buttons the ORIGINAL incident involved and
    remains available for any future call site that specifically needs a
    keyword match rather than a blanket refusal.

    Same "don't filter by window_role" reasoning as _find_submit_button —
    these buttons live on the main window too."""
    if not pos or len(pos) < 2:
        return None
    px, py = pos[0], pos[1]
    return next(
        (e for e in elements
         if e.get("type") in ("buttoncontrol", "button")
         and e.get("bbox")
         and e["bbox"][0] <= px <= e["bbox"][2]
         and e["bbox"][1] <= py <= e["bbox"][3]),
        None,
    )


def _is_leave_blank_value(v: Optional[str]) -> bool:
    """True if a record value means 'leave this field empty' -- robust to
    parens and trailing notes: '(leave blank)', 'leave blank — liability
    only', '(none)', 'none', 'n/a' all count as blank.

    Found 2026-08-10: SIX separate call sites each carried their own
    hardcoded {'(leave blank)', 'none', ...} set, checked via
    `value.lower().strip('()')` on the VALUE being looked up but never
    applied to the SET itself -- so a stripped 'leave blank' could never
    match an unstripped '(leave blank)' set member, and the literal
    placeholder text got typed into the field instead of being skipped.
    One already-correct implementation existed the whole time
    (_lookup_field's own local _is_blank, a prefix match immune to this
    exact bug since it never relied on exact-set-membership against an
    unstripped literal) -- promoted to module level so every call site
    shares it instead of maintaining its own copy that can silently drift
    out of sync, the same generalization already applied to
    _find_destructive_button_at and _SAFE_HOTKEY_COMBOS elsewhere in this
    file.
    """
    if not v:
        return True
    n = v.lower().strip().strip("()").strip()
    return (n in {"none", "n/a", "na", ""}
            or n.startswith("leave blank")
            or n.startswith("none "))


_TAB_PANEL_TYPES = {"editcontrol", "comboboxcontrol", "checkboxcontrol",
                     "buttoncontrol", "button", "panecontrol"}


def _detect_active_tab_idx_raw(tabs: List[Dict[str, Any]],
                                all_elems: List[Dict[str, Any]]) -> Optional[int]:
    """
    Which tab (by index into `tabs`) is actually visible on screen right now,
    or None if no tab's panel could be confirmed. wx moves inactive tab panels
    to negative screen coordinates, so the active tab is the one whose panel
    has at least one real, on-screen (bbox[1] >= 0, below the tab strip)
    interactive element. No hardcoded tab names or count — works for any
    tabbed form.

    Extracted 2026-08-08 from _try_advance_tab so it can ALSO be used at
    startup to verify drill mode's assumed start_tab_idx against reality —
    see LLMAgent.run()'s own startup-verification block for why that case
    needs the OPPOSITE trust rule from _try_advance_tab's mid-run one (this
    function itself is neutral — callers decide what "disagreement" means).

    Also requires e["visible"] (default True). Found 2026-08-08, live,
    immediately after wiring UIA's real IsOffscreen signal into ui_observer.py
    for the scroll-visibility fix: THIS function's own "wx moves inactive
    panels to negative coordinates" assumption turned out to be wrong for
    this form — tab index 0's (Policy) panel elements kept reporting
    bbox[1] >= 0 even long after switching away from it, so this loop
    (checking tabs in order, returning the FIRST match) returned 0 on every
    single call for the rest of the run, no matter which tab was actually
    showing. _try_advance_tab's own drift-guard correctly distrusted that
    and fell back to its tracker each time — but the tracker has no way to
    confirm a click actually landed, so it just kept incrementing blindly,
    cycling through all 8 tabs repeatedly without ever confirming a real
    switch or filling anything. IsOffscreen is UIA's own authoritative
    "is this actually rendered right now" answer — Policy's covered-over
    panel content reports IsOffscreen=True even though its raw bbox never
    moved, which the geometric check alone couldn't see.
    """
    for _tii, _tab_el in enumerate(tabs):
        _tbbox = _tab_el.get("bbox")
        if not _tbbox:
            continue
        _tab_cy = (_tbbox[1] + _tbbox[3]) / 2
        _panel_elems = [
            e for e in all_elems
            if e.get("window_role") != "background"
            and e.get("bbox")
            and e["bbox"][1] >= 0
            and e["bbox"][1] > _tab_cy   # below the tab strip
            and e.get("visible", True)
            and e.get("type") in _TAB_PANEL_TYPES
        ]
        if _panel_elems:
            return _tii
    return None


def _open_dropdown_items(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Real, on-screen list items in `elements` — non-empty exactly when some
    combobox's dropdown is currently open.

    A combobox click TOGGLES, it doesn't just open — clicking one that's
    already open closes it instead. Found live 2026-08-07, the actual root
    cause of a loop that survived both an earlier type-name fix and a poll-
    timing widening: a PRECEDING step's click (nominally aimed at a
    different, adjacent field) landed on this combobox's screen position
    and popped it open by accident. The dedicated combobox-fill handler then
    blindly clicked "to open" it again, which closed it — so every
    subsequent poll correctly found 0 items, because there was genuinely
    nothing open to find. Callers should check this BEFORE clicking to open
    a combobox, and skip the click if it's already open.
    """
    return [e for e in elements
            if e.get("type") in _LISTITEM_TYPES
            and e.get("window_role") != "background"
            and e.get("bbox")]


def _tokenize_option(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _option_matches(value: str, option: str) -> bool:
    """
    True if `option` (a dropdown item's text) should count as matching the
    wanted `value`, beyond a bare identical string.

    Three tiers, each covering a real live case:
      1. Exact match (case/whitespace-insensitive).
      2. Prefix match, either direction — 'Full Coverage' vs 'Full Coverage
         (Comprehensive)'.
      3. Whole-word-token containment — 'Sedan' vs '4-Door Sedan'. Found
         live 2026-08-07: prefix-only matching couldn't catch this (neither
         string is a prefix of the other), so the agent looped forever
         opening the dropdown, never finding 'Sedan', closing it, and
         trying again — 25+ times in one run.

    Deliberately NOT a raw substring check — 'Active' must never match
    'Inactive' (an explicit constraint from when this matching was first
    written). Splitting into whole tokens on non-alphanumeric boundaries and
    requiring the value's token sequence to appear as a contiguous run
    within the option's tokens preserves that: 'inactive' tokenizes to a
    single token ['inactive'], so 'active' can never match it as a
    substring of that token.
    """
    v = value.strip().lower()
    o = option.strip().lower()
    if not v or not o:
        return False
    if v == o or o.startswith(v) or v.startswith(o):
        return True
    v_tokens = _tokenize_option(value)
    o_tokens = _tokenize_option(option)
    if not v_tokens:
        return False
    n = len(v_tokens)
    return any(o_tokens[i:i + n] == v_tokens for i in range(len(o_tokens) - n + 1))


_MAX_VERIFY_RETRIES = 2


def _find_element_by_id(elements: List[Dict[str, Any]], element_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Look up an element in a state's elements list by its element_id — used
    to find the same field's post-action value in a later observation."""
    if element_id is None:
        return None
    return next((e for e in elements if e.get("element_id") == element_id), None)


def _is_leave_blank_prediction(prediction: Dict[str, Any]) -> bool:
    """
    True only when `prediction` is still actually a type/keyboard action AND
    its resolved text means "leave this field empty" (blank / none / n-a /
    "leave blank ..."). Callers should Tab past + mark-attempted only when
    this returns True.

    Found 2026-08-07, live: `_merge()` can silently turn a "type" decision
    into a "click" (comboboxes need a click to open their dropdown before a
    value can be selected). A click prediction has no "text" key at all, so
    naively checking `prediction.get("text")` after that override always
    reads as empty and looks exactly like "leave blank" — even when the LLM
    correctly resolved a real value (e.g. 'Sport 2.0T'). That falsely
    Tab-skipped away before the click could even open the dropdown, and the
    transformer then clicked straight back onto the same combobox next step
    — a confirmed live infinite loop. Gating on action_type=="keyboard" first
    means a click override is never mistaken for a deliberate leave-blank.
    """
    if prediction.get("action_type") != "keyboard":
        return False
    txt  = (prediction.get("text") or "").strip()
    norm = txt.lower().strip().strip("()").strip()
    return (not txt) or norm in {"none", "n/a", "na"} or norm.startswith("leave blank")


def _verify_fill_matches(elements_after: List[Dict[str, Any]],
                          focused_id: Optional[str],
                          expected_text: str) -> bool:
    """
    True if the field that was focused before typing now holds exactly the
    text the agent intended to type. Pure comparison, no I/O — the caller
    (LLMAgent's main loop) owns the retry/re-observe orchestration, since
    that requires live executor/observer calls this function can't make.

    Found 2026-08-07, from a direct user report about the old (deleted)
    Intern iteration lacking any verify-at-fill step: StateValidator only
    checks whether SOMETHING changed after an action (focus moved, value
    differs from before) — never whether it changed to the RIGHT thing. A
    field can type successfully (validator says "ok") while still holding
    the wrong value, and nothing used to catch that.
    """
    elem = _find_element_by_id(elements_after, focused_id)
    actual_text = (elem.get("value") or "").strip() if elem else ""
    return actual_text == expected_text.strip()


def _history_to_text(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No actions taken yet."

    # Count consecutive no_change failures at the tail
    consecutive_failures = 0
    last_failed_target = ""
    for h in reversed(history):
        if h.get("validation") == "no_change":
            consecutive_failures += 1
            last_failed_target = h.get("target", "")
        else:
            break

    lines = []
    for i, h in enumerate(history[-3:], 1):
        at     = h.get("action_type", "?")
        txt    = h.get("typed_text", "")
        target = h.get("target", "")
        val    = h.get("validation", "")
        status = f"  [{val}]" if val else ""
        if at == "click":
            lines.append(f"  {i}. click {target!r}{status}")
        elif at == "keyboard":
            lines.append(f"  {i}. typed {txt!r}{status}" if txt else f"  {i}. keyboard{status}")
        else:
            lines.append(f"  {i}. {at}{status}")

    if consecutive_failures >= 2:
        lines.append(
            f"\n  WARNING: last {consecutive_failures} actions on {last_failed_target!r} "
            f"all failed with no_change. This target is not responding — try a different action or skip it."
        )
    return "\n".join(lines)


def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """Strip markdown fences, thinking blocks, and parse JSON from any LLM response."""
    import re as _re
    raw = raw.strip()
    # Strip <think>...</think> blocks emitted by reasoning models (QwQ, DeepSeek-R1, etc.)
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════════
#  Element resolver — finds pixel coords for an LLM-named target
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_target(target: str, state: Dict[str, Any]) -> Optional[List[float]]:
    """
    Given a label like "First Name", find the matching interactive element in
    the active window and return its center [x, y].

    Matching strategy (in order):
      1. Exact text/label match (case-insensitive)
      2. Partial containment match
      3. Nearest label element + adjacent input element
    """
    if not target:
        return None

    tl = target.lower().strip()
    # Search ALL elements — the form may be a background window when the
    # terminal that launched the agent has OS focus, but pyautogui clicks
    # by absolute screen coordinates so it works regardless of focus.
    active_elems = state.get("elements", [])

    _INTERACTIVE = {
        "input", "button", "checkbox", "radio", "combobox",
        "listitem", "tabitem", "splitbutton", "link", "list",
        "editcontrol", "buttoncontrol", "checkboxcontrol", "radiobuttoncontrol",
        "comboboxcontrol", "listitemcontrol", "tabitemcontrol", "listcontrol",
        "hyperlinkcontrol", "splitbuttoncontrol",
    }
    # Tab navigation controls are deprioritized — they share names with form sections
    # (e.g. "Policy" tab header vs "Policy Number" field).  Only fall back to them
    # when nothing else matches, so LLM clicks like "click Policy" land on inputs first.
    _TAB_TYPES = {"tabitem", "tabitemcontrol"}
    _INTERACTIVE_NONTAB = _INTERACTIVE - _TAB_TYPES

    def _center(e: Dict) -> List[float]:
        b = e.get("bbox", [0, 0, 0, 0])
        return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]

    # 1. Exact match on non-tab interactive element text
    for e in active_elems:
        if e.get("type") in _INTERACTIVE_NONTAB:
            txt = (e.get("text") or e.get("label") or e.get("value") or "").lower()
            if txt == tl:
                return _center(e)

    # 2. Exact match on label element → return nearest non-tab interactive element
    for e in active_elems:
        txt = (e.get("text") or e.get("label") or "").lower()
        if txt == tl:
            cx, cy = _center(e)
            best, best_dist = None, float("inf")
            for other in active_elems:
                if other.get("type") not in _INTERACTIVE_NONTAB:
                    continue
                ox, oy = _center(other)
                dist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = other
            if best and best_dist < 300:
                return _center(best)

    # 3. Partial match on non-tab interactive element text
    for e in active_elems:
        if e.get("type") in _INTERACTIVE_NONTAB:
            txt = (e.get("text") or e.get("label") or e.get("value") or "").lower()
            if tl in txt or txt in tl:
                return _center(e)

    # 4. Partial match on label → nearest non-tab interactive
    for e in active_elems:
        txt = (e.get("text") or e.get("label") or "").lower()
        if tl in txt or txt in tl:
            cx, cy = _center(e)
            best, best_dist = None, float("inf")
            for other in active_elems:
                if other.get("type") not in _INTERACTIVE_NONTAB:
                    continue
                ox, oy = _center(other)
                dist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = other
            if best and best_dist < 300:
                return _center(best)

    # 5. Match by current element value (LLM sometimes uses the value as target)
    for e in active_elems:
        if e.get("type") in _INTERACTIVE_NONTAB:
            val = (e.get("value") or "").lower().strip()
            if val and val == tl:
                return _center(e)

    # 6. Last resort: repeat passes 1+3 including tab items (explicit tab switching)
    for e in active_elems:
        if e.get("type") in _TAB_TYPES:
            txt = (e.get("text") or e.get("label") or e.get("value") or "").lower()
            if txt == tl or tl in txt or txt in tl:
                return _center(e)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  LLMAgent
# ══════════════════════════════════════════════════════════════════════════════

class LLMAgent:
    """
    Goal-driven agentic loop with pluggable LLM providers.

    Parameters
    ----------
    goal        : Natural-language task description.
    provider    : "anthropic" | "groq" | "gemini" | "lmstudio" | "none"
    api_key     : API key for the chosen provider (not needed for lmstudio/none).
    model_id    : Override the default model for the provider.
    lmstudio_url: Base URL for LM Studio server (default: http://localhost:1234/v1).
    model_path  : Path to TransformerAgentNetwork checkpoint.
    dry_run     : Log actions but do not fire real OS input.
    max_steps   : Hard cap on loop iterations.
    step_delay  : Seconds between steps.
    llm_every   : Call LLM every N steps (higher = fewer API calls).
    device_str  : Torch device for the transformer.
    """

    def __init__(
        self,
        goal:          str,
        provider:      str            = "none",
        api_key:       Optional[str]  = None,
        model_id:      Optional[str]  = None,
        lmstudio_url:  str            = "http://localhost:1234/v1",
        model_path:    str            = "tasks/form_filling/model.pt",
        dry_run:       bool           = False,
        max_steps:     int            = 50,
        step_delay:    float          = 1.2,
        llm_every:     int            = 2,
        device_str:    str            = "auto",
        record_num:    int            = 1,
        end_record:    Optional[int]  = None,  # cap multi-record advance at this record (inclusive);
                                                 # None = advance through every record the intake text has
        use_ocr:       bool           = False,
        visual_cache:  Optional[Dict[str, str]] = None,  # Gemini-extracted {field: value}
        visual_reader: Optional[Any]  = None,   # VisualDataReader for per-tab scanning
        source_window:     str            = "",     # title fragment of source data window
        task_plugin:       Optional[Any]  = None,   # TaskPlugin for task-specific logic
        pure_transformer:  bool           = False,  # skip all hardcoded handlers; transformer+LLM only
        disable_auto_handlers: bool       = False,  # skip legacy heuristics but KEEP LLM+transformer merge
        disable_transformer: bool         = False,  # ablation test: skip the model, force the SAME
                                                     # low-confidence fallback it already uses when unsure
        start_tab_idx:     int            = 0,      # start agent at this tab index (drill testing)
        scope:             Optional[Any]  = None,   # ScopeConfig — app-specific tabs/sections/records
        observer:          Optional[Any]  = None,   # perception adapter (snapshot()→schema); default=UIA
        data_source:       Optional[Any]  = None,   # DataSource for field values; default=Notepad
        route_capsule:     bool           = True,   # False = use model_path as-is (skip capsule router)
        correction_watch_seconds: float   = 4.0,    # DAgger: how long to watch for a human correction after
                                                     # a failed step. 0 = disabled. Default preserves the
                                                     # original DAgger-collection behavior; verification/
                                                     # unattended runs should pass a much smaller value —
                                                     # this blocks in real time and no one's there to correct.
    ):
        # Per-application config (tabs, sections, record delimiter). Default =
        # fully generic: no tabs, no sections, no assumptions. Each scope passes
        # its own; the agent code stays application-blind.
        try:
            from agent.scope import ScopeConfig
        except ImportError:
            from components.agent.scope import ScopeConfig
        self._scope = scope if scope is not None else ScopeConfig()
        self.goal       = goal
        self.provider   = provider.lower().strip()
        # Capsule routing: if a registered capsule matches goal/window, use its model.
        # Skipped when route_capsule=False (caller passed an explicit model_path).
        if route_capsule:
            try:
                from agent.capsule import CapsuleRegistry
                _reg = CapsuleRegistry()
                _window = ""  # window title not known yet at init; re-route on first observe
                _routed = _reg.route(goal, _window, fallback=model_path)
                if _routed != model_path:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "Capsule router: matched '%s' → %s", goal[:60], _routed)
                model_path = _routed
            except Exception:
                pass
        self.model_path = model_path
        # Resolve relative model path against repo root so CWD doesn't matter
        if not os.path.isabs(self.model_path):
            _agent_dir = os.path.dirname(os.path.abspath(__file__))
            _repo_root = os.path.normpath(os.path.join(_agent_dir, "..", ".."))
            _abs_mp    = os.path.normpath(os.path.join(_repo_root, self.model_path))
            if os.path.exists(_abs_mp):
                self.model_path = _abs_mp
        # Fail fast — a missing capsule means every step silently runs as pure LLM with no spatial
        # grounding. Surface this at startup so it's never confused with a real run.
        if provider != "none" and not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"\n\n{'!'*60}\n"
                f"  CAPSULE NOT FOUND: {self.model_path}\n"
                f"  Train the model first:  python -m components.intelligence.model.transformer "
                f"--mode train --data_dir data/output/traces/live\n"
                f"{'!'*60}\n"
            )
        self.dry_run    = dry_run
        self.max_steps  = max_steps
        self.step_delay = step_delay
        self.llm_every  = max(1, llm_every)
        self.device_str = device_str

        _model = model_id or _DEFAULT_MODELS.get(self.provider, "")
        self._llm_client: Optional[Any] = None
        self._llm_model:  str           = _model

        self._init_provider(api_key or "", lmstudio_url)

        # ── Sub-components ────────────────────────────────────────────────────
        from agent.executor import ActionExecutor, _TextResolver, _snap_to_element
        self._snap = _snap_to_element
        try:
            from components.observers.ui_observer import UIAutomationObserver
        except ImportError:
            from observers.ui_observer import UIAutomationObserver

        try:
            from components.agent.state_validator import StateValidator
            from components.recorder.correction_handler import CorrectionHandler
            from components.agent import navigation_protocol as _navproto
        except ImportError:
            from agent.state_validator import StateValidator
            from recorder.correction_handler import CorrectionHandler
            from agent import navigation_protocol as _navproto
        self._navproto = _navproto

        # ghost_cursor=False -- REVERTED 2026-08-11, direct request ("Same old
        # problem, fuck the cursor and the caret, revert to when the play was
        # working properly") after Play broke again during a live test with
        # this session's semantic-click tier layered on top of the overlay.
        # This was re-enabled once already (see git history / DEVELOPERS.md
        # for the four bugs found and fixed that time: broken UIA reads,
        # foreground theft, click-swallowing, and the root-cause wrong-HWND
        # bug) and DID pass a live checkpoint after that round of fixes. But
        # it has now cost two separate live sessions their working state, and
        # the user made clear tonight that the visual overlay itself isn't
        # worth the recurring risk -- turned off, not left as a "one more
        # patch" candidate. If cursor-free clicking is wanted again, the
        # already-shipped _try_semantic_click() (BM_CLICK on real Button-
        # class controls) achieves that without a visual overlay at all, so
        # nothing about "no real mouse movement" is lost by this revert.
        self._executor          = ActionExecutor(dry_run=dry_run, ghost_cursor=False)
        self._text_resolver     = _TextResolver()
        # Perception is an injectable adapter (the seam). Any observer whose
        # snapshot() conforms to observers/schema.py plugs in here — UIA now,
        # Excel/web later — with zero agent changes. Default = UIA.
        # NOTE: the observer config MUST match the recorder's, or inference states
        # are out-of-distribution vs training and the transformer predicts garbage
        # (recorder uses background_apps={notepad,.txt} + default limits).
        if observer is not None:
            self._observer = observer
        else:
            try:
                self._observer  = UIAutomationObserver(
                    background_apps={"notepad", ".txt"},
                )
            except TypeError:
                self._observer  = UIAutomationObserver(
                    max_elements_per_window=300, max_total_elements=700,
                )
        self._schema_checked = False   # validate the adapter once, on first observe
        self._validator         = StateValidator()
        self._correction        = CorrectionHandler()
        self._correction_watch_seconds = correction_watch_seconds
        self._record_num: int               = record_num
        # Discovered from the intake text on the first _refresh_record_cache
        # call (_total_records_in); 1 = single-record default, so a source
        # with no 'RECORD N OF M' structure behaves exactly as before this
        # was added — record_num never exceeds total, one submit always ends
        # the run. Multi-record advance added 2026-08-12: self._record_num
        # used to never be reassigned anywhere in this file (agent.py, see
        # _try_advance_tab's last-tab-exhausted branch), so a run could only
        # ever attempt record 1 no matter how many records the intake held.
        self._total_records: int             = 1
        # User-specified cap (--end_record), independent of _total_records --
        # _refresh_record_cache recomputes _total_records from the intake text
        # every refresh, which would silently overwrite any explicit range the
        # user asked for; this stays fixed for the whole run instead.
        self._end_record: Optional[int]      = end_record
        # Set True by _try_advance_tab when the LAST record has just been
        # submitted — checked at the top of run()'s main loop to end the run
        # cleanly instead of continuing to act on a form that has nothing
        # left to submit.
        self._task_finished: bool            = False
        # Set True by _try_advance_tab when it just advanced to a NEW record
        # (submitted the previous one, more remain). Consumed once at the top
        # of run()'s main loop to reset the per-record navigation-guard state
        # (stuck-position streaks, tab-scroll counts, etc.) that would
        # otherwise carry stale history from the record just finished into
        # the fresh one.
        self._record_just_advanced: bool     = False
        # 'attempted' state-feature (inference side): identities of fields acted on
        # this session, fed to the transformer so it stops re-targeting them (the
        # principled fix for empty-optional-field loops). Reset per record.
        self._attempted_keys: set            = set()
        # Subset of _attempted_keys where the attempt was a genuine keyboard
        # action carrying real text -- not a bare navigation click, not a
        # housekeeping "tab"/"ctrl+a" keystroke. Needed because
        # _attempted_keys deliberately includes undecided navigation clicks
        # too (see self._leave_blank_keys' own comment) -- a set that can't
        # tell "typed into" apart from "merely clicked past" is unsafe to
        # trust for editcontrol/comboboxcontrol's "already filled" checks,
        # since a stray click there is pure navigation, not a decision.
        self._typed_keys: set                = set()
        self._attempted_record_num: int      = record_num
        # Form-window lock: captured on the first observe (the window the user
        # clicks at "GO"), re-asserted foreground every step so a stray click can
        # never drift focus into another window (PowerShell/Notepad) and cascade
        # wrong observations/clicks/scrolls. Also the live viewport for scroll.
        self._locked_hwnd: Optional[int]     = None
        self._locked_title: str              = ""
        # Streak-breaker for _reassert_form_window(), added 2026-08-10 after a
        # real, severe incident: the user tried to click away from the form
        # (to reach the Electron Stop button) and could not, because this
        # method re-steals foreground EVERY step (~1-2s cadence) unconditionally
        # -- they had to hard-shutdown their laptop. If the SAME foreign window
        # keeps recurring across consecutive steps, that's the user deliberately
        # returning to it, not a stray drift -- back off instead of fighting them
        # forever. See _reassert_form_window() for the actual break condition.
        self._reassert_foreign_hwnd: Optional[int] = None
        self._reassert_foreign_streak: int         = 0
        self._visual_cache: Dict[str, str]  = visual_cache or {}   # Gemini pre-scan data
        self._visual_reader: Optional[Any]  = visual_reader        # VisualDataReader instance
        self._source_window: str            = source_window        # Notepad/source window title
        # Data source = where field VALUES come from. Injectable adapter (the seam):
        # NotepadDataSource now; Excel/web/email plug into the same slot. The agent
        # reads source-specific I/O (read_full_text/peek) only through self._source.
        self._source: Any = data_source if data_source is not None \
            else NotepadDataSource(source_window)
        self._history:        List[Dict[str, Any]] = []
        self._results:        List[Dict[str, Any]] = []
        self._heuristic_steps: int                 = 0
        self._checked_fields: set           = set()   # checkboxes already clicked this run
        self._filled_this_tab: set          = set()   # edit fields filled on current tab (prevents re-fill on cycling)
        self._nochange_click_pos: set       = set()   # (rx,ry) click positions that gave no_change this tab
        # (rx,ry) of every tab-strip button _try_advance_tab has deliberately
        # left as "exhausted" so far this record — found live 2026-08-07: the
        # transformer's OWN pointer clicked straight back onto the tab just
        # left, 12-13 times in one run (Vehicle <-> Coverage), each cycle
        # costing ~4 steps of pure tug-of-war. Originally bounded to one entry
        # (replaced, not accumulated) — found live 2026-08-07 that this only
        # blocked an IMMEDIATE reversal: a 2-hop-back click (Coverage ->
        # Policyholder, skipping past the single blacklisted Vehicle entry)
        # sailed straight through, producing a Policyholder/Vehicle/Coverage
        # cycle that never reached Drivers. Matches the same "finish forward,
        # don't come back" principle behind verify-at-fill. Now ACCUMULATES
        # every tab left this record — cleared only at the new-record
        # boundary alongside self._attempted_keys, so a fresh record's tabs
        # aren't pre-blocked by the previous record's history.
        self._advance_blacklist_pos: set    = set()
        # Fields explicitly confirmed leave-blank (LLM/lookup said "nothing
        # goes here" and we deliberately skipped it) — distinct from
        # self._attempted_keys, which also includes fields merely CLICKED
        # for navigation with no decision ever made about them (that
        # distinction is exactly what execution_payment_tab_oscillation_fix's
        # regression was about). Found live 2026-08-07, next run after that
        # regression fix: 'Last Payment Amount'/'Last Payment Date' (both
        # editcontrol, genuinely blank per the record) got correctly
        # Tab-skipped every time via the pre-merge leave-blank check, but
        # _t_is_type's checkbox-only attempted exclusion doesn't cover
        # editcontrol/comboboxcontrol -- so a legitimately-blank field's
        # `value` can NEVER become non-empty by design, meaning it re-enters
        # the fill branch and pays a full ~2s LLM round-trip EVERY time focus
        # lands there again, forever. Checked here (any field type) in
        # addition to the checkbox-only attempted check, so a genuine
        # decision (real value typed, checkbox toggled, OR explicit
        # leave-blank) always sticks — only an undecided navigation click
        # doesn't.
        self._leave_blank_keys: set         = set()
        self._current_tab_idx: int          = start_tab_idx  # tracks which tab we're on
        self._guidance: str = ""
        self._task_name: str = ""   # set via run(task_name=...)
        self._cached_record: Dict[str, str] = {}     # full parsed record from Notepad (bypasses 2000-char UIA cap)
        self._ocr_cache: Dict[str, Any] = {}        # instance-level OCR cache (clears per record)

        self._pure_transformer: bool = pure_transformer
        self._no_autohandlers:  bool = disable_auto_handlers
        # Ablation test flag, added 2026-08-14 ("I have to verify that if
        # we don't have the model we're using right now the performance
        # would drop"). See _predict() for the actual skip + fallback
        # mechanics; _ablation_transformer_calls_skipped counts how many
        # times the real model WOULD have been consulted, for the
        # after-the-fact comparison this exists to enable.
        self._disable_transformer: bool = disable_transformer
        self._ablation_transformer_calls_skipped: int = 0

        # ── Task plugin (optional — encapsulates task-specific logic) ───────────
        self._task_plugin: Optional[Any] = task_plugin
        if task_plugin is not None:
            # Wire the executor and observe function into the plugin
            if hasattr(task_plugin, "_executor") and task_plugin._executor is None:
                task_plugin._executor = self._executor
            elif not hasattr(task_plugin, "_executor"):
                task_plugin._executor = self._executor
            task_plugin._observe_fn = self._observe
            # Sync record-level state to plugin if it wasn't set at construction time
            if hasattr(task_plugin, "_record_num") and task_plugin._record_num == 1 and record_num != 1:
                task_plugin._record_num = record_num
            # Sync visual state
            if hasattr(task_plugin, "_visual_cache") and not task_plugin._visual_cache and visual_cache:
                task_plugin._visual_cache = dict(visual_cache)
            if hasattr(task_plugin, "_visual_reader") and task_plugin._visual_reader is None and visual_reader:
                task_plugin._visual_reader = visual_reader
            if hasattr(task_plugin, "_source_window") and not task_plugin._source_window and source_window:
                task_plugin._source_window = source_window
            # Plan-then-replay support (components/agent/field_planner.py) —
            # wire the real UIA-focus/settle-wait/viewport-bottom methods so
            # the plugin's own local fallbacks are only used when it's run
            # standalone (e.g. tests) without a full LLMAgent behind it.
            if hasattr(task_plugin, "_focus_fn") and task_plugin._focus_fn is None:
                task_plugin._focus_fn = self._focus_element_via_uia
            if hasattr(task_plugin, "_settle_wait_fn") and task_plugin._settle_wait_fn is None:
                task_plugin._settle_wait_fn = self._adaptive_settle_wait
            if hasattr(task_plugin, "_viewport_bottom_fn") and task_plugin._viewport_bottom_fn is None:
                task_plugin._viewport_bottom_fn = self._form_viewport_bottom

        # EXPERIMENTAL — OCR overlay via VisionObserver
        # Disabled by default; enable with use_ocr=True in LLMAgent(...)
        self._use_ocr: bool = use_ocr
        self._vision_observer: Optional[Any] = None
        if use_ocr:
            try:
                try:
                    from components.observers.vlm.vision_observer.vision_observer import VisionObserver
                except ImportError:
                    from observers.vlm.vision_observer.vision_observer import VisionObserver
                self._vision_observer = VisionObserver()
                logger.info("OCR overlay enabled (VisionObserver loaded)")
            except Exception as exc:
                logger.warning("OCR overlay requested but VisionObserver unavailable: %s", exc)

        # ── Load persistent task spec → inject into system prompt ─────────────
        self._system_prompt: str = _SYSTEM_PROMPT
        _mp = self.model_path
        if not os.path.isabs(_mp):
            _mp = os.path.join(os.getcwd(), _mp)
        _spec_path = os.path.normpath(os.path.join(os.path.dirname(_mp), "ruleset.md"))
        if os.path.exists(_spec_path):
            try:
                _spec = open(_spec_path, encoding="utf-8").read().strip()
                if _spec:
                    self._system_prompt = (
                        _SYSTEM_PROMPT
                        + "\n\n## TASK SPECIFICATION (learned from demonstrations)\n"
                        + _spec
                    )
                    logger.info("LLMAgent: loaded task spec from %s", _spec_path)
            except Exception as _se:
                logger.warning("LLMAgent: failed to load task spec: %s", _se)

    # ── provider initialisation ───────────────────────────────────────────────

    def _init_provider(self, api_key: str, lmstudio_url: str) -> None:
        p = self.provider

        if p == "anthropic":
            if not _ANTHROPIC_OK:
                logger.warning("anthropic package not installed — falling back to transformer-only.")
                return
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                logger.warning("No Anthropic API key — falling back to transformer-only.")
                return
            self._llm_client = _anthropic.Anthropic(api_key=key)
            logger.info("LLMAgent: provider=anthropic  model=%s", self._llm_model)

        elif p == "groq":
            if not _GROQ_OK:
                logger.warning("groq package not installed — falling back to transformer-only.")
                return
            key = api_key or os.environ.get("GROQ_API_KEY", "")
            if not key:
                logger.warning("No Groq API key — falling back to transformer-only.")
                return
            self._llm_client = _Groq(api_key=key)
            logger.info("LLMAgent: provider=groq  model=%s", self._llm_model)

        elif p == "gemini":
            if not _GEMINI_OK:
                logger.warning("google-generativeai not installed — falling back to transformer-only.")
                return
            key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not key:
                logger.warning("No Gemini API key — falling back to transformer-only.")
                return
            self._llm_client = _genai.Client(api_key=key)
            logger.info("LLMAgent: provider=gemini  model=%s", self._llm_model)

        elif p == "lmstudio":
            if not _OPENAI_OK:
                logger.warning("openai package not installed — falling back to transformer-only.")
                return
            self._llm_client = _OpenAI(base_url=lmstudio_url, api_key="lm-studio")
            logger.info("LLMAgent: provider=lmstudio  url=%s  model=%s", lmstudio_url, self._llm_model)

        elif p == "none":
            logger.info("LLMAgent: provider=none — transformer-only mode.")

        else:
            logger.warning("Unknown provider %r — transformer-only.", p)

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        max_steps:  Optional[int] = None,
        task_name:  str           = "",
    ) -> List[Dict[str, Any]]:
        n = max_steps if max_steps is not None else self.max_steps
        self._task_name = task_name
        logger.info(
            "LLMAgent.run() — goal=%r  provider=%s  max_steps=%d  dry_run=%s",
            self.goal, self.provider, n, self.dry_run,
        )

        # ── timing: run-level ────────────────────────────────────────────────
        import datetime as _dt
        _run_start_ts             = time.time()
        self._run_start_ts        = _run_start_ts
        self._run_start_iso       = _dt.datetime.now().isoformat()
        self._run_duration_sec    = 0.0
        self._time_to_first_action_sec: Optional[float] = None
        self._manual_interventions = 0

        # Catch crashes in background threads (OCR, recorder, etc.) so they're visible
        import threading as _threading, traceback as _tb_mod
        def _thread_exc_hook(args):
            print(f"\n=== BACKGROUND THREAD CRASH ({args.thread.name}) ===\n"
                  + "".join(_tb_mod.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
                  flush=True)
            logger.error("Background thread %s crashed:\n%s", args.thread.name,
                         "".join(_tb_mod.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
        _threading.excepthook = _thread_exc_hook

        _stuck_pos:   Optional[tuple] = None
        _stuck_count: int             = 0
        _STUCK_LIMIT: int             = 3
        _llm_click_pos:   Optional[tuple] = None
        _llm_click_count: int             = 0
        _LLM_CLICK_LIMIT: int             = 1   # force type after this many repeated clicks
        _no_change_streak: int            = 0
        _NO_CHANGE_LIMIT:  int            = 4   # advance tab after this many consecutive no_change
        _last_focused_id:  Optional[str]  = None  # track focus changes to reset streak
        _open_combobox_label: str         = ""  # label of combobox whose dropdown is currently open
        _tab_just_switched: bool          = True   # True on first step to focus first empty field
        _last_auto_step:   int            = -1  # step_idx of last step where an auto-handler fired
        _DROUGHT_LIMIT:    int            = 10  # scroll form after this many non-fill steps in a row
        _tab_scroll_count: int            = 0   # scrolls performed on current tab
        _MAX_TAB_SCROLLS:  int            = 6   # max drought-scrolls per tab before giving up
        _steps_on_tab:     int            = 0   # steps spent on current tab — forces advance if too many
        _TAB_STEP_LIMIT:   int            = 40  # force tab advance after this many steps on same tab
        _pane_escape_last_field: str      = ""  # last field pane-escape tried to click
        _pane_escape_streak:     int      = 0   # consecutive tries on the same field without escaping
        _confirmed_blank_fields: set      = set()  # fields where peek found no value → treat as blank
        # Consecutive low-confidence/invalid pointer Tab-fallbacks (OPT2 navigate
        # branch). Found live 2026-08-07, directly from the user watching a run:
        # "Too much wasted steps. Whenever there are no longer any targets on the
        # screen... I need you to activate Navigation Protocol so that there will
        # be." A single blind Tab is fine (natural model uncertainty self-corrects
        # most of the time); several in a row with zero progress means the
        # transformer's own pointer is stuck guessing. Reset on any confident click.
        _lowconf_fallback_streak: int    = 0
        # Was 3 (one blind Tab tolerated per low-confidence guess, up to 3
        # times, before redirecting) -- changed to 1 on direct, repeated user
        # instruction 2026-08-08 ("Do not use Tab to navigate"): the very
        # first low-confidence/invalid pointer now goes straight to
        # Navigation Protocol's own known target via click, no Tab-and-hope
        # at all. Tab is still used elsewhere for its OWN legitimate job
        # (committing a just-selected combobox value, moving off a just-
        # filled field) -- this only removes it as a NAVIGATION guess.
        _LOWCONF_FALLBACK_LIMIT: int     = 1
        # Same idea as _lowconf_fallback_streak, but for the reclick guard
        # specifically. Found 2026-08-08, live, direct user report ("still
        # not finding the right view"): the transformer's own pointer aimed
        # at the exact same already-filled field's screen position on 3
        # CONSECUTIVE steps in a row (caught safely each time -- no wasted
        # click, no data corruption -- just a blind Tab and hope) before
        # finally moving on, and this repeated dozens of times across one
        # run. A blind Tab gambles on OS focus-traversal order landing
        # somewhere useful; when the SAME stuck position keeps recurring,
        # that gamble keeps failing. Reset on any real fill/click.
        _reclick_streak: int             = 0
        # Was 2 (one blind Tab tolerated on the first drift onto an already-
        # handled field, before redirecting on the second) -- changed to 1 on
        # direct, repeated user instruction 2026-08-08 ("Do not use Tab to
        # navigate"): the very first drift onto an already-filled/attempted
        # field now redirects straight to a known target via click.
        _RECLICK_REDIRECT_LIMIT: int     = 1
        _redirect_stall_count:   int     = 0
        _REDIRECT_STALL_LIMIT:   int     = 2
        # Found 2026-08-09, live, direct user report ("it didn't want to
        # navigate"): a combobox whose dropdown never renders (confirmed:
        # 8 real polling tries, 3.2s, every single attempt) had NO streak
        # tracking at all -- unlike the sibling "combobox is blank, nothing
        # to fill" case a few lines above it in the same function (which
        # already blacklists via _action_history/_nochange_click_pos), this
        # branch just logged a warning, pressed Escape, and looped forever.
        # 'Payment Frequency' hit this exact untracked path 121 times
        # across ~4 minutes in one run before the user gave up and
        # interrupted it. Per-label counter (not per-position -- the
        # combobox's own click position doesn't drift the way the
        # transformer's pointer does elsewhere) tracking consecutive
        # dropdown-render failures for the SAME field.
        _combobox_dropdown_fail_counts: Dict[str, int] = {}
        _COMBOBOX_DROPDOWN_FAIL_LIMIT:  int             = 2
        _heuristic_steps:        int      = 0      # steps decided by auto-handlers (not LLM/transformer)
        _manual_interventions:   int      = 0      # DAgger corrections the human had to make — cognitive-load proxy
        _llm_unavailable_streak: int      = 0      # consecutive "llm unavailable" — infra failure, NOT "value is blank"
        _LLM_UNAVAILABLE_LIMIT:  int      = 3       # halt rather than silently Tab-skip the whole form

        _record_cache_loaded     = False
        _tc_advance_verified     = False   # True once the full top→bottom scan passes before advance/submit
        _prev_elem_count:    int = 0       # element count from previous step — spike = unexpected dialog

        # Repeat-action detector — fingerprint last N actions; identical streak → Tab out
        from collections import deque as _deque
        _action_history: _deque = _deque(maxlen=6)
        _REPEAT_LIMIT: int = 3

        for step_idx in range(n):
          try:
            # Multi-record advance (2026-08-12): _try_advance_tab sets these
            # two flags after clicking Submit — checked here, once, at the
            # top of every iteration, rather than at each of its ~10 call
            # sites, so none of their existing logic needs to change.
            if self._task_finished:
                logger.info("All records complete (%d/%d) — ending run.",
                            self._record_num, self._total_records)
                break
            if self._record_just_advanced:
                logger.info("New record (%d/%d) — resetting per-record navigation state.",
                            self._record_num, self._total_records)
                self._record_just_advanced = False
                _stuck_pos = None; _stuck_count = 0
                _llm_click_pos = None; _llm_click_count = 0
                _no_change_streak = 0
                _last_focused_id = None
                _open_combobox_label = ""
                _tab_just_switched = True
                _last_auto_step = -1
                _tab_scroll_count = 0
                _steps_on_tab = 0
                _pane_escape_last_field = ""
                _pane_escape_streak = 0
                _confirmed_blank_fields = set()
                _lowconf_fallback_streak = 0
                _reclick_streak = 0
                _redirect_stall_count = 0
            _step_t0 = time.time()
            # Found 2026-08-09, live, direct user report ("We're in a loop
            # and we're missing certain steps"): the generic repeat-action
            # guard further down fingerprints whatever `prediction` ends up
            # being right before execution -- but several checkbox-check
            # code paths execute the REAL action (BM_SETCHECK on a genuinely
            # different checkbox each time) and then reassign `prediction`
            # to a boilerplate "tab to move on" just to advance focus
            # afterward. Three checks of three DIFFERENT checkboxes in a row
            # all end up with the identical trailing "keyboard: tab"
            # prediction, so the guard saw "same action 3x" and force-Tabbed
            # AGAIN through a path that skips the normal ExecutionResult/
            # Validator logging -- confirmed in logs/latest.log around
            # 02:08:19 (step 111): 'New Car Replacement' was checked via
            # BM_SETCHECK, then the repeat guard immediately fired anyway.
            # Real progress WAS happening every step -- the guard just
            # couldn't see it. This flag lets any code path that performs a
            # genuine state-changing action this step say so, so the guard
            # (checked later this same step) knows not to second-guess it.
            _real_progress_this_step = False
            # Found 2026-08-09, live, direct user request ("stop wasting
            # steps"): a plain "navigate to the next field" click and the
            # fill that follows it were always two separate step_idx
            # iterations -- the click lands this step, the top-of-loop
            # fillable-check only notices the newly-focused field on the
            # NEXT step. Confirmed against real logs tonight: nearly every
            # field took exactly this click-step/fill-step pair. This flag
            # marks "the prediction this step is a plain navigate click, no
            # redirect/combobox/checkbox logic involved" so the collapse
            # check after execution (below) only fires for the one
            # unambiguous case it's designed for.
            _is_plain_navigate_click = False
            # 1. Observe — but first re-assert the locked form as foreground so a
            # stray click last step can't leave us observing/acting on a drifted
            # window. Lock is captured on the first observe (form is in front at GO).
            self._reassert_form_window()
            state      = self._observe()
            _t_observed = time.time()
            observe_time_sec = _t_observed - _step_t0
            if self._locked_hwnd is None:
                self._lock_form_window(state)

            # Verify the assumed starting tab against reality, once, on the very
            # first step. self._current_tab_idx starts out as whatever
            # start_tab_idx claimed (0 normally, or --start_tab N in drill mode)
            # — that value is NEVER independently checked, it's just trusted.
            # Found live 2026-08-08, discussed with the user while investigating
            # execution_advance_blacklist's tab-tracking: drill mode depends on
            # the human clicking the right tab BEFORE launching — a typo'd
            # --start_tab, a missed click, or a click that hadn't registered
            # yet would leave the tracker wrong from step 1, and everything
            # downstream that trusts it (the backward-tab-click guard included)
            # would reason from a false premise for the entire run. Unlike
            # _try_advance_tab's OWN mid-run mismatch handling (which trusts the
            # tracker over detection, since mid-run noise from a JUST-completed
            # transition is the normal case) — at this single, fresh, settled
            # first observation there's no transition noise to distrust
            # detection for, so here detection wins and the tracker is corrected
            # to match it, loudly.
            if step_idx == 0:
                _boot_tabs = [e for e in state.get("elements", [])
                              if e.get("type") in ("tabitem", "tabitemcontrol")
                              and e.get("window_role") != "background"
                              and e.get("bbox")]
                _boot_detected = _detect_active_tab_idx_raw(_boot_tabs, state.get("elements", []))
                if _boot_detected is not None and _boot_detected != self._current_tab_idx:
                    logger.warning(
                        "Startup tab check: assumed idx=%d (start_tab_idx) but the form is "
                        "actually showing idx=%d %r — correcting the tracker to match reality.",
                        self._current_tab_idx, _boot_detected,
                        (_boot_tabs[_boot_detected].get("text")
                         or _boot_tabs[_boot_detected].get("label") or "?") if _boot_tabs else "?")
                    self._current_tab_idx = _boot_detected
                elif _boot_detected is not None:
                    logger.info("Startup tab check: confirmed idx=%d matches reality.", _boot_detected)

            llm_action: Dict[str, Any] = {}
            _steps_on_tab += 1
            _cur_elem_count = len(state.get("elements", []))
            logger.info("── Step %d/%d  (%d elements) ──", step_idx + 1, n, _cur_elem_count)

            # Dialog guard: large element-count spike → unexpected dialog opened → Escape
            if _prev_elem_count > 0 and _cur_elem_count - _prev_elem_count > 100:
                logger.warning("Element count spiked %d→%d — unexpected dialog detected, pressing Escape.",
                               _prev_elem_count, _cur_elem_count)
                self._executor.execute({"action_type": "hotkey", "keys": ["escape"]})
                self._adaptive_settle_wait(self.step_delay)
                _prev_elem_count = 0
                continue
            _prev_elem_count = _cur_elem_count

            # Load record cache on first step
            if not _record_cache_loaded:
                self._refresh_record_cache(state)
                _record_cache_loaded = True

            # ── Task plugin delegation ────────────────────────────────────────
            # When a TaskPlugin is registered it takes over all task-specific
            # auto-handler blocks (tab switching, auto-fill, auto-skip, etc.).
            # The plugin returns (handled, should_continue):
            #   (True, True)  → plugin handled this step; loop continues
            #   (True, False) → plugin handled and signals task done; loop breaks
            #   (False, False) → not handled; fall through to transformer/LLM below
            if self._task_plugin is not None:
                _p_handled, _p_continue = self._task_plugin.handle_step(state, step_idx)
                if _p_handled:
                    # Sync no_change_streak from plugin back to local tracking
                    if hasattr(self._task_plugin, "_no_change_streak"):
                        _no_change_streak = self._task_plugin._no_change_streak
                    if not _p_continue:
                        break
                    continue
                # Not handled — fall through to transformer/LLM below

            # When a plugin is active OR pure-transformer mode, skip all legacy
            # form-specific auto-handlers and fall through to transformer/LLM only.
            _plugin_active = (self._task_plugin is not None) or self._pure_transformer or self._no_autohandlers

            # In pure_transformer mode: scroll to top + click first field on tab switch
            # (same as the non-plugin handler below, but without auto-fill logic)
            if self._pure_transformer and _tab_just_switched:
                _tab_just_switched  = False
                _steps_on_tab       = 0
                _tab_scroll_count   = 0
                _no_change_streak   = 0
                _last_auto_step     = step_idx
                if self._visual_reader:
                    self._scan_tab_visual(state)
                self._scroll_form_to_top(state)
                time.sleep(0.6)
                state = self._observe()
                # Click the topmost-left interactive field (by bbox Y then X).
                # Excludes tab-strip elements (tabitemcontrol) which sit at the top
                # of the window but are not form fields.
                _candidates = sorted(
                    [e for e in state.get("elements", [])
                     if e.get("type") in ("editcontrol", "comboboxcontrol", "checkboxcontrol")
                     and e.get("window_role") != "background"
                     and e.get("bbox") and e.get("enabled", True)],
                    key=lambda e: (e["bbox"][1], e["bbox"][0])
                )
                if _candidates:
                    _fe = _candidates[0]
                    _fx1, _fy1, _fx2, _fy2 = _fe["bbox"]
                    self._executor.execute({"action_type": "click",
                                            "click_position": [(_fx1+_fx2)/2, (_fy1+_fy2)/2]})
                    time.sleep(self.step_delay * 0.5)
                continue

            # 1a. After a tab switch: scroll to top, re-observe, then click first empty field
            if not _plugin_active and _tab_just_switched:
                _tab_just_switched    = False
                _steps_on_tab         = 0
                _tab_scroll_count     = 0
                _no_change_streak     = 0
                _last_auto_step       = step_idx  # treat tab switch as an auto step
                _pane_escape_last_field = ""
                _pane_escape_streak     = 0
                self._filled_this_tab.clear()     # new tab — reset filled-field tracking
                self._nochange_click_pos.clear()  # new tab — reset failed-click blacklist
                _confirmed_blank_fields.clear()   # new tab — reset peek-confirmed-blank set
                _tc_advance_verified = False      # new tab — reset full-scan gate
                # Visual scan: bring Notepad to foreground, scroll to this tab's section,
                # capture screenshots with Groq vision to read all field values for this tab.
                self._scan_tab_visual(state)
                self._scroll_form_to_top(state)
                time.sleep(0.6)
                state = self._observe()   # get updated positions after scroll
                # Click whichever field is topmost on-screen after scrolling up —
                # position-based, so it naturally lands on the first visible field.
                self._focus_first_empty_field(state)
                time.sleep(self.step_delay)
                continue   # re-observe so auto-handlers run first before LLM gets a chance

            # 1b. Stuck guard: if no_change repeats OR too many steps on same tab, advance.
            # DISABLED when disable_auto_handlers — we want to see the pure
            # transformer with no rescue (honest navigation test).
            _stuck = ((not self._no_autohandlers)
                      and (_no_change_streak >= _NO_CHANGE_LIMIT
                           or (not _plugin_active and _steps_on_tab >= _TAB_STEP_LIMIT)))
            if _stuck:
                if _steps_on_tab >= _TAB_STEP_LIMIT:
                    logger.info("Stuck guard: %d steps on tab — forcing advance.", _steps_on_tab)
                elif _no_change_streak >= _NO_CHANGE_LIMIT:
                    # Before escalating to a full tab-advance, Tab past the stuck field.
                    # This lets multi-section tabs (Driver 1/2/3) continue past a single
                    # problematic checkbox instead of jumping to the next tab entirely.
                    _foc_id = state.get("focused_element_id")
                    _foc_el = next((e for e in state.get("elements", [])
                                    if e.get("element_id") == _foc_id), None)
                    _foc_nm_bare = ((_foc_el.get("label") or _foc_el.get("text") or "?")
                                    if _foc_el else "?")
                    _foc_nm_sec  = self._detect_section(state, _foc_el) if _foc_el else ""
                    _foc_nm      = f"{_foc_nm_sec} {_foc_nm_bare}" if _foc_nm_sec else _foc_nm_bare
                    logger.info("Stuck guard: no_change x%d on %r — Tab past field.",
                                _no_change_streak, _foc_nm)
                    if _foc_nm_bare and _foc_nm_bare != "?":
                        self._filled_this_tab.add(_foc_nm)
                    self._executor.execute({"action_type": "keyboard",
                                            "key_count": 1, "keystrokes": ["tab"]})
                    _no_change_streak = 0
                    _last_auto_step   = step_idx
                    time.sleep(self.step_delay * 0.5)
                    continue
                if self._try_advance_tab(state):
                    _no_change_streak  = 0
                    _tab_just_switched = True
                    _tab_scroll_count  = 0
                    _last_auto_step    = step_idx
                    self._refresh_record_cache(state)
                    time.sleep(self.step_delay)
                    continue
                else:
                    logger.warning("Stuck guard: no next tab found — resetting streak.")
                    _no_change_streak = 0
                    _steps_on_tab     = 0

            # 1c. Universal tab-complete: when visible fields are all handled, do a
            # full top→bottom scan to confirm nothing is hiding above or below the
            # current viewport before advancing to the next tab or submitting.
            # A helper that checks pending edit/combobox fields in a given state.
            def _tc_key(e, _st=state):
                _fn  = (e.get("label") or e.get("text") or "").strip()
                _sec = self._detect_section(_st, e)
                return f"{_sec} {_fn}" if _sec else _fn
            def _tc_has_pending(_st):
                _fl = {s.lower() for s in self._filled_this_tab}
                _cb_lower = {s.lower() for s in _confirmed_blank_fields}
                for _e in _st.get("elements", []):
                    if (_e.get("window_role") == "background"
                            or _e.get("type") not in ("editcontrol", "comboboxcontrol")
                            or not _e.get("enabled", True)):
                        continue
                    # Inactive tab panels have negative screen coordinates in wxPython.
                    # Skip off-screen elements — they belong to other tabs, not this one.
                    if _e.get("bbox") and _e["bbox"][1] < 0:
                        continue
                    _fn = (_e.get("label") or _e.get("text") or "").strip()
                    _sec = self._detect_section(_st, _e)
                    _fk  = (f"{_sec} {_fn}" if _sec else _fn).lower()
                    if _fk in _fl or _fn.lower() in _fl:
                        continue
                    if _fk in _cb_lower or _fn.lower() in _cb_lower:
                        continue  # peeked and confirmed blank — skip
                    # Skip fields in a section that has no data at all (e.g. Driver 3 when
                    # record has only 2 drivers).  Without this check _lookup_field falls
                    # back to the bare key and returns the wrong value (e.g. Policyholder's
                    # First Name instead of Driver 3 First Name), making the section appear
                    # pending even though it doesn't exist.
                    if _sec and self._cached_record:
                        _sec_lower = _sec.lower()
                        _fn_key_tc = _sec_lower + " first name"
                        _fn_val_tc = next((rv for rk, rv in self._cached_record.items()
                                           if rk.lower() == _fn_key_tc), "")
                        if not (_fn_val_tc and not _is_leave_blank_value(_fn_val_tc)):
                            continue  # section has no real person — not pending
                    _known = self._lookup_field(_fn, section=_sec)
                    _val   = (_e.get("value") or "").strip()
                    if _known:
                        if _is_leave_blank_value(_known):
                            continue  # field should be blank — not pending
                        # Pending only if empty OR value doesn't match expected
                        if not _val or _val.lower() != _known.lower():
                            return True
                        # Current value already matches expected — not pending
                        continue
                    # Empty editcontrol/combobox with unknown value — may still need filling via peek
                    if (not _val and _e.get("type") in ("editcontrol", "comboboxcontrol")):
                        return True
                # Also check unchecked checkboxes that should be checked (or are unknown)
                for _chk in _st.get("elements", []):
                    if (_chk.get("window_role") == "background"
                            or _chk.get("type") != "checkboxcontrol"
                            or not _chk.get("enabled", True)):
                        continue
                    if _chk.get("bbox") and _chk["bbox"][1] < 0:
                        continue
                    _nm = (_chk.get("label") or _chk.get("text") or "").strip()
                    if _nm in self._checked_fields:
                        continue
                    if _nm.lower() in _cb_lower:
                        continue  # confirmed blank after peek
                    _exp = self._lookup_field(_nm)
                    if _exp and _exp.lower().strip().startswith("yes"):
                        return True
                    # Unknown checkbox: no data → not blocking tab advance
                return False

            if not _plugin_active and self._filled_this_tab and not _tc_has_pending(state):
                # Visible fields look done — do a full top→bottom scan before acting
                if not _tc_advance_verified:
                    _tcav_state = state
                    _tcav_found = False
                    # 1. Scroll to top and check for missed fields above viewport
                    self._scroll_form_to_top(_tcav_state)
                    time.sleep(0.4)
                    _tcav_state = self._observe()
                    for _tcav_i in range(6):   # up to 6 scroll-down passes
                        if _tc_has_pending(_tcav_state):
                            _pend_names = [
                                (e.get("label") or e.get("text") or "?")
                                for e in _tcav_state.get("elements", [])
                                if e.get("type") in ("editcontrol", "comboboxcontrol")
                                and e.get("window_role") != "background"
                                and e.get("enabled", True)
                                and self._lookup_field(
                                    (e.get("label") or e.get("text") or "").strip(),
                                    section=self._detect_section(_tcav_state, e))
                                and (e.get("label") or e.get("text") or "").strip().lower()
                                    not in {s.lower() for s in self._filled_this_tab}
                            ][:4]
                            logger.info("Tab-complete scan (pass %d): pending fields found: %s",
                                        _tcav_i, _pend_names)
                            state        = _tcav_state
                            _tcav_found  = True
                            _tc_advance_verified = False
                            self._focus_first_empty_field(_tcav_state)
                            time.sleep(self.step_delay)
                            break
                        self._scroll_form_down(_tcav_state)
                        time.sleep(0.25)
                        _tcav_state = self._observe()
                    if _tcav_found:
                        continue   # back to main loop to fill found fields
                    _tc_advance_verified = True   # full scan passed — safe to advance/submit

                # Full scan verified — advance tab or submit
                if self._try_advance_tab(state):
                    logger.info("Tab complete: all fields done — advancing to next tab.")
                    _no_change_streak    = 0
                    _tab_just_switched   = True
                    _tab_scroll_count    = 0
                    _last_auto_step      = step_idx
                    _tc_advance_verified = False
                    self._filled_this_tab.clear()
                    _confirmed_blank_fields.clear()
                    self._refresh_record_cache(state)
                    time.sleep(self.step_delay)
                    continue
                else:
                    # Last tab — click Submit via UIA (button may be scrolled off-screen)
                    _submitted = False
                    try:
                        import uiautomation as _uia_sub
                        for _btn_name in ("Submit  New", "Submit & New", "Submit"):
                            _btn = _uia_sub.ButtonControl(Name=_btn_name, searchDepth=8)
                            if _btn.Exists(maxSearchSeconds=0.5):
                                logger.info("Tab complete: last tab — UIA click Submit %r", _btn_name)
                                _btn.Click()
                                _submitted = True
                                break
                    except Exception as _sub_exc:
                        logger.warning("UIA Submit click failed: %s", _sub_exc)
                    if not _submitted:
                        # Fallback: coordinate click from element list
                        _submit_el = next(
                            (e for e in state.get("elements", [])
                             if e.get("type") == "buttoncontrol"
                             and "submit" in (e.get("label") or e.get("text") or "").lower()
                             and e.get("window_role") != "background"
                             and e.get("bbox")),
                            None
                        )
                        if _submit_el:
                            x1, y1, x2, y2 = _submit_el["bbox"]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            logger.info("Tab complete: last tab — coord click Submit @ (%.0f, %.0f)", cx, cy)
                            self._executor.execute({"action_type": "click", "click_position": [cx, cy]})
                            _submitted = True
                    if _submitted:
                        break

            # 1b. Pane-focus escape: if focus is on a section header / container (panecontrol),
            # Tab will do nothing. Jump directly to the first empty edit field instead.
            _focused_id  = state.get("focused_element_id")
            _focused_el  = next((e for e in state.get("elements", []) if e.get("element_id") == _focused_id), None)
            if not _plugin_active and _focused_el and _focused_el.get("type") == "panecontrol":
                _pane_label = (_focused_el.get("label") or _focused_el.get("text") or "?")[:40]
                _pane_y     = _focused_el["bbox"][1] if _focused_el.get("bbox") else 0
                # If this pane is a section_ with no data in the record (e.g. Driver 3 when
                # record has only 2 drivers), don't escape into it — advance tab instead.
                # Read the section directly from the pane's own label rather than calling
                # _detect_section (which finds panes ABOVE the element, not the element itself).
                import re as _pre
                _pane_raw = (_focused_el.get("label") or _focused_el.get("text") or "").lower()
                _pm = _pre.match(r"section_(driver|vehicle)_(\d+)$", _pane_raw)
                _pane_sec = f"{_pm.group(1).title()} {_pm.group(2)}" if _pm else ""
                if _pane_sec and self._cached_record:
                    _ps_lower = _pane_sec.lower()
                    # A section exists only if its First Name field has a real value.
                    # Boolean fields like "SR-22 Required: No" are present for every
                    # section and must not be used as presence indicators.
                    _fn_key   = _ps_lower + " first name"
                    _fn_val   = next((rv for rk, rv in self._cached_record.items()
                                      if rk.lower() == _fn_key), "")
                    _sec_has_real = bool(_fn_val and not _is_leave_blank_value(_fn_val))
                    if not _sec_has_real:
                        logger.info("Pane-escape: section %r has no real data — advancing tab.", _pane_sec)
                        if self._try_advance_tab(state):
                            _no_change_streak  = 0
                            _tab_just_switched = True
                            _tab_scroll_count  = 0
                            _last_auto_step    = step_idx
                            self._filled_this_tab.clear()
                            _confirmed_blank_fields.clear()
                            self._refresh_record_cache(state)
                            time.sleep(self.step_delay)
                        _heuristic_steps += 1; continue
                logger.info("Focus on pane %r — clicking first empty field to escape.", _pane_label)
                # Peek at which field we'd click before actually clicking it.
                # If we've clicked the same field 3+ times with no escape (loop!), scroll down first.
                _pane_filled_lower = {s.lower() for s in self._filled_this_tab}
                def _pane_not_handled(_pe):
                    _fn  = (_pe.get("label") or _pe.get("text") or "").strip()
                    _sec = self._detect_section(state, _pe)
                    _fk  = f"{_sec} {_fn}".lower() if _sec else _fn.lower()
                    return _fk not in _pane_filled_lower and _fn.lower() not in _pane_filled_lower
                _pane_candidates = sorted(
                    [_pe for _pe in state.get("elements", [])
                     if (_pe.get("window_role") != "background"
                         and _pe.get("type") == "editcontrol"
                         and _pe.get("bbox") and _pe.get("enabled", True)
                         and _pe["bbox"][1] >= max(100, _pane_y)
                         and _pane_not_handled(_pe))],
                    key=lambda e: (e["bbox"][1], e["bbox"][0])
                )
                _pane_next_el = _pane_candidates[0] if _pane_candidates else None
                _pane_next_name = (_pane_next_el.get("label") or _pane_next_el.get("text") or "").strip() if _pane_next_el else ""
                if _pane_next_name and _pane_next_name == _pane_escape_last_field:
                    _pane_escape_streak += 1
                else:
                    _pane_escape_streak = 1
                    _pane_escape_last_field = _pane_next_name
                if _pane_escape_streak >= 3:
                    logger.warning("Pane-escape: stuck on %r for %d tries — scrolling down to reveal it.",
                                   _pane_next_name, _pane_escape_streak)
                    self._scroll_form_down(state)
                    time.sleep(self.step_delay * 0.5)
                    _pane_escape_streak = 0  # reset after scroll so we try click once fresh
                    _heuristic_steps += 1; continue
                # Press Tab instead of UIA SetFocus — the form's own tab order
                # reliably lands on the first field inside the section pane.
                # UIA SetFocus can bounce back to the previous section when the
                # target field hasn't been scrolled into the form's active region.
                self._executor.execute({"action_type": "keyboard",
                                        "key_count": 1, "keystrokes": ["tab"]})
                time.sleep(self.step_delay * 0.5)
                _heuristic_steps += 1; continue
                # No unhandled field found in this pane — fall through so the
                # universal tab-complete check (above) or LLM handles next steps.

            # 1c. Button-focus escape: if focus landed on a button but there are still
            # pending fields, seek the first unfilled field directly instead of waiting
            # for the LLM to figure out what to do (avoids 10-step wait-loop).
            if (not _plugin_active and _focused_el
                    and _focused_el.get("type") in {
                        "button", "buttoncontrol", "splitbutton", "splitbuttoncontrol",
                        "hyperlinkcontrol", "link",
                        "tabitem", "tabitemcontrol"}
                    and _tc_has_pending(state)):
                _btn_lbl = (_focused_el.get("label") or _focused_el.get("text") or "").strip()
                logger.info("Button-focus escape: focus on non-input %r but fields still pending "
                            "— seeking first unfilled field.", _btn_lbl)
                if self._focus_first_empty_field(state):
                    _no_change_streak = 0
                    time.sleep(self.step_delay * 0.5)
                    _heuristic_steps += 1; continue
                # No empty edit/combobox visible — try unchecked checkboxes directly
                _chk_pending = [
                    e for e in state.get("elements", [])
                    if e.get("type") == "checkboxcontrol"
                    and e.get("window_role") != "background"
                    and e.get("enabled", True)
                    and (e.get("label") or e.get("text") or "").strip() not in self._checked_fields
                ]
                _chk_acted = False
                for _chk_e in _chk_pending:
                    _chk_nm = (_chk_e.get("label") or _chk_e.get("text") or "").strip()
                    _exp_v  = self._lookup_field(_chk_nm)
                    if _exp_v and _exp_v.lower().strip().startswith("yes"):
                        _bbox = _chk_e.get("bbox")
                        if _bbox:
                            try:
                                import win32gui as _wgb; import win32api as _wab
                                _cx2 = (_bbox[0] + _bbox[2]) / 2
                                _cy2 = (_bbox[1] + _bbox[3]) / 2
                                _hw2 = _wgb.WindowFromPoint((int(_cx2), int(_cy2)))
                                if _hw2:
                                    _wab.SendMessage(_hw2, 0x00F1, 1, 0)
                                    logger.info("Button-escape: checked checkbox %r via BM_SETCHECK.", _chk_nm)
                                    self._checked_fields.add(_chk_nm)
                                    self._filled_this_tab.add(_chk_nm)
                                    _chk_acted = True
                            except Exception as _bce:
                                logger.warning("Button-escape checkbox BM_SETCHECK failed: %s", _bce)
                if _chk_acted:
                    time.sleep(self.step_delay * 0.5)
                    _heuristic_steps += 1; continue
                # Scroll down to reveal hidden fields
                if self._scroll_form_down(state):
                    _tab_scroll_count += 1
                    _last_auto_step    = step_idx
                    time.sleep(self.step_delay * 0.5)
                    _heuristic_steps += 1; continue
                # All seeks exhausted — force Tab past the button, never give it to LLM
                logger.info("Button-escape: all seeks failed — forcing Tab past %r.", _btn_lbl)
                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                time.sleep(self.step_delay * 0.5)
                _heuristic_steps += 1; continue

            # 1z. Already-handled guard: focus bounced back to a field we already filled.
            #     Tab past without calling LLM — prevents LLM from overwriting correct values.
            if not _plugin_active:
                _fid_ah = state.get("focused_element_id")
                _fe_ah  = next((e for e in state.get("elements", [])
                                if e.get("element_id") == _fid_ah), None)
                if _fe_ah and _fe_ah.get("type") in ("editcontrol", "comboboxcontrol"):
                    _fn_ah  = (_fe_ah.get("label") or _fe_ah.get("text") or "").strip()
                    _sec_ah = self._detect_section(state, _fe_ah)
                    _fk_ah  = f"{_sec_ah} {_fn_ah}" if _sec_ah else _fn_ah
                    if _fk_ah in self._filled_this_tab:
                        _foc_val_ah = (_fe_ah.get("value") or "").strip()
                        _exp_ah     = self._lookup_field(_fn_ah, section=_sec_ah)
                        if not _exp_ah or _foc_val_ah.lower() == _exp_ah.lower():
                            logger.info("Already-handled guard: %r already filled (%r) — Tab.",
                                        _fk_ah, _foc_val_ah[:30])
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            _heuristic_steps += 1; continue

            # 2. Auto-skip: if focused field already has the correct value, Tab past it
            if not _plugin_active and self._llm_client and self._auto_skip(state):
                logger.info("Auto-skip: focused field already has correct value — Tab.")
                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                _no_change_streak = 0
                _steps_on_tab     = 0
                # NOTE: _last_auto_step is intentionally NOT reset here.
                # Resetting on auto-skip would mask LLM back-cycling (clicking already-filled
                # fields), preventing the drought guard from firing and scrolling down.
                time.sleep(self.step_delay)
                _heuristic_steps += 1; continue

            # 2a. Auto-fill: if focused field is empty (or has wrong leftover value), type correct value
            auto_text = (self._auto_fill(state) if not _plugin_active else None)
            # 2a-peek: focused empty editcontrol with no cached value — peek Notepad to populate cache.
            # Gated off when disable_auto_handlers — these are task guards (Notepad
            # peek + dup-label fill) that fill fields without the transformer.
            if not auto_text and not self._no_autohandlers:
                _fid_p = state.get("focused_element_id")
                _fe_p  = next((e for e in state.get("elements", [])
                               if e.get("element_id") == _fid_p), None)
                if (_fe_p and _fe_p.get("type") in ("editcontrol", "comboboxcontrol")
                        and not (_fe_p.get("value") or "").strip()):
                    _fn_p      = (_fe_p.get("label") or _fe_p.get("text") or "").strip()
                    _sec_p     = self._detect_section(state, _fe_p)
                    _fk_p      = f"{_sec_p} {_fn_p}" if _sec_p else _fn_p
                    _fn_lower_p = _fn_p.lower()
                    if (_fn_p
                            and not self._lookup_field(_fn_p, section=_sec_p)
                            and _fn_lower_p not in {s.lower() for s in _confirmed_blank_fields}
                            and _fn_lower_p not in {s.lower() for s in self._filled_this_tab}):
                        logger.info("Auto-fill: '%s' not in cache — peeking Notepad.", _fn_p)
                        self._peek_notepad(state, _fn_p)
                        auto_text = self._auto_fill(state)
                        if not auto_text:
                            # Peek exhausted — no value found; treat as blank and tab past
                            logger.info("Auto-fill: '%s' not found after peek — treating as blank, Tab.",
                                        _fn_p)
                            _confirmed_blank_fields.add(_fn_lower_p)
                            self._filled_this_tab.add(_fk_p)
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            _no_change_streak = 0
                            _last_auto_step   = step_idx
                            time.sleep(self.step_delay)
                            _heuristic_steps += 1; continue
            # 2a-dup: duplicate UIA label — field is empty but label already in _filled_this_tab.
            # Peek Notepad for the next uncached field after that label and fill it.
            if not auto_text and not self._no_autohandlers:
                _fid_dup = state.get("focused_element_id")
                _fe_dup  = next((e for e in state.get("elements", [])
                                 if e.get("element_id") == _fid_dup), None)
                if (_fe_dup and _fe_dup.get("type") in ("editcontrol", "comboboxcontrol")
                        and not (_fe_dup.get("value") or "").strip()):
                    _fn_dup     = (_fe_dup.get("label") or _fe_dup.get("text") or "").strip()
                    _fn_dup_low = _fn_dup.lower()
                    _sec_dup    = self._detect_section(state, _fe_dup)
                    _fk_dup     = f"{_sec_dup} {_fn_dup}" if _sec_dup else _fn_dup
                    if (_fn_dup
                            and _fk_dup in self._filled_this_tab
                            and _fn_dup_low not in {s.lower() for s in _confirmed_blank_fields}):
                        logger.info("Dup-label: '%s' already filled this tab — peeking next field.", _fn_dup)
                        _next = self._peek_next_field_after(state, _fn_dup)
                        if _next:
                            _nk, _nv = _next
                            self._cached_record[_nk] = _nv
                            self._visual_cache[_nk]  = _nv
                            logger.info("Dup-label fill: actual field=%r  value=%r  (UIA label=%r)",
                                        _nk, _nv, _fn_dup)
                            self._ensure_form_foreground(state)
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["ctrl+a"]})
                            self._executor.execute({
                                "action_type": "keyboard",
                                "key_count": len(_nv),
                                "keystrokes": list(_nv),
                                "text": _nv,
                            })
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            self._filled_this_tab.add(_nk)
                            _no_change_streak = 0
                            _steps_on_tab     = 0
                            _last_auto_step   = step_idx
                            time.sleep(self.step_delay)
                            _heuristic_steps += 1; continue
                        else:
                            logger.info("Dup-label: no next field found — treating as blank, Tab.")
                            _confirmed_blank_fields.add(_fn_dup_low + "#dup")
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            _no_change_streak = 0
                            _last_auto_step   = step_idx
                            time.sleep(self.step_delay)
                            _heuristic_steps += 1; continue

            if auto_text:
                field_name_log, text_val, needs_clear = auto_text
                self._peek_notepad(state, field_name_log)
                self._ensure_form_foreground(state)
                if needs_clear:
                    logger.info("Auto-fill (overwrite): '%s' → %r", field_name_log, text_val[:40])
                    self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["ctrl+a"]})
                else:
                    logger.info("Auto-fill: '%s' → %r", field_name_log, text_val[:40])
                self._executor.execute({
                    "action_type": "keyboard",
                    "key_count": len(text_val),
                    "keystrokes": list(text_val),
                    "text": text_val,
                })
                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                logger.info("Auto-Tab after auto-fill.")
                _no_change_streak = 0
                _steps_on_tab     = 0
                _last_auto_step   = step_idx
                self._filled_this_tab.add(field_name_log)
                time.sleep(self.step_delay)
                _heuristic_steps += 1; continue

            # 2a2. Auto-check: if focused checkbox should be checked per background data, click it
            auto_chk = (self._auto_check(state) if not _plugin_active else None)
            if auto_chk is not None:
                field_name_log, should_check = auto_chk
                _did_new_check = False
                if should_check:
                    if field_name_log in self._checked_fields:
                        # Already checked this run — don't toggle it off
                        logger.info("Auto-check: '%s' already checked this run — Tab.", field_name_log)
                    else:
                        elements   = state.get("elements", [])
                        focused_id = state.get("focused_element_id")
                        focused_el = next((e for e in elements if e.get("element_id") == focused_id), None)
                        if focused_el and focused_el.get("bbox"):
                            x1, y1, x2, y2 = focused_el["bbox"]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            # Check if already checked via Win32 BM_GETCHECK
                            already_checked = False
                            try:
                                import win32gui as _wg
                                import win32api as _wa
                                BM_GETCHECK = 0x00F0
                                _hwnd = _wg.WindowFromPoint((int(cx), int(cy)))
                                if _hwnd:
                                    already_checked = (_wa.SendMessage(_hwnd, BM_GETCHECK, 0, 0) == 1)
                            except Exception:
                                pass
                            if already_checked:
                                logger.info("Auto-check: '%s' already checked — Tab.", field_name_log)
                                self._checked_fields.add(field_name_log)
                            else:
                                self._peek_notepad(state, field_name_log)
                                logger.info("Auto-check: '%s' → clicking @ (%.0f, %.0f)", field_name_log, cx, cy)
                                self._executor.execute({"action_type": "click", "click_position": [cx, cy]})
                                self._checked_fields.add(field_name_log)
                                _did_new_check = True
                        else:
                            logger.warning("Auto-check: '%s' — no bbox, pressing space", field_name_log)
                            self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["space"]})
                            self._checked_fields.add(field_name_log)
                            _did_new_check = True
                else:
                    logger.info("Auto-check: '%s' should remain unchecked — Tab.", field_name_log)
                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                _no_change_streak = 0
                if _did_new_check:
                    # Only reset drought/tab counters when a new checkbox was actually clicked.
                    # Revisits (already checked, should-remain-unchecked) let the guards accumulate
                    # so the drought and stuck guards eventually advance the tab.
                    _steps_on_tab   = 0
                    _last_auto_step = step_idx
                time.sleep(self.step_delay)
                _heuristic_steps += 1; continue

            # 2a3. Peek for unknown focused checkbox (not in cache → peek, then re-check)
            _fid_chk = state.get("focused_element_id")
            _fe_chk  = next((e for e in state.get("elements", [])
                             if e.get("element_id") == _fid_chk), None)
            if (not _plugin_active and _fe_chk and _fe_chk.get("type") == "checkboxcontrol"):
                _fn_chk      = (_fe_chk.get("label") or _fe_chk.get("text") or "").strip()
                _fn_chk_low  = _fn_chk.lower()
                if (_fn_chk
                        and not self._lookup_field(_fn_chk)
                        and _fn_chk_low not in {s.lower() for s in _confirmed_blank_fields}
                        and _fn_chk_low not in {s.lower() for s in self._checked_fields}):
                    logger.info("Auto-check: '%s' not in cache — peeking Notepad.", _fn_chk)
                    self._peek_notepad(state, _fn_chk)
                    auto_chk2 = self._auto_check(state)
                    if auto_chk2 is not None:
                        field_name_log2, should_check2 = auto_chk2
                        if should_check2 and field_name_log2 not in self._checked_fields:
                            if _fe_chk.get("bbox"):
                                x1, y1, x2, y2 = _fe_chk["bbox"]
                                cx2, cy2 = (x1 + x2) / 2, (y1 + y2) / 2
                                logger.info("Auto-check (post-peek): '%s' → clicking", field_name_log2)
                                self._executor.execute({"action_type": "click", "click_position": [cx2, cy2]})
                            else:
                                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["space"]})
                            self._checked_fields.add(field_name_log2)
                        else:
                            logger.info("Auto-check (post-peek): '%s' should remain unchecked — Tab.", _fn_chk)
                        self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                        _no_change_streak = 0
                        _last_auto_step   = step_idx
                        time.sleep(self.step_delay)
                        _heuristic_steps += 1; continue
                    else:
                        # Peek found nothing — mark confirmed blank, tab past
                        logger.info("Auto-check: '%s' not found after peek — treating as blank, Tab.", _fn_chk)
                        _confirmed_blank_fields.add(_fn_chk_low)
                        self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                        _no_change_streak = 0
                        _last_auto_step   = step_idx
                        time.sleep(self.step_delay)
                        _heuristic_steps += 1; continue

            # 2b. Auto-fix combobox: if focused combobox has WRONG value, select correct option
            fix = (self._combobox_needs_fix(state) if not _plugin_active else None)
            if fix:
                field_name, current_val, expected_val = fix
                elements   = state.get("elements", [])
                # Check if dropdown is already open (listitems visible). Standard
                # UIA "ListItem" maps to "listitem" (no "control" suffix) in
                # ui_observer.py's _CTRL_TYPE_MAP -- "listitemcontrol" alone
                # misses every real dropdown using the standard, mapped type.
                listitems = [e for e in elements
                             if e.get("type") in ("listitem", "listitemcontrol")
                             and e.get("window_role") != "background"]
                if listitems:
                    # Find the matching listitem and click it. _option_matches
                    # handles 'Sedan' matching an option like '4-Door Sedan' —
                    # a bare exact-only check here would miss it the same way
                    # the other two combobox handlers did before being fixed.
                    target = next(
                        (e for e in listitems
                         if _option_matches(expected_val, e.get("text") or e.get("label") or "")),
                        None,
                    )
                    if target and target.get("bbox"):
                        x1, y1, x2, y2 = target["bbox"]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        logger.info("Combobox-fix: clicking listitem %r @ (%.0f,%.0f)",
                                    expected_val, cx, cy)
                        self._executor.execute({"action_type": "click",
                                                "click_position": [cx, cy]})
                        _steps_on_tab   = 0
                        _last_auto_step = step_idx
                        time.sleep(self.step_delay)
                        _heuristic_steps += 1; continue
                    else:
                        logger.warning("Combobox-fix: listitem %r not found in open list", expected_val)
                else:
                    # Dropdown is closed — open it
                    focused_id = state.get("focused_element_id")
                    focused    = next((e for e in elements if e.get("element_id") == focused_id), None)
                    if focused and focused.get("bbox"):
                        x1, y1, x2, y2 = focused["bbox"]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        self._peek_notepad(state, field_name)
                        logger.info("Combobox-fix: '%s' = %r → need %r — opening dropdown",
                                    field_name, current_val, expected_val)
                        self._executor.execute({"action_type": "click",
                                                "click_position": [cx, cy]})
                        _steps_on_tab   = 0
                        _last_auto_step = step_idx
                        time.sleep(self.step_delay)
                        _heuristic_steps += 1; continue
                        
            _drought = step_idx - _last_auto_step
            if not _plugin_active and _drought >= _DROUGHT_LIMIT:
                if _tab_scroll_count < _MAX_TAB_SCROLLS:
                    logger.info("Drought guard: %d steps without auto-handler — scrolling form (scroll %d/%d).",
                                _drought, _tab_scroll_count + 1, _MAX_TAB_SCROLLS)
                    if self._scroll_form_down(state):
                        _tab_scroll_count += 1
                        _last_auto_step    = step_idx   # reset drought so it doesn't fire next step
                        time.sleep(self.step_delay * 0.75)
                        state = self._observe()          # re-observe so we see newly revealed fields
                        self._focus_first_empty_field(state, after_scroll=True)  # skip already-filled
                        time.sleep(0.3)
                        _heuristic_steps += 1; continue   # re-observe with scrolled form + focused field
                else:
                    # All scroll attempts exhausted — all visible fields are done on this tab.
                    # Proactively advance to the next tab instead of waiting for the stuck guard.
                    logger.info("Drought guard: scrolls exhausted (%d/%d) — advancing tab.",
                                _tab_scroll_count, _MAX_TAB_SCROLLS)
                    if self._try_advance_tab(state):
                        _no_change_streak  = 0
                        _tab_just_switched = True
                        _tab_scroll_count  = 0
                        _last_auto_step    = step_idx
                        self._refresh_record_cache(state)
                        time.sleep(self.step_delay)
                        continue

            # ── Navigation Protocol (runs in EVERY mode) ───────────────────────────
            # System-level responsibility, not a cloned behavior: keep an actionable,
            # empty target visible so the transformer only ever has to decide WHERE
            # among what's on screen, never learn WHEN/HOW FAR to scroll from
            # demonstrations that essentially never scrolled (0/11,062 pre-campaign
            # steps). decide() is a pure function of the state snapshot + how many
            # consecutive scrolls have already failed to reveal anything new on this
            # tab — see components/agent/navigation_protocol.py for the full design
            # rationale and unit tests.
            _nav_vb = self._form_viewport_bottom(state) - 8   # small safety margin, matches prior behavior
            _nav = self._navproto.decide(
                state, _nav_vb, _tab_scroll_count, max_dead_scrolls=2,
                attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key,
            )
            if _nav.action == self._navproto.NavAction.WAIT:
                _tab_scroll_count = 0                # actionable field visible → reset cap
            elif _nav.action == self._navproto.NavAction.SCROLL:
                # Scroll ONCE, re-observe, and let the NEXT decide() call (top
                # of the loop, on the fresh post-scroll state) make the real
                # WAIT/SCROLL/ADVANCE_TAB call — no acting on anything here.
                #
                # Found 2026-08-08, live, immediately after the optimal-view
                # redesign shipped: "you're just scrolling slowly, filling
                # nothing." This branch used to click whatever field the
                # scroll revealed and `continue` straight back to the top,
                # skipping the transformer's turn entirely — a leftover from
                # the OLD decide() (any visible target -> WAIT next time). The
                # NEW decide() only returns WAIT once cur == best (the
                # genuinely optimal view), so that stale click-and-continue
                # kept re-triggering SCROLL every single iteration: the same
                # nearly-stationary field got refocused (never typed into) on
                # every step, log-confirmed ('First Name' clicked 4 times in a
                # row, 25px apart, nothing ever filled). Two pieces of logic
                # built for different rules were fighting each other.
                #
                # Fixed by deleting the speculative click. decide() is now the
                # single source of truth for when scrolling stops; once cur
                # reaches best, WAIT fires on its own and the normal per-step
                # transformer/OPT2 flow below — which already knows how to
                # click AND fill, verified — takes over.
                _cur_before, _best_before = self._navproto.optimal_view_counts(
                    state, _nav_vb, attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                _sig_before = self._navproto.visible_field_signature(state, _nav_vb)
                # Track one specific off-screen field across this scroll and
                # measure its ACTUAL pixel movement — kept from the earlier
                # diagnostic round, still useful evidence for scroll-precision
                # questions independent of the WAIT/SCROLL decision itself.
                _diag_track_el = self._navproto.find_visible_empty_target(
                    state, 1e9, attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                _diag_track_key = (self._attempt_key(_diag_track_el, elements=state.get("elements", []))
                                    if _diag_track_el else None)
                _diag_track_y0  = (_diag_track_el["bbox"][1] if _diag_track_el and _diag_track_el.get("bbox") else None)
                _scrolled   = self._scroll_form_down(state)
                time.sleep(self.step_delay * 0.6)
                _state_after = self._observe()
                _vb_after    = self._form_viewport_bottom(_state_after)
                _view_moved  = (_scrolled and
                                self._navproto.visible_field_signature(_state_after, _vb_after) != _sig_before)
                if _diag_track_key is not None and _diag_track_y0 is not None:
                    _diag_track_after = next(
                        (e for e in _state_after.get("elements", [])
                         if self._attempt_key(e, elements=_state_after.get("elements", [])) == _diag_track_key
                         and e.get("bbox")), None)
                    if _diag_track_after is not None:
                        _diag_dy = _diag_track_y0 - _diag_track_after["bbox"][1]
                        logger.info(
                            "[SCROLL-DIAG] tracked target %r: bbox_top %.0f → %.0f  (moved %.0f px up)",
                            (_diag_track_el.get("label") or _diag_track_el.get("text") or "?")[:30],
                            _diag_track_y0, _diag_track_after["bbox"][1], _diag_dy)
                    else:
                        logger.info("[SCROLL-DIAG] tracked target %r vanished from the post-scroll snapshot.",
                                    (_diag_track_el.get("label") or _diag_track_el.get("text") or "?")[:30])
                state = _state_after if _view_moved else state
                _cur_after, _best_after = self._navproto.optimal_view_counts(
                    state, _vb_after, attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                if _view_moved and _cur_after > _cur_before:
                    # Genuine progress toward the optimal view (more targets
                    # visible now than before this scroll) -- reset the
                    # dead-scroll cap so a long tab can keep scrolling as many
                    # times as it genuinely has real content.
                    _last_auto_step = step_idx
                    _heuristic_steps += 1
                    _tab_scroll_count = 0
                    logger.info("Navigation Protocol: scroll improved the view (%d → %d of %d possible targets).",
                                _cur_before, _cur_after, _best_after)
                else:
                    _tab_scroll_count += 1
                    logger.info(
                        "Navigation Protocol: scroll did not improve the view (%d/%d targets, dead=%d/2).",
                        _cur_after, _best_after, _tab_scroll_count)
                continue
            elif _nav.action == self._navproto.NavAction.ADVANCE_TAB:
                logger.info("Navigation Protocol: %s", _nav.reason)
                if self._try_advance_tab(state):
                    _tab_just_switched = True
                    _tab_scroll_count  = 0
                    _last_auto_step    = step_idx
                    self._refresh_record_cache(self._observe())
                    time.sleep(self.step_delay)
                    continue

            # ── OPT2 BATCH FAST-FILL: fill EVERY currently-visible known-value
            # field in one pass, not just the focused one. Added 2026-08-14,
            # direct request ("it needs to be instant") after live log
            # evidence showed the single-field fast-fill below had already
            # cut model calls to almost nothing (16 transformer calls in a
            # full run) yet total time barely moved (4m09s vs last night's
            # 4m33s) — steps landed ~1s apart REGARDLESS of whether the model
            # was skipped, because each field still paid for its own
            # observe()+act() cycle. The fixed per-step cost (a fresh UIA
            # snapshot of 100+ elements, every single field) was the real
            # bottleneck, not reasoning.
            #
            # Navigation Protocol (just above) already guarantees at least
            # one actionable target is visible before this line is reached —
            # find_all_visible_empty_targets (navigation_protocol.py) reuses
            # that exact same eligibility rule (fillable type, empty value,
            # not yet attempted, on-screen) to return ALL of them instead of
            # just the first. Each one is filled directly via the same
            # already-tested, already-shipped mechanisms as the single-field
            # path below (WM_SETTEXT / CB_SETCURSEL / BM_SETCHECK via
            # SetFocus, no click) — reused, not reimplemented — with no Tab
            # key and no re-observe between fields in the same batch, only a
            # cheap Tab-to-blur after each write. One settle-wait for the
            # WHOLE batch, then one continue: the next step's single observe()
            # picks up whatever's left (unknowns, or fields a scroll would
            # reveal), so a full tab's known fields cost one observe() plus
            # N direct writes instead of N observe()+act() cycles.
            #
            # Any single field this loop can't resolve (no label, lookup
            # miss, HWND resolution fails, combobox option doesn't match)
            # is just skipped — it stays visible and empty, so it's picked
            # up by the single-field fast-fill / dead-spot rescue /
            # transformer fallback below on a later step, exactly as if
            # batch fast-fill had never run. filled_count starts at 0 and
            # only a nonzero count short-circuits the rest of this step.
            if self._no_autohandlers:
                _bf_targets = self._navproto.find_all_visible_empty_targets(
                    state, _nav_vb, attempted_keys=self._attempted_keys,
                    attempt_key_fn=self._attempt_key)
                _bf_filled = 0
                if _bf_targets:
                    self._ensure_foreground(state)
                for _bf_el in _bf_targets:
                    _bf_ty = (_bf_el.get("type") or "").lower()
                    _bf_label = (_bf_el.get("label") or _bf_el.get("text") or "").strip()
                    if not _bf_label:
                        continue
                    _bf_key = self._attempt_key(_bf_el, elements=state.get("elements", []))
                    if _bf_ty == "editcontrol":
                        if (_bf_key in self._leave_blank_keys
                                or _bf_key in self._typed_keys):
                            continue
                        _bf_sec = self._detect_section(state, _bf_el)
                        # Three-attempt escalation (plain lookup, cache
                        # refresh + retry, Notepad peek + retry) -- the
                        # exact same trust bar _ask_llm's own blank fast
                        # path already earned live (2026-08-08: a naive
                        # single-lookup miss got a real field hallucinated
                        # by the LLM 3 times before this escalation finally
                        # confirmed it blank on the 4th). A confirmed-blank
                        # result costs no more than what the reactive path
                        # was already paying for the SAME field -- this
                        # just skips the transformer call in front of it.
                        # A source that cannot identify this field is not
                        # saying it is blank - it is saying it does not know,
                        # and those must not be collapsed. Scope #1 never had to
                        # tell them apart: a Notepad record's keys ARE the form's
                        # labels, so a miss really did mean blank. Scope #2 asks
                        # a grade book for "Course Abad, Andrea A." while its
                        # columns are PROGRAM and FINAL GRADE - so every field
                        # missed, every one was ruled confirmed-blank, and a live
                        # run tabbed past 109 of them without filling one. Left
                        # unmarked and unfilled, the field stays visible and
                        # empty for the transformer/LLM path below, which is
                        # where deciding what a portal column wants from a record
                        # in another vocabulary belongs.
                        if not self._source_knows(_bf_label):
                            logger.info("[OPT2] batch fast-fill '%s' -> the data source "
                                        "does not recognise this field; leaving it for "
                                        "the transformer/LLM", _bf_label)
                            continue
                        _bf_val = self._resolve_field_value_with_escalation(state, _bf_label, section=_bf_sec)
                        if not _bf_val:
                            logger.info("[OPT2] batch fast-fill '%s' → confirmed blank, Tab past "
                                        "(no transformer, no LLM, no click)", _bf_label)
                            self._leave_blank_keys.add(_bf_key)
                            self._mark_attempted(_bf_el, elements=state.get("elements", []))
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            _bf_filled += 1
                            continue
                        _bf_ctrl = self._resolve_field_control(state, _bf_label, _bf_el.get("bbox"))
                        _bf_hwnd = None
                        if _bf_ctrl is not None:
                            try:
                                _bf_ctrl.SetFocus()
                                _bf_hwnd = _bf_ctrl.NativeWindowHandle
                            except Exception as _bf_exc:
                                logger.debug("Batch fast-fill focus/handle resolution failed for %r — %s",
                                             _bf_label, _bf_exc)
                        if not _bf_hwnd:
                            continue
                        logger.info("[OPT2] batch fast-fill '%s' → %r (no transformer, no LLM, no click)",
                                    _bf_label, _bf_val[:40])
                        self._executor.execute({
                            "action_type": "keyboard", "text": _bf_val,
                            "key_count": len(_bf_val), "keystrokes": list(_bf_val),
                            "direct_fill_hwnd": _bf_hwnd,
                        })
                        self._mark_attempted(_bf_el, elements=state.get("elements", []))
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        _bf_filled += 1
                    elif _bf_ty == "comboboxcontrol":
                        if (_bf_key in self._leave_blank_keys
                                or _bf_key in self._typed_keys):
                            continue
                        _bf_sec = self._detect_section(state, _bf_el)
                        # A source that cannot identify this field is not
                        # saying it is blank - it is saying it does not know,
                        # and those must not be collapsed. Scope #1 never had to
                        # tell them apart: a Notepad record's keys ARE the form's
                        # labels, so a miss really did mean blank. Scope #2 asks
                        # a grade book for "Course Abad, Andrea A." while its
                        # columns are PROGRAM and FINAL GRADE - so every field
                        # missed, every one was ruled confirmed-blank, and a live
                        # run tabbed past 109 of them without filling one. Left
                        # unmarked and unfilled, the field stays visible and
                        # empty for the transformer/LLM path below, which is
                        # where deciding what a portal column wants from a record
                        # in another vocabulary belongs.
                        if not self._source_knows(_bf_label):
                            logger.info("[OPT2] batch fast-fill '%s' -> the data source "
                                        "does not recognise this field; leaving it for "
                                        "the transformer/LLM", _bf_label)
                            continue
                        _bf_val = self._resolve_field_value_with_escalation(state, _bf_label, section=_bf_sec)
                        if not _bf_val:
                            logger.info("[OPT2] batch fast-fill '%s' → confirmed blank, Tab past "
                                        "(no transformer, no LLM, no click)", _bf_label)
                            self._leave_blank_keys.add(_bf_key)
                            self._mark_attempted(_bf_el, elements=state.get("elements", []))
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            _bf_filled += 1
                            continue
                        _bf_ctrl = self._resolve_field_control(state, _bf_label, _bf_el.get("bbox"))
                        _bf_hwnd = None
                        if _bf_ctrl is not None:
                            try:
                                _bf_ctrl.SetFocus()
                                _bf_hwnd = _bf_ctrl.NativeWindowHandle
                            except Exception as _bf_exc:
                                logger.debug("Batch fast-fill focus/handle resolution failed for %r — %s",
                                             _bf_label, _bf_exc)
                        if not _bf_hwnd:
                            continue
                        _bf_cb_result = self._executor.execute({
                            "action_type": "combobox_select", "text": _bf_val,
                            "combobox_hwnd": _bf_hwnd,
                        })
                        if not _bf_cb_result.success:
                            continue   # no matching option — leave for the click-based fallback
                        logger.info("[OPT2] batch fast-fill '%s' → %r (no transformer, no LLM, no click)",
                                    _bf_label, _bf_val[:40])
                        self._mark_attempted(_bf_el, elements=state.get("elements", []))
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        _bf_filled += 1
                    elif _bf_ty in ("checkboxcontrol", "checkbox"):
                        if _bf_label in self._checked_fields:
                            continue
                        _bf_should_check = self._lookup_checkbox_should_check(_bf_label)
                        if _bf_should_check is None:
                            continue
                        if _bf_should_check:
                            _bf_bbox = _bf_el.get("bbox")
                            if not _bf_bbox:
                                continue
                            try:
                                import win32gui as _bfwg
                                import win32api as _bfwa
                                _bf_cx = (_bf_bbox[0] + _bf_bbox[2]) / 2
                                _bf_cy = (_bf_bbox[1] + _bf_bbox[3]) / 2
                                _bf_hw = _bfwg.WindowFromPoint((int(_bf_cx), int(_bf_cy)))
                                if not _bf_hw:
                                    continue
                                _bfwa.SendMessage(_bf_hw, 0x00F1, 1, 0)  # BM_SETCHECK, BST_CHECKED
                            except Exception as _bf_exc:
                                logger.debug("Batch fast-fill checkbox check failed for %r — %s",
                                             _bf_label, _bf_exc)
                                continue
                            logger.info("[OPT2] batch fast-fill checkbox '%s' → checked "
                                        "(no transformer, no LLM, no click)", _bf_label)
                        else:
                            logger.info("[OPT2] batch fast-fill checkbox '%s' → leave unchecked "
                                        "(no transformer, no LLM, no click)", _bf_label)
                        self._checked_fields.add(_bf_label)
                        self._mark_attempted(_bf_el, elements=state.get("elements", []))
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        _bf_filled += 1
                if _bf_filled > 0:
                    logger.info("[OPT2] batch fast-fill: %d field(s) filled in one pass, no per-field re-observe.",
                                _bf_filled)
                    self._adaptive_settle_wait(self.step_delay * 0.2)
                    continue

            # ── OPT2 FAST-FILL: skip the transformer AND the LLM entirely when
            # the focused field's value is already known via the same
            # deterministic lookup _ask_llm's own fast path already trusts.
            # Built 2026-08-14, direct request ("don't stop unless there's
            # something to reason about", 3-day demo deadline). Mirrors the
            # exact exclusion logic _t_is_type uses a few lines below
            # (_attempt_key / _leave_blank_keys / _typed_keys) so this can
            # never disagree with the existing reactive path about what's
            # already handled or deliberately left blank. Falls through to
            # the completely unmodified code below on ANY miss (no label, no
            # known value, HWND resolution fails, combobox option doesn't
            # match) -- reasoning only costs anything on genuine absence,
            # never on a guess about the future.
            # Extended same night to comboboxcontrol via CB_SETCURSEL (a
            # direct message that selects an option with no click and no
            # dropdown ever opened, live-tested against the real form) --
            # editcontrol uses WM_SETTEXT (direct_fill_hwnd), comboboxcontrol
            # uses CB_SETCURSEL (combobox_hwnd); checkboxes already have
            # their own separate, working mechanism (BM_SETCHECK) and stay
            # on today's existing path untouched.
            if self._no_autohandlers:
                _ff_fid = state.get("focused_element_id")
                _ff_fel = next((e for e in state.get("elements", [])
                                if e.get("element_id") == _ff_fid), None)
                _ff_ty  = (_ff_fel.get("type") or "").lower() if _ff_fel else ""
                _ff_val = (_ff_fel.get("value") or "").strip() if _ff_fel else ""
                _ff_key = (self._attempt_key(_ff_fel, elements=state.get("elements", []))
                           if _ff_fel else None)
                if (_ff_fel and _ff_ty in ("editcontrol", "comboboxcontrol") and not _ff_val
                        and _ff_key not in self._leave_blank_keys
                        and _ff_key not in self._typed_keys):
                    _ff_label = (_ff_fel.get("label") or _ff_fel.get("text") or "").strip()
                    _ff_sec   = self._detect_section(state, _ff_fel)
                    _ff_val_known = self._lookup_field(_ff_label, section=_ff_sec) if _ff_label else ""
                    if _ff_label and _ff_val_known:
                        _ff_ctrl = self._resolve_field_control(state, _ff_label, _ff_fel.get("bbox"))
                        _ff_hwnd = None
                        if _ff_ctrl is not None:
                            try:
                                _ff_ctrl.SetFocus()
                                _ff_hwnd = _ff_ctrl.NativeWindowHandle
                            except Exception as _ff_exc:
                                logger.debug("Fast-fill focus/handle resolution failed for %r — %s",
                                             _ff_label, _ff_exc)
                        if _ff_hwnd and _ff_ty == "editcontrol":
                            self._ensure_foreground(state)
                            logger.info("[OPT2] fast-fill '%s' → %r (no transformer, no LLM, no click)",
                                        _ff_label, _ff_val_known[:40])
                            self._executor.execute({
                                "action_type": "keyboard", "text": _ff_val_known,
                                "key_count": len(_ff_val_known), "keystrokes": list(_ff_val_known),
                                "direct_fill_hwnd": _ff_hwnd,
                            })
                            self._mark_attempted(_ff_fel, elements=state.get("elements", []))
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            # Tightened 2026-08-14, direct request ahead of a
                            # ~1-minute demo target: the settle-wait here is
                            # ALREADY adaptive (returns early once the state
                            # stabilizes) -- this only lowers the worst-case
                            # ceiling, and we already have hard, synchronous
                            # proof the value landed (the WM_SETTEXT readback
                            # matched, checked before this line ever runs),
                            # unlike the general-purpose fill path this same
                            # helper also guards, which has no such proof.
                            self._adaptive_settle_wait(self.step_delay * 0.2)
                            continue
                        elif _ff_hwnd and _ff_ty == "comboboxcontrol":
                            self._ensure_foreground(state)
                            _ff_cb_result = self._executor.execute({
                                "action_type": "combobox_select", "text": _ff_val_known,
                                "combobox_hwnd": _ff_hwnd,
                            })
                            if _ff_cb_result.success:
                                logger.info("[OPT2] fast-fill '%s' → %r (no transformer, no LLM, no click)",
                                            _ff_label, _ff_val_known[:40])
                                self._mark_attempted(_ff_fel, elements=state.get("elements", []))
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                self._adaptive_settle_wait(self.step_delay * 0.2)
                                continue
                            # No matching real option (or the direct-select
                            # attempt otherwise failed) -- fall through to
                            # today's existing click-based open+select path
                            # below, unchanged.
                elif (_ff_fel and _ff_ty in ("checkboxcontrol", "checkbox")
                      and (_ff_fel.get("label") or _ff_fel.get("text") or "").strip()
                          not in self._checked_fields):
                    # ── OPT2 CHECKBOX FAST-FILL: added 2026-08-14, same
                    # night, direct evidence from a real live run's log --
                    # 25 of 57 remaining live model decisions in one full
                    # run were checkboxes, every one shown already having a
                    # deterministically known answer via the exact same
                    # lookup mechanism text/combobox fields already use.
                    # Checkboxes were excluded from the fast-fill checks
                    # above because they need different eligibility rules
                    # (no reliable .value to read -- see self._checked_fields'
                    # own comment -- and a different, already-proven
                    # mechanism: BM_SETCHECK via WindowFromPoint on the
                    # bbox center, the exact same call shape already used
                    # at this file's other checkbox sites, not the name-
                    # based UIA lookup text/combobox fields use).
                    # _auto_check() already bundles the lookup + "yes"-
                    # prefix parsing in one call -- reused exactly, not
                    # reimplemented.
                    _chk_result = self._auto_check(state)
                    if _chk_result is not None:
                        _chk_label, _chk_should_check = _chk_result
                        _chk_bbox = _ff_fel.get("bbox")
                        if _chk_should_check and _chk_bbox:
                            try:
                                import win32gui as _cfwg
                                import win32api as _cfwa
                                _chk_cx = (_chk_bbox[0] + _chk_bbox[2]) / 2
                                _chk_cy = (_chk_bbox[1] + _chk_bbox[3]) / 2
                                _chk_hw = _cfwg.WindowFromPoint((int(_chk_cx), int(_chk_cy)))
                                if _chk_hw:
                                    _cfwa.SendMessage(_chk_hw, 0x00F1, 1, 0)  # BM_SETCHECK, BST_CHECKED
                                    logger.info("[OPT2] fast-fill checkbox '%s' → checked "
                                                "(no transformer, no LLM, no click)", _chk_label)
                                    self._checked_fields.add(_chk_label)
                                    self._mark_attempted(_ff_fel, elements=state.get("elements", []))
                                    self._executor.execute({"action_type": "keyboard",
                                                            "key_count": 1, "keystrokes": ["tab"]})
                                    self._adaptive_settle_wait(self.step_delay * 0.2)
                                    continue
                            except Exception as _chk_exc:
                                logger.debug("Fast-fill checkbox check failed for %r — %s",
                                             _chk_label, _chk_exc)
                        elif not _chk_should_check:
                            # Wx checkboxes already default to unchecked
                            # (execution_checkbox_reset_default) -- a known
                            # "NO" answer needs no action at all, just Tab
                            # past and mark it handled, same as the existing
                            # type-intercept path's own unchecked branch.
                            logger.info("[OPT2] fast-fill checkbox '%s' → leave unchecked "
                                        "(no transformer, no LLM, no click)", _chk_label)
                            self._checked_fields.add(_chk_label)
                            self._mark_attempted(_ff_fel, elements=state.get("elements", []))
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            self._adaptive_settle_wait(self.step_delay * 0.2)
                            continue
                    # Unknown checkbox value -- fall through to today's
                    # existing reactive path (transformer + LLM decide),
                    # exactly as before this change.
                elif _ff_fel and _ff_ty not in ("editcontrol", "comboboxcontrol",
                                                "checkboxcontrol", "checkbox"):
                    # ── OPT2 DEAD-SPOT RESCUE: added 2026-08-14, direct
                    # request ahead of a ~1-minute demo target, from real log
                    # evidence -- a live run showed Tab periodically landing
                    # on a genuinely non-fillable element (a section-divider
                    # pane between groups of fields), which paid for a full
                    # transformer decision every single time before recovery.
                    # Before paying for that live model call, try the exact
                    # same deterministic, model-free "find the next visible,
                    # empty, unattempted field" mechanism already used
                    # elsewhere in this file for the identical purpose
                    # (agent.py's own redirect-stuck sites) -- no new
                    # navigation logic invented, just tried earlier, before
                    # the model, instead of only after it as an escalation.
                    # Falls straight through to today's unmodified code
                    # (the transformer call) if this finds nothing or fails
                    # -- never worse than today, only sometimes faster.
                    _dsr_vb = self._form_viewport_bottom(state) - 8
                    _dsr_target = self._navproto.find_visible_empty_target(
                        state, _dsr_vb, attempted_keys=self._attempted_keys,
                        attempt_key_fn=self._attempt_key)
                    if _dsr_target and _dsr_target.get("bbox"):
                        _dsr_label = (_dsr_target.get("label") or _dsr_target.get("text") or "").strip()
                        if _dsr_label:
                            _dsr_ctrl = self._resolve_field_control(
                                state, _dsr_label, _dsr_target.get("bbox"))
                            if _dsr_ctrl is not None:
                                try:
                                    _dsr_ctrl.SetFocus()
                                    logger.info("[OPT2] dead-spot rescue: focused '%s' directly "
                                                "(no transformer, no click)", _dsr_label)
                                    self._adaptive_settle_wait(self.step_delay * 0.2)
                                    continue
                                except Exception as _dsr_exc:
                                    logger.debug("Dead-spot rescue focus failed for %r — %s",
                                                 _dsr_label, _dsr_exc)

            # 2. Transformer always runs — learned behavioral engine
            llm_action = None

            t_pred = self._predict(state)
            t_type = t_pred.get("action_type", "no_op")
            t_conf = t_pred.get("confidence", max(t_pred.get("_scores", {}).values(), default=0.0))
            logger.info("[TRANSFORMER] action=%-8s  conf=%.2f", t_type, t_conf)

            # Routing: the transformer's ACTION-TYPE head collapses (always one
            # class), so we never let it act alone on action choice. The LLM
            # always decides WHAT (action + value); the transformer supplies
            # WHERE (click position) via _merge. Transformer's strength is
            # positioning (click_acc ~0.76), not action selection.
            _HIGH_CONF   = 1.01   # effectively: never skip the LLM
            _MED_CONF    = 0.50

            _decision_maker = "transformer"

            # ── OPTION 2: transformer-navigation + LLM value-oracle ──────────
            # Transformer's PROVEN strength is WHICH field (pointer, click_acc
            # ~0.76); its action-type head is unreliable, so we don't use it.
            # Rule:
            #   - If a fillable field is focused & empty → LLM supplies its value
            #     (one call, the "understanding"), then type it.
            #   - Otherwise → click the transformer's pointer target (navigate to
            #     the next field). The transformer DRIVES navigation.
            if self._no_autohandlers:
                _fid2 = state.get("focused_element_id")
                _fe2  = next((e for e in state.get("elements", [])
                              if e.get("element_id") == _fid2), None)

                # ── OPTION B (DEVELOPERS.md → Decisions): the FOCUSED widget's TYPE
                # decides fill-vs-navigate — NOT the transformer's action-type head
                # (which whipsaws click↔keyboard across retrains). A fillable+empty
                # focused field → FILL it (LLM value); otherwise the transformer's
                # POINTER navigates (click), and when its pointer lands on a tab /
                # button that IS the tab-switch / submit. Navigation — which element,
                # what order, WHEN to switch tabs — stays 100% transformer; only this
                # universal "is the focused thing a fillable empty field?" check
                # replaces the unstable head. Widget mechanics are universal GUI
                # semantics, not form-specific rules.
                _fe2_ty  = (_fe2.get("type") or "").lower() if _fe2 else ""
                _fe2_val = (_fe2.get("value") or "").strip() if _fe2 else ""
                # Exclude already-attempted CHECKBOXES specifically — ui_observer never
                # reports a real checked/unchecked state via ValuePattern (wx.CheckBox
                # doesn't expose it), so a checkbox's `value` reads empty FOREVER, checked
                # or not, and `not _fe2_val` alone can never recognize "already done" for
                # this one type. Confirmed live 2026-08-07 ("we reached Payment tab, but
                # there was a loop at the end"): 'Auto-Pay Enrolled' (already correctly
                # checked) oscillated with 'Last Payment Amount' for 18+ steps.
                #
                # REGRESSION, found live 2026-08-07 in the VERY NEXT run ("Got considerably
                # worse... did you save the best run?"): this check was first written to
                # cover editcontrol/comboboxcontrol too, not just checkboxes. That broke
                # them — clicking a text/combobox field to bring it into FOCUS (e.g. the
                # low-confidence-fallback escalation's direct click, execution_lowconf_
                # navprotocol_escalation) makes _record_attempt() mark it attempted
                # immediately, before any value is ever typed into it. For a checkbox a
                # click IS the fill action, so that's correct; for an editcontrol/combobox
                # a click is just navigation — the field stayed genuinely empty but was
                # permanently excluded from ever being filled again. Confirmed in the log:
                # 'Expiration Date' was clicked exactly once, by the escalation fix, and
                # never mentioned again for the rest of the run — never typed into.
                # editcontrol/comboboxcontrol don't need this at all: `not _fe2_val` is
                # already accurate for them (typing/selecting genuinely changes `value`,
                # a bare click doesn't) — scoped to checkboxes only, where it's the only
                # type this gap actually applies to.
                _fe2_key = self._attempt_key(_fe2, elements=state.get("elements", [])) if _fe2 else None
                _fe2_chk_attempted = (
                    bool(_fe2) and _fe2_ty in ("checkboxcontrol", "checkbox")
                    and _fe2_key in self._attempted_keys
                )
                # Any field type, deliberately confirmed leave-blank earlier this record
                # (see self._leave_blank_keys' own comment) — re-asking is guaranteed to
                # get the same "nothing goes here" answer every time, at the cost of a
                # full LLM round-trip per visit. Found live 2026-08-07, next run after
                # the checkbox-only regression fix: 'Last Payment Amount'/'Last Payment
                # Date' (editcontrol, genuinely blank) got correctly Tab-skipped each
                # time but kept re-entering the fill branch forever, since their `value`
                # can never become non-empty by design.
                _fe2_confirmed_blank = bool(_fe2) and _fe2_key in self._leave_blank_keys
                # Trust a genuine-typed-text record over a live "is it empty"
                # read when they conflict, for ANY type -- not just checkboxes
                # (_fe2_chk_attempted, above, already did this but only there,
                # since checkboxes never expose a real value at all). Found
                # 2026-08-08, live: 'Years Continuously Insured' -- typed and
                # CONFIRMED filled once already, several steps earlier in the
                # same run -- got navigated back to, read as empty by this
                # exact live snapshot, and re-entered the fill branch fully:
                # a real LLM lookup, a real paste, a real 2-retry verify-at-
                # fill cycle, 4 times in a row. Same "two sources of truth
                # disagreeing, trust the one already proven right" lesson as
                # every other fix tonight: a live value read has proven
                # unreliable for at least this one field.
                #
                # Deliberately checks self._typed_keys, NOT self._attempted_
                # keys -- the general attempted set also includes fields
                # merely CLICKED for navigation with no value ever typed
                # (see self._leave_blank_keys' own comment, and execution_
                # payment_tab_oscillation_fix's regression two comment blocks
                # above this one). Using the general set here would silently
                # reintroduce that exact bug: a stray navigation click onto a
                # genuinely-empty editcontrol/comboboxcontrol would mark it
                # "already attempted" and exclude it from ever being filled.
                # _typed_keys only gains an entry when a keyboard action
                # carried real text, so a bare click can never trip this.
                _fe2_already_attempted = bool(_fe2) and _fe2_key in self._typed_keys
                _t_is_type = (_fe2_ty in ("editcontrol", "input", "comboboxcontrol",
                                          "checkboxcontrol", "checkbox")
                              and not _fe2_val and not _fe2_chk_attempted and not _fe2_confirmed_blank
                              and not _fe2_already_attempted)

                # DIAGNOSTIC, added 2026-08-09: direct user report ("skipped
                # most fields in Vehicle after Current Mileage") -- log
                # evidence showed a plain navigate click land somewhere
                # ("Focus moved after click" = ok) right where 'Annual Miles
                # Est.' should sit by field order, but the very next step
                # never once logged "LLM focused-field lookup" for it --
                # meaning _t_is_type was False, but nothing in the log says
                # WHY: _fe2 could be None (a focused_element_id that doesn't
                # match anything in this fresh scan -- the same element-id-
                # churn bug class already fixed twice elsewhere in this file,
                # for state_validator.py and _attempt_key), or _fe2 could be
                # found but excluded for a real reason (wrong type, already
                # non-empty, already attempted). Rather than guess which,
                # log the exact case so the next live run answers it
                # directly. No behavior change -- logging only.
                if _fid2 and not _t_is_type and not (_fe2 and _fe2_ty in ("checkboxcontrol", "checkbox")):
                    if _fe2 is None:
                        logger.info(
                            "[FOCUS-DIAG] focused_element_id=%r not found among %d scanned elements "
                            "after a navigate click -- possible id churn.", _fid2, len(state.get("elements", [])))
                    else:
                        logger.info(
                            "[FOCUS-DIAG] focused %r type=%r value=%r not treated as fillable: "
                            "chk_attempted=%s confirmed_blank=%s already_attempted=%s",
                            (_fe2.get("label") or _fe2.get("text") or "?")[:40], _fe2_ty, _fe2_val[:40],
                            _fe2_chk_attempted, _fe2_confirmed_blank, _fe2_already_attempted)

                if _t_is_type and self._llm_client:
                    _lowconf_fallback_streak = 0   # real progress — a fillable target is focused
                    _reclick_streak = 0
                    # transformer chose to FILL → LLM supplies the value (the WHAT).
                    # LLM owns value-filling per the architecture.
                    llm_action = self._ask_llm(state)

                    if _is_llm_unavailable(llm_action):
                        _llm_unavailable_streak += 1
                        logger.warning(
                            "LLM unavailable (%d/%d) — NOT treating as leave-blank; retrying, not skipping.",
                            _llm_unavailable_streak, _LLM_UNAVAILABLE_LIMIT,
                        )
                        if _llm_unavailable_streak >= _LLM_UNAVAILABLE_LIMIT:
                            logger.error(
                                "LLM unavailable %d times in a row — halting rather than blank-filling "
                                "the rest of the form. Check the provider (e.g. LM Studio's local server).",
                                _llm_unavailable_streak,
                            )
                            break
                        time.sleep(self.step_delay)
                        continue
                    _llm_unavailable_streak = 0

                    if llm_action.get("action_type") == "done":
                        logger.info("LLM: task complete."); break

                    _flabel = ((_fe2.get("label") or _fe2.get("text") or "?")[:30]
                               if _fe2 else "(no focus)")

                    # Recognize "leave blank" on llm_action BEFORE _merge() runs, not
                    # after. _merge() overrides ANY "type" decision into a "click"
                    # whenever the transformer's (collapsed, near-constant-confidence)
                    # action-type head says click — regardless of whether the LLM's
                    # intended text was real or empty. _is_leave_blank_prediction is
                    # gated on action_type=="keyboard" specifically so a legitimate
                    # combobox click-override is never mistaken for leave-blank (see
                    # its own docstring) — but that gate ALSO means a genuinely empty
                    # answer (nothing in the record for this field) never gets
                    # recognized once merge has already turned it into a click, and
                    # the resulting click uses the transformer's own (~69%-accurate)
                    # pointer position, not this field's — landing almost anywhere.
                    # Confirmed live 2026-08-07 ("we reached Payment tab, but there was
                    # a loop at the end"): 'Last Payment Amount' had no record value,
                    # the LLM correctly said "type" with empty text, merge overrode it
                    # to a click at the transformer's drifted pointer position instead
                    # of Tabbing past a legitimately blank field, contributing to an
                    # 18+ step oscillation with 'Auto-Pay Enrolled' next to it.
                    if llm_action.get("action_type") in ("type", "keyboard") and _is_leave_blank_prediction(
                            {"action_type": "keyboard", "text": llm_action.get("text", "")}):
                        logger.info("[OPT2] %r → LLM says leave-blank/empty (pre-merge) — "
                                    "Tab past (skip), not letting merge turn this into a stray click.",
                                    _flabel)
                        if _fe2 is not None:
                            self._mark_attempted(_fe2, elements=state.get("elements", []))
                            # Distinct from _attempted_keys (see self._leave_blank_keys'
                            # own comment) -- this is a genuine, deliberate "nothing
                            # goes here" decision, not just a navigation click, so it
                            # must stick regardless of field type.
                            self._leave_blank_keys.add(
                                self._attempt_key(_fe2, elements=state.get("elements", [])))
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        self._adaptive_settle_wait(self.step_delay * 0.4)
                        continue

                    prediction = self._merge(t_pred, t_conf, llm_action, state)
                    _decision_maker = "llm"
                    logger.info("[OPT2] TRANSFORMER chose TYPE → LLM value for '%s' → %r",
                                _flabel, prediction.get("text", "")[:40])

                    # When merge overrides this FILL decision into a click, anchor it to
                    # the ALREADY-FOCUSED field's own bbox, not the transformer's
                    # independently-predicted click_position. Found live 2026-08-07,
                    # "Same loop on Auto Enrolled" (1175 log lines): _merge()'s override
                    # (~L4560) uses t_pred['click_position'] -- the transformer's own,
                    # SEPARATELY-learned navigation pointer, trained to answer "where do I
                    # click next", not "where is the field currently in focus". Those are
                    # different questions with different answers: at one visit the pointer
                    # guessed (1484,426), which missed 'Auto-Pay Enrolled''s real bbox
                    # entirely -- the click landed on some unrelated element, WHICH THEN
                    # got marked attempted by _record_attempt() instead of the checkbox
                    # actually intended, so 'Auto-Pay Enrolled' itself was never recorded
                    # as attempted and kept re-entering the fill branch every time focus
                    # cycled back to it. We already KNOW exactly where the focused field
                    # is (_fe2['bbox']) -- there's no reason to trust a second, independent
                    # guess for "click the thing that's already focused". Fixes this for
                    # every type this override applies to (checkbox, combobox), not just
                    # the one that happened to surface it.
                    if prediction.get("action_type") == "click" and _fe2 and _fe2.get("bbox"):
                        _feb = _fe2["bbox"]
                        prediction = dict(prediction)
                        prediction["click_position"] = [(_feb[0] + _feb[2]) / 2, (_feb[1] + _feb[3]) / 2]

                    # Toggle-avoidance: _merge() can override a "type" decision into
                    # a "click" (comboboxes need a click to open before a value can
                    # be selected). If a dropdown is ALREADY open, executing that
                    # click via the generic path below would CLOSE it instead of
                    # opening it — a click toggles, it doesn't just open (see
                    # _open_dropdown_items' docstring). This is the THIRD place this
                    # exact bug class showed up, found live 2026-08-07: 'Education
                    # Level' and 'Number of Doors' each burned 8-11 steps toggling
                    # open/closed before accidentally landing on the right value
                    # through a different, already-fixed handler. The two dedicated
                    # combobox handlers already guard against this; this is the one
                    # place _merge()'s own override reaches the generic execution
                    # path, bypassing both.
                    if prediction.get("action_type") == "click":
                        _already_open_items = _open_dropdown_items(state.get("elements", []))
                        if _already_open_items:
                            _wanted = (llm_action.get("text") or "").strip()
                            _oh = next((e for e in _already_open_items
                                        if _wanted and _option_matches(_wanted, e.get("text") or e.get("label") or "")),
                                       None)
                            if _oh and _oh.get("bbox"):
                                _ohb = _oh["bbox"]
                                logger.info("[OPT2] Dropdown for %r already open — selecting %r directly "
                                            "instead of re-clicking (would toggle it closed).", _flabel, _wanted)
                                self._executor.execute({"action_type": "click",
                                                        "click_position": [(_ohb[0]+_ohb[2])/2, (_ohb[1]+_ohb[3])/2]})
                                time.sleep(0.25)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                            else:
                                logger.info("[OPT2] Dropdown for %r already open, no match for %r — "
                                            "Escape instead of re-clicking (would toggle it closed).",
                                            _flabel, _wanted)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["escape"]})
                            time.sleep(self.step_delay * 0.4)
                            continue

                    # Leave-blank guard: if the resolved value means "leave empty"
                    # (record said '(leave blank)' / none / n-a, possibly with a note),
                    # DON'T type the placeholder literally — Tab past + mark attempted.
                    # See _is_leave_blank_prediction for why this must gate on
                    # action_type=="keyboard" first (a live infinite-loop fix).
                    if _is_leave_blank_prediction(prediction):
                        logger.info("[OPT2] %r → leave-blank/empty — Tab past (skip).", _flabel)
                        if _fe2 is not None:
                            self._mark_attempted(_fe2, elements=state.get("elements", []))
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        self._adaptive_settle_wait(self.step_delay * 0.4)
                        continue
                else:
                    # Transformer pointer navigates to the next field
                    _decision_maker = "transformer"
                    _pos2 = t_pred.get("click_position")
                    # Low-confidence click gate. Found 2026-08-07, live: a
                    # ptr_conf=0.20 guess landed on the tab strip and ended a
                    # tab pass 9 fields early (Policy: 4/13 filled). The
                    # confidence was already being logged, never acted on.
                    # This does NOT hand WHERE to the LLM (that stays
                    # forbidden in pure/Option-B mode, see _merge()) — it
                    # only declines to act on a guess the model itself
                    # flagged as unreliable, falling back to the SAME
                    # generic Tab-advance already used below for a
                    # structurally invalid pointer. Tab is a universal
                    # navigation primitive, not task-specific, so this holds
                    # for any form this agent is pointed at, not just this one.
                    # Threshold matches the gate already established for the
                    # non-Option-B merge path (agent.py's _merge(), ~L4045).
                    _click_conf2 = t_pred.get("_click_conf", 0.0)
                    _elements2   = state.get("elements", [])
                    # Union with _advance_blacklist_pos: don't let the transformer's
                    # own pointer immediately undo a tab the system just deliberately
                    # left as exhausted (found live 2026-08-07 — see that attribute's
                    # comment for the full story, a 12-13x repeating tug-of-war).
                    _gated_pos2  = _gate_low_confidence_click(
                        _pos2, _click_conf2, elements=_elements2,
                        blacklist=self._nochange_click_pos | self._advance_blacklist_pos)
                    if _gated_pos2 is None and _pos2:
                        _floor2 = (_TAB_CLICK_CONF_FLOOR if _is_tab_strip_click(_pos2, _elements2)
                                   else _CLICK_CONF_FLOOR)
                        logger.info("[OPT2] pointer low-confidence (%.2f < %.2f) — Tab fallback instead of acting on it",
                                    _click_conf2, _floor2)
                    _pos2 = _gated_pos2
                    if _pos2 and (_pos2[0] > 1 or _pos2[1] > 1):
                        _snap2 = self._snap(_pos2, state) or _pos2
                        # Guard against re-clicking an ALREADY-FILLED, already-attempted
                        # combobox. Found live 2026-08-07 investigating a user report
                        # ("Stuck on Marital Status"): the transformer's pointer drifted
                        # back onto 'Marital Status' at ptr_conf=0.58 (correctly filled
                        # with 'Married' 4 steps earlier) -- moderate confidence, well
                        # above the gate floor, so nothing blocked it. A plain click on a
                        # closed combobox TOGGLES it open (+6 elements); the VERY NEXT
                        # step's click landed on a list-item position and closed it again
                        # (-6 elements) via the generic path, which has no value-matching
                        # logic at all -- unlike the dedicated fill handler, it can't tell
                        # whether it just re-selected 'Married' or silently overwrote it
                        # with whatever was under the cursor. The run's own metrics can't
                        # even detect this: Value Accuracy only tracks typed text fields,
                        # never combobox selections. Unlike the already-attempted-EMPTY-
                        # combobox fix (execution_attempted_combobox_position_drift, which
                        # only ever wastes a step), this is a real correctness risk, so it's
                        # checked for ANY already-filled combobox, not just blank ones.
                        _reclick_elem = self._elem_at(state, _snap2)
                        _reclick_ty = (_reclick_elem.get("type") or "").lower() if _reclick_elem else ""
                        _reclick_key = (self._attempt_key(_reclick_elem, elements=state.get("elements", []))
                                        if _reclick_elem else None)
                        # Was: value-truthy AND in self._attempted_keys. Changed
                        # 2026-08-08 to self._typed_keys alone, same reasoning as
                        # the editcontrol case below (_reclick_edit_filled) --
                        # a live value read has proven unreliable (stale-empty)
                        # for at least one field this same night, and
                        # self._typed_keys is already scoped to genuine
                        # keyboard+text actions, so it doesn't need the value
                        # check to avoid the click-only-marks-attempted
                        # regression either.
                        _reclick_combobox_filled = (
                            _reclick_ty in ("comboboxcontrol", "combobox")
                            and _reclick_key in self._typed_keys
                        )
                        # Checkboxes extended in here 2026-08-07: even after the fill-click
                        # anchoring fix (execution_fill_click_anchored_to_focused_bbox), the
                        # NAVIGATE branch's own pointer can still independently drift back
                        # onto an already-checked checkbox's position on its own (a separate
                        # prediction, unrelated to the fill branch this guard was originally
                        # built for). Confirmed live: "Same problem still" -- 'Auto-Pay
                        # Enrolled' no longer re-entered the fill branch (that part of the
                        # bug is fixed), but the navigate branch's own clicks still landed on
                        # it 12 times in one run, each one caught downstream by the Win32
                        # BM_GETCHECK guard and converted to Tab -- not destructive, but a
                        # wasted click+Tab every time instead of Tabbing immediately. A
                        # checkbox has no "value" to check (see the checkbox-attempted note
                        # above _fe2_chk_attempted) -- attempted alone is enough here, unlike
                        # comboboxes which also require a non-empty value.
                        _reclick_checkbox_attempted = (
                            _reclick_ty in ("checkboxcontrol", "checkbox")
                            and _reclick_key in self._attempted_keys
                        )
                        # Plain text fields extended in here 2026-08-08, reported
                        # directly: "Wasted steps in Coverage due to loops" (traced
                        # to Vehicle -- the run hadn't reached Coverage yet, but
                        # same mechanism). Confirmed live: the navigate branch's
                        # pointer oscillated between TWO already-filled fields for
                        # 24+ steps straight -- 'Number of Doors' (a combobox,
                        # already caught by the guard above) and 'Model' (a plain
                        # editcontrol, NOT covered). Each individual re-click on
                        # 'Model' is cheap in isolation (a click + Tab, no LLM call,
                        # no data-corruption risk since a text-field click doesn't
                        # change its value) -- which is exactly why this case was
                        # deliberately left open in execution_payment_tab_
                        # oscillation_fix's round-3 writeup ("bounded by the same
                        # general navigation dynamics... not a special-cased loop
                        # anymore"). That reasoning covered a single stray click,
                        # not a genuine, repeating, multi-field OSCILLATION -- the
                        # cumulative cost across dozens of steps is real.
                        #
                        # Used to also require a non-empty live value read (on
                        # the theory that unlike a checkbox, a plain field's
                        # `value` accurately reflects real state). Found
                        # 2026-08-08, live, same run as the OPT2 fill-gate fix
                        # above: 'Years Continuously Insured' read back EMPTY
                        # from a live snapshot moments after being typed and
                        # verify-at-fill-confirmed -- the same stale-read
                        # issue, not unique to the fill-decision gate. This
                        # guard would have failed to catch a stray re-click on
                        # that exact field for the identical reason. Dropped
                        # the value check -- but checks self._typed_keys, NOT
                        # self._attempted_keys (unlike the checkbox case just
                        # above, where click IS the fill action so the general
                        # attempted set is correct). For an editcontrol, the
                        # general attempted set also includes fields merely
                        # clicked for navigation with no value ever typed;
                        # trusting it here would block a genuinely-empty field
                        # from ever being re-clicked/filled after one stray
                        # navigation click landed on it — see _typed_keys' own
                        # comment and execution_payment_tab_oscillation_fix's
                        # original regression.
                        _reclick_edit_filled = (
                            _reclick_ty in ("editcontrol", "input")
                            and _reclick_key in self._typed_keys
                        )
                        # Found live 2026-08-12: 831 of ~894 steps in one 40-minute
                        # run (93%) were the navigate pointer oscillating between two
                        # correctly-blank fields, 'Lienholder Address' and
                        # 'Lienholder/Lender' -- real fills stopped completely for
                        # the last ~34 minutes. Neither field was EVER in
                        # self._typed_keys (nothing gets typed into a field the
                        # record deliberately wants blank -- that's the whole point
                        # of self._leave_blank_keys, see its own comment), so
                        # _reclick_edit_filled above never fired for them, and they
                        # got zero protection from this guard at all -- unlike a
                        # merely-attempted-but-not-yet-typed field (which correctly
                        # stays reclickable, see _reclick_edit_filled's own
                        # reasoning above), a leave-blank decision IS final and
                        # deliberate, same category as a genuine typed value, not a
                        # stray navigation click. Not type-gated (editcontrol/
                        # comboboxcontrol/checkboxcontrol can all be left-blank) --
                        # self._leave_blank_keys is already the authoritative "this
                        # is done, nothing goes here" signal regardless of type.
                        _reclick_leave_blank = _reclick_key in self._leave_blank_keys
                        if (_reclick_combobox_filled or _reclick_checkbox_attempted
                                or _reclick_edit_filled or _reclick_leave_blank):
                            _reclick_label = (_reclick_elem.get("label") or _reclick_elem.get("text") or "?")[:30]
                            _reclick_reason = ("filled" if (_reclick_combobox_filled or _reclick_edit_filled)
                                               else "left-blank" if _reclick_leave_blank
                                               else "checked")
                            _reclick_streak += 1
                            # Found 2026-08-08, live, direct user report ("still not
                            # finding the right view"): the pointer aimed at THIS
                            # SAME already-filled field's screen position on 3
                            # consecutive steps before finally moving on -- a plain
                            # Tab each time gambles on OS focus-traversal order
                            # landing somewhere useful, and that gamble kept
                            # failing. Once the SAME stuck position recurs, stop
                            # gambling: redirect straight to a known-good target,
                            # the same deterministic fallback the low-confidence
                            # streak escalation already uses.
                            if _reclick_streak >= _RECLICK_REDIRECT_LIMIT:
                                # Found 2026-08-09, live, direct user report ("it's
                                # still revealing them one at a time"): this used to
                                # search only the CURRENT viewport
                                # (find_visible_empty_target) for a redirect target --
                                # but the actual action taken is _focus_element_via_uia
                                # (UIA SetFocus, not a click), and this wx form auto-
                                # scrolls whatever gets focused into view as its own
                                # native behavior (established earlier this session).
                                # That meant every redirect quietly revealed exactly
                                # ONE field via that side effect, before Navigation
                                # Protocol's own explicit "scroll to the densest
                                # cluster" logic ever got a chance to run -- decide()
                                # never saw cur==0 because the redirect kept finding
                                # SOMETHING nearby first. Switched to
                                # find_scroll_target_element so a redirect aims at the
                                # DEEPEST field in the next dense CLUSTER -- the same
                                # auto-scroll side effect then reveals the whole
                                # cluster at once instead of one field at a time.
                                # REAL viewport height, not unbounded -- the window
                                # width IS the density calculation; passing an
                                # effectively-infinite width degenerates it into "just
                                # the single deepest field on the whole tab," losing
                                # the "fits in one screen" density concept entirely.
                                #
                                # Found 2026-08-09, live, direct user report ("it
                                # skipped a lot of Policyholder tabs"): always
                                # reaching for the deepest field of the cluster
                                # overshot when a real, currently-visible, zero-
                                # scroll-needed target already existed. Log +
                                # source cross-check: 'Suffix' (known blank) sat
                                # right next to 'Marital Status'/'Occupation'/etc
                                # -- already on screen, no scroll required -- but
                                # the redirect reached past them for 'DL Number'
                                # (12 real fields further down: Marital Status,
                                # Occupation, Education Level, Credit Score,
                                # Email, Home/Cell/Work Phone, Street Address 1/2,
                                # City, State, ZIP, County, Country, Years at
                                # Address). SetFocus's native auto-scroll landed
                                # DL Number in view but scrolled PAST all twelve
                                # -- element count dropped 162->155, confirming
                                # things scrolled off the TOP, not just revealed
                                # below. Since this module only ever scrolls DOWN,
                                # never back up, those twelve fields were gone for
                                # the rest of the run -- real, silent data loss.
                                # Try the NEAREST already-visible target FIRST
                                # (zero scroll, zero overshoot risk by
                                # construction); only reach for the deep-cluster
                                # strategy when nothing is visible right now at
                                # all -- preserving the original 2026-08-08 fix
                                # (a redirect must not settle for revealing just
                                # one field when scrolling further would reveal a
                                # whole cluster) without letting it discard
                                # perfectly good, already-visible fields to get
                                # there.
                                _rc_target = self._navproto.find_visible_empty_target(
                                    state, self._form_viewport_bottom(state) - 8,
                                    attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                if _rc_target is None:
                                    _rc_target = self._navproto.find_scroll_target_element(
                                        state, self._form_viewport_bottom(state) - 8,
                                        attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                if _rc_target and _rc_target.get("bbox"):
                                    _rcb = _rc_target["bbox"]
                                    _rc_label = (_rc_target.get("label") or _rc_target.get("text") or "?")[:30]
                                    _rc_key = self._attempt_key(_rc_target, elements=state.get("elements", []))
                                    logger.info(
                                        "[OPT2] pointer stuck on already-%s %s %r %d times in a row "
                                        "— redirecting to known target %r instead of another blind Tab.",
                                        _reclick_reason, _reclick_ty, _reclick_label, _reclick_streak, _rc_label)
                                    # UIA SetFocus first -- no screen coordinates, no
                                    # "semantic positioning" guess about where this
                                    # visually sits (see _focus_element_via_uia's own
                                    # docstring for why). Falls back to the click only if
                                    # UIA can't find/focus the control at all.
                                    _rc_full_label = (_rc_target.get("label") or _rc_target.get("text") or "").strip()
                                    _rc_pos = [(_rcb[0] + _rcb[2]) / 2, (_rcb[1] + _rcb[3]) / 2]
                                    # _find_destructive_button_at guard (2026-08-13) -- see
                                    # the checkbox-redirect fix elsewhere in this loop for
                                    # the full incident writeup; same risk shape, same fix.
                                    if (not self._focus_element_via_uia(_rc_full_label, expected_bbox=_rcb)
                                            and not _find_destructive_button_at(state.get("elements", []), _rc_pos)):
                                        self._executor.execute({
                                            "action_type": "click",
                                            "click_position": _rc_pos,
                                        })
                                    self._adaptive_settle_wait(self.step_delay * 0.4)
                                    _reclick_streak = 0
                                    # VERIFY the redirect actually moved focus, don't
                                    # just assume it worked and `continue` blind.
                                    #
                                    # Found 2026-08-08, live, direct user report ("It's
                                    # not working... it's using Tab to fucking
                                    # navigate"): a run got stuck oscillating between
                                    # this redirect and a plain Tab for 25+ consecutive
                                    # steps, zero new fields filled. The transformer's
                                    # own raw pointer prediction never changed step to
                                    # step (frozen at the same coordinate) -- if the
                                    # redirect click HAD actually focused the target,
                                    # the next step's own auto-fill fast path (which
                                    # checks state["focused_element_id"], independent
                                    # of the transformer's pointer entirely) should
                                    # have fired immediately. It never did across 13
                                    # consecutive attempts, meaning the click wasn't
                                    # actually landing focus on the target -- the same
                                    # "assume it worked" mistake this project has hit
                                    # and fixed multiple times already (verify-at-fill,
                                    # the scroll branch), just in a new spot.
                                    _rc_check = self._observe()
                                    _rc_landed = self._attempt_key(
                                        next((e for e in _rc_check.get("elements", [])
                                              if e.get("element_id") == _rc_check.get("focused_element_id")), {}),
                                        elements=_rc_check.get("elements", []),
                                    ) == _rc_key
                                    if _rc_landed:
                                        _redirect_stall_count = 0
                                        # Found 2026-08-09, live, direct user
                                        # report ("revealing one input field
                                        # one-by-one again"): the SAME night's
                                        # earlier "prefer an already-visible
                                        # target" fix means this redirect now
                                        # usually lands via find_visible_empty_
                                        # target (nearest reachable field) --
                                        # but "reachable per the geometric
                                        # viewport estimate" isn't the same as
                                        # "genuinely on screen already." If
                                        # landing here STILL grew the element
                                        # count, the near target wasn't
                                        # actually risk-free -- it required a
                                        # real reveal, same as the deep-aim
                                        # strategy would have needed, just a
                                        # smaller, incomplete one. Log
                                        # evidence: element count crept
                                        # 162->169->170 across separate
                                        # redirects, each one exploiting only
                                        # a sliver of whatever got revealed,
                                        # with zero "revealed cluster via" or
                                        # "Scroll-form" lines anywhere in the
                                        # run -- the deliberate big-reveal
                                        # strategy essentially never fired.
                                        # When a reveal DID happen, finish
                                        # what it started: aim the SAME
                                        # deep-cluster search at the NEW state
                                        # once, completing the reveal instead
                                        # of leaving it half-done.
                                        _n_before_land = len(state.get("elements", []))
                                        state = _rc_check
                                        if len(state.get("elements", [])) > _n_before_land:
                                            _rc_deep = self._navproto.find_scroll_target_element(
                                                state, self._form_viewport_bottom(state) - 8,
                                                attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                            if _rc_deep and _rc_deep.get("bbox"):
                                                _rc_deep_key = self._attempt_key(_rc_deep, elements=state.get("elements", []))
                                                _rc_deep_label = (_rc_deep.get("label") or _rc_deep.get("text") or "").strip()
                                                if _rc_deep_key != _rc_key and _rc_deep_label and self._focus_element_via_uia(
                                                        _rc_deep_label, expected_bbox=_rc_deep["bbox"]):
                                                    logger.info(
                                                        "[OPT2] %r's landing already revealed more content — "
                                                        "completing the reveal via %r before refocusing shallow.",
                                                        _rc_label, _rc_deep_label[:30])
                                                    state = self._observe()
                                        # Found 2026-08-09, live, direct user
                                        # report ("Marital Status was not
                                        # even filled up until Cell Phone" /
                                        # "it navigated away after filling
                                        # SSN"): landing SetFocus on the
                                        # DEEPEST field of a cluster (to
                                        # reveal the whole cluster via its
                                        # auto-scroll side effect) also makes
                                        # THAT deepest field the next thing
                                        # that gets filled -- the auto-fill
                                        # fast path always acts on whatever is
                                        # currently focused. Every shallower
                                        # field in the SAME newly-revealed
                                        # cluster (Marital Status, Occupation,
                                        # Education Level, Credit Score) got
                                        # silently skipped, since none of THEM
                                        # ever became focused -- only the deep
                                        # target used to trigger the reveal
                                        # did. Re-focus onto the SHALLOWEST
                                        # now-visible unattempted field before
                                        # continuing, so revealing the cluster
                                        # and filling it happen on different
                                        # fields -- top to bottom, not
                                        # bottom-first.
                                        _shallow_target = self._navproto.find_visible_empty_target(
                                            state, self._form_viewport_bottom(state) - 8,
                                            attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                        if _shallow_target and _shallow_target.get("bbox"):
                                            _shallow_key = self._attempt_key(
                                                _shallow_target, elements=state.get("elements", []))
                                            if _shallow_key != _rc_key:
                                                _shallow_label = (_shallow_target.get("label")
                                                                   or _shallow_target.get("text") or "").strip()
                                                _shallow_bbox = _shallow_target["bbox"]
                                                if _shallow_label and self._focus_element_via_uia(
                                                        _shallow_label, expected_bbox=_shallow_bbox):
                                                    logger.info(
                                                        "[OPT2] revealed cluster via %r — refocusing shallower "
                                                        "target %r to fill top-down.", _rc_label, _shallow_label[:30])
                                                    state = self._observe()
                                        continue
                                    _redirect_stall_count += 1
                                    logger.warning(
                                        "[OPT2] redirect click on %r did NOT move focus there (stall %d/%d) "
                                        "— the click isn't sticking, not just a stray pointer guess.",
                                        _rc_label, _redirect_stall_count, _REDIRECT_STALL_LIMIT)
                                    if _redirect_stall_count >= _REDIRECT_STALL_LIMIT:
                                        # Found 2026-08-08, live, direct user report ("What
                                        # the fuck was that why did it go to Vehicle?"): this
                                        # used to blame the WHOLE TAB and advance past it the
                                        # instant ONE target failed to stick twice -- but the
                                        # failure was specific to 'Street Address 1' (its
                                        # exact click coordinates, twice), not evidence the
                                        # tab itself was exhausted. A dozen other genuinely
                                        # fillable Policyholder fields (City, State, ZIP,
                                        # County, DL info, ...) never even got tried before
                                        # the tab got abandoned -- twice in a row, compounding
                                        # into a 2-tab jump straight to Vehicle. Fixed: mark
                                        # ONLY the specific unreachable target attempted (so
                                        # it stops getting re-offered) and look for a
                                        # DIFFERENT target before concluding the tab itself is
                                        # exhausted. Known, accepted trade-off: the unreachable
                                        # field itself will not get filled this run -- better
                                        # than either looping on it forever or discarding
                                        # every other real field alongside it.
                                        #
                                        # Found 2026-08-08, live, SAME night, immediately
                                        # after shipping the fix above: it fired exactly as
                                        # designed ("'Street Address 1' unreachable and no
                                        # other target visible either -- advancing") and
                                        # STILL jumped past the tab wrongly. Why: the search
                                        # below was viewport-limited (_form_viewport_bottom),
                                        # so it only asked "is anything else visible RIGHT
                                        # NOW" -- and at that exact moment, nothing else
                                        # happened to be on screen yet. The very next scroll
                                        # immediately revealed 'Street Address 2' sitting just
                                        # below the fold, which then failed the SAME way,
                                        # compounding into the reported jump straight to
                                        # Vehicle. Search the WHOLE tab (1e9, unbounded --
                                        # same pattern already used by the SCROLL-DIAG
                                        # tracking above), not just the current viewport: if
                                        # something exists anywhere on this tab, even
                                        # off-screen, `continue` and let Navigation Protocol's
                                        # own decide() scroll to it naturally next iteration --
                                        # only advance when truly nothing is left anywhere.
                                        _redirect_stall_count = 0
                                        self._attempted_keys.add(_rc_key)
                                        _alt_target = self._navproto.find_visible_empty_target(
                                            _rc_check, 1e9, attempted_keys=self._attempted_keys,
                                            attempt_key_fn=self._attempt_key)
                                        if _alt_target is None:
                                            logger.warning(
                                                "[OPT2] %r unreachable and nothing else remains anywhere on "
                                                "this tab (checked on- and off-screen) — advancing.", _rc_label)
                                            if self._try_advance_tab(state):
                                                _tab_just_switched = True
                                                _tab_scroll_count  = 0
                                                _last_auto_step    = step_idx
                                                self._refresh_record_cache(self._observe())
                                                time.sleep(self.step_delay)
                                        else:
                                            logger.info(
                                                "[OPT2] %r unreachable, but %r is still visible and "
                                                "reachable — staying on this tab instead of advancing.",
                                                _rc_label,
                                                (_alt_target.get("label") or _alt_target.get("text") or "?")[:30])
                                    continue
                            logger.info(
                                "[OPT2] pointer drifted back onto already-%s %s %r — Tab instead "
                                "of a wasted re-click.",
                                _reclick_reason, _reclick_ty, _reclick_label)
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            self._adaptive_settle_wait(self.step_delay * 0.4)
                            continue
                        # COMBOBOX-AS-FILL: demos action comboboxes as CLICKS, so the
                        # model clicks them; but a plain click only toggles the
                        # dropdown → open/close oscillation. A click on an EMPTY
                        # combobox = intent to SET its value → route to FILL (focus it,
                        # then the LLM value + open/select handler below).
                        _cbox = None
                        if not self._pure_transformer and self._llm_client:
                            _cbox = next(
                                (e for e in state.get("elements", [])
                                 if e.get("type") == "comboboxcontrol"
                                 and not (e.get("value") or "").strip()
                                 and e.get("bbox")
                                 and e["bbox"][0] - 2 <= _snap2[0] <= e["bbox"][2] + 2
                                 and e["bbox"][1] - 2 <= _snap2[1] <= e["bbox"][3] + 2),
                                None)
                        if _cbox is not None and self._attempt_key(
                                _cbox, elements=state.get("elements", [])) in self._leave_blank_keys:
                            # Already confirmed blank earlier this record — don't repeat
                            # the click/check/escape dance. Found live 2026-08-07: the
                            # position-based dead-end blacklist (below) only catches 3
                            # identical clicks in a row at the SAME bucketed position;
                            # the transformer's click-position estimate for the same
                            # field wobbles enough between separate approaches (drifting
                            # back to it minutes apart, not consecutively) to dodge that
                            # 10px bucket each time. 'Suffix' got the full click-open-
                            # check-escape-tab treatment 4 separate times in one
                            # Policyholder pass (10 wasted steps) despite being
                            # _mark_attempted()'d after the first. label-based (stable
                            # regardless of pixel drift), so check it FIRST and skip
                            # straight to Tab — no click needed, there's nothing to open
                            # or verify about a field already known blank.
                            #
                            # Found 2026-08-09, live, direct user report ("we're
                            # improving"): ground-truth check against
                            # data_entry_intake.txt showed 'Primary Use' has a real
                            # value ('Commute') that never got filled -- a bare
                            # low-confidence-fallback navigate click had landed on it
                            # earlier (no lookup ever attempted, just passing through),
                            # which still marked it in the GENERAL self._attempted_keys
                            # via _record_attempt. This check used to read that SAME
                            # general set, concluding "confirmed blank" for a field
                            # that was never actually looked up. Switched to
                            # self._leave_blank_keys — populated ONLY when
                            # self._lookup_field was genuinely called and genuinely
                            # found nothing (a few hundred lines below, the "combobox
                            # %r — no value, Tab" branch) — the same distinction
                            # self._leave_blank_keys already exists to make elsewhere
                            # in this file (see its own comment), just not yet applied
                            # here. A field merely touched by a stray click now
                            # correctly still gets a real fill attempt.
                            _cb_label_skip = (_cbox.get("label") or _cbox.get("text") or "").strip()
                            # Found 2026-08-08, live, direct user report ("It's using Tab
                            # to navigate. Its not finding the actual optimal view."):
                            # this branch had NO streak tracking at all -- unlike the
                            # sibling "already-filled" reclick guard below (which
                            # escalates to a verified redirect after 2 repeats), this one
                            # just Tabbed every single time, forever, with no memory of
                            # how many times it had just recurred. Log evidence: 'Suffix'
                            # hit this exact branch 3 separate times in a row (steps
                            # 11-13), then again twice more later in the same run --
                            # moderate transformer confidence (0.72+, not low enough to
                            # trip the separate low-confidence fallback), just repeatedly
                            # re-guessing a coordinate it had already been told is a dead
                            # end. Reuses the SAME _reclick_streak counter and verified-
                            # redirect mechanism as the already-filled guard -- one
                            # unified "stop re-visiting a dead field" escalation instead
                            # of two separate, inconsistently-capable copies of it.
                            _reclick_streak += 1
                            if _reclick_streak >= _RECLICK_REDIRECT_LIMIT:
                                # Same fix as the sibling guard above: aim at the
                                # deepest field of the densest cluster (REAL viewport
                                # height, not unbounded -- see that guard's comment for
                                # why an infinite window width breaks the density
                                # calculation) instead of the nearest single field, so
                                # this redirect's own SetFocus auto-scrolls the whole
                                # next cluster into view.
                                #
                                # Found 2026-08-09, live, direct user report ("it
                                # skipped a lot of Policyholder tabs") -- this is
                                # the EXACT incident: this guard fired on 'Suffix'
                                # (correctly known blank), and reached past
                                # 'Marital Status'/'Occupation'/etc -- already
                                # visible, zero scroll needed -- for 'DL Number'
                                # (12 real fields further down). Same fix as the
                                # sibling guard above: try the nearest already-
                                # visible target first (zero overshoot risk by
                                # construction); only reach for the deep-cluster
                                # strategy when nothing is visible right now.
                                _cb_rc_target = self._navproto.find_visible_empty_target(
                                    state, self._form_viewport_bottom(state) - 8,
                                    attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                if _cb_rc_target is None:
                                    _cb_rc_target = self._navproto.find_scroll_target_element(
                                        state, self._form_viewport_bottom(state) - 8,
                                        attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                if _cb_rc_target and _cb_rc_target.get("bbox"):
                                    _cbrcb = _cb_rc_target["bbox"]
                                    _cb_rc_label = (_cb_rc_target.get("label") or _cb_rc_target.get("text") or "?")[:30]
                                    _cb_rc_key = self._attempt_key(_cb_rc_target, elements=state.get("elements", []))
                                    logger.info(
                                        "[OPT2] combobox %r already attempted (known blank) %d times in a "
                                        "row — redirecting to known target %r instead of another blind Tab.",
                                        _cb_label_skip[:30], _reclick_streak, _cb_rc_label)
                                    # Same UIA SetFocus-first fix as the sibling guard above.
                                    _cb_rc_full_label = (_cb_rc_target.get("label") or _cb_rc_target.get("text") or "").strip()
                                    _cb_rc_pos = [(_cbrcb[0] + _cbrcb[2]) / 2, (_cbrcb[1] + _cbrcb[3]) / 2]
                                    # _find_destructive_button_at guard (2026-08-13) -- same
                                    # risk shape as every other find_visible_empty_target
                                    # redirect that falls through to a raw coordinate click:
                                    # the target could be ANY on-screen element, including a
                                    # button. Only relevant if the UIA-focus-by-name attempt
                                    # above fails and this falls through to the raw click.
                                    if (not self._focus_element_via_uia(_cb_rc_full_label, expected_bbox=_cbrcb)
                                            and not _find_destructive_button_at(state.get("elements", []), _cb_rc_pos)):
                                        self._executor.execute({
                                            "action_type": "click",
                                            "click_position": _cb_rc_pos,
                                        })
                                    self._adaptive_settle_wait(self.step_delay * 0.4)
                                    _reclick_streak = 0
                                    # Same verify-don't-assume fix as the sibling guard.
                                    _cb_rc_check = self._observe()
                                    _cb_rc_landed = self._attempt_key(
                                        next((e for e in _cb_rc_check.get("elements", [])
                                              if e.get("element_id") == _cb_rc_check.get("focused_element_id")), {}),
                                        elements=_cb_rc_check.get("elements", []),
                                    ) == _cb_rc_key
                                    if _cb_rc_landed:
                                        _redirect_stall_count = 0
                                        # Same "complete an accidental partial
                                        # reveal" fix as the sibling guard
                                        # above -- found 2026-08-09, live,
                                        # direct user report ("revealing one
                                        # input field one-by-one again").
                                        _cb_n_before_land = len(state.get("elements", []))
                                        state = _cb_rc_check
                                        if len(state.get("elements", [])) > _cb_n_before_land:
                                            _cb_deep = self._navproto.find_scroll_target_element(
                                                state, self._form_viewport_bottom(state) - 8,
                                                attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                            if _cb_deep and _cb_deep.get("bbox"):
                                                _cb_deep_key = self._attempt_key(_cb_deep, elements=state.get("elements", []))
                                                _cb_deep_label = (_cb_deep.get("label") or _cb_deep.get("text") or "").strip()
                                                if _cb_deep_key != _cb_rc_key and _cb_deep_label and self._focus_element_via_uia(
                                                        _cb_deep_label, expected_bbox=_cb_deep["bbox"]):
                                                    logger.info(
                                                        "[OPT2] %r's landing already revealed more content — "
                                                        "completing the reveal via %r before refocusing shallow.",
                                                        _cb_rc_label, _cb_deep_label[:30])
                                                    state = self._observe()
                                        # Same "refocus shallowest before
                                        # filling" fix as the sibling guard
                                        # above.
                                        _cb_shallow_target = self._navproto.find_visible_empty_target(
                                            state, self._form_viewport_bottom(state) - 8,
                                            attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                        if _cb_shallow_target and _cb_shallow_target.get("bbox"):
                                            _cb_shallow_key = self._attempt_key(
                                                _cb_shallow_target, elements=state.get("elements", []))
                                            if _cb_shallow_key != _cb_rc_key:
                                                _cb_shallow_label = (_cb_shallow_target.get("label")
                                                                      or _cb_shallow_target.get("text") or "").strip()
                                                _cb_shallow_bbox = _cb_shallow_target["bbox"]
                                                if _cb_shallow_label and self._focus_element_via_uia(
                                                        _cb_shallow_label, expected_bbox=_cb_shallow_bbox):
                                                    logger.info(
                                                        "[OPT2] revealed cluster via %r — refocusing shallower "
                                                        "target %r to fill top-down.",
                                                        _cb_rc_label, _cb_shallow_label[:30])
                                                    state = self._observe()
                                        continue
                                    _redirect_stall_count += 1
                                    if _redirect_stall_count >= _REDIRECT_STALL_LIMIT:
                                        # Same fix as the sibling guard above: don't blame the
                                        # whole tab for one unreachable target, try a
                                        # different one first.
                                        _redirect_stall_count = 0
                                        self._attempted_keys.add(_cb_rc_key)
                                        _nav_vb_cbalt = self._form_viewport_bottom(_cb_rc_check) - 8
                                        _cb_alt_target = self._navproto.find_visible_empty_target(
                                            _cb_rc_check, _nav_vb_cbalt, attempted_keys=self._attempted_keys,
                                            attempt_key_fn=self._attempt_key)
                                        if _cb_alt_target is None:
                                            logger.warning(
                                                "[OPT2] %r unreachable and no other target visible either "
                                                "— this tab's remaining content isn't reachable right now, "
                                                "advancing.", _cb_rc_label)
                                            if self._try_advance_tab(state):
                                                _tab_just_switched = True
                                                _tab_scroll_count  = 0
                                                _last_auto_step    = step_idx
                                                self._refresh_record_cache(self._observe())
                                                time.sleep(self.step_delay)
                                        else:
                                            logger.info(
                                                "[OPT2] %r unreachable, but %r is still visible and "
                                                "reachable — staying on this tab instead of advancing.",
                                                _cb_rc_label,
                                                (_cb_alt_target.get("label") or _cb_alt_target.get("text") or "?")[:30])
                                    continue
                            logger.info("[OPT2] combobox %r already attempted (known blank) — Tab, no re-click.",
                                        _cb_label_skip[:30])
                            self._executor.execute({"action_type": "keyboard",
                                                    "key_count": 1, "keystrokes": ["tab"]})
                            self._adaptive_settle_wait(self.step_delay * 0.5)
                            continue
                        if _cbox is not None:
                            _cb_label = (_cbox.get("label") or _cbox.get("text") or "").strip()
                            # Found 2026-08-09, live, direct user report ("you were
                            # doing so well until the Clear All modal"): this branch
                            # clicks _snap2 directly -- the SAME raw
                            # transformer-position variable the sibling navigate
                            # branch (a few hundred lines below) already guards
                            # against landing on a footer action button, via
                            # _find_destructive_button_at. That guard was only ever
                            # wired into the navigate branch; this branch reuses the
                            # identical variable but never got the same check. Log
                            # evidence: right after a failed 'Payment Frequency'
                            # dropdown-open click (a field that sits near the bottom
                            # of its tab, close to the fixed footer row), a 'Clear
                            # All' confirmation dialog (class '#32770', the real
                            # Windows dialog class -- not a false positive) turned
                            # up open on the very next step, with no other click in
                            # between. Same generalization already applied once
                            # (_find_destructive_button_at matches ANY button-type
                            # element at the position, not a keyword list) — just
                            # extended to the other call site sharing the same risky
                            # input, instead of only the one already caught live.
                            # Found 2026-08-09, live, direct user report ("It closed
                            # the fucking Terminal", TWICE, deterministically
                            # reproduced across 3 separate runs -- always at the same
                            # field, 'Payment Frequency', always right after this
                            # exact unconditional click and nothing else). Traced end
                            # to end: the earlier alt+f4-prompt/hotkey-allowlist fix
                            # addressed a real but DIFFERENT vulnerability -- this
                            # crash has no keyboard action anywhere near it, it's a
                            # bare CLICK. The sibling navigate-click branch (a few
                            # hundred lines below) already refuses to click outside
                            # the locked form window via self._point_in_form -- this
                            # combobox-open branch reuses the SAME _snap2 target but
                            # never got that check, only _find_destructive_button_at
                            # (which only catches known BUTTON elements inside the
                            # form's own accessibility tree -- it has no idea a point
                            # lands in a completely different window, like the
                            # terminal, because that window's controls were never
                            # scanned into state['elements'] at all). Same class of
                            # gap _find_destructive_button_at itself was generalized
                            # for once already -- one more copy of the same missing
                            # guard, not a new mechanism.
                            if not self._point_in_form(_snap2, state):
                                logger.warning(
                                    "[GUARD] combobox-open target (%.0f,%.0f) OUTSIDE form window — "
                                    "Tab instead of drifting into another window.", _snap2[0], _snap2[1])
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                time.sleep(self.step_delay * 0.5)
                                continue
                            _cb_open_btn = _find_destructive_button_at(state.get("elements", []), _snap2)
                            if _cb_open_btn is not None:
                                _cb_btn_name = (_cb_open_btn.get("text") or _cb_open_btn.get("label") or "button")
                                logger.warning(
                                    "[GUARD] combobox-open target (%.0f,%.0f) lands on button %r — "
                                    "Tab instead of clicking an action button.", _snap2[0], _snap2[1], _cb_btn_name)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                time.sleep(self.step_delay * 0.5)
                                continue
                            logger.info("[OPT2] CLICK on empty combobox %r → treat as FILL", _cb_label[:30])
                            _cb_sec = self._detect_section(state, _cbox)
                            _cb_val = self._lookup_field(_cb_label, section=_cb_sec)
                            # ONE click opens the dropdown; select inline + continue.
                            # (Don't route to the type-handler — its own open-click would
                            #  TOGGLE the dropdown shut → no options → Escape loop.)
                            #
                            # Skip the click entirely if a dropdown is ALREADY open —
                            # see _open_dropdown_items' docstring for why (a click
                            # toggles, it doesn't just open).
                            if not _open_dropdown_items(state.get("elements", [])):
                                # Target the arrow button, not the center -- see
                                # _combobox_dropdown_arrow_pos's own docstring.
                                _cb_arrow_pos = (self._combobox_dropdown_arrow_pos(_cbox["bbox"])
                                                 if _cbox and _cbox.get("bbox") else _snap2)
                                self._executor.execute({"action_type": "click", "click_position": _cb_arrow_pos})
                            if not _cb_val:
                                # Optional field with no record value (e.g. Suffix).
                                # Escape + Tab past it, and MARK it attempted so the
                                # transformer stops re-targeting it (the real fix is the
                                # 'attempted' state-feature; this just records the hit).
                                logger.info("[OPT2] combobox %r — no value, Tab", _cb_label)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["escape"]})
                                self._mark_attempted(_cbox, elements=state.get("elements", []))
                                # Found 2026-08-09, live, direct user report ("we're
                                # improving") -- ground-truth check against
                                # data_entry_intake.txt showed 'Primary Use' has a
                                # real value ('Commute') that never got filled.
                                # Traced it: a bare low-confidence-fallback
                                # navigate click landed on 'Primary Use' earlier
                                # (no lookup ever attempted -- just passing
                                # through), which still marked it in the GENERAL
                                # self._attempted_keys via _record_attempt. The
                                # sibling "combobox already attempted (known
                                # blank)" skip-check a few hundred lines above
                                # then read that SAME general set and concluded
                                # "confirmed blank," permanently skipping a field
                                # that was never actually looked up. _cb_val is
                                # empty HERE specifically because self._lookup_field
                                # was genuinely called and genuinely found nothing
                                # -- the real "confirmed blank" signal, distinct
                                # from _attempted_keys for exactly the same reason
                                # self._leave_blank_keys already exists (see its
                                # own comment a few hundred lines above). Reusing
                                # that established set instead of inventing a new
                                # one, and pointing the skip-check at it.
                                self._leave_blank_keys.add(
                                    self._attempt_key(_cbox, elements=state.get("elements", [])))
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                # Fingerprint this exact outcome (position, not just
                                # label — same defense the general repeat-guard below
                                # uses) so a stuck loop is detected even though this
                                # branch `continue`s past that guard's own check.
                                # Found live 2026-08-07: attempted-marking alone didn't
                                # stop the transformer's own high-confidence (0.91)
                                # pointer from clicking straight back onto this same
                                # empty combobox 30+ times — Escape+Tab correctly
                                # recognized "nothing to fill" every time but never
                                # actually broke the cycle. Once this exact position
                                # repeats _REPEAT_LIMIT times, blacklist it so the
                                # confidence gate rejects it outright next time,
                                # regardless of how confident the model is.
                                _cb_fp = ("combobox_empty", _round_click_pos(_snap2))
                                _action_history.append(_cb_fp)
                                if list(_action_history)[-_REPEAT_LIMIT:] == [_cb_fp] * _REPEAT_LIMIT:
                                    logger.warning(
                                        "Repeat-action guard: %r stuck empty %dx in a row — blacklisting @ (%.0f,%.0f).",
                                        _cb_label, _REPEAT_LIMIT, _snap2[0], _snap2[1])
                                    self._nochange_click_pos.add(_round_click_pos(_snap2))
                                    _action_history.clear()
                                time.sleep(self.step_delay * 0.5)
                                continue
                            # Standard UIA "ListItem" maps to "listitem" (no "control"
                            # suffix) in ui_observer.py's _CTRL_TYPE_MAP.
                            # Poll count/interval: found live 2026-08-07 — 'Sedan' (a
                            # real, correct option) still came up with 0 list items
                            # found after the fix above, in the SAME run where other
                            # dropdowns (Make, Policy Type, ...) succeeded moments
                            # apart. Not a matching bug — the popup just hadn't
                            # finished rendering within the old 4x0.35s=1.4s window
                            # every time. Widened to 8x0.4s=3.2s to give slow renders
                            # more room without meaningfully slowing the common fast
                            # case (the loop still exits the instant items appear).
                            #
                            # Widened to 12x0.5s=6.0s on 2026-08-10, then REVERTED
                            # back to 8x0.4s=3.2s the same day once live evidence
                            # proved that theory wrong: a Play-launched run showed
                            # 'Policy Term' and 'Policy Status' BOTH still "empty
                            # after 12 tries (6.0s)" — the dropdown wasn't rendering
                            # LATE, it wasn't rendering AT ALL, so the extra 2.8s of
                            # patience bought nothing except a slower give-up on a
                            # failure that was never going to resolve by waiting
                            # longer. Direct user report right after ("still so
                            # fucking slow") confirmed the widening made things
                            # worse, not better. The REAL cause of the dropdown
                            # never rendering is still open (see
                            # execution_combobox_value_reverts_after_fill and this
                            # bug class's own history above) — not something more
                            # patience fixes, so this constant went back to the
                            # value that's actually been verified NOT to make
                            # genuine failures slower for no benefit.
                            _POLL_TRIES, _POLL_INTERVAL = 8, 0.4
                            _items = []
                            for _try in range(_POLL_TRIES):
                                time.sleep(_POLL_INTERVAL)
                                _cs = self._observe()
                                _items = [e for e in _cs.get("elements", [])
                                          if e.get("type") in ("listitem", "listitemcontrol")
                                          and e.get("window_role") != "background" and e.get("bbox")]
                                if _items:
                                    break
                            if not _items:
                                logger.warning(
                                    "Combobox: dropdown for %r still empty after %d tries (%.1fs) — giving up.",
                                    _cb_label, _POLL_TRIES, _POLL_TRIES * _POLL_INTERVAL)
                            _o = lambda e: (e.get("text") or e.get("label") or "").strip()
                            _hit = next((e for e in _items if _option_matches(_cb_val, _o(e))), None)
                            # No dropdown ever rendered at all (not "rendered late" --
                            # confirmed live this really can happen for some real,
                            # working comboboxes, see _select_combobox_value_via_
                            # keyboard's own docstring). Try real keyboard navigation
                            # before falling into the escalate-and-give-up path below.
                            _kb_filled = False
                            if not _hit and not _items:
                                _kb_filled = self._select_combobox_value_via_keyboard(
                                    _cb_label, _cb_val, _snap2[0], _snap2[1])
                            if _hit or _kb_filled:
                                if _hit:
                                    _b = _hit["bbox"]
                                    self._executor.execute({"action_type": "click",
                                                            "click_position": [(_b[0]+_b[2])/2, (_b[1]+_b[3])/2]})
                                logger.info("Combobox(click-fill): %r → selected %r", _cb_label, _cb_val)
                                time.sleep(0.25)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                self._filled_this_tab.add(_cb_label)
                                _combobox_dropdown_fail_counts.pop(_cb_label, None)
                            else:
                                if _items:
                                    logger.warning("Combobox(click-fill): %r not in options %s",
                                                   _cb_val, [_o(e) for e in _items][:12])
                                else:
                                    logger.warning("Combobox(click-fill): dropdown for %r did not render", _cb_label)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["escape"]})
                                # Found 2026-08-09, live, direct user report
                                # ("it didn't want to navigate"): with no
                                # streak tracking here, this branch retried
                                # the SAME stuck combobox forever -- 121
                                # times across ~4 minutes for 'Payment
                                # Frequency' in one run, zero progress, only
                                # stopped when the user gave up. Escalate
                                # after repeated failures on the SAME field
                                # instead of retrying identically forever.
                                _combobox_dropdown_fail_counts[_cb_label] = (
                                    _combobox_dropdown_fail_counts.get(_cb_label, 0) + 1)
                                if _combobox_dropdown_fail_counts[_cb_label] >= _COMBOBOX_DROPDOWN_FAIL_LIMIT:
                                    logger.warning(
                                        "[OPT2] %r failed to fill %d times in a row (dropdown never rendered "
                                        "or option never matched) — marking attempted and moving on.",
                                        _cb_label, _combobox_dropdown_fail_counts[_cb_label])
                                    _combobox_dropdown_fail_counts.pop(_cb_label, None)
                                    self._mark_attempted(_cbox, elements=state.get("elements", []))
                                    # Same reasoning as the type-into-combobox
                                    # sibling branch's own 2026-08-12 fix: a
                                    # 2-strike give-up is a final, deliberate
                                    # decision, same category as leave-blank --
                                    # reusing that set lets the reclick guard
                                    # protect this field from the transformer's
                                    # own navigate-click pointer drifting back
                                    # onto it later, not just this immediate visit.
                                    self._leave_blank_keys.add(
                                        self._attempt_key(_cbox, elements=state.get("elements", [])))
                                    _cd_target = self._navproto.find_visible_empty_target(
                                        state, self._form_viewport_bottom(state) - 8,
                                        attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                    if _cd_target and _cd_target.get("bbox"):
                                        _cd_label = (_cd_target.get("label") or _cd_target.get("text") or "").strip()
                                        _cdb = _cd_target["bbox"]
                                        _cd_pos = [(_cdb[0] + _cdb[2]) / 2, (_cdb[1] + _cdb[3]) / 2]
                                        # _find_destructive_button_at guard (2026-08-13) -- see
                                        # the same-shaped fix above this record's own checkbox
                                        # redirect for the full incident writeup.
                                        if (not (_cd_label and self._focus_element_via_uia(_cd_label, expected_bbox=_cdb))
                                                and not _find_destructive_button_at(state.get("elements", []), _cd_pos)):
                                            self._executor.execute({
                                                "action_type": "click",
                                                "click_position": _cd_pos,
                                            })
                            time.sleep(self.step_delay * 0.5)
                            continue
                        else:
                            # VISITED-ADVANCE crutch (gated off in pure/no-autohandler mode)
                            if not self._no_autohandlers:
                                _vis = getattr(self, "_visited_pos", None)
                                if _vis is None:
                                    _vis = self._visited_pos = set()
                                _vk = (round(_snap2[0] / 15) * 15, round(_snap2[1] / 15) * 15)
                                if _vk in _vis:
                                    _cands = sorted(
                                        [e for e in state.get("elements", [])
                                         if e.get("window_role") != "background"
                                         and e.get("type") in ("editcontrol", "input",
                                                               "comboboxcontrol", "combobox")
                                         and e.get("bbox")],
                                        key=lambda e: (e["bbox"][1], e["bbox"][0]))
                                    for _e in _cands:
                                        _b = _e["bbox"]
                                        _cx, _cy = (_b[0] + _b[2]) / 2, (_b[1] + _b[3]) / 2
                                        _ck = (round(_cx / 15) * 15, round(_cy / 15) * 15)
                                        if _ck not in _vis:
                                            _snap2 = [_cx, _cy]; _vk = _ck
                                            logger.info("[VISITED-ADVANCE] pointer repeated — next unvisited field @ (%.0f,%.0f)", _cx, _cy)
                                            break
                                _vis.add(_vk)
                            if not self._point_in_form(_snap2, state):
                                # Hallucinated target outside the form (e.g. (114,72)
                                # into another window, or below the viewport). Don't
                                # drift — Tab instead (stays in-form; wx auto-scrolls
                                # the focused field into view).
                                logger.warning("[GUARD] target (%.0f,%.0f) OUTSIDE form window — Tab instead of drifting.",
                                               _snap2[0], _snap2[1])
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                self._adaptive_settle_wait(self.step_delay * 0.5)
                                continue
                            # Found 2026-08-08, live: this branch's low-confidence
                            # navigate-pointer landed on a real 'Clear All' button
                            # (ptr_conf=0.53 -- unremarkable, no different from
                            # dozens of other navigation clicks), popping a genuine
                            # Windows confirmation dialog that then blocked the rest
                            # of the run. First fix only matched destructive-sounding
                            # keywords; direct pushback ("it's not just the Clear
                            # All") plus reading the form's OWN footer button list
                            # found 'Load Record' (silently overwrites every field
                            # from a file) and 'Print Preview' (its own separate
                            # modal) are just as capable of derailing a run and
                            # match no destructive keyword at all. This branch never
                            # has a legitimate reason to click ANY button while
                            # navigating between fields — see
                            # _find_destructive_button_at's own comment for why a
                            # blanket refusal generalizes where a keyword list can't.
                            _navigate_btn = _find_destructive_button_at(
                                state.get("elements", []), _snap2)
                            if _navigate_btn is not None:
                                _nb_name = (_navigate_btn.get("text")
                                            or _navigate_btn.get("label") or "button")
                                logger.warning(
                                    "[GUARD] pointer target (%.0f,%.0f) lands on button "
                                    "%r — Tab instead of clicking an action button while navigating.",
                                    _snap2[0], _snap2[1], _nb_name)
                                self._executor.execute({"action_type": "keyboard",
                                                        "key_count": 1, "keystrokes": ["tab"]})
                                self._adaptive_settle_wait(self.step_delay * 0.5)
                                continue
                            prediction = {"action_type": "click", "click_position": _snap2}
                            _is_plain_navigate_click = True
                            logger.info("[OPT2] TRANSFORMER navigates → click @ (%.0f,%.0f)  ptr_conf=%.2f",
                                        _snap2[0], _snap2[1], t_pred.get("_click_conf", 0.0))
                            _lowconf_fallback_streak = 0
                            _reclick_streak = 0
                    else:
                        # Found live 2026-08-07, directly from the user watching a run:
                        # "Too much wasted steps. Whenever there are no longer any
                        # targets on the screen... activate Navigation Protocol so
                        # that there will be." A single blind Tab is normal (model
                        # uncertainty usually self-corrects); several low-confidence/
                        # invalid pointer guesses IN A ROW with zero progress means
                        # the transformer itself is stuck — hand navigation to the
                        # same deterministic Navigation Protocol logic that already
                        # decided a target exists on this screen, instead of
                        # continuing to blind-Tab and hope.
                        _lowconf_fallback_streak += 1
                        if _lowconf_fallback_streak >= _LOWCONF_FALLBACK_LIMIT:
                            _nav_vb_lc = self._form_viewport_bottom(state) - 8
                            _target = self._navproto.find_visible_empty_target(
                                state, _nav_vb_lc, attempted_keys=self._attempted_keys,
                                attempt_key_fn=self._attempt_key)
                            if _target and _target.get("bbox"):
                                _tb = _target["bbox"]
                                _tlabel = (_target.get("label") or _target.get("text") or "?")[:30]
                                _tcx, _tcy = (_tb[0] + _tb[2]) / 2, (_tb[1] + _tb[3]) / 2
                                logger.info(
                                    "[OPT2] %d low-confidence fallbacks in a row — Navigation "
                                    "Protocol taking over: known target %r @ (%.0f,%.0f).",
                                    _lowconf_fallback_streak, _tlabel, _tcx, _tcy)
                                # Found 2026-08-09, live, direct user report ("Agent
                                # is stuck"): this branch fell through to a plain
                                # coordinate click on the pipeline's normal
                                # execution path below, with NO verification that
                                # it actually landed -- unlike every sibling
                                # redirect guard in this file (the reclick-streak
                                # guard, the combobox-attempted guard, the
                                # checkbox-already-checked guard), all of which use
                                # _focus_element_via_uia + re-observe-and-verify.
                                # Log evidence: 'ZIP Code' stayed the focused
                                # element for 6+ consecutive steps while this
                                # branch kept "successfully" finding 'Occupation'
                                # as a target, clicking its coordinates, and
                                # getting no_change every single time -- because
                                # finding a target reset _lowconf_fallback_streak
                                # to 0 regardless of whether the click actually
                                # worked, this exact cycle could repeat forever
                                # with nothing ever noticing. Same "verify what
                                # actually happened, don't assume the click
                                # landed" fix already proven on every sibling
                                # guard, just never carried into this one.
                                _tlabel_full = (_target.get("label") or _target.get("text") or "").strip()
                                _t_key = self._attempt_key(_target, elements=state.get("elements", []))
                                # _find_destructive_button_at guard (2026-08-13) -- see the
                                # checkbox-redirect fix elsewhere in this loop for the full
                                # incident writeup; same risk shape, same fix.
                                if (not self._focus_element_via_uia(_tlabel_full, expected_bbox=_tb)
                                        and not _find_destructive_button_at(state.get("elements", []), [_tcx, _tcy])):
                                    self._executor.execute({
                                        "action_type": "click",
                                        "click_position": [_tcx, _tcy],
                                    })
                                self._adaptive_settle_wait(self.step_delay * 0.4)
                                _lowconf_fallback_streak = 0
                                _reclick_streak = 0
                                _lc_check = self._observe()
                                _lc_landed = self._attempt_key(
                                    next((e for e in _lc_check.get("elements", [])
                                          if e.get("element_id") == _lc_check.get("focused_element_id")), {}),
                                    elements=_lc_check.get("elements", []),
                                ) == _t_key
                                if _lc_landed:
                                    _redirect_stall_count = 0
                                    # Found 2026-08-09, live, direct user report
                                    # ("it did it again in the Navigation
                                    # Protocol, one input field at a time"):
                                    # this is a THIRD sibling redirect guard
                                    # with the exact same shape as the reclick-
                                    # streak guard and the combobox-attempted-
                                    # blank guard -- but it never got either of
                                    # their "complete an accidental partial
                                    # reveal" or "refocus to the shallowest
                                    # revealed field" fixes. Log evidence:
                                    # this exact branch fired repeatedly on the
                                    # Vehicle tab (Annual Miles Est., Primary
                                    # Use, Purchase Date, Purchase Price...),
                                    # each redirect occasionally growing the
                                    # element count ("3 new interactive
                                    # element(s) appeared", "1 new interactive
                                    # element(s) appeared") with nothing ever
                                    # noticing or exploiting it -- one field
                                    # revealed and filled per redirect, never
                                    # a cluster. Same fix as the two sibling
                                    # guards, applied here for the first time.
                                    _n_before_lc_land = len(state.get("elements", []))
                                    state = _lc_check
                                    if len(state.get("elements", [])) > _n_before_lc_land:
                                        _lc_deep = self._navproto.find_scroll_target_element(
                                            state, self._form_viewport_bottom(state) - 8,
                                            attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                        if _lc_deep and _lc_deep.get("bbox"):
                                            _lc_deep_key = self._attempt_key(_lc_deep, elements=state.get("elements", []))
                                            _lc_deep_label = (_lc_deep.get("label") or _lc_deep.get("text") or "").strip()
                                            if _lc_deep_key != _t_key and _lc_deep_label and self._focus_element_via_uia(
                                                    _lc_deep_label, expected_bbox=_lc_deep["bbox"]):
                                                logger.info(
                                                    "[OPT2] %r's landing already revealed more content — "
                                                    "completing the reveal via %r before refocusing shallow.",
                                                    _tlabel, _lc_deep_label[:30])
                                                state = self._observe()
                                    _lc_shallow = self._navproto.find_visible_empty_target(
                                        state, self._form_viewport_bottom(state) - 8,
                                        attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                    if _lc_shallow and _lc_shallow.get("bbox"):
                                        _lc_shallow_key = self._attempt_key(_lc_shallow, elements=state.get("elements", []))
                                        if _lc_shallow_key != _t_key:
                                            _lc_shallow_label = (_lc_shallow.get("label") or _lc_shallow.get("text") or "").strip()
                                            if _lc_shallow_label and self._focus_element_via_uia(
                                                    _lc_shallow_label, expected_bbox=_lc_shallow["bbox"]):
                                                logger.info(
                                                    "[OPT2] revealed cluster via %r — refocusing shallower "
                                                    "target %r to fill top-down.", _tlabel, _lc_shallow_label[:30])
                                                state = self._observe()
                                    prediction = {"action_type": "no_op"}
                                else:
                                    _redirect_stall_count += 1
                                    logger.warning(
                                        "[OPT2] low-confidence-fallback redirect to %r did NOT move "
                                        "focus there (stall %d/%d) — the click isn't sticking.",
                                        _tlabel, _redirect_stall_count, _REDIRECT_STALL_LIMIT)
                                    if _redirect_stall_count >= _REDIRECT_STALL_LIMIT:
                                        _redirect_stall_count = 0
                                        self._attempted_keys.add(_t_key)
                                    prediction = {"action_type": "no_op"}
                            else:
                                # Nothing left visible either — this IS the "no
                                # targets on screen" case. Scroll (or advance the
                                # tab if scrolling is already exhausted) right now
                                # instead of another blind Tab that just burns a
                                # step hoping natural tab-order helps.
                                logger.info(
                                    "[OPT2] %d low-confidence fallbacks in a row, nothing left "
                                    "visible either — Navigation Protocol scrolling/advancing.",
                                    _lowconf_fallback_streak)
                                if _tab_scroll_count < _MAX_TAB_SCROLLS:
                                    self._scroll_form_down(state)
                                    _tab_scroll_count += 1
                                else:
                                    self._try_advance_tab(state)
                                    _tab_scroll_count = 0
                                _lowconf_fallback_streak = 0
                                _reclick_streak = 0
                                _last_auto_step = step_idx
                                time.sleep(self.step_delay * 0.5)
                                continue
                        else:
                            prediction = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
                            logger.info("[OPT2] pointer invalid — Tab fallback")

            elif self._llm_client and t_conf < _HIGH_CONF:
                llm_action = self._ask_llm(state)
                action_type = llm_action.get("action_type", "wait")
                reason = llm_action.get("reason", "")
                logger.info("[LLM:%s] action=%-8s", self.provider, action_type)

                if action_type == "done":
                    logger.info("LLM: task complete.")
                    break
                if action_type == "wait":
                    time.sleep(2.0)
                    continue

                # ── Merge: LLM decides what, transformer decides where ──────
                prediction = self._merge(t_pred, t_conf, llm_action, state)
                _decision_maker = "llm"
                print(f"\n  [LLM TOOK OVER]  reason: {reason}\n", flush=True)

                # Stuck-click guard
                if prediction.get("action_type") == "click":
                    pos = tuple(int(v) for v in prediction.get("click_position", [0, 0]))
                    if pos == _llm_click_pos:
                        _llm_click_count += 1
                    else:
                        _llm_click_pos, _llm_click_count = pos, 1
                    if _llm_click_count > _LLM_CLICK_LIMIT:
                        text = self._value_for_focused(state)
                        if text:
                            logger.warning("LLM stuck clicking — forcing type %r", text[:40])
                            prediction = {"action_type": "keyboard", "key_count": len(text),
                                          "keystrokes": list(text), "text": text}
                            _llm_click_pos, _llm_click_count = None, 0
                else:
                    _llm_click_pos, _llm_click_count = None, 0

            # 2c. Transformer acts alone — high confidence OR no LLM provider
            else:
                if self._llm_client:
                    logger.info("[TRANSFORMER] high-conf (%.2f) — acting alone, LLM skipped", t_conf)
                else:
                    logger.info("[TRANSFORMER] no LLM provider — acting alone")
                prediction = t_pred

                # Anti-scroll-loop: if scroll keeps producing no_change, the model
                # is stuck over-predicting scroll (overfit on small data). Swap to
                # its next-best non-scroll action so it progresses.
                if (prediction.get("action_type") == "scroll"
                        and getattr(self, "_scroll_nochange", 0) >= 2):
                    _alt = sorted(
                        ((p, a) for a, p in t_pred.get("_scores", {}).items() if a != "scroll"),
                        reverse=True,
                    )
                    if _alt:
                        prediction = dict(prediction)
                        prediction["action_type"] = _alt[0][1]
                        logger.warning("Scroll-loop (%dx no_change) — switching to next-best: %s (%.2f)",
                                       self._scroll_nochange, _alt[0][1], _alt[0][0])
                    self._scroll_nochange = 0

                if prediction.get("action_type") == "click":
                    snapped = self._snap(prediction.get("click_position", [0, 0]), state)
                    if snapped:
                        prediction = dict(prediction)
                        prediction["click_position"] = snapped

                    snap_tuple = tuple(int(v) for v in (snapped or prediction.get("click_position", [0, 0])))
                    if snap_tuple == _stuck_pos:
                        _stuck_count += 1
                    else:
                        _stuck_pos, _stuck_count = snap_tuple, 1

                    # VISITED-ADVANCE protocol (transformer-only runs): after a
                    # field is clicked, assume it's handled. If the (collapsed)
                    # pointer targets an already-visited field, advance to the
                    # next UNVISITED fillable field by position order — lets the
                    # run cover the form instead of looping on one spot.
                    # Gated OFF in pure mode — the model's is_visited feature
                    # handles progression; this crutch fights it.
                    if not self._llm_client and not self._no_autohandlers:
                        _vis = getattr(self, "_visited_pos", None)
                        if _vis is None:
                            _vis = self._visited_pos = set()
                        _key = (round(snap_tuple[0] / 15) * 15, round(snap_tuple[1] / 15) * 15)
                        if _key in _vis:
                            _cands = sorted(
                                [e for e in state.get("elements", [])
                                 if e.get("window_role") != "background"
                                 and e.get("type") in ("editcontrol", "input",
                                                       "comboboxcontrol", "combobox")
                                 and e.get("bbox")],
                                key=lambda e: (e["bbox"][1], e["bbox"][0]))
                            for _e in _cands:
                                _b = _e["bbox"]
                                _cx, _cy = (_b[0] + _b[2]) / 2, (_b[1] + _b[3]) / 2
                                _ck = (round(_cx / 15) * 15, round(_cy / 15) * 15)
                                if _ck not in _vis:
                                    prediction = {"action_type": "click",
                                                  "click_position": [_cx, _cy]}
                                    snap_tuple = (int(_cx), int(_cy))
                                    _key = _ck
                                    logger.info("[VISITED-ADVANCE] pointer repeated — next unvisited field @ (%.0f,%.0f)", _cx, _cy)
                                    break
                        _vis.add(_key)
                        _stuck_count = 0   # visited-advance handles progression

                    if _stuck_count >= _STUCK_LIMIT:
                        logger.warning("Loop detected @ %s %dx — forcing keyboard.", snap_tuple, _stuck_count)
                        prediction   = {"action_type": "keyboard", "key_count": 1, "keystrokes": []}
                        _stuck_count = 0
                else:
                    _stuck_pos, _stuck_count = None, 0

                # TextResolver turns the transformer's source_elem_idx (which
                # Notepad element to read) into actual typed text. Required in
                # pure mode too — without it the model can navigate but never
                # types Notepad values (the "doesn't know how to get from the
                # Notepad" symptom).
                if prediction.get("action_type") == "keyboard":
                    src_idx = prediction.get("source_elem_idx", -1)
                    text = self._text_resolver.resolve(state, source_elem_idx=src_idx)
                    if text:
                        prediction = dict(prediction)
                        prediction["text"] = text

            # 3. Execute — guard against typing into non-edit elements or clicking submit early
            _fid = state.get("focused_element_id")
            _fel = next((e for e in state.get("elements", []) if e.get("element_id") == _fid), None)
            _flabel     = ((_fel.get("label") or _fel.get("text") or "?").strip() if _fel else "unknown")
            _flabel_sec = self._detect_section(state, _fel) if _fel else ""
            _flabel_full = f"{_flabel_sec} {_flabel}" if _flabel_sec else _flabel

            if (not self._pure_transformer
                    and prediction.get("action_type") == "keyboard" and prediction.get("text")):
                logger.info("Type target: focused=[%s] %r", _fel.get("type","?") if _fel else "?", _flabel_full)
                # If focused field already has the correct value → Tab instead of re-typing
                if _fel and _fel.get("type") in ("editcontrol", "input"):
                    _cur_val = (_fel.get("value") or "").strip()
                    _new_val = prediction["text"].strip()
                    if _cur_val and _cur_val == _new_val:
                        # Field already holds the right value — don't re-type. Tab to
                        # release focus and let the TRANSFORMER decide the next move.
                        # (No hardcoded "go to empty field" / "click Submit" crutch —
                        # the end-game is the model's job; gaps there are a data
                        # problem to fix with demos, not agent navigation.)
                        logger.info("Re-type guard: %r already has %r — Tab instead.", _flabel_full, _cur_val[:40])
                        self._filled_this_tab.add(_flabel_full)
                        _no_change_streak += 1
                        prediction = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
                # If the focused element is NOT an editable text field, typing will do nothing
                # (or corrupt a pane/tab/button). Replace with Tab to advance focus instead.
                if _fel and _fel.get("type") not in ("editcontrol", "input", "comboboxcontrol"):
                    if _fel.get("type") == "checkboxcontrol":
                        # LLM typed a truthy value (e.g. "YES (check)") into a checkbox — check it
                        _chk_text = prediction.get("text", "").lower().strip()
                        _should_chk = _chk_text not in ("", "no", "false", "0", "unchecked")
                        if _should_chk and _flabel not in self._checked_fields:
                            _chk_bbox = _fel.get("bbox")
                            if _chk_bbox:
                                try:
                                    import win32gui as _wgc; import win32api as _wac
                                    _cx = (_chk_bbox[0] + _chk_bbox[2]) / 2
                                    _cy = (_chk_bbox[1] + _chk_bbox[3]) / 2
                                    _hw = _wgc.WindowFromPoint((int(_cx), int(_cy)))
                                    if _hw:
                                        _wac.SendMessage(_hw, 0x00F1, 1, 0)  # BM_SETCHECK, BST_CHECKED
                                        logger.info("Checkbox %r checked via BM_SETCHECK (type intercept).", _flabel_full)
                                        self._checked_fields.add(_flabel_full)
                                        self._filled_this_tab.add(_flabel_full)
                                        _real_progress_this_step = True
                                        # Belt-and-suspenders alongside the
                                        # ui_observer.py TogglePattern fix
                                        # (2026-08-09): mark it attempted
                                        # immediately so even THIS step's own
                                        # redirect search (which still runs
                                        # against the pre-check state) can't
                                        # offer this same checkbox back to
                                        # itself before the next observation
                                        # cycle picks up its new value.
                                        self._mark_attempted(_fel, elements=state.get("elements", []))
                                except Exception as _cbe:
                                    logger.warning("Checkbox BM_SETCHECK failed: %s", _cbe)
                        elif not _should_chk and _flabel not in self._checked_fields:
                            # Symmetric with the check-branch above -- was
                            # entirely missing until 2026-08-12, found live:
                            # 'Renewal Policy' was already checked (a form-
                            # reset quirk, not anything the agent did -- it
                            # was checked the very first time the agent ever
                            # looked at it this record) but record 2's own
                            # data wanted it 'NO'. With no code path to
                            # actively uncheck a box, the click-guard a few
                            # lines below (correctly, on its own terms)
                            # permanently blocked every click on it -- it
                            # only ever knows "already checked" means "leave
                            # it alone", never "checked but shouldn't be".
                            # 25 identical redirects over ~80s before the
                            # run moved on. This is the first record in the
                            # whole project that has ever wanted a checked
                            # box turned off, so the gap never surfaced
                            # before now. Don't assume the form always
                            # starts a record in the right state -- actively
                            # drive it to the target state either direction,
                            # same as a human would.
                            _chk_bbox = _fel.get("bbox")
                            if _chk_bbox:
                                try:
                                    import win32gui as _wgu; import win32api as _wau
                                    _cx = (_chk_bbox[0] + _chk_bbox[2]) / 2
                                    _cy = (_chk_bbox[1] + _chk_bbox[3]) / 2
                                    _hw = _wgu.WindowFromPoint((int(_cx), int(_cy)))
                                    if _hw:
                                        BM_GETCHECK, BM_SETCHECK, BST_CHECKED = 0x00F0, 0x00F1, 1
                                        if _wau.SendMessage(_hw, BM_GETCHECK, 0, 0) == BST_CHECKED:
                                            _wau.SendMessage(_hw, BM_SETCHECK, 0, 0)  # BST_UNCHECKED
                                            logger.info("Checkbox %r unchecked via BM_SETCHECK "
                                                        "(type intercept, target=NO).", _flabel_full)
                                        self._filled_this_tab.add(_flabel_full)
                                        _real_progress_this_step = True
                                        # Same belt-and-suspenders reasoning as the
                                        # check-branch's own mark_attempted call.
                                        self._mark_attempted(_fel, elements=state.get("elements", []))
                                except Exception as _cbe:
                                    logger.warning("Checkbox BM_SETCHECK (uncheck) failed: %s", _cbe)
                    else:
                        logger.warning("LLM type into non-edit [%s] %r — pressing Tab instead.",
                                       _fel.get("type","?"), _flabel[:40])
                    # Found 2026-08-09, live, direct user report ("Marital
                    # Status was not even filled up until Cell Phone but we
                    # navigated away"): this blind Tab (used to move on
                    # after checking a checkbox via the raw BM_SETCHECK
                    # message, which never moves real keyboard focus itself)
                    # handed movement to WINDOWS' OWN native tab order --
                    # which, distinct from anything this code decided,
                    # triggered ANOTHER auto-scroll of its own (element count
                    # dropped 163->157 right at this exact point in the log).
                    # Stacked on top of the redirect-to-deepest-cluster fix
                    # from earlier tonight, that uncontrolled second scroll
                    # left several shallower fields (Marital Status,
                    # Occupation, Education Level, Credit Score) scrolled
                    # past -- and since this module only ever scrolls DOWN,
                    # never back up, they were gone for the rest of the run.
                    # Redirect to a KNOWN visible target instead of gambling
                    # on native tab order, same principle as every other
                    # Tab-as-navigation removal tonight.
                    _cb_next = self._navproto.find_visible_empty_target(
                        state, self._form_viewport_bottom(state) - 8,
                        attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                    # _find_destructive_button_at guard, applied here for the first
                    # time -- found live 2026-08-13: this exact redirect landed on
                    # 'Clear All' (real confirmation dialog popped, run derailed for
                    # several steps). The guard already exists, already covers the
                    # navigate and combobox-open-dropdown branches (same risk shape:
                    # clicking a coordinate resolved from search/scan logic, not a
                    # position already known by construction to belong to a field) --
                    # this branch has the identical risk and was simply never wired
                    # through it. Not a new keyword or a Clear-All-specific rule; the
                    # guard blocks ANY button-type element at the target position,
                    # by construction, regardless of label.
                    _cb_next_pos = None
                    if _cb_next and _cb_next.get("bbox"):
                        _cnb = _cb_next["bbox"]
                        _cb_next_pos = [(_cnb[0] + _cnb[2]) / 2, (_cnb[1] + _cnb[3]) / 2]
                    if _cb_next_pos and _find_destructive_button_at(state.get("elements", []), _cb_next_pos):
                        logger.warning("Checkbox redirect target was a destructive button %r — Tab instead.",
                                        (_cb_next.get("text") or _cb_next.get("label") or "?"))
                        prediction = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
                    elif _cb_next_pos:
                        prediction = {"action_type": "click", "click_position": _cb_next_pos}
                    else:
                        prediction = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}

            if prediction.get("action_type") == "click":
                _cp = prediction.get("click_position", [0, 0])
                # Guard: block LLM from re-clicking already-checked checkboxes (would uncheck them)
                _chk_at_cp = next(
                    (e for e in state.get("elements", [])
                     if e.get("type") in ("checkboxcontrol", "checkbox")
                     and e.get("window_role") != "background"
                     and e.get("bbox")
                     and e["bbox"][0] <= _cp[0] <= e["bbox"][2]
                     and e["bbox"][1] <= _cp[1] <= e["bbox"][3]),
                    None
                )
                if _chk_at_cp:
                    _chk_label_bare = (_chk_at_cp.get("label") or _chk_at_cp.get("text") or "").strip()
                    _chk_sec   = self._detect_section(state, _chk_at_cp)
                    _chk_label = f"{_chk_sec} {_chk_label_bare}" if _chk_sec else _chk_label_bare
                    _chk_cx = (_chk_at_cp["bbox"][0] + _chk_at_cp["bbox"][2]) / 2
                    _chk_cy = (_chk_at_cp["bbox"][1] + _chk_at_cp["bbox"][3]) / 2
                    # Use Win32 BM_SETCHECK — pyautogui clicks don't toggle wx checkboxes
                    try:
                        import win32gui as _wg2; import win32api as _wa2
                        BM_GETCHECK = 0x00F0; BM_SETCHECK = 0x00F1; BST_CHECKED = 1
                        _hwnd2 = _wg2.WindowFromPoint((int(_chk_cx), int(_chk_cy)))
                        if _hwnd2:
                            _already = (_wa2.SendMessage(_hwnd2, BM_GETCHECK, 0, 0) == BST_CHECKED)
                            if _chk_label in self._checked_fields or _already:
                                # Found 2026-08-09, live, direct user report
                                # ("Navigation error"): the transformer's own
                                # click prediction landed on this SAME
                                # already-checked checkbox 10 CONSECUTIVE
                                # steps in a row ('Uninsured/Underinsured
                                # Motorist'). This guard correctly blocked
                                # every one of those clicks (never
                                # accidentally unchecked it), but its
                                # fallback was a blind Tab -- a fourth spot
                                # using Tab-as-navigation that got missed in
                                # tonight's broader sweep (this path is
                                # completely separate from the reclick-streak
                                # guard, which only covers edit/combobox
                                # fields, not this specific "block re-
                                # clicking a checked checkbox" check). Blind
                                # Tab happened to still work here since a
                                # plain Tab through a checklist naturally
                                # lands on the next checkbox most of the
                                # time -- but it's the same gamble on OS
                                # focus order this project has repeatedly
                                # found unreliable elsewhere. Redirect to a
                                # known visible target instead, same
                                # principle as every other guard tonight.
                                logger.warning("Checkbox %r already checked — redirecting instead of blind Tab.",
                                                _chk_label)
                                _chk_next = self._navproto.find_visible_empty_target(
                                    state, self._form_viewport_bottom(state) - 8,
                                    attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                                if _chk_next and _chk_next.get("bbox"):
                                    _chk_next_label = (_chk_next.get("label") or _chk_next.get("text") or "").strip()
                                    if _chk_next_label and self._focus_element_via_uia(
                                            _chk_next_label, expected_bbox=_chk_next["bbox"]):
                                        # "no_op", not None -- prediction.get(...) is
                                        # called unconditionally a few lines below
                                        # this block; None would crash there.
                                        prediction = {"action_type": "no_op"}
                                    else:
                                        _cnb = _chk_next["bbox"]
                                        prediction = {"action_type": "click",
                                                      "click_position": [(_cnb[0] + _cnb[2]) / 2, (_cnb[1] + _cnb[3]) / 2]}
                                else:
                                    prediction = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
                            else:
                                _wa2.SendMessage(_hwnd2, BM_SETCHECK, BST_CHECKED, 0)
                                logger.info("Checkbox %r checked via Win32 BM_SETCHECK.", _chk_label)
                                self._checked_fields.add(_chk_label)
                                self._filled_this_tab.add(_chk_label)
                                self._mark_attempted(_chk_at_cp, elements=state.get("elements", []))
                                _real_progress_this_step = True
                                _no_change_streak = 0
                                _last_auto_step   = step_idx
                                prediction = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
                    except Exception as _ce:
                        logger.warning("BM_SETCHECK failed for %r: %s", _chk_label, _ce)

                # NOTE: do NOT filter by window_role here — Submit/Submit & New buttons live
                # on the main form window which is tagged "background" by the state scanner,
                # and we must block clicks on them regardless of their window role.
                _btn_at_cp = next(
                    (e for e in state.get("elements", [])
                     if e.get("type") in ("buttoncontrol", "button")
                     and e.get("bbox")
                     and any(kw in (e.get("text") or e.get("label") or "").lower()
                             for kw in _SUBMIT_KW)
                     and e["bbox"][0] <= _cp[0] <= e["bbox"][2]
                     and e["bbox"][1] <= _cp[1] <= e["bbox"][3]),
                    None
                )
                if _btn_at_cp and not self._no_autohandlers:
                    _btn_name = (_btn_at_cp.get("text") or _btn_at_cp.get("label") or "button")
                    logger.warning("Blocking premature click on button %r — advancing tab instead.", _btn_name)
                    if self._try_advance_tab(state):
                        _no_change_streak  = 0
                        _tab_just_switched = True
                        _tab_scroll_count  = 0
                        _last_auto_step    = step_idx
                        self._refresh_record_cache(state)
                        time.sleep(self.step_delay)
                        continue
                    prediction = {"action_type": "no_op"}

            # Intercept LLM clicks that target a tab element — resolved coords can be wrong.
            # Route through _try_advance_tab which uses the correct indexed bbox instead.
            if prediction.get("action_type") == "click":
                _cp = prediction.get("click_position", [0, 0])
                _all_tabs = [e for e in state.get("elements", [])
                             if e.get("type") in ("tabitem", "tabitemcontrol")
                             and e.get("window_role") != "background"
                             and e.get("bbox")]
                _tab_hit = next(
                    (e for e in _all_tabs
                     if e["bbox"][0] <= _cp[0] <= e["bbox"][2]
                     and e["bbox"][1] <= _cp[1] <= e["bbox"][3]),
                    None,
                )
                if _tab_hit:
                    _hit_name = (_tab_hit.get("text") or _tab_hit.get("label") or "?").strip()
                    # Go to the tab the model ACTUALLY clicked — NOT current+1.
                    # (Routing every tab-click through advance-to-next made repeated
                    # clicks on one tab race through ALL tabs, filling none.)
                    _sorted_tabs = sorted(_all_tabs, key=lambda e: e["bbox"][0])
                    _hit_idx = _sorted_tabs.index(_tab_hit)
                    # Block backward tab-strip clicks outright — found live 2026-08-08
                    # via the drill tool: "it jumped to VIN damn". A drill run started
                    # fresh at Payment (idx 7, never having visited Vehicle idx 2 this
                    # session, so _advance_blacklist_pos had nothing to block it) —
                    # the transformer's own pointer clicked Vehicle's tab strip at
                    # ptr_conf=0.54, comfortably clearing the general 0.50 tab-strip
                    # floor (execution_tab_click_stricter_confidence), which doesn't
                    # distinguish direction. Every successful full-form run this
                    # session progressed strictly forward, never backward — same
                    # "finish forward, don't come back" principle already validated
                    # for _advance_blacklist_pos, extended here to cover tabs never
                    # yet visited this session (which that mechanism can't, since it
                    # only blacklists tabs the SYSTEM itself has deliberately left).
                    # Not confidence-gated like the general tab-strip floor — no
                    # confidence observed live has ever made a backward jump correct.
                    if _hit_idx < self._current_tab_idx:
                        logger.info(
                            "[GUARD] pointer clicked backward to %r (idx %d, currently on idx %d) "
                            "— Tab instead, this workflow never goes backward.",
                            _hit_name, _hit_idx, self._current_tab_idx)
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        self._adaptive_settle_wait(self.step_delay * 0.4)
                        continue
                    # Found 2026-08-09, live, direct user report ("Coverage
                    # was skipped"): the backward-block guard above stops a
                    # click landing BEHIND the current tab, but nothing
                    # stopped one landing more than one tab AHEAD either.
                    # Log evidence: while on Vehicle (idx 2), the
                    # transformer's own raw pointer clicked directly on
                    # Drivers (idx 4) -- "Tab-click -> navigating to
                    # 'Drivers' (idx 4)" -- and since idx 4 is not LESS than
                    # idx 2, the backward guard never triggered. Coverage
                    # (idx 3) was never visited at all this run; not one of
                    # its fields (Bodily Injury, MedPay Limit, UM/UIM Limit,
                    # ...) appears anywhere before this jump. The comment
                    # above this block explains why the code deliberately
                    # goes to whichever tab the model ACTUALLY clicked
                    # rather than always current+1 -- that was fixing a
                    # real, different bug (repeated clicks landing on one
                    # tab racing through every tab in between). Preserving
                    # that fix for the legitimate case (_hit_idx ==
                    # current+1, or re-clicking the current tab) while
                    # blocking the specific case neither guard covered: a
                    # jump skipping one or more entirely unvisited tabs.
                    # Redirect to the immediately NEXT tab instead of either
                    # blind-Tabbing (the backward guard's response, which
                    # doesn't apply here since this isn't a stray guess --
                    # the model clearly intended SOME forward click) or
                    # trusting the skip.
                    if _hit_idx > self._current_tab_idx + 1:
                        logger.info(
                            "[GUARD] pointer clicked forward past unvisited tab(s) to %r "
                            "(idx %d, currently on idx %d) — going to idx %d instead, "
                            "this workflow never skips a tab.",
                            _hit_name, _hit_idx, self._current_tab_idx, self._current_tab_idx + 1)
                        _next_tab = _sorted_tabs[self._current_tab_idx + 1]
                        _nx1, _ny1, _nx2, _ny2 = _next_tab["bbox"]
                        _next_name = (_next_tab.get("text") or _next_tab.get("label") or "?").strip()
                        self._executor.execute({"action_type": "click",
                                                "click_position": [(_nx1 + _nx2) / 2, (_ny1 + _ny2) / 2]})
                        self._current_tab_idx = self._current_tab_idx + 1
                        _no_change_streak  = 0
                        _tab_just_switched = True
                        _tab_scroll_count  = 0
                        _last_auto_step    = step_idx
                        self._filled_this_tab.clear()
                        _confirmed_blank_fields.clear()
                        self._refresh_record_cache(state)
                        time.sleep(self.step_delay)
                        continue
                    if _hit_idx != self._current_tab_idx:
                        # Found 2026-08-09, live, direct user report ("that
                        # was a good Policyholder -> Payment, problem is we
                        # skipped so much fucking input fields"): the
                        # forward-skip guard above only blocks a click
                        # landing MORE than one tab ahead -- a click on the
                        # very NEXT tab was always allowed through
                        # unconditionally, with no check that the CURRENT
                        # tab was actually finished. Log evidence: on
                        # Coverage, only 8 fields got filled (Bodily Injury,
                        # Property Damage, MedPay Limit, Collision/
                        # Comprehensive Deductible, Rental Limit, UM/UIM
                        # Limit, one checkbox) before the transformer's own
                        # raw pointer happened to click the Drivers tab
                        # header directly -- Roadside Assistance, GAP
                        # Insurance, Accident Forgiveness, Diminishing
                        # Deductible, Total Premium, and others were still
                        # sitting there, never touched. The ONLY path that
                        # already verifies "is this tab actually done" is
                        # the separate stuck-guard/Navigation-Protocol
                        # advance flow (used correctly for every OTHER tab
                        # transition in this same run) -- this raw-pointer
                        # path bypassed that check entirely. Same principle
                        # as _find_destructive_button_at (never trust the
                        # uncertain raw pointer for a structural, high-
                        # consequence action) and the backward/forward-skip
                        # guards above -- extended to cover "leaving a tab
                        # early" too. Search the WHOLE current tab
                        # (unbounded, on- or off-screen -- same 1e9 pattern
                        # already used elsewhere tonight for "is there
                        # truly nothing left") before trusting the switch;
                        # if something remains, redirect to IT instead.
                        _remaining = self._navproto.find_visible_empty_target(
                            state, 1e9, attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                        if _remaining and _remaining.get("bbox"):
                            _rem_label = (_remaining.get("label") or _remaining.get("text") or "").strip()
                            _rem_bbox = _remaining["bbox"]
                            logger.info(
                                "[GUARD] pointer clicked tab %r (idx %d) but %r is still unfilled on "
                                "the current tab — redirecting there instead of leaving early.",
                                _hit_name, _hit_idx, _rem_label[:30])
                            _rem_pos = [(_rem_bbox[0] + _rem_bbox[2]) / 2, (_rem_bbox[1] + _rem_bbox[3]) / 2]
                            # _find_destructive_button_at guard (2026-08-13) -- see the
                            # checkbox-redirect fix above for the full incident writeup;
                            # same risk shape, same fix.
                            if (not (_rem_label and self._focus_element_via_uia(_rem_label, expected_bbox=_rem_bbox))
                                    and not _find_destructive_button_at(state.get("elements", []), _rem_pos)):
                                self._executor.execute({
                                    "action_type": "click",
                                    "click_position": _rem_pos,
                                })
                            time.sleep(self.step_delay * 0.4)
                            continue
                        x1, y1, x2, y2 = _tab_hit["bbox"]
                        logger.info("Tab-click → navigating to %r (idx %d).", _hit_name, _hit_idx)
                        self._executor.execute({"action_type": "click",
                                                "click_position": [(x1 + x2) / 2, (y1 + y2) / 2]})
                        self._current_tab_idx = _hit_idx
                        _no_change_streak  = 0
                        _tab_just_switched = True
                        _tab_scroll_count  = 0
                        _last_auto_step    = step_idx
                        self._filled_this_tab.clear()
                        _confirmed_blank_fields.clear()
                        self._refresh_record_cache(state)
                        time.sleep(self.step_delay)
                        continue
                    # Model re-clicked the tab it's ALREADY on → don't race. Fill the
                    # next unfilled field instead; only if none left, advance to next tab.
                    if self._focus_first_empty_field(state):
                        logger.info("Tab %r already active — focusing next unfilled field instead.", _hit_name)
                        time.sleep(0.3)
                        _heuristic_steps += 1
                        continue
                    if self._try_advance_tab(state):
                        _no_change_streak  = 0
                        _tab_just_switched = True
                        _tab_scroll_count  = 0
                        _last_auto_step    = step_idx
                        self._filled_this_tab.clear()
                        _confirmed_blank_fields.clear()
                        self._refresh_record_cache(state)
                        time.sleep(self.step_delay)
                        continue
                    prediction = {"action_type": "no_op"}

            # Combobox: type-into-combobox → open dropdown + click matching listitem
            if (not self._pure_transformer
                    and prediction.get("action_type") == "keyboard"
                    and prediction.get("text")
                    and _fel and _fel.get("type") == "comboboxcontrol"
                    and _fel.get("bbox")):
                _combo_value = prediction["text"]
                _bx1, _by1, _bx2, _by2 = _fel["bbox"]
                _ccx, _ccy = (_bx1 + _bx2) / 2, (_by1 + _by2) / 2
                # Check whether the dropdown is ALREADY open before clicking to
                # "open" it — see _open_dropdown_items' docstring for why (a
                # click toggles, it doesn't just open); this was the actual root
                # cause of a loop that survived both the earlier type-name fix
                # and the poll-timing widening.
                _already_open = _open_dropdown_items(state.get("elements", []))
                if _already_open:
                    logger.info("Combobox: dropdown for %r already open (%d items) — skipping the open-click.",
                                _flabel, len(_already_open))
                else:
                    logger.info("Combobox: clicking to open dropdown for %r → %r", _flabel, _combo_value)
                    # Target the arrow button, not the center -- see
                    # _combobox_dropdown_arrow_pos's own docstring.
                    _cb_arrow_pos2 = self._combobox_dropdown_arrow_pos(_fel["bbox"])
                    self._executor.execute({"action_type": "click", "click_position": _cb_arrow_pos2})
                # The dropdown can take a moment to render. Poll a few times before
                # giving up — otherwise we Escape on an empty observe and waste a
                # whole open→escape→reopen cycle (the #1 combobox time-sink).
                # ui_observer.py's _CTRL_TYPE_MAP maps the standard UIA "ListItem"
                # control type to "listitem" (no "control" suffix) -- only raw,
                # unmapped ControlTypeName strings fall through as
                # "listitemcontrol". Checking for "listitemcontrol" alone missed
                # every real dropdown whose items report the standard, mapped
                # type -- found live 2026-08-07: a run looped forever opening
                # "Body Type"'s dropdown, finding 0 list items every time despite
                # 8 new elements genuinely appearing, and Escaping, over and over.
                #
                # That fix alone wasn't enough, though: re-tested live and 'Sedan'
                # (a genuinely correct, exact-match option) STILL came up with 0
                # items in the SAME run where other dropdowns succeeded moments
                # apart. Turned out to be neither a matching nor a timing bug —
                # the REAL cause was the toggle-click issue guarded against just
                # above (_already_open): once that's fixed, an open→escape→
                # reopen cycle should be rare, but the poll stays widened as a
                # second line of defense against genuinely slow renders.
                #
                # Widened AGAIN 2026-08-10, then REVERTED the same day -- matches
                # the sibling poll above (same file) and its own reasoning: live
                # evidence showed the dropdown wasn't rendering LATE, it wasn't
                # rendering AT ALL, so the wider window only made a real failure
                # take longer to give up on, which is exactly what "still so
                # fucking slow" was reporting. Back to the value that's actually
                # been verified not to waste time for no benefit.
                _POLL_TRIES, _POLL_INTERVAL = 8, 0.4
                _listitems = list(_already_open)
                for _try in range(_POLL_TRIES):
                    if _listitems:
                        break
                    time.sleep(_POLL_INTERVAL)
                    _combo_state = self._observe()
                    _listitems = [e for e in _combo_state.get("elements", [])
                                  if e.get("type") in _LISTITEM_TYPES
                                  and e.get("window_role") != "background"
                                  and e.get("bbox")]
                    if _listitems:
                        break
                if not _listitems:
                    logger.warning("Combobox: dropdown for %r still empty after %d tries (%.1fs) — giving up.",
                                    _combo_value, _POLL_TRIES, _POLL_TRIES * _POLL_INTERVAL)
                def _opt(e): return (e.get("text") or e.get("label") or "").strip()
                # _option_matches: exact, then prefix-fuzzy ('Full Coverage' vs
                # 'Full Coverage (Comprehensive)'), then whole-word-token
                # containment (e.g. a value matching only part of a longer option)
                # — without ever letting
                # 'Active' match 'Inactive'. See its docstring for the live bug
                # (25+-iteration loop) that motivated the third tier.
                _match = next((e for e in _listitems if _option_matches(_combo_value, _opt(e))), None)
                if not _match and _listitems:
                    logger.warning("Combobox: %r not in options %s",
                                   _combo_value, [_opt(e) for e in _listitems][:12])
                # Keyboard fallback (_select_combobox_value_via_keyboard) -- two
                # real, distinct reasons to reach it, both meaning "the rendered
                # listitems can't be trusted, read the control's own live value
                # instead":
                #   1. No dropdown ever rendered at all (original case).
                #   2. Real live evidence 2026-08-12: a long alphabetical list
                #      (US states, 'State' field) only ever showed the first
                #      ~12 items (Alabama..Idaho) no matter how many times the
                #      dropdown was reopened -- 37 identical retries over 96s,
                #      all seeing the exact same truncated set, hunting
                #      'Texas'. The listbox only materializes UIA elements for
                #      whatever's currently scrolled into view; re-clicking to
                #      reopen never scrolls it, so a target past the visible
                #      window was permanently invisible to _listitems no
                #      matter how many times this branch retried -- gated out
                #      of the ONLY existing escape hatch (this function) by
                #      "and not _listitems", since 12 (wrong) items is a
                #      non-empty list. Arrow-key navigation reads
                #      ValuePattern.Value directly instead of rendered
                #      elements, so it doesn't care what's scrolled into view.
                # Escape first in case #2 (a real, wrong dropdown IS open) --
                # puts the control into the same closed/focused state case #1
                # already left it in, which is the state this fallback was
                # actually verified against live.
                _kb_filled = False
                if not _match:
                    if _listitems:
                        self._executor.execute({"action_type": "keyboard", "key_count": 1,
                                                "keystrokes": ["escape"]})
                        time.sleep(0.15)
                    _kb_filled = self._select_combobox_value_via_keyboard(
                        _flabel, _combo_value, _ccx, _ccy)
                if _match or _kb_filled:
                    if _match:
                        _lx1, _ly1, _lx2, _ly2 = _match["bbox"]
                        self._executor.execute({"action_type": "click",
                                                "click_position": [(_lx1+_lx2)/2, (_ly1+_ly2)/2]})
                    logger.info("Combobox: selected %r", _combo_value)
                    _no_change_streak = 0
                    _last_auto_step   = step_idx
                    # Tab past to advance focus — without this, LLM re-opens the same combobox
                    time.sleep(0.25)
                    self._executor.execute({"action_type": "keyboard", "key_count": 1,
                                            "keystrokes": ["tab"]})
                    # Mark in action history so repeat-guard catches any re-attempt
                    _action_history.append(("combobox_done", _flabel_full))
                    self._filled_this_tab.add(_flabel_full)
                else:
                    logger.warning("Combobox: %r not in dropdown — pressing Escape", _combo_value)
                    self._executor.execute({"action_type": "keyboard", "key_count": 1,
                                            "keystrokes": ["escape"]})
                    _action_history.append(("combobox_fail", _flabel))
                    # Found live 2026-08-12, direct report ("Another loop
                    # damn"): 'Bodily Injury (k$/k$)' retried this exact
                    # failing search 71 times across the run -- the record's
                    # own target value ('30/60') genuinely isn't one of this
                    # control's real options (['25/50', '50/100', '100/300',
                    # '250/500', '500/500', '300/300']), a real data problem
                    # no amount of retrying can ever resolve. This branch had
                    # no escalation/give-up counter at all, unlike its
                    # sibling click-fill branch above, which already solved
                    # the identical shape of bug 2026-08-09 ('Payment
                    # Frequency', 121 retries) via _combobox_dropdown_fail_
                    # counts -- that same, already-proven mechanism, reused
                    # here rather than inventing a second one.
                    _combobox_dropdown_fail_counts[_flabel_full] = (
                        _combobox_dropdown_fail_counts.get(_flabel_full, 0) + 1)
                    if _combobox_dropdown_fail_counts[_flabel_full] >= _COMBOBOX_DROPDOWN_FAIL_LIMIT:
                        logger.warning(
                            "[OPT2] %r failed to fill %d times in a row (option never matched "
                            "or dropdown never rendered) — marking attempted and moving on.",
                            _flabel_full, _combobox_dropdown_fail_counts[_flabel_full])
                        _combobox_dropdown_fail_counts.pop(_flabel_full, None)
                        if _fel is not None:
                            self._mark_attempted(_fel, elements=state.get("elements", []))
                            # A 2-strike give-up is just as final and deliberate
                            # a decision as a genuine leave-blank answer (see
                            # self._leave_blank_keys' own comment) -- reusing
                            # the same set means the reclick guard (extended
                            # 2026-08-12 for leave-blank fields) also protects
                            # this field from the TRANSFORMER's own navigate-
                            # click pointer drifting back onto it later in the
                            # run, not just from this immediate re-ask.
                            self._leave_blank_keys.add(
                                self._attempt_key(_fel, elements=state.get("elements", [])))
                        # Found live 2026-08-12, immediately after shipping the
                        # counter above: 'Bodily Injury (k$/k$)' STILL retried
                        # right after "marking attempted and moving on" logged
                        # -- marking attempted is bookkeeping only, it doesn't
                        # move keyboard focus. Focus stayed on the same failed
                        # combobox, so the very next step's LLM lookup (which
                        # doesn't check attempted-state at all -- it just
                        # answers from the record) re-triggered this entire
                        # branch from scratch. The click-fill sibling branch
                        # above already actively redirects focus after its own
                        # escalation for exactly this reason -- mirrored here.
                        _cd_target = self._navproto.find_visible_empty_target(
                            state, self._form_viewport_bottom(state) - 8,
                            attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
                        if _cd_target and _cd_target.get("bbox"):
                            _cd_label = (_cd_target.get("label") or _cd_target.get("text") or "").strip()
                            _cdb = _cd_target["bbox"]
                            _cd_pos = [(_cdb[0] + _cdb[2]) / 2, (_cdb[1] + _cdb[3]) / 2]
                            # _find_destructive_button_at guard (2026-08-13) -- see the
                            # checkbox-redirect fix above for the full incident writeup;
                            # same risk shape, same fix.
                            if (not (_cd_label and self._focus_element_via_uia(_cd_label, expected_bbox=_cdb))
                                    and not _find_destructive_button_at(state.get("elements", []), _cd_pos)):
                                self._executor.execute({
                                    "action_type": "click",
                                    "click_position": _cd_pos,
                                })
                time.sleep(self.step_delay * 0.5)
                continue

            # Repeat-action guard: fingerprint this prediction; if last N identical → Tab
            if self._pure_transformer:
                _action_history.clear()  # don't accumulate in pure-transformer mode
            if _real_progress_this_step:
                # Found 2026-08-09, live, direct user report ("We're in a
                # loop and we're missing certain steps"): a checkbox-check
                # code path already made real, distinguishable progress this
                # step (a genuinely different checkbox, confirmed via
                # BM_SETCHECK) and only reassigned `prediction` afterward to
                # a boilerplate "tab to move on." Fingerprinting THAT
                # compares bookkeeping, not the actual decision -- three
                # DIFFERENT successful checks in a row all end with the
                # identical trailing tab, so this guard used to fire anyway,
                # forcing an extra Tab through a path that skips the normal
                # ExecutionResult/Validator logging (confirmed missing from
                # logs/latest.log around step 111). Skip the guard entirely
                # and let the normal keyboard-tab path below run instead,
                # with its usual logging -- and clear history, since real
                # progress legitimately breaks any actual stuck streak too.
                _action_history.clear()
            else:
                _atype = prediction.get("action_type", "no_op")
                if _atype == "keyboard":
                    _fp = ("keyboard", (prediction.get("text") or "".join(prediction.get("keystrokes", [])))[:40])
                elif _atype == "click":
                    _cp2 = prediction.get("click_position", [0, 0])
                    _fp  = ("click", round(_cp2[0] / 20) * 20, round(_cp2[1] / 20) * 20)
                else:
                    _fp = (_atype,)
                _action_history.append(_fp)
                if (len(_action_history) >= _REPEAT_LIMIT
                        and len(set(_action_history)) == 1):
                    logger.warning("Repeat-action guard: same action %dx in a row — forcing Tab.", _REPEAT_LIMIT)
                    _action_history.clear()
                    _no_change_streak = 0
                    self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]})
                    self._adaptive_settle_wait(self.step_delay * 0.5)
                    continue

            if prediction.get("action_type") == "keyboard":
                self._ensure_form_foreground(state)
                # Clear existing field value before typing to prevent append corruption
                # (e.g. "Maria" + "James" → "MariaJames").  Skip for nav-only actions.
                _pred_text = prediction.get("text", "")
                _pred_keys = prediction.get("keystrokes", [])
                _nav_keys  = {"tab", "return", "enter", "escape", "Key.tab",
                               "Key.return", "Key.enter", "Key.escape"}
                _is_nav_only = not _pred_text and all(k in _nav_keys for k in _pred_keys)
                if not _is_nav_only and _focused_el:
                    # Focus the field before typing into it. Found 2026-08-08,
                    # live: even with verify-at-fill's settle-poll fix, one
                    # field ('Years Continuously Insured', a plain editcontrol,
                    # no different from any working field) kept failing verify
                    # for multiple REAL seconds -- too long to be settle time.
                    # But verify-at-fill's own RETRY path, which re-clicks the
                    # field before retyping, reliably fixed it every time --
                    # evidence the type had reached the field UIA reports as
                    # focused, but real OS keyboard focus hadn't actually
                    # landed there yet: a focus-transition race, not a
                    # value-settle race. The original fix here was a
                    # coordinate click + a flat 0.1s sleep before the first
                    # attempt -- still failing live 2026-08-14 on this exact
                    # field, and separately on 'Years at Address', where a
                    # direct check (reading its real current bbox from the
                    # live UIA tree) found the click coordinate 474px off
                    # from the field's real position -- a second, different
                    # mechanism (stale coordinate, not a race) hitting the
                    # same raw-coordinate-click code path.
                    #
                    # Both are fixed by the same real, generalizing change:
                    # _focus_element_via_uia (already built and proven for
                    # exactly this class of bug -- 'Street Address 1/2',
                    # 2026-08-08, "simulated click missed despite a correct
                    # bbox") uses UIA's SetFocus() directly instead of a
                    # simulated coordinate click. That sidesteps the
                    # focus-transition race entirely (SetFocus is a direct,
                    # synchronous accessibility call, not a queued OS input
                    # event) and sidesteps coordinate staleness entirely (it
                    # looks the control up by its current accessible Name,
                    # not a bbox captured earlier in the decision cycle).
                    # Falls back to the coordinate click only if UIA lookup
                    # fails -- same fallback shape already used at every
                    # other _focus_element_via_uia call site in this file.
                    # Universal for any typed field, not specific to either
                    # one that surfaced it.
                    if _focused_el.get("bbox"):
                        _fcb = _focused_el["bbox"]
                        _fc_label = (_focused_el.get("label") or _focused_el.get("text") or "").strip()
                        _fc_focused = False
                        if _fc_label and prediction.get("direct_fill_eligible"):
                            # Direct-fill eligible: resolve the live Control ourselves
                            # instead of going through _focus_element_via_uia's
                            # bool-only wrapper, so the same lookup that already
                            # focuses the field also hands us its NativeWindowHandle
                            # for the WM_SETTEXT fast path below -- same underlying
                            # UIA call, same SetFocus(), nothing resolved twice.
                            _fc_ctrl = self._resolve_field_control(state, _fc_label, _fcb)
                            if _fc_ctrl is not None:
                                try:
                                    _fc_ctrl.SetFocus()
                                    _fc_focused = True
                                    prediction["direct_fill_hwnd"] = _fc_ctrl.NativeWindowHandle
                                except Exception as _fc_exc:
                                    logger.debug("Direct-fill focus/handle resolution failed for %r — %s",
                                                 _fc_label, _fc_exc)
                        elif _fc_label:
                            _fc_focused = self._focus_element_via_uia(_fc_label, expected_bbox=_fcb)
                        if not _fc_focused:
                            self._executor.execute({"action_type": "click",
                                                    "click_position": [(_fcb[0] + _fcb[2]) / 2,
                                                                        (_fcb[1] + _fcb[3]) / 2]})
                        time.sleep(0.1)
                    _foc_val = (_focused_el.get("value") or "").strip()
                    if _foc_val:
                        logger.info("Pre-type clear: field has %r — ctrl+a.", _foc_val[:40])
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["ctrl+a"]})
            # Foreground guard: before a keystroke, re-assert the window we OBSERVED.
            # If a stray click knocked the form out of front, keystrokes would leak
            # into whatever stole focus (e.g. the terminal → "Windows PowerShell").
            # Universal (re-focus the observed active window), not form-specific.
            if prediction.get("action_type") == "keyboard":
                self._ensure_foreground(state)
            _t_decided = time.time()
            decide_time_sec = _t_decided - _t_observed
            result = self._executor.execute(prediction)
            _t_executed = time.time()
            execute_time_sec = _t_executed - _t_decided
            logger.info("%s", result)

            # Detect tab click: if the executed click landed on a tab element, mark a tab switch
            # so the next step scrolls to top and focuses the first empty field.
            if prediction.get("action_type") == "click":
                _cp = prediction.get("click_position", [0, 0])
                _all_tabs = [e for e in state.get("elements", [])
                             if e.get("type") in ("tabitem", "tabitemcontrol")
                             and e.get("window_role") != "background"
                             and e.get("bbox")]
                for _ti_idx, _ti_el in enumerate(_all_tabs):
                    _tx1, _ty1, _tx2, _ty2 = _ti_el["bbox"]
                    if _tx1 <= _cp[0] <= _tx2 and _ty1 <= _cp[1] <= _ty2:
                        self._current_tab_idx = _ti_idx
                        _tab_just_switched    = True
                        _tab_scroll_count     = 0
                        _no_change_streak     = 0
                        _steps_on_tab         = 0
                        logger.info("Tab click detected: switched to tab idx=%d  '%s'",
                                    _ti_idx,
                                    _ti_el.get("text") or _ti_el.get("label") or "?")
                        # Notify plugin so it also resets its tab-switch state
                        if self._task_plugin is not None and hasattr(self._task_plugin, "notify_tab_click"):
                            self._task_plugin.notify_tab_click(_ti_idx, state)
                        break

            # Auto-Tab DISABLED — it was a navigation crutch (heuristic advancing
            # fields instead of the transformer clicking the next field). With it
            # off, the transformer MUST navigate (click next field) itself, so we
            # measure honestly whether it learned navigation.
            if (False and self._llm_client
                    and prediction.get("action_type") == "keyboard"
                    and prediction.get("text")):
                tab_pred = {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
                self._executor.execute(tab_pred)
                logger.info("Auto-Tab after type.")

            # A typed/pasted value can take a beat to actually land in the
            # control's text buffer before UIA will report it -- taking one
            # snapshot immediately (zero delay) can read the field as still-
            # empty even though the paste genuinely succeeded. Found
            # 2026-08-08, live: 'Years Continuously Insured' failed
            # verify-at-fill 100% of the time across 4 separate visits (always
            # exactly '' on the first check AND both retries -- never partial
            # or differently-wrong) yet held the correct value by the end of
            # the run -- the signature of a race, not a real typing failure.
            # A single fixed delay isn't reliable either: the retry path
            # already sleeps 0.3*step_delay before its own re-check and STILL
            # read empty both times, so the real settle time varies and can
            # exceed a guessed constant. Poll instead -- short checks, break
            # the moment the value actually shows up, so fast fields aren't
            # slowed down and slow ones aren't given up on prematurely.
            # Scoped to keyboard actions only (clicks' "focus moved" check
            # isn't reading typed content, so it isn't exposed to this race).
            # 4. Validate
            # For keyboard actions, check IMMEDIATELY first (zero extra cost
            # for the ~175 fields/step that were never affected by the race
            # above) and only poll with short sleeps if that first read isn't
            # correct yet -- avoids paying the settle-wait on every single
            # typed field just to cover the rare one that actually needs it.
            state_after = None
            if prediction.get("action_type") == "keyboard" and prediction.get("text"):
                _settle_text   = prediction["text"].strip()
                _settle_budget = self.step_delay * 0.4
                _settle_waited = 0.0
                _settle_poll   = 0.1
                state_after = self._observe()
                while _settle_waited < _settle_budget:
                    _settle_fid = state_after.get("focused_element_id")
                    if _verify_fill_matches(state_after.get("elements", []), _settle_fid, _settle_text):
                        break
                    time.sleep(_settle_poll)
                    _settle_waited += _settle_poll
                    state_after = self._observe()
            if state_after is None:
                state_after = self._observe()

            # ── Collapse navigate+fill into one step ──────────────────────────
            # Found 2026-08-09, live, direct user request ("stop wasting
            # steps... look up a lot at once"): a plain navigate click and
            # the fill that follows were always two separate step_idx
            # iterations, even though by the time this line runs, state_after
            # already shows exactly what got focused as a direct result of
            # THIS step's own click -- there's no reason to wait a full extra
            # step just to notice it. Scoped deliberately narrow: only a
            # plain navigate click (_is_plain_navigate_click, not a redirect/
            # combobox-open/checkbox click, all of which have their own
            # already-correct multi-step mechanics), only editcontrol/input
            # (combobox/checkbox fills are multi-click sequences of their
            # own, not a simple collapse candidate), and only when a real
            # record value is found (a genuinely-blank field is left for the
            # next step's own established blank-handling logic, not
            # duplicated here). Reassigning `prediction` to a keyboard/text
            # shape and re-observing `state_after` means every check below
            # this point (verify-at-fill, validation, _record_attempt) runs
            # exactly as it already does for a normal type action -- no
            # separate verification path to build or trust.
            if (_is_plain_navigate_click and prediction.get("action_type") == "click"
                    and state_after.get("focused_element_id")
                    and state_after.get("focused_element_id") != state.get("focused_element_id")):
                _cn_fid = state_after["focused_element_id"]
                _cn_els = state_after.get("elements", [])
                _cn_el = next((e for e in _cn_els if e.get("element_id") == _cn_fid), None)
                if (_cn_el and (_cn_el.get("type") or "").lower() in ("editcontrol", "input")
                        and not (_cn_el.get("value") or "").strip()):
                    _cn_label = (_cn_el.get("label") or _cn_el.get("text") or "").strip()
                    _cn_key = self._attempt_key(_cn_el, elements=_cn_els)
                    if (_cn_label and _cn_key not in self._typed_keys
                            and _cn_key not in self._leave_blank_keys):
                        _cn_sec = self._detect_section(state_after, _cn_el)
                        _cn_val = self._lookup_field(_cn_label, section=_cn_sec)
                        if _cn_val and _cn_el.get("bbox"):
                            logger.info(
                                "[OPT2] collapsing navigate+fill: %r focused this step, "
                                "known value %r — typing now instead of next step.",
                                _cn_label, _cn_val[:40])
                            _cnb = _cn_el["bbox"]
                            # Same _focus_element_via_uia fix as the standard
                            # type path above -- this branch had ZERO settle
                            # time at all between click and paste (not even
                            # the 0.1s the standard path had), the same
                            # raw-coordinate-click risk (focus-transition
                            # race / stale coordinate), collapsed into one
                            # step specifically to save time. Real focus via
                            # UIA SetFocus first; coordinate click only as
                            # fallback.
                            if not (_cn_label and self._focus_element_via_uia(_cn_label, expected_bbox=_cnb)):
                                self._executor.execute({"action_type": "click",
                                                        "click_position": [(_cnb[0] + _cnb[2]) / 2,
                                                                            (_cnb[1] + _cnb[3]) / 2]})
                            self._executor.execute({"action_type": "keyboard", "text": _cn_val})
                            prediction = {"action_type": "keyboard", "text": _cn_val}
                            state_after = self._observe()
                            # _record_attempt()'s keyboard branch (called
                            # further below) reads state["focused_element_id"]
                            # -- the field focused BEFORE this step began, since
                            # that's the correct signal for a normal type
                            # action. Here the target field only became
                            # focused DURING this same step (via the click
                            # just above), so that lookup would find the
                            # WRONG (previous) field. Mark the real target
                            # directly instead of relying on that downstream
                            # call for this one case.
                            self._mark_attempted(_cn_el, elements=_cn_els)
                            self._typed_keys.add(_cn_key)

            # ── Verify-at-fill: catch a WRONG value the moment it's typed, not later ──
            # StateValidator (right below) only checks whether SOMETHING changed —
            # focus moved, value differs from before — never whether it changed to
            # the RIGHT thing. A field can type successfully (validator says "ok")
            # while still holding the wrong value, and nothing used to catch that.
            # Found 2026-08-07 from a direct user report about the old (deleted)
            # Intern iteration lacking exactly this: "we don't want to keep coming
            # back to something... a constant checkback always consumes too much
            # time." Fix is inline and bounded, not a separate re-check pass —
            # verify against the value the agent itself just decided to type
            # (prediction["text"]), retry a couple of times if it doesn't match,
            # then move on regardless so a stubborn field can't stall the run.
            if prediction.get("action_type") == "keyboard" and prediction.get("text"):
                _expected_text = prediction["text"].strip()
                for _verify_attempt in range(_MAX_VERIFY_RETRIES + 1):
                    # Use THIS snapshot's own reported focused_element_id, not the
                    # pre-typing one from `state`. element_id is assigned purely by
                    # scan position ("elem_{offset+count}") — self-consistent WITHIN
                    # one observation, but not stable ACROSS separate observations
                    # (any element gaining/losing text between scans shifts every
                    # id after it). Found 2026-08-07, live: the very first field
                    # ('Policy Number') triggered 2 wasted retries because the
                    # pre-typing id no longer pointed at the same field in the
                    # post-typing snapshot, reading a false empty mismatch even
                    # though the type had actually already succeeded.
                    _foc_id_vf_now = state_after.get("focused_element_id")
                    if _verify_fill_matches(state_after.get("elements", []), _foc_id_vf_now, _expected_text):
                        if _verify_attempt > 0:
                            logger.info("Verify-at-fill: corrected after retry %d.", _verify_attempt)
                        break
                    _actual_elem = _find_element_by_id(state_after.get("elements", []), _foc_id_vf_now)
                    _actual_text = (_actual_elem.get("value") or "").strip() if _actual_elem else ""
                    if _verify_attempt >= _MAX_VERIFY_RETRIES:
                        logger.warning(
                            "Verify-at-fill: still mismatched after %d retries — moving on "
                            "(expected %r, got %r).",
                            _MAX_VERIFY_RETRIES, _expected_text[:40], _actual_text[:40])
                        # "Moving on" must actually move focus off this field — found
                        # live 2026-08-07: it didn't, so a field whose paste genuinely
                        # fails (real OS keyboard focus lagging UIA's reported focus —
                        # the same lag class documented in
                        # execution_stuck_loop_wrong_tab_field) stayed focused+empty,
                        # OPT2's fill-branch re-selected it next step on pure geometry
                        # (focused+empty+fillable), and the run never left that one
                        # field for the rest of the session (87 repeats, one run).
                        # This mirrors the disabled stuck-guard's own escape action
                        # (~L1216) — not re-enabling that guard, just giving this
                        # already-approved give-up path the Tab it was already
                        # claiming to take.
                        self._executor.execute({"action_type": "keyboard",
                                                "key_count": 1, "keystrokes": ["tab"]})
                        time.sleep(self.step_delay * 0.3)
                        break
                    logger.warning("Verify-at-fill: expected %r, got %r — retrying (%d/%d).",
                                    _expected_text[:40], _actual_text[:40],
                                    _verify_attempt + 1, _MAX_VERIFY_RETRIES)
                    # Re-click the field before retyping — don't just trust that real
                    # OS keyboard focus still matches UIA's reported focused element.
                    # A blind ctrl+a/paste retry can't recover from exactly the
                    # failure mode it's trying to fix if focus itself is the problem.
                    if _actual_elem and _actual_elem.get("bbox"):
                        _aeb = _actual_elem["bbox"]
                        self._executor.execute({"action_type": "click",
                                                "click_position": [(_aeb[0] + _aeb[2]) / 2,
                                                                    (_aeb[1] + _aeb[3]) / 2]})
                        time.sleep(0.1)
                    self._executor.execute({"action_type": "keyboard",
                                            "key_count": 1, "keystrokes": ["ctrl+a"]})
                    self._executor.execute({"action_type": "keyboard", "text": _expected_text})
                    time.sleep(self.step_delay * 0.3)
                    state_after = self._observe()

            validation  = self._validator.validate(state, state_after, prediction)
            logger.info("Validator → %s: %s", validation.status, validation.reason)

            # Record the field acted on this step (attempted feature) — mirrors the
            # train-time derivation (every click/type target becomes 'attempted').
            self._record_attempt(state, prediction, state_after=state_after)

            if validation.status == "done":
                logger.info("StateValidator: task appears complete.")
                break

            _cur_focused_id = state.get("focused_element_id")
            # Track scroll-specific no-change so we can break scroll loops
            if validation.status == "no_change" and prediction.get("action_type") == "scroll":
                self._scroll_nochange = getattr(self, "_scroll_nochange", 0) + 1
            else:
                self._scroll_nochange = 0
            if validation.status == "no_change":
                if _cur_focused_id and _cur_focused_id != _last_focused_id:
                    _no_change_streak = 0
                else:
                    _no_change_streak += 1
                # Blacklist click position so transformer won't override LLM type there again
                if prediction.get("action_type") == "click":
                    _fcp = prediction.get("click_position", [])
                    if len(_fcp) >= 2:
                        self._nochange_click_pos.add((round(_fcp[0] / 10) * 10, round(_fcp[1] / 10) * 10))
            else:
                _no_change_streak = 0
            _last_focused_id = _cur_focused_id

            # Combobox open/close tracking — mark field filled when dropdown closes
            _n_before = len(state.get("elements", []))
            _n_after  = len(state_after.get("elements", []))
            _delta    = _n_after - _n_before
            if _delta > 2:
                # Dropdown opened — record which combobox is open
                _foc_id_cb = state.get("focused_element_id")
                _foc_cb = next((e for e in state.get("elements", [])
                                if e.get("element_id") == _foc_id_cb), None)
                if _foc_cb and _foc_cb.get("type") in ("comboboxcontrol", "combobox"):
                    _ocb_bare = (_foc_cb.get("label") or _foc_cb.get("text") or "").strip()
                    _ocb_sec  = self._detect_section(state, _foc_cb)
                    _open_combobox_label = f"{_ocb_sec} {_ocb_bare}" if _ocb_sec else _ocb_bare
            elif _delta < -2 and _open_combobox_label and validation.status == "ok":
                # Dropdown closed after a successful action — mark combobox as filled
                logger.info("Combobox-close tracking: marking %r as filled.", _open_combobox_label)
                self._filled_this_tab.add(_open_combobox_label)
                _open_combobox_label = ""

            # Notify plugin of validation result so it can update its own streak tracking
            if self._task_plugin is not None:
                self._task_plugin.notify_validation(validation.status, _cur_focused_id)
                # Sync plugin streak back to local so stuck guard in run() stays accurate
                if hasattr(self._task_plugin, "_no_change_streak"):
                    _no_change_streak = self._task_plugin._no_change_streak

            if (validation.status in ("no_change", "unexpected", "error") and self._task_name
                    and self._correction_watch_seconds > 0):
                logger.info("Validation failed (%s) — watching for user correction (%.1fs) …",
                            validation.status, self._correction_watch_seconds)
                steps = self._correction.watch(self._observer, seconds=self._correction_watch_seconds)
                if steps:
                    saved = self._correction.save(self._task_name, steps)
                    if saved:
                        logger.info("Correction saved → %s", saved)
                        _manual_interventions += 1

            # 7. Record
            pos = prediction.get("click_position", [0.0, 0.0])
            res = state.get("screen_resolution", [1920, 1080])
            W, H = float(res[0]) or 1920, float(res[1]) or 1080
            self._history.append({
                "state":       state,
                "action_type": prediction.get("action_type", "no_op"),
                # Set only by a semantic-mode model's predict() (see
                # transformer.py's SEMANTIC_VERBS block) -- None for legacy
                # models. Lets predict() read its own history back in the
                # exact verb space it was trained on instead of reconstructing
                # it (lossily) from the collapsed legacy action_type string.
                "verb":        prediction.get("verb"),
                "click_xy":    [pos[0] / W, pos[1] / H] if pos else [0.0, 0.0],
                "key_count":   prediction.get("key_count", 0),
                "typed_text":  prediction.get("text", ""),
                "target":      llm_action.get("target", "") if self._llm_client and llm_action else "",
                "validation":  validation.status,
            })
            _step_time_sec = time.time() - _step_t0
            if (self._time_to_first_action_sec is None
                    and prediction.get("action_type") not in (None, "no_op", "noop", "wait")):
                self._time_to_first_action_sec = time.time() - _run_start_ts
            self._results.append({
                "step":              step_idx + 1,
                "state":             state,
                "action":            prediction,
                "result":            str(result),
                "validation":        validation.status,
                "guidance":          self._guidance,
                "elements":          len(state.get("elements", [])),
                "decision_by":       _decision_maker,
                "t_conf":            t_conf,
                "timestamp":         _dt.datetime.now().isoformat(),
                "observe_time_sec":  round(observe_time_sec, 4),
                "decide_time_sec":   round(decide_time_sec, 4),
                "execute_time_sec":  round(execute_time_sec, 4),
                "step_time_sec":     round(_step_time_sec, 4),
            })

            if not result.success:
                # Skip-and-continue: one failed action shouldn't kill the whole
                # run. Count consecutive failures and only abort if stuck.
                logger.warning("Execution failed — skipping: %s", result.error)
                _exec_fail_streak = getattr(self, "_exec_fail_streak", 0) + 1
                self._exec_fail_streak = _exec_fail_streak
                if _exec_fail_streak >= 8:
                    logger.error("8 consecutive execution failures — halting.")
                    break
                time.sleep(self.step_delay)
                continue
            self._exec_fail_streak = 0

            # Adaptive settle-wait (2026-08-14), not a blind time.sleep(self.
            # step_delay) -- see _adaptive_settle_wait's own docstring. Only
            # the SUCCESS path (this one) gets the adaptive treatment; the
            # failure-path sleep just above (L4936) keeps its full, blind
            # wait deliberately -- a failed execution is exactly the case
            # this project's own prior incident (checking the screen too
            # soon) warns against being clever about.
            self._adaptive_settle_wait(self.step_delay)

          except Exception as _step_exc:
            import traceback as _tb
            _tb_str = _tb.format_exc()
            logger.error("STEP %d CRASHED:\n%s", step_idx + 1, _tb_str)
            print(f"\n=== STEP {step_idx+1} CRASHED ===\n{_tb_str}", flush=True)
            break

        self._heuristic_steps       = _heuristic_steps
        self._manual_interventions  = _manual_interventions
        self._run_duration_sec = time.time() - _run_start_ts
        logger.info(
            "LLMAgent finished — %d step(s)  (%d heuristic)  duration=%.1fs  time_to_first_action=%s",
            len(self._results), _heuristic_steps, self._run_duration_sec,
            f"{self._time_to_first_action_sec:.1f}s" if self._time_to_first_action_sec is not None else "n/a",
        )
        self._export_run_traces(self._results, task_name)
        return list(self._results)

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def _export_run_traces(self, results: List[Dict[str, Any]], task_name: str) -> None:
        """
        Convert completed run results to trace JSONs and write to disk.

        Only exports steps where the agent's action had a real effect (validation != no_change
        and action_type != no_op) — bad steps teach bad behavior.

        Output: data/output/traces/agent_runs/session_{timestamp}/live_step_{idx:04d}.json
        Compatible with TrajectoryDataset — ContinualLearner picks these up automatically.
        """
        good = [
            r for r in results
            if r.get("validation") != "no_change"
            and r.get("action", {}).get("action_type", "no_op") not in ("no_op", "noop", "wait", None)
        ]
        if len(good) < 3:
            logger.info("Trace export: only %d usable steps — skipping (run too short/failed).", len(good))
            return

        import datetime as _dt, json as _json
        session_ts  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_tag = task_name or "run"
        out_dir     = os.path.join(_ROOT, "data", "output", "traces", "agent_runs",
                                   f"session_{session_ts}_{session_tag}")
        os.makedirs(out_dir, exist_ok=True)

        written = 0
        for i, r in enumerate(good):
            state  = r.get("state", {})
            action = r.get("action", {})
            a_type = action.get("action_type", "no_op")

            res = state.get("screen_resolution", [1920, 1080])
            W   = float(res[0]) or 1920.0

            if a_type == "click":
                pos = action.get("click_position") or [0.0, 0.0]
                mouse_actions = [{"position": [float(pos[0]), float(pos[1])],
                                  "type": "click",
                                  "timestamp": r.get("state", {}).get("timestamp", "")}]
                kb_actions = []
            elif a_type == "keyboard":
                mouse_actions = []
                text = action.get("text", "") or ""
                key_count = action.get("key_count", 0)
                if text:
                    strokes = [{"pasted_text": text, "key": ""}]
                else:
                    strokes = [{"key": "tab", "pasted_text": ""}] * max(1, key_count)
                kb_actions = [{"strokes": strokes}]
            else:
                continue  # skip anything else (scroll, done markers, etc.)

            trace = {
                "trace_id":  f"live_step_{i:04d}",
                "timestamp": session_ts,
                "duration":  1.5,
                "type":      session_tag,
                "state":     state,
                "mouse":     {"actions": mouse_actions},
                "keyboard":  {"actions": kb_actions},
                "diff":      {},
                "action":    a_type,
            }
            out_path = os.path.join(out_dir, f"live_step_{i:04d}.json")
            try:
                with open(out_path, "w", encoding="utf-8") as _f:
                    _json.dump(trace, _f, ensure_ascii=False)
                written += 1
            except Exception as _e:
                logger.debug("Trace export: failed writing step %d: %s", i, _e)

        logger.info("Trace export: %d step(s) → %s", written, out_dir)

    # ── Auto-skip helper ─────────────────────────────────────────────────────

    def _auto_skip(self, state: Dict[str, Any]) -> bool:
        """
        Return True if the currently focused field already contains the value
        that BACKGROUND DATA says it should have — no LLM call needed.
        Only applies to non-empty fields (text inputs and dropdowns).
        """
        elements   = state.get("elements", [])
        focused_id = state.get("focused_element_id")
        if not focused_id:
            return False
        focused = next((e for e in elements if e.get("element_id") == focused_id), None)
        if not focused:
            return False
        # Skip buttons — they are not fillable fields
        if focused.get("type") in ("buttoncontrol",):
            return False
        current = (focused.get("value") or "").strip()

        field_name = (focused.get("label") or focused.get("text") or "").strip()
        if not field_name:
            return False

        # Use cached record (full Notepad text); fall back to bg_blobs
        rec: Dict[str, str] = {}
        if self._cached_record:
            rec = self._cached_record
        else:
            bg_elems = [e for e in elements if e.get("window_role") == "background"]
            bg_blobs = [(e.get("value") or "").strip() for e in bg_elems if e.get("value")]
            for blob in sorted(bg_blobs, key=len, reverse=True):
                r = _parse_records(blob)
                if r:
                    rec = r.get(self._record_num, r.get(min(r), {}))
                    break

        if not rec:
            return False

        # Detect Driver/Vehicle section context for section-prefixed lookup.
        section  = self._detect_section(state, focused)
        filled_key = f"{section} {field_name}" if section else field_name
        # Duplicate-label guard: if we already processed a field with this label this tab,
        # don't auto-skip the second occurrence — it's a different field (e.g. Prior Policy No.
        # reported as 'Prior Insurer' in UIA). Let the dup-label handler deal with it.
        if filled_key in self._filled_this_tab:
            return False
        fl = field_name.lower()
        expected = None
        if section:
            sec_key = f"{section} {field_name}"
            expected = rec.get(sec_key) or next(
                (v for k, v in rec.items() if k.lower() == sec_key.lower()), None)
        if expected is None:
            expected = rec.get(field_name) or next(
                (v for k, v in rec.items() if k.lower() == fl), None)
        if expected is None:
            return False   # field not in record — let LLM decide

        # (leave blank) / (none) in data → leave field as-is regardless of current value
        if _is_leave_blank_value(expected):
            logger.info("Auto-skip: '%s' = (leave blank) — Tab.", filled_key)
            self._filled_this_tab.add(filled_key)
            return True

        if not current:
            return False   # empty field that needs filling — let auto-fill handle it

        match = current.lower() == expected.lower()
        if match:
            logger.info("Auto-skip: '%s' already = %r", filled_key, current)
            self._filled_this_tab.add(filled_key)
        return match

    def _read_notepad_full_text(self, state: Dict[str, Any]) -> str:
        """
        Read the full source text via the injected data source. Source-specific
        I/O (Win32 WM_GETTEXT for Notepad, COM for Excel, …) lives in the adapter,
        not here — this is just the agent's call into the seam.
        """
        try:
            return self._source.read_full_text(state) or ""
        except Exception:
            return ""

    def _refresh_record_cache(self, state: Dict[str, Any]) -> None:
        """
        Populate self._cached_record with {field: value} for the current record.
        Priority:
          1. Visual cache (Gemini pre-scan) — used when available; no Win32 needed.
          2. Win32 Notepad text read → parse records.
          3. UIA background element blobs / OCR fallback.

        Skips the whole read+parse when this exact record's cache is
        already confirmed populated. Found live 2026-08-14 ("any more
        improvements"): a single-record run logged 31 refreshes, every
        one returning the IDENTICAL 176-field sample already sitting in
        self._cached_record — 13 separate call sites across run()
        (nearly every one a tab-advance handler, 7 tab-advances/record)
        each pay for a real Win32 Notepad text read + full regex parse
        on every single tab switch, even though the record's own data
        never changes mid-record (this is a static intake snapshot, not
        a live-edited file). self._attempted_record_num already tracks,
        reliably, which record_num the cache last completed a genuine
        structured parse for (set only after `_parse_records` actually
        succeeds, a few lines below) — reusing that existing invariant
        instead of inventing a new one. Skipping when it already matches
        self._record_num returns exactly the same cache a real re-read
        would have produced, just without paying for the read again.
        Record-advance still refreshes for real: _record_num changes
        before this is called for the new record, so the guard doesn't
        match and the reset logic (attempted-keys, tab-idx, etc. — a few
        lines below) still runs exactly as it always has.

        Deliberately excludes visual_reader mode: that branch (below)
        never touches self._attempted_record_num at all, so the
        "confirmed populated" invariant this guard relies on was never
        established for it -- skipping there would be trusting a
        guarantee that path never actually made. Left exactly as it
        already behaves today; not the path any real live run uses
        (every log this session shows "VLM pre-scan skipped").
        """
        if (not self._visual_reader and self._cached_record
                and self._record_num == self._attempted_record_num):
            return
        import re

        # When visual_reader is active, use ONLY what the VLM has seen on screen.
        # Win32 reads Notepad memory directly (agent "knows" without looking) — skip it.
        if self._visual_reader:
            self._cached_record = dict(self._visual_cache)
            sample = list(self._cached_record.items())[:5]
            logger.info("Record cache: VLM-only mode — %d field(s) from visual cache  sample=%r",
                        len(self._cached_record), sample)
            return

        full_text = self._read_notepad_full_text(state)

        # Fallback: reconstruct text from background elements (UIA or OCR)
        if not full_text:
            elements = state.get("elements", [])
            bg_elems = [e for e in elements if e.get("window_role") == "background"]

            # Try UIA blobs first (they carry full multi-line text in a single value)
            blobs = sorted(
                [(e.get("value") or "").strip() for e in bg_elems],
                key=len, reverse=True
            )
            for blob in blobs:
                if blob and _parse_records(blob):
                    full_text = blob
                    logger.info("Record cache: Win32 failed — using UIA blob (%d chars)", len(blob))
                    break
            if not full_text and blobs and blobs[0]:
                full_text = blobs[0]   # longest UIA blob even if unstructured
                logger.info("Record cache: Win32 failed — using largest UIA blob (%d chars)", len(full_text))

            # If UIA blobs were empty, try OCR elements: group by y-coord to rebuild lines
            if not full_text:
                ocr_elems = [e for e in bg_elems
                             if e.get("source") == "ocr" and (e.get("text") or "")]
                if ocr_elems:
                    ocr_elems.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]) if e.get("bbox") else (0, 0))
                    lines, cur_line, cur_y = [], [], None
                    for e in ocr_elems:
                        y = e["bbox"][1] if e.get("bbox") else 0
                        if cur_y is None or abs(y - cur_y) <= 10:
                            cur_line.append(e.get("text", ""))
                        else:
                            if cur_line:
                                lines.append(" ".join(cur_line))
                            cur_line = [e.get("text", "")]
                        cur_y = y
                    if cur_line:
                        lines.append(" ".join(cur_line))
                    full_text = "\n".join(lines)
                    logger.info("Record cache: Win32 failed — rebuilt %d lines from OCR boxes", len(lines))

        if not full_text:
            logger.warning("Record cache: no Notepad text available — cache unchanged (%d fields)", len(self._cached_record))
            return

        records = _parse_records(full_text)
        if records:
            self._total_records = _total_records_in(records)
            rec = records.get(self._record_num, records.get(min(records), {}))
            self._cached_record = rec
            # New record = new session → clear attempted history so fields on the
            # fresh record aren't pre-marked from the previous one. Also clear the
            # accumulated tab-advance blacklist (_advance_blacklist_pos) — tabs
            # left "exhausted" on the previous record must not stay blocked on
            # this one.
            if self._record_num != self._attempted_record_num:
                self._attempted_keys.clear()
                self._typed_keys.clear()
                self._advance_blacklist_pos.clear()
                self._leave_blank_keys.clear()
                # Found live 2026-08-13: _checked_fields was the one piece of
                # per-record state missing from this reset. Its own __init__
                # comment says "checkboxes already clicked this run" but it
                # was never cleared anywhere -- so once a checkbox got
                # checked in ANY earlier record, both the uncheck-branch's
                # own guard (`_flabel not in self._checked_fields`, ~L3935)
                # and the click-guard (`_chk_label in self._checked_fields`,
                # ~L4032) treated it as permanently off-limits, even when a
                # later record's data wants the opposite state. Reproduced
                # directly from a live run's log: 'Renewal Policy' checked
                # during an earlier record, record 2 wants 'NO' (unchecked) --
                # the uncheck-branch never fires (guard always False), so the
                # agent just redirects off the field and back onto it forever
                # (4+ identical no_change steps before the run was stopped).
                self._checked_fields.clear()
                # car_insurance_form_wx.py's own _on_submit() always calls
                # self.nb.SetSelection(0) — a real Submit puts the FORM on
                # Policy tab regardless of where the agent started (drill mode
                # included), so the agent's own idea of "current tab" must
                # follow suit here or it goes stale — found live 2026-08-08
                # while fixing execution_backward_tab_click_guard: without
                # this, a backward-tab-click guard added at the same time
                # would have wrongly blocked the legitimate Payment -> Policy
                # reset for the next record, mistaking it for the exact
                # erroneous backward jump that guard exists to catch.
                self._current_tab_idx = 0
                self._attempted_record_num = self._record_num
            sample = list(rec.items())[:5]
            logger.info("Record cache refreshed: %d fields for record %d  sample=%r",
                        len(rec), self._record_num, sample)
        else:
            # Plain key:value text — parse directly
            data: Dict[str, str] = {}
            for line in full_text.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = re.sub(r"[\[\]]", "", key).strip()
                    val = re.sub(r"\s*\[.*", "", val)
                    val = re.sub(r"\s*←.*", "", val).strip()
                    if key and key not in data and val and val.lower() not in {"(none)", "none", "(leave blank)"}:
                        data[key] = val
            self._cached_record = data
            sample = list(data.items())[:5]
            logger.info("Record cache refreshed (plain text): %d fields  sample=%r", len(data), sample)

        # ── Overlay visual cache on top: Groq/Gemini values win on conflicts ──
        if self._visual_cache:
            before = len(self._cached_record)
            self._cached_record.update(self._visual_cache)
            logger.info("Record cache: merged %d visual cache field(s) on top of %d Win32 fields",
                        len(self._visual_cache), before)

    def _scan_tab_visual(self, state: Dict[str, Any]) -> None:
        """
        When a form tab switches, scroll the source Notepad to the matching
        TAB N — TABNAME section for the current record, then use the visual
        reader (Groq/Gemini) to capture all field values by scrolling downward.
        New fields are merged into self._visual_cache and self._cached_record.
        """
        if not self._visual_reader or not self._source_window:
            return

        # Resolve current tab name from _current_tab_idx + state tab elements
        _KNOWN_TABS = self._scope.tab_names   # scope-provided; empty → no tab filter
        elements = state.get("elements", [])
        tabs = [
            e for e in elements
            if e.get("type") in ("tabitem", "tabitemcontrol")
            and e.get("window_role") != "background"
            and e.get("bbox")
            and (not _KNOWN_TABS
                 or (e.get("text") or e.get("label") or "").strip().lower() in _KNOWN_TABS)
        ]
        if not tabs or self._current_tab_idx >= len(tabs):
            return
        tab_name = (tabs[self._current_tab_idx].get("text")
                    or tabs[self._current_tab_idx].get("label") or "").strip()
        if not tab_name:
            return

        # Get full Notepad text for line-number lookup
        full_text = self._read_notepad_full_text(state) or ""
        if not full_text:
            return

        logger.info("Visual tab scan: scanning Notepad for tab=%r record=%d",
                    tab_name, self._record_num)
        new_fields = self._visual_reader.scan_tab(
            tab_name, self._record_num, full_text, self._source_window
        )

        if new_fields:
            self._visual_cache.update(new_fields)
            self._cached_record.update(new_fields)
            logger.info("Visual tab scan: %d new field(s) added to cache (total visual=%d)",
                        len(new_fields), len(self._visual_cache))

    def _peek_next_field_after(self, state: Dict[str, Any], after_label: str) -> Optional[tuple]:
        """
        Read Notepad text and return (key, value) of the first field that appears
        AFTER after_label in the document and has NOT yet been cached in _cached_record.
        Used to resolve duplicate-UIA-label fields (e.g. 'Prior Policy No.' reported as
        'Prior Insurer' by UIA).  Returns None if not found.
        """
        full_text = self._read_notepad_full_text(state)
        if not full_text:
            return None
        lines = full_text.splitlines()
        fl = after_label.lower()
        found_target = False
        for line in lines:
            stripped = line.strip()
            if not found_target:
                if fl in stripped.lower() and ":" in stripped:
                    found_target = True
                continue
            if ":" in stripped:
                colon_idx = stripped.index(":")
                k = stripped[:colon_idx].strip()
                v = stripped[colon_idx + 1:].strip()
                if k and v and k not in self._cached_record:
                    logger.info("_peek_next_field_after: '%s' → next uncached field='%s' value=%r",
                                after_label, k, v)
                    return (k, v)
        return None

    def _peek_notepad(self, state: Dict[str, Any], field_name: str) -> None:
        """
        Scroll Notepad to the line containing field_name, then hover the
        mouse over it — gives the visual impression of reading.
        Uses win32 EM_LINESCROLL so focus never leaves the form.
        Works with both classic Notepad and Windows 11 Notepad (WinUI3).
        When VLM (visual_reader) is active:
          - Field already in visual cache → skip (VLM already read it).
          - Field NOT in visual cache → Win32 fallback: scroll, extract value, cache it, log.
        When no VLM: original scroll-and-hover UX behaviour.
        """
        _is_vlm_mode = bool(self._visual_reader)
        if field_name in self._cached_record:
            return   # already have a value — no need to peek
        try:
            import pyautogui
            import win32gui
            import win32api

            EM_GETFIRSTVISIBLELINE = 0x00CE
            EM_LINESCROLL          = 0x00B6

            # Find the raw text and background window title from state.
            # Prefer text-file windows (.txt / Notepad) over terminal windows
            # (PowerShell/CMD accumulate huge log buffers that confuse detection).
            # Separate two concerns:
            #   bg_window_title — used ONLY to find the Notepad hwnd; prefer .txt windows
            #   raw_text        — used ONLY for line-number lookup; use largest blob from
            #                     any background window (Win11 Notepad exposes only 1 char
            #                     via UIA, but PowerShell accumulates full file content)
            _TERMINAL_HINTS = {"powershell", "terminal", "command prompt", "cmd.exe"}
            elements = state.get("elements", [])
            bg_window_title = ""   # hwnd-lookup title (.txt preferred)
            raw_text        = ""   # largest blob for line-number lookup
            for e in elements:
                if e.get("window_role") != "background":
                    continue
                win_title = (e.get("window_title") or "").strip()
                blob      = (e.get("value") or "").strip()
                # Track the .txt / Notepad window for hwnd regardless of blob size
                if not bg_window_title:
                    is_textfile = ".txt" in win_title or "notepad" in win_title.lower()
                    if is_textfile:
                        bg_window_title = win_title
                # Track largest blob from any background window for line lookup
                if blob and len(blob) > len(raw_text):
                    raw_text = blob

            # Prefer full Win32 text (bypasses the 2000-char UIA cap) for accurate line lookup.
            # Fall back to raw_text from bg_blobs if Win32 read fails.
            full_np_text = self._read_notepad_full_text(state)
            if full_np_text:
                raw_text = full_np_text

            logger.info("_peek_notepad: field=%r  bg_title=%r  raw_text_len=%d",
                        field_name, bg_window_title, len(raw_text))
            if not raw_text:
                logger.info("_peek_notepad: no background text found — skipping scroll")
                return

            lines = raw_text.splitlines()
            from data_sources.notepad_source import _find_field_line
            hit = _find_field_line(lines, field_name)
            target_line = hit[0] if hit else 0
            fl = field_name.lower()

            # ── Find the Notepad window ───────────────────────────────────────
            np_hwnd = None

            # Strategy 1: use exact title from UIA background element (most reliable)
            if bg_window_title:
                np_hwnd = win32gui.FindWindow(None, bg_window_title)
                if not np_hwnd:
                    # Title might include app name suffix, e.g. "file.txt - Notepad"
                    np_hwnd = win32gui.FindWindow(None, bg_window_title + " - Notepad")

            # Strategy 2: enumerate top-level windows by class / title heuristic
            if not np_hwnd:
                def _find_np(hwnd, _):
                    nonlocal np_hwnd
                    if np_hwnd:
                        return
                    if not win32gui.IsWindowVisible(hwnd):
                        return
                    title = win32gui.GetWindowText(hwnd)
                    cls   = win32gui.GetClassName(hwnd)
                    if cls == "Notepad":
                        np_hwnd = hwnd
                    elif ".txt" in title or "notepad" in title.lower():
                        np_hwnd = hwnd
                win32gui.EnumWindows(_find_np, None)

            logger.info("_peek_notepad: np_hwnd=%s  title=%r",
                        np_hwnd, win32gui.GetWindowText(np_hwnd) if np_hwnd else None)
            if not np_hwnd:
                return

            # ── Find the edit control (classic or Win11) ─────────────────────
            _EDIT_CLASSES = {"Edit", "RichEditD2DPT", "RichEdit20W", "RICHEDIT50W"}
            edit_hwnd = None

            def _find_edit(hwnd, _):
                nonlocal edit_hwnd
                if edit_hwnd:
                    return
                try:
                    cls = win32gui.GetClassName(hwnd)
                    if cls in _EDIT_CLASSES:
                        edit_hwnd = hwnd
                except Exception:
                    pass

            # Enumerate direct children first
            win32gui.EnumChildWindows(np_hwnd, _find_edit, None)

            # Win11 Notepad nests the editor deeper — walk one level of grandchildren
            if not edit_hwnd:
                child = win32gui.GetWindow(np_hwnd, 5)   # GW_CHILD
                while child and not edit_hwnd:
                    win32gui.EnumChildWindows(child, _find_edit, None)
                    child = win32gui.GetWindow(child, 2)  # GW_HWNDNEXT

            logger.info("_peek_notepad: edit_hwnd=%s  target_line=%d", edit_hwnd, target_line)
            if not edit_hwnd:
                return

            # ── Scroll to target line ─────────────────────────────────────────
            first_visible = win32api.SendMessage(edit_hwnd, EM_GETFIRSTVISIBLELINE, 0, 0)
            delta = target_line - first_visible - 3   # show 3 lines above target
            logger.info("_peek_notepad: first_visible=%d  delta=%d", first_visible, delta)
            if delta != 0:
                win32api.SendMessage(edit_hwnd, EM_LINESCROLL, 0, delta)

            # ── Default: parse value directly from Win32 text (no API call) ──
            # Exact key match preferred; whole-word fallback inside _find_field_line.
            if hit and hit[1]:
                _val = hit[1]
                self._visual_cache[field_name]  = _val
                self._cached_record[field_name] = _val
                logger.info("_peek_notepad: Win32 text found field=%r  value=%r",
                            field_name, _val)
                return

            if _is_vlm_mode and self._visual_reader:
                # ── VLM fallback: scroll Notepad until Groq sees the field ──────
                form_hwnd = win32gui.GetForegroundWindow()   # save form focus
                try:
                    self._visual_reader._force_foreground(np_hwnd)
                    time.sleep(0.4)
                    # pyautogui.scroll targets cursor window — move mouse over Notepad
                    try:
                        rect = win32gui.GetWindowRect(np_hwnd)
                        pyautogui.moveTo((rect[0]+rect[2])//2, (rect[1]+rect[3])//2,
                                         duration=0.1)
                    except Exception:
                        pass

                    found_val   = None
                    seen_hashes : set = set()
                    fl_lower    = field_name.lower()

                    for attempt in range(10):   # scroll up to 10 screens to find field
                        screenshot = pyautogui.screenshot()
                        img_hash   = hash(screenshot.tobytes())

                        if img_hash not in seen_hashes:
                            seen_hashes.add(img_hash)
                            new_fields = self._visual_reader._extract_from_screenshot(screenshot)
                            for k, v in new_fields.items():
                                norm = k.strip()
                                if norm and norm not in self._visual_cache:
                                    self._visual_cache[norm]  = str(v).strip()
                                    self._cached_record[norm] = str(v).strip()
                                if norm.lower() == fl_lower:
                                    found_val = str(v).strip()

                        if found_val:
                            logger.info("_peek_notepad: VLM fallback found field=%r  value=%r"
                                        "  (attempt %d)", field_name, found_val, attempt + 1)
                            break

                        # Field not seen yet — scroll down and try again
                        logger.info("_peek_notepad: VLM fallback scrolling down for field=%r"
                                    "  (attempt %d)", field_name, attempt + 1)
                        pyautogui.scroll(-10)
                        time.sleep(0.35)

                    if not found_val:
                        logger.warning("_peek_notepad: VLM fallback exhausted — "
                                       "field=%r not found on screen", field_name)
                finally:
                    try:
                        win32gui.SetForegroundWindow(form_hwnd)
                    except Exception:
                        pass
            else:
                # ── Non-VLM mode: hover mouse over Notepad (purely visual UX) ──
                rect = win32gui.GetWindowRect(np_hwnd)
                cx   = (rect[0] + rect[2]) / 2
                cy   = (rect[1] + rect[3]) / 2
                orig = pyautogui.position()
                pyautogui.moveTo(cx, cy, duration=0.25)
                time.sleep(0.4)
                pyautogui.moveTo(orig.x, orig.y, duration=0.2)

        except Exception:
            pass   # never block the agent over a cosmetic action

    def _lock_form_window(self, state: Dict[str, Any]) -> None:
        """Capture the form's window handle once (the window in front at 'GO'),
        so we can re-assert it every step and never drift into another window."""
        if self._locked_hwnd is not None:
            return
        try:
            import win32gui
            title = (state.get("window_title") or "").strip()
            hwnd = win32gui.FindWindow(None, title) if title else win32gui.GetForegroundWindow()
            if hwnd:
                self._locked_hwnd  = hwnd
                self._locked_title = title
                logger.info("Form window LOCKED: %r (hwnd=%s)", title[:40], hwnd)
        except Exception as exc:
            logger.debug("Form-lock failed: %s", exc)

    def _reassert_form_window(self) -> None:
        """Bring the locked form back to foreground if focus drifted away. Runs at
        the top of every step → the model physically cannot act on another window.

        SAFETY: backs off if the same foreign window keeps recurring across
        consecutive steps -- see the streak-breaker just below. Added
        2026-08-10 after a real, severe incident: this method used to fight
        for foreground completely unconditionally, every single step
        (~1-2s cadence), including using a fake Alt-keypress specifically to
        defeat Windows' own anti-focus-stealing protection. The user tried
        to click over to the Electron Stop button and it kept getting
        yanked back before they could act -- they had to hard-shutdown
        their laptop to regain control of their own mouse. That is never
        acceptable, regardless of how well-intentioned the anti-drift
        design was."""
        if not self._locked_hwnd:
            return
        try:
            import win32gui
            fg = win32gui.GetForegroundWindow()
            if fg == self._locked_hwnd:
                self._reassert_foreign_hwnd   = None
                self._reassert_foreign_streak = 0
                return
            if not win32gui.IsWindow(self._locked_hwnd):
                return

            # Streak-breaker: a single stray drift-and-recover is normal,
            # expected behavior (e.g. a combobox dropdown transiently taking
            # foreground) and this method should still handle it. But if the
            # SAME foreign window is back again on the VERY NEXT step despite
            # having just been stolen from, that's not drift -- it's the user
            # deliberately returning to it (most critically: trying to reach
            # Stop). Back off for the rest of THIS window's occurrences
            # rather than fighting them again.
            if fg == self._reassert_foreign_hwnd:
                self._reassert_foreign_streak += 1
            else:
                self._reassert_foreign_hwnd   = fg
                self._reassert_foreign_streak = 1
            if self._reassert_foreign_streak >= 2:
                # DIAGNOSTIC, added 2026-08-10: direct live report, "clicked
                # the form, literally nothing happened" -- the bare hwnd
                # number this warning used to log is useless after the fact
                # (the window may no longer exist, and a number alone can't
                # distinguish "the user genuinely clicked away" from "some
                # OTHER window -- the ghost overlay, an unrelated popup, or
                # a same-process wx dialog our own modal-dismiss logic a few
                # lines below never got a chance to run on, since THIS
                # branch returns before reaching it -- is silently sitting
                # on top". Best-effort title/class/same-process lookup so
                # the NEXT occurrence gives a real answer instead of another
                # guess. Never allowed to raise or block backing off, which
                # is the actual safety-critical behavior here.
                _fg_title = _fg_class = "?"
                _fg_same_process = None
                try:
                    _fg_title = win32gui.GetWindowText(fg) or "(no title)"
                    _fg_class = win32gui.GetClassName(fg) or "?"
                    import win32process as _wp
                    _, _fg_pid = _wp.GetWindowThreadProcessId(fg)
                    _, _form_pid = _wp.GetWindowThreadProcessId(self._locked_hwnd)
                    _fg_same_process = (_fg_pid == _form_pid)
                except Exception:
                    pass
                logger.warning(
                    "NOT re-stealing foreground -- hwnd=%s title=%r class=%r "
                    "same_process_as_form=%s has held focus across %d "
                    "consecutive steps despite being reclaimed; treating this "
                    "as deliberate (e.g. the user reaching for Stop), not drift.",
                    fg, _fg_title, _fg_class, _fg_same_process, self._reassert_foreign_streak)
                return

            # A MODAL dialog owned by the SAME process as the locked form (e.g. a
            # wx.MessageBox popped by clicking Submit) blocks SetForegroundWindow
            # from ever winning the form back — Windows won't foreground an owner
            # window while its modal child is open, so re-asserting alone loops
            # forever. Detect this generically (same PID, different hwnd — no
            # hardcoded dialog titles/text) and dismiss it with Escape first.
            try:
                import win32process
                _, form_pid = win32process.GetWindowThreadProcessId(self._locked_hwnd)
                _, fg_pid   = win32process.GetWindowThreadProcessId(fg)
                if fg and fg_pid == form_pid:
                    # DIAGNOSTIC, added 2026-08-08: direct user report, "still
                    # going one field at a time" — log evidence showed THIS
                    # warning firing on every single _try_advance_tab click,
                    # unconditionally, for the rest of a run that never
                    # recovered (SCROLL-DIAG kept tracking the SAME field
                    # across 7+ supposedly-different tabs, meaning the real
                    # on-screen tab likely never actually changed). Two
                    # competing explanations, both unconfirmed: (a) this really
                    # is a modal dialog and Escaping it is correct, but
                    # something else is separately preventing the tab switch
                    # from sticking; or (b) wx's Notebook implementation gives
                    # each tab panel its own child HWND, so a plain tab switch
                    # legitimately changes GetForegroundWindow() to a normal
                    # sibling window (same PID, not a dialog at all) — and
                    # THIS code's own Escape + forced SetForegroundWindow back
                    # to the ORIGINAL locked hwnd is what's undoing/blocking
                    # the switch, every single time. Class name distinguishes
                    # them for certain: "#32770" is a real Windows dialog box;
                    # anything else is very likely just a sibling content
                    # window, not something that needs dismissing at all.
                    try:
                        _fg_cls   = win32gui.GetClassName(fg)
                        _fg_title = win32gui.GetWindowText(fg)
                    except Exception:
                        _fg_cls, _fg_title = "?", "?"
                    logger.warning(
                        "Foreign window from the form's own process is blocking it "
                        "(hwnd=%s, class=%r, title=%r) — likely a modal dialog. "
                        "Dismissing with Escape.", fg, _fg_cls, _fg_title)
                    import pyautogui
                    pyautogui.press("escape")
                    time.sleep(0.15)
            except Exception as _dlg_exc:
                logger.debug("Modal-dialog dismiss check failed: %s", _dlg_exc)

            # Alt keypress satisfies the SetForegroundWindow focus-steal restriction.
            try:
                import win32com.client
                win32com.client.Dispatch("WScript.Shell").SendKeys('%')
            except Exception:
                pass
            win32gui.SetForegroundWindow(self._locked_hwnd)
            time.sleep(0.05)
            logger.info("Re-asserted form foreground (focus had drifted off the form).")
        except Exception as exc:
            logger.debug("Re-assert form failed: %s", exc)

    def _select_combobox_value_via_keyboard(
        self, cb_label: str, cb_val: str, cx: float, cy: float, max_steps: int = 20,
    ) -> bool:
        """Selects a combobox's value with real keyboard navigation instead
        of clicking to open a visual dropdown -- a fallback for comboboxes
        that never open a real dropdown popup at all, tried only after the
        existing click-and-poll path already failed.

        Verified live 2026-08-10, directly, not guessed: after a real
        click on this project's own car insurance form's 'Policy Status'
        combobox, a full UIA tree walk found ZERO list-type elements
        anywhere on screen -- the "still empty after N tries" warnings
        were completely accurate, not a timing race (two earlier fixes
        this same day both assumed a timing race and both turned out
        wrong). This control also has no Invoke/Toggle/SelectionItem/
        ExpandCollapse pattern at all, and its ValuePattern claims
        writable (IsReadOnly=False) but SetValue() raises a raw COM error
        -- a minimally, almost-brokenly UIA-exposed control, not something
        any pattern-based shortcut can drive. What DOES work, confirmed
        directly against the real live control: type-ahead (pressing the
        target's first letter jumps straight to a matching option, same
        as any standard Windows combobox) and Up/Down arrow navigation,
        both of which visibly change ValuePattern.Value in real time even
        though nothing about a "dropdown" ever becomes visible or
        UIA-discoverable at any point.

        This is a general fallback, not a fix for one field on one form:
        any comboboxcontrol whose real dropdown genuinely never renders
        (confirmed by the existing poll already coming up empty) gets this
        tried next, regardless of which app it's in. Reads the live value
        with a single, targeted UIA call after every keystroke -- NOT a
        full self._observe() snapshot, which walks the whole visible UI
        tree and would be far too slow to call after every single
        keypress in a bounded search loop like this one.

        SKIPPED ENTIRELY while the ghost-cursor overlay is active --
        added 2026-08-10, same day as a real incident. That overlay covers
        the entire screen for its whole lifetime, and UIA's
        ControlFromPoint silently resolves to ITS window instead of the
        real control the whole time it's running (confirmed live) --
        meaning every _live_value() read below is guaranteed None, so
        every attempt through this function has a 0% success rate whenever
        the overlay is active. A same-day attempt to fix the read itself
        (stopping/restarting the overlay around it) caused a worse
        regression -- real Tcl/Tk corruption from repeated in-process
        cycling, leaving a broken, no-longer-click-through window blocking
        every click on the user's screen -- and was reverted (see
        DEVELOPERS.md's execution_ghost_overlay_breaks_uia_reads entry).

        NOTE: ghost_cursor was disabled for the rest of 2026-08-10, then
        RE-ENABLED 2026-08-11 once its root cause was actually found and
        fixed -- root.winfo_id() was targeting the wrong (child) window
        for every Win32 style change, which explains both the invisible
        rendering AND the click-swallowing (see
        ghost_cursor_wrong_hwnd_render_bug in DEVELOPERS.md). That fix
        does NOT touch this specific guard's reason for existing, though:
        it only corrects real mouse hit-testing and visual compositing,
        neither of which governs UIA's OWN ControlFromPoint -- confirmed
        directly, separately, live, that UIA's hit-testing ignores
        WS_EX_TRANSPARENT entirely regardless of which window it's
        correctly applied to. So self._executor.ghost was non-None again
        in every real run for a while, and this guard was ACTIVE --
        genuinely the only thing standing between a real run and the
        guaranteed-to-fail 20-40-keystroke search loop this whole
        docstring describes.

        REVERTED AGAIN 2026-08-11 (ghost_cursor=False, direct request) --
        self._executor.ghost is back to None in every real run, so this
        guard is currently a no-op. Left in place regardless: it's correct
        for whichever way ghost_cursor is currently set, costs nothing
        when the overlay is off, and protects the next re-enable attempt
        (this is the third time that flag has flipped) from silently
        reintroducing the exact failure mode described below.

        Even without that crash, a guaranteed-to-fail 20-40-keystroke
        search loop with zero visible on-screen feedback (the dropdown
        never renders, so nothing visibly moves) reads to a user watching
        the screen as "the agent froze" for 5-10+ real seconds, confirmed
        directly from the very next live report after the revert: "The
        Agent couldn't click on the form at all. It's frozen shut." Since
        this function currently cannot succeed under the one condition
        that actually matters (a real live run), skipping it outright and
        going straight to the existing 2-strike give-up-and-move-on
        escalation (_combobox_dropdown_fail_counts, already proven working
        live) cuts that same failure from ~11s of silent dead time down to
        the original ~3.2s poll, with the exact same eventual outcome
        (failure, marked attempted, agent moves on). Only skipped when
        self._executor.ghost is not None -- dry_run/tests (where ghost is
        None) still exercise the real fallback logic below unaffected.
        """
        if getattr(self._executor, "ghost", None) is not None:
            return False

        try:
            import uiautomation as _uia
        except ImportError:
            return False

        def _live_value():
            try:
                ctrl = _uia.ControlFromPoint(int(cx), int(cy))
                if ctrl is None:
                    return None
                vp = ctrl.GetPattern(_uia.PatternId.ValuePattern)
                return vp.Value if vp else None
            except Exception:
                return None

        # NOT stopping/restarting the ghost overlay here anymore --
        # reverted 2026-08-10, the same day it was added, after a real,
        # severe live regression: "couldn't click a damn thing in the
        # form." This function is called once per combobox whose dropdown
        # never renders, and on a real run that happened twice in a row
        # (two failed attempts on 'Policy Status' before the existing
        # 2-strike back-off correctly gave up and moved on) -- meaning
        # GhostOverlay.hide_for_uia_read()/restore_after_uia_read() (each
        # a full stop()-then-start() of its Tkinter window/thread) fired
        # twice in quick succession within the SAME live process. The very
        # next combobox click after that logged "Tcl_AsyncDelete: async
        # handler deleted by the wrong thread" -- the exact Tcl/Tk
        # corruption signature this project's own test_ghost_overlay.py
        # module docstring already documents as a precursor to a hard,
        # process-wide crash when a Tk root's teardown doesn't stay
        # strictly confined to a single, isolated process. The run then
        # went completely silent for 17 seconds until the user force-killed
        # it -- consistent with a corrupted, no-longer-click-through,
        # full-screen topmost window left sitting over the whole desktop.
        #
        # This function's own _live_value() reads WILL still silently
        # resolve to the ghost overlay instead of the real control while
        # the overlay is running (the actual bug is real and still not
        # fixed at the read level) -- but that only means this fallback
        # reports failure more often than it should, which the ALREADY
        # WORKING _combobox_dropdown_fail_counts 2-strike back-off in
        # agent.py's OPT2 branch bounds to two ~17s attempts before moving
        # on, confirmed live in the very same run that surfaced this crash.
        # A wrong-but-bounded fallback beats a fix that can silently take
        # over the user's entire screen input. A real fix (e.g. resizing
        # the overlay to a small, positioned window instead of a
        # full-screen one, so it stops needing to be destroyed at all) is
        # a bigger, separate piece of work, not a same-day patch.

        try:
            if cb_val and cb_val[0].isalnum():
                self._executor.execute({"action_type": "keyboard", "key_count": 1,
                                        "keystrokes": [cb_val[0].lower()]})
                time.sleep(0.2)
                _v = _live_value()
                if _v is not None and _option_matches(cb_val, _v):
                    logger.info("Combobox(keyboard): %r → %r via type-ahead", cb_label, _v)
                    return True

            # Type-ahead alone can't reach every option (ambiguous first
            # letters, or a control that doesn't support it at all) -- walk
            # the real option list with arrow keys next. Confirmed live this
            # list does NOT wrap around (stops at the last item rather than
            # cycling back to the first), so Down alone can miss anything
            # positioned "before" wherever the control currently sits;
            # walking back Up from wherever Down stopped covers the rest.
            #
            # Stops the instant a keypress doesn't change the value, rather
            # than always spending the full max_steps -- found live
            # 2026-08-10: a real run pressed Down 18 times in a row with
            # the value stuck at the list's last item after press #5,
            # wasting ~2s of real time before ever trying Up at all
            # (direct user report right after: "so fucking slow"). The
            # list is confirmed non-wrapping, so a keypress that doesn't
            # move the value at all means the boundary has been reached
            # and every further press in that direction is guaranteed
            # wasted, not just unlucky.
            # Found live 2026-08-12: 'Primary Use' burned the FULL max_steps
            # budget in both directions (39 keystrokes, ~8s) without ever
            # matching -- visually "the up and down buttons looping a lot",
            # prompting a manual scroll to try to unstick it. The existing
            # non-wrapping-boundary break above only fires on two REAL,
            # non-None reads that happen to be equal -- if _live_value()
            # returns None on every single attempt (ControlFromPoint not
            # resolving the right control, a stale/moved position, or a
            # genuine read failure), that check never fires at all, since
            # None is never "not None and == prev". A working control (per
            # this function's own docstring) produces real, changing string
            # reads throughout the search -- three unreadable reads in a row
            # is a strong, safe signal this control isn't going to become
            # readable by pressing more keys, not evidence the target is
            # just further down the list. Doesn't change behavior for any
            # currently-working combobox (they never hit this), only cuts a
            # guaranteed-blind search short instead of exhausting it.
            _UNREADABLE_LIMIT = 3
            _prev = _live_value()
            _unreadable_streak = 0
            for _ in range(max_steps):
                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["down"]})
                time.sleep(0.15)
                _v = _live_value()
                if _v is not None and _option_matches(cb_val, _v):
                    logger.info("Combobox(keyboard): %r → %r via Down-arrow", cb_label, _v)
                    return True
                if _v is None:
                    _unreadable_streak += 1
                    if _unreadable_streak >= _UNREADABLE_LIMIT:
                        logger.warning("Combobox(keyboard): %r — value unreadable %d reads in a "
                                       "row during Down-search — stopping early instead of "
                                       "burning the rest of the %d-step budget.",
                                       cb_label, _unreadable_streak, max_steps)
                        return False
                else:
                    _unreadable_streak = 0
                    if _v == _prev:
                        break  # hit the bottom of the (non-wrapping) list
                _prev = _v

            _prev = _live_value()
            _unreadable_streak = 0
            for _ in range(max_steps):
                self._executor.execute({"action_type": "keyboard", "key_count": 1, "keystrokes": ["up"]})
                time.sleep(0.15)
                _v = _live_value()
                if _v is not None and _option_matches(cb_val, _v):
                    logger.info("Combobox(keyboard): %r → %r via Up-arrow", cb_label, _v)
                    return True
                if _v is None:
                    _unreadable_streak += 1
                    if _unreadable_streak >= _UNREADABLE_LIMIT:
                        logger.warning("Combobox(keyboard): %r — value unreadable %d reads in a "
                                       "row during Up-search — stopping early instead of "
                                       "burning the rest of the %d-step budget.",
                                       cb_label, _unreadable_streak, max_steps)
                        return False
                else:
                    _unreadable_streak = 0
                    if _v == _prev:
                        break  # hit the top of the (non-wrapping) list
                _prev = _v
        except Exception as exc:
            logger.debug("Combobox keyboard-navigation fallback failed for %r: %s", cb_label, exc)
        return False

    def _form_rect(self, state: Dict[str, Any]):
        """Live (l, t, r, b) of the locked form window."""
        try:
            import win32gui
            hwnd = self._locked_hwnd or win32gui.GetForegroundWindow()
            return win32gui.GetWindowRect(hwnd)
        except Exception:
            sw, sh = state.get("screen_resolution", [1920, 1200])
            return (0, 0, sw, sh)

    def _point_in_form(self, pos, state: Dict[str, Any], margin: int = 6) -> bool:
        """True if a click target falls inside the locked form window (so the
        transformer can't drift a click into another window)."""
        if not pos or len(pos) < 2:
            return False
        l, t, r, b = self._form_rect(state)
        return (l - margin) <= pos[0] <= (r + margin) and (t - margin) <= pos[1] <= (b + margin)

    def _form_viewport_bottom(self, state: Dict[str, Any]) -> float:
        """Live bottom edge of the form's scroll viewport (locked window rect,
        clamped to the screen). Used to tell on-screen fields from scrolled-off
        ones — the window size varies per run, so this must be read live."""
        sh = state.get("screen_resolution", [1920, 1200])[1]
        try:
            import win32gui
            hwnd = self._locked_hwnd or win32gui.GetForegroundWindow()
            _, _, _, wb = win32gui.GetWindowRect(hwnd)
            return min(wb, sh)
        except Exception:
            return sh

    def _ensure_foreground(self, state: Dict[str, Any]) -> None:
        """Re-assert the observed active window as foreground before typing, so
        keystrokes can't leak into a window that stole focus (e.g. the terminal).
        Best-effort + universal — re-focuses whatever window we just observed."""
        try:
            import win32gui
            title = (state.get("window_title") or "").strip()
            if not title:
                return
            fg = win32gui.GetForegroundWindow()
            if (win32gui.GetWindowText(fg) or "").strip() == title:
                return                      # already in front — nothing to do
            hwnd = win32gui.FindWindow(None, title)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.05)
                logger.info("Foreground guard: re-focused %r before typing", title[:40])
        except Exception:
            pass

    def _detect_section(self, state: Dict[str, Any], focused_el: Dict) -> str:
        """
        Return the repeated-section name (e.g. 'Driver 2') the focused element
        belongs to, or '' if none / the scope has no sections.

        Generic mechanism: the section is the lowest section-pane whose top edge
        is at/above the focused field's vertical centre. The pane prefix, the
        name pattern, and the display format all come from the injected
        ScopeConfig — so a non-sectioned app (default scope, pattern=None) gets
        '' for free, and no app-specific names live here.
        """
        import re as _re
        _pat = getattr(self._scope, "section_pattern", None)
        if not _pat:                         # scope has no sections → no-op
            return ""
        if not focused_el or not focused_el.get("bbox"):
            return ""
        _prefix = getattr(self._scope, "section_prefix", "section_")
        fx1, fy1, fx2, fy2 = focused_el["bbox"]
        fy_center = (fy1 + fy2) / 2

        section_panes = sorted(
            [e for e in state.get("elements", [])
             if e.get("type") == "panecontrol"
             and e.get("window_role") != "background"
             and e.get("bbox")
             and (e.get("label") or e.get("text") or "").startswith(_prefix)],
            key=lambda e: e["bbox"][1]
        )

        current_label = ""
        for pane in section_panes:
            if pane["bbox"][1] <= fy_center:
                current_label = pane.get("label") or pane.get("text") or ""
            else:
                break

        if not current_label:
            return ""
        m = _re.match(_pat, current_label.lower())
        if m:
            return self._scope.section_format(*m.groups())
        return ""

    def _lookup_field(self, field_name: str, section: str = "") -> str:
        """
        Look up a field value from the cached record.
        If `section` is provided (e.g. 'Driver 2'), try the section-prefixed key
        first (e.g. 'Driver 2 First Name') then fall back to the bare key.
        Returns empty string if not found or if value is a placeholder.
        """
        if not self._cached_record:
            return ""
        rec = self._cached_record

        import re as _re

        _is_blank = _is_leave_blank_value

        def _get(key: str) -> str:
            kl = key.lower()
            v = rec.get(key) or next((rv for rk, rv in rec.items() if rk.lower() == kl), "")
            return "" if _is_blank(v) else v

        def _get_fuzzy(key: str) -> str:
            """Strip punctuation/spaces and match any record key that contains all words."""
            kl_words = set(_re.sub(r"[^a-z0-9 ]", " ", key.lower()).split())
            if not kl_words:
                return ""
            for rk, rv in rec.items():
                rk_words = set(_re.sub(r"[^a-z0-9 ]", " ", rk.lower()).split())
                if kl_words <= rk_words or rk_words <= kl_words:
                    if not _is_blank(rv):
                        return rv
            return ""

        if section:
            val = _get(f"{section} {field_name}")
            if val:
                return val
        val = _get(field_name)
        if not val:
            val = _get_fuzzy(field_name)
        return val

    def _auto_fill(self, state: Dict[str, Any]) -> Optional[tuple]:
        """
        If the focused element is an empty text field that has a known value in
        the record cache (or BACKGROUND DATA as fallback), return (field_name, value).
        Only fires for editcontrol — not comboboxes/checkboxes.
        """
        elements   = state.get("elements", [])
        focused_id = state.get("focused_element_id")
        if not focused_id:
            return None
        focused = next((e for e in elements if e.get("element_id") == focused_id), None)
        if not focused:
            return None
        if focused.get("type") not in ("editcontrol",):
            return None
        current = (focused.get("value") or "").strip()

        field_name = (focused.get("label") or focused.get("text") or "").strip()
        if not field_name:
            return None

        # Detect which Driver/Vehicle section this field is in (e.g. "Driver 2")
        # so we look up "Driver 2 First Name" instead of bare "First Name".
        section = self._detect_section(state, focused)
        # Use section-prefixed key in _filled_this_tab so "Driver 2 First Name"
        # and "Driver 3 First Name" are tracked independently.
        filled_key = f"{section} {field_name}" if section else field_name

        # Skip re-filling a field we already filled this tab (prevents cycling loops
        # where the form clears numeric/spin fields on re-focus).
        if filled_key in self._filled_this_tab:
            return None

        if current:
            # Field has a value — check if it's wrong (leftover from a previous run).
            # If wrong, return it flagged for overwrite (Ctrl+A before typing).
            expected_check = self._lookup_field(field_name, section=section)
            if (expected_check
                    and not _is_leave_blank_value(expected_check)
                    and expected_check.lower().strip("()") != "yes (check)"
                    and current.lower() != expected_check.lower()):
                logger.info("Auto-fill: '%s' has wrong value %r — will overwrite with %r",
                            filled_key, current, expected_check)
                return (filled_key, expected_check, True)   # True = needs Ctrl+A clear first
            return None   # correct value or no expected — let auto_skip handle it

        # ── Primary: use the cached full record (bypasses 2000-char UIA cap) ──
        expected = self._lookup_field(field_name, section=section)

        # ── Fallback: parse bg_blobs from UIA state ───────────────────────────
        if not expected:
            bg_elems = [e for e in elements if e.get("window_role") == "background"]
            bg_blobs = [(e.get("value") or "").strip() for e in bg_elems if e.get("value")]
            sec_key  = f"{section} {field_name}" if section else ""
            for blob in sorted(bg_blobs, key=len, reverse=True):
                r = _parse_records(blob)
                if r:
                    rec = r.get(self._record_num, r.get(min(r), {}))
                    fl  = field_name.lower()
                    # Try section-prefixed key first, then bare key
                    if sec_key:
                        expected = rec.get(sec_key) or next(
                            (v for k, v in rec.items() if k.lower() == sec_key.lower()), "")
                    if not expected:
                        expected = rec.get(field_name) or next(
                            (v for k, v in rec.items() if k.lower() == fl), "")
                    if expected:
                        break

        if not expected:
            return None

        if _is_leave_blank_value(expected) or expected.lower().strip("()") == "yes (check)":
            return None

        # Return filled_key (section-prefixed if in a driver/vehicle section) as the
        # first element so the caller uses it for _filled_this_tab tracking, keeping
        # "Driver 2 First Name" and "Driver 3 First Name" independent.
        return (filled_key, expected, False)   # False = field is empty, no clear needed

    def _combobox_needs_fix(self, state: Dict[str, Any]) -> Optional[tuple]:
        """
        If the focused element is a combobox whose current value does NOT match
        BACKGROUND DATA, return (field_name, current_value, expected_value).
        Otherwise return None.
        """
        elements   = state.get("elements", [])
        focused_id = state.get("focused_element_id")
        focused    = next((e for e in elements if e.get("element_id") == focused_id), None)
        if not focused:
            return None
        if focused.get("type") != "comboboxcontrol":
            return None
        current = (focused.get("value") or "").strip()
        # NOTE: do NOT short-circuit on empty current — comboboxes now have a blank
        # first option, so current="" is the default state and may need to be fixed.

        field_name = (focused.get("label") or focused.get("text") or "").strip()
        if not field_name:
            return None
        section  = self._detect_section(state, focused)

        # Check raw record value so we can detect "(leave blank)" sentinels that
        # _lookup_field strips out.
        if self._cached_record:
            _rec = self._cached_record
            _sec_key = f"{section} {field_name}" if section else ""
            _raw = ""
            if _sec_key:
                _raw = _rec.get(_sec_key) or next(
                    (v for k, v in _rec.items() if k.lower() == _sec_key.lower()), "")
            if not _raw:
                _raw = _rec.get(field_name) or next(
                    (v for k, v in _rec.items() if k.lower() == field_name.lower()), "")
            if _raw and _is_leave_blank_value(_raw):
                if current:
                    # Non-empty combobox should be cleared — select the blank option
                    return (field_name, current, "")
                return None   # already blank

        expected = self._lookup_field(field_name, section=section)
        if not expected or current.lower() == expected.lower():
            return None
        return (field_name, current, expected)

    def _lookup_checkbox_should_check(self, field_name: str) -> Optional[bool]:
        """Look up a checkbox's known answer in background data and parse it
        into a check/leave-unchecked decision. Returns None if not found
        (let LLM decide). Extracted from _auto_check 2026-08-14 so batch-fill
        can ask this question for ANY checkbox element, not just whichever
        one happens to be focused right now — same parsing rule, one place."""
        if not field_name:
            return None
        expected = self._lookup_field(field_name)
        if not expected:
            return None   # not in background data — let LLM decide
        # "YES (check)", "yes", "true", "checked" → check it
        ev = expected.lower().strip()
        return ev.startswith("yes") or ev in {"check", "true", "1", "checked"}

    def _auto_check(self, state: Dict[str, Any]) -> Optional[tuple]:
        """
        If the focused element is a checkbox, look it up in BACKGROUND DATA.
        Returns (field_name, should_check: bool) if the field is found, else None.
        When should_check is True the caller clicks the checkbox; when False it tabs past.
        """
        elements   = state.get("elements", [])
        focused_id = state.get("focused_element_id")
        if not focused_id:
            return None
        focused = next((e for e in elements if e.get("element_id") == focused_id), None)
        if not focused:
            return None
        if focused.get("type") != "checkboxcontrol":
            return None

        field_name = (focused.get("label") or focused.get("text") or "").strip()
        if not field_name:
            return None

        should_check = self._lookup_checkbox_should_check(field_name)
        if should_check is None:
            return None   # not in background data — let LLM decide
        return (field_name, should_check)

    @staticmethod
    def _combobox_dropdown_arrow_pos(bbox) -> List[float]:
        """Position of a combobox's own dropdown-arrow button, not its
        text-area center. Found live 2026-08-14, investigating a real
        emergency-stop incident: the click-open mechanism has always
        clicked dead center of the combobox's bbox, gambling on the whole
        control responding to a click the same way a button does. Direct
        UIA inspection of the real form (not assumed) showed every
        combobox exposes its ACTUAL dropdown toggle as a SEPARATE
        ButtonControl with a real InvokePattern, flush against the right
        edge of the combobox's own bounding rectangle (measured ~19px wide
        in this form) -- the combobox control itself only exposes
        ValuePattern, no ExpandCollapsePattern. Center-clicking happens to
        still work for most comboboxes (the underlying native control
        responds to a plain click too), but isn't the semantically correct
        target -- same "use semantic, not precise" principle already
        shipped for buttons (executor.py's BM_CLICK tier). Targeting the
        arrow position lets the SAME existing click pipeline
        (_try_uia_invoke, tried before any raw coordinate click) actually
        invoke the real toggle control instead of hoping a center-click
        lands on something responsive. Generic geometry (right-edge offset
        from the combobox's OWN bbox), not tied to any specific field.
        NOT a confirmed fix for any one field's dropdown-render failures
        (root cause for those remains open, per this file's own prior
        investigation) -- a real hardening found along the way, not a
        guess at why one specific combobox fails while others don't.
        """
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        return [max(x1, x2 - 15), (y1 + y2) / 2]

    def _focus_element_via_uia(self, label: str, expected_bbox=None) -> bool:
        """
        Focus a control directly via UIA's SetFocus(), bypassing simulated
        mouse clicks at computed screen coordinates entirely -- no bbox
        center, no pixel math, no guessing where something visually sits.
        Returns True if a matching control was found and SetFocus() was
        issued without error.

        Added 2026-08-08, live, direct user instruction ("Don't use
        semantic positioning") immediately after 'Street Address 1' and
        then 'Street Address 2' -- two different fields, click coordinates
        only 10px apart -- both failed to receive focus via a simulated
        click at their bbox center, twice each, confirmed by re-observing
        afterward. A pixel-coordinate click can miss for reasons that have
        nothing to do with WHICH field is being targeted (edge clipping
        near the viewport boundary, visual occlusion, a few pixels of bbox
        drift) -- SetFocus() sidesteps that whole class of failure since it
        never goes through screen coordinates at all, just the control
        object UIA already knows about.

        `expected_bbox` (optional, from the caller's own already-picked
        target element) disambiguates repeated-section forms -- see
        _find_uia_control_by_name's own docstring for why this matters.
        """
        _ctrl = self._find_uia_control_by_name(label, expected_bbox=expected_bbox)
        if _ctrl is None:
            return False
        try:
            _ctrl.SetFocus()
            return True
        except Exception as exc:
            logger.debug("UIA SetFocus on %r failed — %s", label, exc)
            return False

    def _find_uia_control_by_name(self, label: str, expected_bbox=None):
        """
        Shared lookup used by _focus_element_via_uia and
        _scroll_into_view_via_uia -- finds a live UIA control by its
        accessible Name among Edit/ComboBox/CheckBox controls in the
        locked form window. Returns None if UIA is unavailable or nothing
        matches; never raises.

        Found 2026-08-09, live, direct user report ("another loop in the
        Drivers, why do you hate me?"): logs/latest.log showed a redirect
        repeatedly picking 'First Name' as the shallowest target in Driver
        3's newly-revealed cluster, calling SetFocus() on it, then the VERY
        NEXT step's fresh observation showing the transformer's pointer
        already back on an unrelated already-filled field -- as if the
        SetFocus never landed on the field that was actually chosen.
        Root cause: this lookup searched by accessible Name ALONE, with no
        way to tell apart the SAME bare label repeated across Driver 1/2/3
        sections (no section prefix in the raw UIA Name property) -- the
        exact repeated-section collision class already fixed twice
        elsewhere in this file (state_validator.py's stable-key matching,
        agent.py's own _attempt_key rank disambiguation), just never
        carried into this UIA-level lookup. `EditControl(Name="First
        Name")` always returns the FIRST match in the UIA tree -- Driver
        1's, already filled and usually off-screen -- regardless of which
        driver's cluster the caller actually meant. SetFocus happily
        succeeded every time, just on the wrong element: the auto-fill fast
        path then saw an already-filled field focused, took the "navigate"
        branch instead of filling anything, and the transformer's learned
        pointer (frozen on an unrelated coordinate) re-triggered the exact
        same "already-filled, redirect" guard next step -- a tight loop
        with no field ever actually getting typed into.

        Fixed by disambiguating on position when the caller can supply one:
        every existing caller already has the specific target ELEMENT (with
        its own bbox) in hand before calling this, from
        navigation_protocol's own scan -- they were just discarding that
        bbox and passing the bare label through. When `expected_bbox` is
        given and more than one control shares the Name, walk all matches
        (via UIA's own `foundIndex`) and return whichever one's own
        BoundingRectangle center is closest to the expected position,
        instead of blindly trusting tree order. `expected_bbox` is optional
        and defaults to the old first-match behavior when omitted, so any
        future caller without a specific position in hand still works.
        """
        try:
            import uiautomation as _uia
            import win32gui as _w32g
        except Exception:
            return None
        try:
            hwnd = self._locked_hwnd or _w32g.GetForegroundWindow()
            root = _uia.ControlFromHandle(hwnd)
            for _finder in (root.EditControl, root.ComboBoxControl, root.CheckBoxControl):
                if expected_bbox is None:
                    _c = _finder(searchDepth=25, Name=label)
                    if _c.Exists(maxSearchSeconds=0.3):
                        return _c
                    continue
                _ecx = (expected_bbox[0] + expected_bbox[2]) / 2
                _ecy = (expected_bbox[1] + expected_bbox[3]) / 2
                _best, _best_dist = None, None
                for _idx in range(1, 11):  # generous cap; real forms never repeat a label 10x
                    _c = _finder(searchDepth=25, Name=label, foundIndex=_idx)
                    if not _c.Exists(maxSearchSeconds=0.3):
                        break
                    try:
                        _r = _c.BoundingRectangle
                        _dist = ((_r.left + _r.right) / 2 - _ecx) ** 2 + ((_r.top + _r.bottom) / 2 - _ecy) ** 2
                    except Exception:
                        continue
                    if _best_dist is None or _dist < _best_dist:
                        _best, _best_dist = _c, _dist
                if _best is not None:
                    return _best
            return None
        except Exception as exc:
            logger.debug("UIA control lookup for %r failed — %s", label, exc)
            return None

    def _resolve_field_control(self, state: Dict[str, Any], label: str, bbox):
        """Resolves a live UIA control for a field via
        _find_uia_control_by_name -- but only pays for that function's
        expensive position-based disambiguation search (a foundIndex=1..10
        loop, each "does this index exist" check costing a real UIA
        timeout when it doesn't) when the label is ACTUALLY ambiguous in
        the current observed state. For a unique label -- the overwhelming
        majority of real fields -- passes expected_bbox=None instead,
        which takes _find_uia_control_by_name's own fast path (one
        Exists() check per finder, returns as soon as found, no
        timeout-driven "does a second match exist" probing at all).

        Found live 2026-08-14, direct request ("still too slow, at least
        <60s"): timed OPT2 batch fast-fill's own log -- 13 fields filled
        in 7 seconds with NO observe() between them (the thing batching
        was built to eliminate), meaning something else was costing
        ~0.5s/field. Traced into _find_uia_control_by_name itself:
        every batch/single-field/dead-spot-rescue caller always passed
        expected_bbox, forcing the slow disambiguation path on every
        field, even though disambiguation only matters for the small
        minority of labels that genuinely repeat (e.g. "First Name"
        across Driver 1/2/3 sections -- the real reason expected_bbox
        exists at all, added 2026-08-09). Worse for non-editcontrol
        types: the function tries EditControl/ComboBoxControl/
        CheckBoxControl in that fixed order, so a checkbox field paid for
        TWO wrong-type timeouts before ever reaching the right one.

        Fix doesn't touch _find_uia_control_by_name itself or weaken its
        disambiguation guarantee -- a genuinely-duplicate label still
        gets expected_bbox exactly as before, so still resolves to the
        correct one of several matches, unchanged. Only the "is this
        label unique on screen right now" check (an in-memory count over
        elements already in hand, not a new UIA call) decides whether the
        slow path is even needed. Never slower than before, only faster
        for the common case.
        """
        if not label:
            return None
        elements = state.get("elements", [])
        _dup = sum(1 for e in elements
                   if (e.get("label") or e.get("text") or "").strip() == label)
        # Point-based fast path, added 2026-08-14 ("any more improvements"):
        # try this FIRST, before ever falling back to the name-search below.
        _point_ctrl = self._resolve_control_via_point(bbox, label)
        if _point_ctrl is not None:
            return _point_ctrl
        return self._find_uia_control_by_name(label, expected_bbox=bbox if _dup > 1 else None)

    # Native window classes _resolve_control_via_point trusts as a real,
    # known-type control under a point -- the same classes _keyboard_direct/
    # _combobox_direct (executor.py) already check on the write side, kept
    # here only as a coarse "is this a real edit/combobox control at all"
    # gate. Deliberately loose (accepts either edit OR combobox classes
    # regardless of which type the caller wanted) since the specific write
    # mechanism re-verifies its own exact class before ever writing.
    _POINT_RESOLVE_CLASSES = {"Edit", "RichEditD2DPT", "RichEdit20W", "RICHEDIT50W",
                               "RICHEDIT60W", "RichEdit", "ComboBox"}

    def _resolve_control_via_point(self, bbox, expected_label: str,
                                    uia_mod=None, win32gui_mod=None, user32=None):
        """Resolves a live UIA control directly from its own on-screen
        position via WindowFromPoint -- the exact mechanism this file's
        checkbox fast-fill already uses and has proven live -- instead of
        a UIA accessible-Name SEARCH. WindowFromPoint is one plain, cheap
        Win32 call (no accessibility-tree walk, no search timeout);
        wrapping the resulting handle via UIA's ControlFromHandle (already
        used inside _find_uia_control_by_name itself, on a handle FOUND by
        searching -- here on a handle we already have from screen
        position, no search involved either way) is likewise cheap.

        Found live 2026-08-14, direct request ("any more improvements"):
        checked a real log after the disambiguation-skip fix
        (_resolve_field_control) shipped and total run time barely
        moved. The SAME batch showed checkboxes (already using this exact
        WindowFromPoint mechanism, never went through
        _find_uia_control_by_name at all) landing 3 fields within about a
        second, while editcontrol/comboboxcontrol fields right next to
        them -- still doing a UIA name-search even after the
        disambiguation fix -- landed roughly 1/second. Direct,
        apples-to-apples evidence in the same batch that the search
        itself, not the disambiguation loop around it, was the remaining
        cost.

        Verifies the resolved control's own native window class AND its
        accessible Name before trusting it, so a stale bbox or an
        unexpected control under that point (e.g. WindowFromPoint landing
        on a parent panel instead of the field itself) can't silently
        write into the wrong control -- returns None on any mismatch,
        letting the caller fall back to the proven UIA-search path.
        Writing the wrong TEXT into the wrong field is a worse failure
        mode than checkboxes ever risked (a wrongly-checked box is still
        visibly wrong and correctable; wrong text can look plausible), so
        this checks more than the checkbox path does before trusting the
        point-based hit.

        `uia_mod`/`win32gui_mod`/`user32` are optional, test-injectable
        overrides for the real `uiautomation`/`win32gui`/
        `ctypes.windll.user32` -- same shape and same reason as
        executor.py's own `user32=None` parameters (_try_semantic_click,
        _keyboard_direct, _combobox_direct): `ctypes.windll.user32` is a
        live, C-backed, process-wide object, unsafe to patch in place in
        a test.
        """
        if not bbox or len(bbox) != 4 or not expected_label:
            return None
        try:
            import ctypes
            if uia_mod is None:
                import uiautomation as uia_mod
            if win32gui_mod is None:
                import win32gui as win32gui_mod
            if user32 is None:
                user32 = ctypes.windll.user32
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            hwnd = win32gui_mod.WindowFromPoint((int(cx), int(cy)))
            if not hwnd:
                return None
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buf, 256)
            if buf.value not in self._POINT_RESOLVE_CLASSES:
                return None
            ctrl = uia_mod.ControlFromHandle(hwnd)
            if (ctrl.Name or "").strip().lower() != expected_label.strip().lower():
                return None
            return ctrl
        except Exception as exc:
            logger.debug("Point-based control resolution for %r failed — %s", expected_label, exc)
            return None

    def _scroll_into_view_via_uia(self, label: str, expected_bbox=None) -> bool:
        """
        Bring a specific control into view via UIA's
        ScrollItemPattern.ScrollIntoView() -- ONE native call, no pixel
        distance, no calibration, no loop of repeated scroll increments.
        UIA itself figures out exactly how far its container needs to move
        to make the control visible. Returns True if the control was found
        and the call succeeded, False otherwise (caller should fall back
        to the increment-based _scroll_form_down_uia).

        Added 2026-08-08, live, direct user request: "scroll on it once
        and then boom" -- rejected an earlier plan to measure a
        pixels-per-scroll-increment ratio and fire a computed number of
        SmallIncrement calls in a tight loop, on the grounds that it was
        still fundamentally guessing (a ratio that could be wrong under
        different DPI/window states) dressed up as one action. The user's
        own question -- "don't you have the UI Accessibility Tree to just
        find what isn't in focus" -- pointed at the actually-correct
        primitive: ask UIA to bring a specific control into view directly,
        the same way any accessibility-aware application would. The target
        element itself is chosen by navigation_protocol.find_scroll_target_
        element -- the deepest element in the densest reachable cluster,
        so bringing IT into view should reveal the whole cluster above it.
        """
        # DIAGNOSTIC, added 2026-08-08: found live, immediately after
        # shipping this route, that it was silently falling back to
        # _scroll_form_down_uia every time (log showed "SmallIncrement
        # issued", never "ScrollIntoView landed on..."), with the actual
        # reason swallowed at debug level. Log EVERY failure branch at
        # info/warning so the next run says definitively why -- control not
        # found vs. no ScrollItemPattern support vs. an actual exception --
        # instead of guessing a fix blind.
        _ctrl = self._find_uia_control_by_name(label, expected_bbox=expected_bbox)
        if _ctrl is None:
            logger.info("[SCROLL-INTO-VIEW-DIAG] no UIA control found for %r — falling back.", label)
            return False
        try:
            _sip = _ctrl.GetScrollItemPattern()
            if _sip is None:
                logger.info(
                    "[SCROLL-INTO-VIEW-DIAG] control %r found but does not expose "
                    "ScrollItemPattern — falling back.", label)
                return False
            _sip.ScrollIntoView()
            return True
        except Exception as exc:
            logger.warning("[SCROLL-INTO-VIEW-DIAG] ScrollIntoView on %r raised — %s — falling back.",
                            label, exc)
            return False

    def _scroll_form_down_uia(self, state: Dict[str, Any]) -> bool:
        """
        Fire ONE native UIA ScrollPattern.Scroll(SmallIncrement) on the active
        form's scrollable pane, found by walking UP from a real on-screen
        field through the UIA parent chain. Returns True once the call was
        ISSUED, False if no scrollable pane was found or UIA isn't available
        — callers should fall back to the mouse-wheel route in that case.
        SmallIncrement, not LargeIncrement — see the comment at the actual
        Scroll() call below for why.

        Revived 2026-08-08 from this repo's own history (commit 0ec5688a's
        _scrollbar_drag, later removed as dead code) after the user asked
        whether a single precise scroll — like a separate, since-deleted
        "old Intern" project reportedly had — was possible here too. The
        historical version hardcoded a list of field names to anchor on
        ("Last Name", "Email Address", ...) which would violate the
        no-hardcoding-for-specific-tasks rule; this version anchors on
        whatever editcontrol labels are actually present in the CURRENT
        state, so it works on any form.

        Deliberately does NOT verify the pane actually moved here — that
        would be a second, internal observe-and-decide cycle stacked on the
        caller's own (in the SCROLL branch of the Navigation Protocol
        block), which already caused a "two systems, two chances to
        disagree" regression once this same session (see the self-
        calibrating-scroll comment on the wheel-scroll fallback below). The
        caller re-observes once and checks the visible-field signature to
        confirm the scroll actually took effect.
        """
        _panel = self._find_scrollable_pane_uia(state)
        if _panel is None:
            return False
        try:
            import uiautomation as _uia
            # SmallIncrement, not LargeIncrement -- found 2026-08-08, live,
            # immediately after reviving this route: LargeIncrement is a full
            # native "page jump," much bigger than the old wheel-fallback's
            # small -5/-15/-25 unit nudges. Combined with the new optimal-view
            # decide() (which already loops, re-scrolling as many times as
            # needed until cur==best), one LargeIncrement call overshot so far
            # that the very first field on the tab ('First Name') vanished
            # from the post-scroll snapshot entirely -- confirmed by
            # [SCROLL-DIAG] in the log. Since this protocol only scrolls DOWN,
            # never back up, overshooting permanently loses access to
            # whatever got skipped -- directly responsible for a 2.3% task
            # completion rate that run. decide()'s loop is exactly what makes
            # several small scrolls fine (and expected) instead of needing one
            # large one.
            _panel.GetScrollPattern().Scroll(_uia.ScrollAmount.NoAmount, _uia.ScrollAmount.SmallIncrement)
            logger.info("Scroll-form: UIA ScrollPattern SmallIncrement issued on pane %r",
                        getattr(_panel, "Name", "?"))
            return True
        except Exception as exc:
            logger.debug("Scroll-form: UIA ScrollPattern route unavailable — %s", exc)
            return False

    def _find_scrollable_pane_uia(self, state: Dict[str, Any]):
        """
        Shared by _scroll_form_down_uia (SmallIncrement) and
        _scroll_form_down_uia_percent (absolute jump): walks UP from a real
        on-screen field through the UIA parent chain to find the pane whose
        ScrollPattern.VerticallyScrollable is True. Returns the UIA control
        or None if UIA is unavailable or no scrollable ancestor was found.
        """
        try:
            import uiautomation as _uia
            import win32gui as _w32g
        except Exception:
            return None
        try:
            hwnd = self._locked_hwnd or _w32g.GetForegroundWindow()
            root = _uia.ControlFromHandle(hwnd)

            elements = state.get("elements", [])
            _anchor_names = [
                (e.get("label") or e.get("text") or "").strip()
                for e in elements
                if e.get("type") == "editcontrol"
                and e.get("window_role") != "background"
                and (e.get("label") or e.get("text"))
            ]
            _anchor = None
            for _nm in _anchor_names:
                _c = root.EditControl(searchDepth=25, Name=_nm)
                if _c.Exists(maxSearchSeconds=0.2):
                    _anchor = _c
                    break
            if _anchor is None:
                _anchor = _uia.GetFocusedControl()
            if _anchor is None:
                return None

            _panel, _cur = None, _anchor
            for _ in range(15):
                if _cur is None:
                    break
                try:
                    _sp = _cur.GetScrollPattern()
                    if _sp is not None and _sp.VerticallyScrollable:
                        _panel = _cur
                        break
                except Exception:
                    pass
                try:
                    _cur = _cur.GetParentControl()
                except Exception:
                    break
            return _panel
        except Exception as exc:
            logger.debug("Scroll-form: pane lookup failed — %s", exc)
            return None

    def _scroll_form_down_uia_percent(self, state: Dict[str, Any], target_bbox_top: float) -> bool:
        """
        Jump the scrollable pane DIRECTLY to a computed absolute position
        via ScrollPattern.SetScrollPercent() -- one call, no repeated
        increments, no measured pixels-per-call ratio.

        The trick: ScrollPattern.VerticalViewSize is a REQUIRED, standard
        property (not optional like ScrollItemPattern) reporting what
        fraction of the total scrollable content the current viewport shows
        (e.g. 30 means "30% of the content is visible right now"). Combined
        with the pane's own on-screen pixel height, that gives an exact
        total-content-height figure, and from there an exact pixels-per-
        scroll-percent ratio -- computed from ONE read of self-reported UIA
        data, not measured by scrolling and observing the result. No
        calibration nudge, no internal observe-and-adjust cycle.

        Added 2026-08-08, live, immediately after _scroll_into_view_via_uia
        (ScrollItemPattern-based) turned out to not be supported by this
        form's controls -- falling back to SmallIncrement every time, which
        only ever revealed one field per call ("it only revealed one
        input"). ScrollPattern itself (as opposed to ScrollItemPattern) is
        already proven working on this pane -- SmallIncrement uses the same
        object successfully -- so SetScrollPercent is far more likely to
        actually be implemented here too.

        Returns True if the jump was issued, False if the pane doesn't
        expose the needed properties (falls back to SmallIncrement).
        """
        _panel = self._find_scrollable_pane_uia(state)
        if _panel is None:
            return False
        try:
            import uiautomation as _uia
            _sp = _panel.GetScrollPattern()
            if _sp is None or not _sp.VerticallyScrollable:
                return False
            _view_size = _sp.VerticalViewSize     # % of content currently visible
            _cur_percent = _sp.VerticalScrollPercent
            if not _view_size or _view_size <= 0 or _cur_percent < 0:
                logger.info(
                    "[SCROLL-PERCENT-DIAG] pane %r reports VerticalViewSize=%r "
                    "VerticalScrollPercent=%r — can't compute a ratio, falling back.",
                    getattr(_panel, "Name", "?"), _view_size, _cur_percent)
                return False
            _rect = _panel.BoundingRectangle
            _panel_height_px = _rect.bottom - _rect.top
            if _panel_height_px <= 0:
                return False
            _total_content_px = _panel_height_px / (_view_size / 100.0)
            _px_per_percent = _total_content_px / 100.0
            _target_percent = _cur_percent + (target_bbox_top / _px_per_percent)
            _target_percent = max(0.0, min(100.0, _target_percent))
            _sp.SetScrollPercent(_uia.ScrollPattern.NoScrollValue, _target_percent)
            logger.info(
                "Scroll-form: UIA SetScrollPercent jumped pane %r from %.1f%% to %.1f%% "
                "(one absolute motion, target %.0fpx below current top).",
                getattr(_panel, "Name", "?"), _cur_percent, _target_percent, target_bbox_top)
            return True
        except Exception as exc:
            logger.info("[SCROLL-PERCENT-DIAG] SetScrollPercent route failed — %s — falling back.", exc)
            return False

    def _scroll_form_down(self, state: Dict[str, Any]) -> bool:
        """
        Scroll the active form window down to reveal fields hidden below the
        visible area. THREE routes, tried in order:

        1. _scroll_into_view_via_uia on navigation_protocol's own computed
           scroll target (the deepest element in the densest reachable
           cluster below) — ONE native ScrollIntoView call, lands exactly
           on the optimal cluster, no pixel math, no repeated increments.
           Added 2026-08-08 per direct user request ("scroll on it once
           and then boom") after rejecting an increment-counting approach
           for still fundamentally being a guess (a pixels-per-increment
           ratio) dressed up as one action.
        2. _scroll_form_down_uia_percent — jumps the pane directly via
           SetScrollPercent(), computed from the pane's own self-reported
           VerticalViewSize (one read, no calibration nudge). Added
           2026-08-08, live, immediately after route 1 turned out to not be
           supported here (ScrollItemPattern is optional; ScrollPattern
           itself, which this uses, is already proven working via route 3).
        3. _scroll_form_down_uia — one native SmallIncrement page-scroll,
           used if neither of the above computed a usable jump.
        4. _scroll_form_down_wheel — simulated mouse-wheel, the original
           fallback, if UIA isn't available at all.

        Returns True if a scroll was attempted.
        """
        # REAL viewport height, not unbounded -- find_scroll_target_element's
        # width parameter IS the density calculation itself; an effectively-
        # infinite width degenerates it into "just the single deepest field
        # on the whole tab," losing the "fits in one screen" concept this
        # whole module exists to compute. Found 2026-08-09 while writing
        # tests for the redirect-guard version of this exact call.
        _target = self._navproto.find_scroll_target_element(
            state, self._form_viewport_bottom(state) - 8,
            attempted_keys=self._attempted_keys, attempt_key_fn=self._attempt_key)
        if _target is None:
            logger.info("[SCROLL-INTO-VIEW-DIAG] find_scroll_target_element returned nothing to target — falling back.")
        else:
            _target_label = (_target.get("label") or _target.get("text") or "").strip()
            if self._scroll_into_view_via_uia(_target_label, expected_bbox=_target.get("bbox")):
                logger.info(
                    "Scroll-form: UIA ScrollIntoView landed on %r — one motion, no increments.",
                    _target_label[:30])
                return True
            _target_bbox = _target.get("bbox")
            if _target_bbox and self._scroll_form_down_uia_percent(state, _target_bbox[1]):
                return True
        if self._scroll_form_down_uia(state):
            return True
        return self._scroll_form_down_wheel(state)

    def _scroll_form_down_wheel(self, state: Dict[str, Any]) -> bool:
        """
        Scroll the active form window down to reveal fields hidden below the visible area.
        Uses pyautogui.scroll over the center of the active window elements.
        Returns True if scroll was attempted.
        """
        try:
            import pyautogui
            elements = state.get("elements", [])
            # Exclude comboboxcontrol — scroll wheel over a combobox changes its value.
            _SAFE_TYPES = {
                "editcontrol", "checkboxcontrol",
                "radiobuttoncontrol", "tabitemcontrol", "buttoncontrol",
            }
            active = [e for e in elements
                      if e.get("type") in _SAFE_TYPES
                      and e.get("window_role") != "background"
                      and e.get("bbox")]
            # Fallback: any non-background, non-combobox element with bbox
            if not active:
                active = [e for e in elements
                          if e.get("type") != "comboboxcontrol"
                          and e.get("window_role") != "background" and e.get("bbox")]
            if not active:
                return False
            # Use centroid of active form elements, capped to the middle of the
            # screen so repeated scrolls don't push the target off the bottom edge.
            xs = [(e["bbox"][0] + e["bbox"][2]) / 2 for e in active]
            ys = [(e["bbox"][1] + e["bbox"][3]) / 2 for e in active]
            cx = sum(xs) / len(xs)
            cy = min(sum(ys) / len(ys), state.get("screen_resolution", [1920, 1080])[1] * 0.55)
            # Final safety: if the computed point is inside a combobox, shift left
            _cb_list = [e for e in elements
                        if e.get("type") == "comboboxcontrol" and e.get("bbox")]
            for _cb in _cb_list:
                bx1, by1, bx2, by2 = _cb["bbox"]
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    cx = max(bx1 - 40, 10)
                    logger.debug("Scroll-form: shifted cx=%.0f to avoid combobox", cx)
                    break
            # Land the NEXT unfilled field at the TOP of the viewport in one
            # motion, instead of nudging by a fixed/guessed amount -- found
            # 2026-08-08, live: a fixed small scroll only ever revealed one
            # field, not the "optimal view" (max actionable targets visible
            # at once) this module exists to maximize. A first attempt scaled
            # scroll magnitude by a guessed count-based heuristic; told
            # directly by the user this can be done precisely instead, and it
            # can -- the scroll-units-to-pixels ratio isn't known in advance,
            # but it doesn't need to be guessed: scroll a small known amount,
            # measure how far the SAME element actually moved on screen
            # Scroll amount scales with how much unfilled fillable content is
            # still off-screen below the viewport -- bigger gap, bigger scroll.
            #
            # Found 2026-08-08, live, across several rounds tonight: a fixed
            # small scroll only revealed one field at a time. Tried making
            # this function self-calibrating instead (scroll a known nudge,
            # re-observe, measure exactly how far the target moved, compute
            # the precise remaining distance) -- but that added a SECOND,
            # internal observe-and-decide cycle inside this function, on top
            # of the CALLER's own separate observe-and-click cycle right
            # after this returns. Two independent observation cycles
            # stacked on each other is exactly the "two systems, two chances
            # to disagree" bug class already found and fixed three times
            # earlier tonight (verify-at-fill, the dead-scroll counter, the
            # focus-transition race) -- and it reproduced here too: repeated
            # "barely moved" / "actionable field revealed" cycles with
            # nothing ever getting filled. Simplified back to ONE decisive
            # scroll here, informed by a count-based estimate rather than a
            # precise measured one, and left the single "observe, find
            # target, click" cycle entirely to the caller (already correct,
            # already fixed for this exact class of bug once already).
            _vb = self._form_viewport_bottom(state)
            _remaining_below = sum(
                1 for e in elements
                if e.get("type") in ("editcontrol", "comboboxcontrol", "checkboxcontrol")
                and e.get("window_role") != "background"
                and not (e.get("value") or "").strip()
                and e.get("bbox") and e["bbox"][1] > _vb
            )
            _scroll_units = -5 if _remaining_below <= 1 else -15 if _remaining_below <= 4 else -25
            orig = pyautogui.position()
            pyautogui.moveTo(cx, cy, duration=0.1)
            pyautogui.scroll(_scroll_units)
            pyautogui.moveTo(orig.x, orig.y, duration=0.1)
            logger.info("Scroll-form: scrolled down %d units at form center (%.0f, %.0f) "
                        "— %d unfilled fields still below viewport",
                        -_scroll_units, cx, cy, _remaining_below)
            return True
        except Exception as exc:
            logger.warning("Scroll-form: failed — %s", exc)
            return False

    # ── EXPERIMENTAL: OCR background-window reader ───────────────────────────
    # Uses Tesseract to screenshot the background window (Notepad, PDF, etc.)
    # and convert detected text boxes into trace-format elements tagged
    # source="ocr", window_role="background".  Only fires when UIA returns
    # thin background data (≤1 char).  Result is cached per window title so
    # Tesseract only runs once per unique background document, not every step.

    def _ocr_background_window(self, state: Dict[str, Any]) -> List[Dict]:
        """
        Screenshot the background window and run Tesseract OCR on it.
        Returns a list of trace-compatible elements (or [] on any error).
        Results are cached by window title so OCR only runs once per document.
        """
        try:
            import pytesseract
            from PIL import ImageGrab
            import win32gui
        except ImportError:
            return []

        # Find the background window hwnd and title
        bg_hwnd  = None
        bg_title = ""
        elements = state.get("elements", [])
        for e in elements:
            if e.get("window_role") != "background":
                continue
            win_title = (e.get("window_title") or "").strip()
            if not win_title:
                continue
            hwnd = win32gui.FindWindow(None, win_title)
            if hwnd:
                bg_hwnd  = hwnd
                bg_title = win_title
                break

        if not bg_hwnd or not bg_title:
            return []

        # Return cached result if we've already OCR'd this window
        if bg_title in self._ocr_cache:
            logger.debug("OCR cache hit for %r (%d elements)", bg_title,
                         len(self._ocr_cache[bg_title]))
            return self._ocr_cache[bg_title]

        logger.info("OCR: scanning background window %r …", bg_title)

        try:
            # Screenshot only the background window region
            rect = win32gui.GetWindowRect(bg_hwnd)   # (left, top, right, bottom)
            img  = ImageGrab.grab(bbox=rect)
            win_x, win_y = rect[0], rect[1]

            # Run Tesseract — page segmentation mode 11 (sparse text) works
            # well for mixed label+value layouts like insurance forms / PDFs.
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--psm 11",
            )

            ocr_elems = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                conf = int(data["conf"][i])
                text = (data["text"][i] or "").strip()
                if conf < 30 or not text:
                    continue

                # Convert from image-relative coords to screen coords
                x1 = win_x + data["left"][i]
                y1 = win_y + data["top"][i]
                x2 = x1    + data["width"][i]
                y2 = y1    + data["height"][i]

                # Infer element type from text shape
                if text.endswith(":"):
                    elem_type = "label"
                elif len(text) > 3 and text[0].isdigit():
                    elem_type = "input"
                else:
                    elem_type = "label"

                ocr_elems.append({
                    "element_id":   f"ocr_{i}",
                    "type":         elem_type,
                    "control_type": "OCR",
                    "bbox":         [x1, y1, x2, y2],
                    "text":         text,
                    "value":        text,
                    "label":        text,
                    "enabled":      True,
                    "visible":      True,
                    "focused":      False,
                    "confidence":   conf / 100.0,
                    "source":       "ocr",
                    "window_role":  "background",
                    "window_title": bg_title,
                    "app":          "",
                    "metadata":     {"ocr_conf": conf},
                })

            logger.info("OCR: found %d text boxes in %r", len(ocr_elems), bg_title)
            self._ocr_cache[bg_title] = ocr_elems
            return ocr_elems

        except Exception as exc:
            logger.warning("OCR scan failed: %s", exc)
            return []

    def _ensure_form_foreground(self, state: Dict[str, Any]) -> None:
        """
        Bring the form window to the foreground before sending keyboard events.
        pyautogui sends keystrokes to the OS foreground window — if Notepad or
        the terminal grabbed focus, keystrokes land there instead of the form.
        Uses the hwnd of the first active (non-background) element in state.
        """
        try:
            import win32gui
            elements = state.get("elements", [])
            active_el = next((e for e in elements
                              if e.get("window_role") != "background"
                              and e.get("bbox")), None)
            if not active_el:
                return
            bbox = active_el["bbox"]
            cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            hwnd = win32gui.WindowFromPoint((cx, cy))
            if not hwnd:
                return
            # Walk up to the top-level window
            parent = win32gui.GetParent(hwnd)
            while parent:
                hwnd   = parent
                parent = win32gui.GetParent(hwnd)
            fg = win32gui.GetForegroundWindow()
            if fg == hwnd:
                return  # already foreground — nothing to do
            win32gui.SetForegroundWindow(hwnd)
            import time as _t; _t.sleep(0.05)
        except Exception:
            pass

    def _scroll_form_to_top(self, state: Dict[str, Any]) -> None:
        """
        Scroll the active form window back to the top so that _focus_first_empty_field
        always starts from the first visible field, not mid-page.
        Uses Ctrl+Home then multiple large scroll-up passes to guarantee reaching the top.
        """
        try:
            import pyautogui
            elements = state.get("elements", [])
            # Exclude comboboxcontrol — scroll wheel over a combobox changes its value.
            _SAFE_TYPES = {
                "editcontrol", "checkboxcontrol",
                "radiobuttoncontrol", "tabitemcontrol", "buttoncontrol",
            }
            active = [e for e in elements
                      if e.get("type") in _SAFE_TYPES
                      and e.get("window_role") != "background"
                      and e.get("bbox")]
            if not active:
                active = [e for e in elements
                          if e.get("type") != "comboboxcontrol"
                          and e.get("window_role") != "background" and e.get("bbox")]
            if not active:
                return
            xs = [(e["bbox"][0] + e["bbox"][2]) / 2 for e in active]
            ys = [(e["bbox"][1] + e["bbox"][3]) / 2 for e in active]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            # Final safety: if the computed point is inside a combobox, shift left
            _cb_list2 = [e for e in elements
                         if e.get("type") == "comboboxcontrol" and e.get("bbox")]
            for _cb2 in _cb_list2:
                bx1, by1, bx2, by2 = _cb2["bbox"]
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    cx = max(bx1 - 40, 10)
                    break
            orig = pyautogui.position()
            pyautogui.moveTo(cx, cy, duration=0.15)
            # Click the form body so it receives keyboard events, then Ctrl+Home
            # to jump unconditionally to scroll position (0, 0).  Follow with
            # several large upward scrolls as insurance for panels that don't
            # honour Ctrl+Home.
            pyautogui.click(cx, cy)
            time.sleep(0.05)
            pyautogui.hotkey("ctrl", "Home")
            time.sleep(0.15)
            for _ in range(5):
                pyautogui.scroll(200)
                time.sleep(0.05)
            pyautogui.moveTo(orig.x, orig.y, duration=0.1)
            logger.info("Scroll-form-top: scrolled to top at (%.0f, %.0f)", cx, cy)
        except Exception as exc:
            logger.warning("Scroll-form-top: failed — %s", exc)

    def _uia_focus_first_field(self) -> bool:
        """
        Walk the UIA tree of the foreground window and SetFocus on the first
        EditControl or ComboBox not yet handled this tab pass.
        Restricts the walk to the currently active tab pane so that fields on
        inactive tabs (whose deeply-nested controls can still have positive
        BoundingRectangles in UIA even after wx hides the panel) are never
        mistakenly focused.
        Returns True if a field was found and focused.
        """
        try:
            import win32gui as _w32g
            import uiautomation as _uia
            fg = _w32g.GetForegroundWindow()
            if not fg:
                return False
            root = _uia.ControlFromHandle(fg)
            _filled_lower = {s.lower() for s in self._filled_this_tab}
            _TARGET_NAMES = {"Edit", "ComboBox"}

            # ── Detect active tab pane ────────────────────────────────────────
            # Bug found 2026-08-07, live: iterating _TAB_PANE_NAMES in list order
            # and taking the FIRST one that "exists with a positive-coord child"
            # is not the same as finding the ACTUALLY active one — verified live
            # this caused the agent to lock onto an earlier tab's pane and skip
            # straight past Policyholder (Policy -> Vehicle -> Coverage, only
            # 3/13 Policy fields touched) because that earlier pane satisfied the
            # loop's check first. self._current_tab_idx is already reliably
            # maintained elsewhere (it correctly detects "already on this tab"),
            # so use it directly instead of guessing from coordinates.
            _TAB_PANE_NAMES = self._scope.tab_pane_names   # scope-provided; [] → none
            _search_root = root   # fallback: search entire tree
            if 0 <= self._current_tab_idx < len(_TAB_PANE_NAMES):
                _pname = _TAB_PANE_NAMES[self._current_tab_idx]
                # Bug found 2026-08-07, live, right after the fix above landed:
                # this lookup ran ~3s after the tab-switch click and still came
                # up empty for a genuinely-active, 31-field tab (Policyholder) —
                # wx hadn't finished registering the new page's controls into
                # the UIA tree yet. A single maxSearchSeconds=0.05 check doesn't
                # give it room to settle. Retry briefly before accepting "not
                # found" and falling back to the coordinate-guess scan, which
                # is exactly the path that then wrongly skips the whole tab.
                _pane = None
                for _attempt in range(4):   # ~0.05 + 0.2*3 = up to ~0.65s total
                    _pane = root.PaneControl(searchDepth=6, Name=_pname)
                    if _pane.Exists(maxSearchSeconds=0.05):
                        break
                    time.sleep(0.2)
                if _pane is not None and _pane.Exists(maxSearchSeconds=0.05):
                    _search_root = _pane
                    logger.info("_uia_focus_first_field: active pane = %r (tab idx %d)",
                                _pname, self._current_tab_idx)
                else:
                    logger.warning("_uia_focus_first_field: pane %r (tab idx %d) not found after retries — "
                                    "falling back to coordinate-guess scan.",
                                    _pname, self._current_tab_idx)
                    # Diagnostics — the retry fix didn't hold up on a live re-test
                    # (still "not found" ~4s after the click, ruling out a simple
                    # timing gap). Dump exactly what's really in the tree instead
                    # of guessing at another fix blind: which window this actually
                    # queried, and what pane names genuinely exist at this moment.
                    try:
                        _fg_title = _w32g.GetWindowText(fg)
                        logger.warning("_uia_focus_first_field DIAG: foreground hwnd=%s title=%r", fg, _fg_title)
                        _found_panes = []
                        def _dump(ctrl, depth=0):
                            if depth > 5:
                                return
                            try:
                                if ctrl.ControlTypeName == "PaneControl":
                                    _found_panes.append((ctrl.Name, depth))
                            except Exception:
                                pass
                            for c in ctrl.GetChildren():
                                _dump(c, depth + 1)
                        _dump(root)
                        logger.warning("_uia_focus_first_field DIAG: panes actually in tree = %s", _found_panes[:20])
                    except Exception as _diag_exc:
                        logger.warning("_uia_focus_first_field DIAG: dump failed — %s", _diag_exc)
            if _search_root is root:
                # Fallback only: current-tab-idx lookup unavailable/failed. Same
                # coordinate-based guess as before — kept for robustness, not trusted.
                for _pname in _TAB_PANE_NAMES:
                    _pane = root.PaneControl(searchDepth=6, Name=_pname)
                    if not _pane.Exists(maxSearchSeconds=0.05):
                        continue
                    for _ch in _pane.GetChildren():
                        try:
                            _r = _ch.BoundingRectangle
                            if _r.left >= 0 and _r.top >= 0 and _r.width > 0:
                                _search_root = _pane   # restrict walk to this pane
                                break
                        except Exception:
                            pass
                    if _search_root is not root:
                        logger.info("_uia_focus_first_field: active pane = %r (coordinate guess)", _pname)
                        break

            # ── Walk the active pane (or full tree as fallback) ───────────────
            # Collect ALL candidates, then sort by screen position (top → left)
            # so we always focus the visually-topmost field regardless of
            # the UIA child enumeration order (which differs from creation order
            # on wxPython ScrolledPanel controls).
            _candidates: list = []
            _diag_seen_target_type  = 0   # Edit/ComboBox controls seen at all
            _diag_filtered_filled   = 0   # excluded: name already in _filled_this_tab
            _diag_filtered_unnamed  = 0   # excluded: no Name at all
            _diag_filtered_offscreen = 0  # excluded: bbox not on-screen

            def _walk(ctrl, depth=0):
                nonlocal _diag_seen_target_type, _diag_filtered_filled
                nonlocal _diag_filtered_unnamed, _diag_filtered_offscreen
                if depth > 12:
                    return
                try:
                    ctn = ctrl.ControlTypeName
                except Exception:
                    ctn = ""
                if ctn in _TARGET_NAMES:
                    _diag_seen_target_type += 1
                    name = (ctrl.Name or "").strip()
                    if not name:
                        _diag_filtered_unnamed += 1
                    elif name.lower() in _filled_lower:
                        _diag_filtered_filled += 1
                    else:
                        try:
                            rect = ctrl.BoundingRectangle
                            # Only include controls that are actually on-screen
                            # (positive coords, non-zero size).  This filters out
                            # controls on inactive tab panels that wx parks off-screen.
                            if (rect.left >= 0 and rect.top >= 0
                                    and rect.width > 0 and rect.height > 0):
                                _candidates.append((rect.top, rect.left, ctrl))
                            else:
                                _diag_filtered_offscreen += 1
                        except Exception:
                            pass
                for child in ctrl.GetChildren():
                    _walk(child, depth + 1)

            _walk(_search_root)
            if not _candidates:
                # Found 2026-08-08, live: this returned False silently, with NO
                # log line, right after navigation_protocol.has_visible_empty_
                # target() had JUST confirmed something actionable was on
                # screen -- a caller couldn't tell WHY this and that function
                # disagreed (they use two different "already handled" trackers:
                # this one filters via self._filled_this_tab, that one via
                # attempted_keys). Log the actual breakdown instead of nothing,
                # so the next occurrence has real evidence instead of a guess.
                logger.warning(
                    "_uia_focus_first_field: zero candidates (saw %d Edit/ComboBox "
                    "controls total: %d filtered as already-filled, %d unnamed, "
                    "%d off-screen) — but navigation protocol thought something "
                    "was actionable here.",
                    _diag_seen_target_type, _diag_filtered_filled,
                    _diag_filtered_unnamed, _diag_filtered_offscreen)
                return False
            _candidates.sort(key=lambda t: (t[0], t[1]))
            target = _candidates[0][2]
            target.SetFocus()
            logger.info("Tab-advance focus [pane-scoped]: UIA SetFocus on %r (first unhandled)", (target.Name or "").strip())
            return True
        except Exception as exc:
            logger.warning("_uia_focus_first_field: failed — %s", exc)
            return False

    def _focus_first_empty_field(
        self,
        state:        Dict[str, Any],
        after_scroll: bool = False,
        min_y:        float = 0,
    ) -> bool:
        """
        Click the first editcontrol on the current tab that has not yet been
        handled this tab pass.  Includes both empty fields (need filling) and
        non-empty fields with wrong values (need overwriting) — the auto-fill
        overwrite / auto-skip logic handles each field correctly once focused.
        Returns True if a field was found and clicked, False otherwise.

        after_scroll=True : historical flag (kept for callers); filtering is now
                            always done via _filled_this_tab.
        min_y             : only consider fields whose top-edge y >= min_y (used
                            by pane-escape to avoid jumping back above the pane).

        Bug found 2026-08-07 via a live stuck-loop (repeatedly re-focusing a
        field from an INACTIVE tab, forever): the candidate scan below trusts
        `state["elements"]` bboxes to tell active-tab fields from inactive-tab
        ones, which does not hold — verified live that a hidden tab's field
        can report the exact same positive on-screen bbox as when visible.
        What IS reliable, also verified live: the inactive tab's own PANE
        genuinely stops existing in the UIA tree (Exists()==False), while the
        active tab's pane exists with real children. `_uia_focus_first_field`
        already implements exactly that pane-scoped search correctly.

        First fix attempt tried it, then fell back to the scan below on
        failure — still broken, caught live: when the active pane is
        genuinely out of unhandled fields, `_uia_focus_first_field` correctly
        returns False, but the "fallback" then went and found a WRONG-tab
        field via the very scan that can't tell tabs apart, masking the one
        signal (`False` = nothing left here) that callers already use
        correctly to advance to the next tab (`_try_advance_tab`). So for the
        common case (min_y<=0, true for every current caller), the pane-scoped
        result is now authoritative — no unsafe fallback. The state-dict scan
        only still runs for the min_y>0 pane-escape case, which
        `_uia_focus_first_field` doesn't support and no caller currently uses.
        """
        if min_y <= 0:
            return self._uia_focus_first_field()
        elements = state.get("elements", [])
        # Compute actual tab-strip bottom so clicks never land on the tab bar.
        _tab_bottoms = [
            e["bbox"][3] for e in elements
            if e.get("type") in ("tabitem", "tabitemcontrol")
            and e.get("window_role") != "background"
            and e.get("bbox")
        ]
        _tab_floor = (max(_tab_bottoms) + 5) if _tab_bottoms else 110
        # For tab-switch (min_y=0) use no y-floor: fields above the viewport have negative
        # screen-y in a scrolled panel but UIA SetFocus reaches them regardless of position.
        # For pane-escape (min_y=_pane_y>0) keep the floor so we don't jump above the pane.
        _min_y = min_y if min_y > 0 else float("-inf")
        # Candidates: any enabled editcontrol or combobox not yet handled this tab pass.
        # _filled_this_tab stores section-prefixed keys (e.g. "Driver 2 First Name") for
        # driver/vehicle fields and bare names for top-level fields.
        _filled_lower = {s.lower() for s in self._filled_this_tab}
        fillable = []
        for e in elements:
            if (e.get("window_role") == "background"
                    or e.get("type") not in ("editcontrol", "comboboxcontrol")
                    or not e.get("bbox")
                    or not e.get("enabled", True)
                    or e["bbox"][1] < _min_y):
                continue
            fn   = (e.get("label") or e.get("text") or "").strip()
            sec  = self._detect_section(state, e)
            fkey = f"{sec} {fn}".lower() if sec else fn.lower()
            if fkey in _filled_lower or fn.lower() in _filled_lower:
                continue   # already handled this tab pass
            fillable.append(e)
        if not fillable:
            return False
        # Sort top-to-bottom so we get the topmost unfilled field
        fillable.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
        e = fillable[0]
        x1, y1, x2, y2 = e["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        field_label = (e.get("label") or e.get("text") or "?").strip()
        # Try UIA SetFocus first — avoids coordinate mis-clicks near the tab strip.
        # Falls back to coordinate click if UIA search fails or times out.
        if field_label and field_label != "?":
            try:
                import win32gui as _w32g
                import uiautomation as _uia_mod
                _fg = _w32g.GetForegroundWindow()
                _root = _uia_mod.ControlFromHandle(_fg)
                # Try EditControl first, then ComboBox (for combobox fields)
                _ctrl = _root.EditControl(searchDepth=10, Name=field_label)
                if not _ctrl.Exists(maxSearchSeconds=0.2):
                    _ctrl = _root.ComboBoxControl(searchDepth=10, Name=field_label)
                if _ctrl.Exists(maxSearchSeconds=0.2):
                    _ctrl.SetFocus()
                    logger.info("Tab-advance focus [state-scan FALLBACK]: UIA SetFocus on %r (first unhandled)", field_label)
                    return True
            except Exception:
                pass  # fall through to coordinate click
        # Fallback: coordinate click, clipped to be within the field and below the tab strip
        cy = max(cy, _tab_floor)
        cy = min(cy, y2 - 2)
        logger.info("Tab-advance focus: clicking first unhandled field %r @ (%.0f, %.0f)", field_label, cx, cy)
        self._executor.execute({"action_type": "click", "click_position": [cx, cy]})
        return True

    def _try_advance_tab(self, state: Dict[str, Any]) -> bool:
        """
        When stuck (too many no_change steps), click the next form tab.
        Returns True if a tab was found and clicked.
        """
        elements = state.get("elements", [])
        # Accept both UIA type names: "tabitem" (standard) and "tabitemcontrol" (wxPython)
        tabs = [
            e for e in elements
            if e.get("type") in ("tabitem", "tabitemcontrol")
            and e.get("window_role") != "background"
            and e.get("bbox")
        ]
        logger.info("_try_advance_tab: found %d tab(s): %s",
                    len(tabs),
                    [((e.get("text") or e.get("label") or "?"), e.get("type")) for e in tabs])
        if not tabs:
            # Last resort: re-observe after scrolling form back to top so tab bar is visible
            logger.warning("_try_advance_tab: no tabs found — scroll to top and re-observe.")
            self._scroll_form_to_top(state)
            # `time` is already imported at module level (line 51) -- this used
            # to needlessly re-import it locally, which shadows the module-level
            # name for this WHOLE function (Python scoping: any local
            # assignment/import makes a name local throughout its function, even
            # before the import line executes). Harmless while this was the only
            # time.sleep() call in _try_advance_tab; broke the multi-record
            # advance branch added 2026-08-12, which calls time.sleep() via a
            # different code path that never reaches this import.
            time.sleep(0.5)
            state2 = self._observe()
            tabs = [
                e for e in state2.get("elements", [])
                if e.get("type") in ("tabitem", "tabitemcontrol")
                and e.get("window_role") != "background"
                and e.get("bbox")
            ]
            logger.info("_try_advance_tab: after re-observe found %d tab(s)", len(tabs))
            if not tabs:
                return False

        # Detect the active tab (see _detect_active_tab_idx_raw's own docstring
        # for the mechanism). Mid-run, a detection/tracker MISMATCH is treated as
        # noise and the tracker wins — see that function's caller-guidance note
        # for why this differs from the fresh-startup case.
        _all_elems  = state.get("elements", [])
        _detected   = _detect_active_tab_idx_raw(tabs, _all_elems)
        _active_idx = _detected if _detected is not None else self._current_tab_idx
        if _active_idx != self._current_tab_idx:
            logger.warning(
                "_try_advance_tab: pane detection drifted (detected=%d, tracked=%d) — using tracker.",
                _active_idx, self._current_tab_idx)
            _active_idx = self._current_tab_idx

        logger.info("_try_advance_tab: active tab detected as idx=%d (fallback=%d)",
                    _active_idx, self._current_tab_idx)

        next_idx = _active_idx + 1
        if next_idx >= len(tabs):
            # Found live 2026-08-08, directly reported: "that's good. The only
            # problem is that it didn't submit." The log already said
            # "signalling done" right here — but that was only ever a LOG
            # MESSAGE, no caller ever acted on it; every call site just
            # treats a False return as "advance failed", falls through to
            # normal per-step processing, and hopes the transformer someday
            # clicks Submit on its own (it doesn't reliably learn to, and
            # never has in any run this session). This is the one place in
            # the whole file that already knows, with certainty, "there is no
            # next tab AND this tab's own content is exhausted" (callers only
            # reach here via the stuck/drought guards and Navigation
            # Protocol's ADVANCE_TAB, all of which already confirmed
            # scrolling reveals nothing new) — the exact, generic condition
            # for "finished, click the way to finish this record." Reuses
            # _find_submit_button()/_SUBMIT_KW (the same generic keyword
            # matching already used elsewhere to BLOCK premature submit
            # clicks) — no form-specific button text, tab name, or tab count
            # assumed; this applies to any tabbed form with a submit-like
            # button, driven purely by tab POSITION (last) and Navigation
            # Protocol's own exhaustion signal, both already form-agnostic.
            logger.info("Stuck guard: already on last tab (%d tabs total) — exhausted.",
                        len(tabs))
            _submit_btn = _find_submit_button(state.get("elements", []))
            if _submit_btn:
                _sb1, _sy1, _sb2, _sy2 = _submit_btn["bbox"]
                _sname = (_submit_btn.get("text") or _submit_btn.get("label") or "?").strip()
                logger.info("Stuck guard: last tab exhausted — clicking %r @ (%.0f, %.0f).",
                            _sname, (_sb1 + _sb2) / 2, (_sy1 + _sy2) / 2)
                self._executor.execute({"action_type": "click",
                                        "click_position": [(_sb1 + _sb2) / 2, (_sy1 + _sy2) / 2]})
                # car_insurance_form_wx.py's own _on_submit() always resets to
                # tab 0 — force the agent's own tracking to match immediately.
                # Belt-and-suspenders with _refresh_record_cache's own reset
                # below (which fires on a record_num change) — without this
                # explicit line, the backward-tab-click guard
                # (execution_backward_tab_click_guard) would wrongly block the
                # legitimate next click onto tab 0, mistaking the post-submit
                # reset for the exact erroneous backward jump it exists to catch.
                self._current_tab_idx = 0
                # Multi-record advance (2026-08-12) — see _next_record_or_finished
                # and self._total_records's own comments for the history this
                # replaces (self._record_num used to never change here at all).
                # --end_record caps this independent of how many records the
                # intake text actually has, e.g. --start_record 2 --end_record 3
                # to attempt only records 2-3 of a 5-record file.
                _effective_total = (min(self._total_records, self._end_record)
                                     if self._end_record is not None else self._total_records)
                _next = _next_record_or_finished(self._record_num, _effective_total)
                if _next is not None:
                    logger.info("Record %d/%d submitted — advancing to record %d.",
                                self._record_num, _effective_total, _next)
                    self._record_num = _next
                    time.sleep(self.step_delay)   # let the form's own reset finish
                    self._refresh_record_cache(self._observe())
                    self._record_just_advanced = True
                else:
                    logger.info("Record %d/%d submitted — no further records, run complete.",
                                self._record_num, _effective_total)
                    self._task_finished = True
                return True
            logger.info("Stuck guard: last tab exhausted, but no submit-like button found — "
                        "signalling done.")
            return False

        next_tab = tabs[next_idx]
        x1, y1, x2, y2 = next_tab["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tab_name = (next_tab.get("text") or next_tab.get("label") or "?").strip()
        logger.info("Stuck guard: advancing to tab %r @ (%.0f, %.0f)", tab_name, cx, cy)
        # Blacklist the tab we're LEAVING (not the one we're going to) — see
        # _advance_blacklist_pos's own comment for why. ACCUMULATES (adds to
        # the set, doesn't replace it) so a click back to ANY already-left
        # tab is blocked, not just the single most recent one.
        _left_tab = tabs[_active_idx]
        if _left_tab.get("bbox"):
            _lx1, _ly1, _lx2, _ly2 = _left_tab["bbox"]
            self._advance_blacklist_pos.add(_round_click_pos([(_lx1 + _lx2) / 2, (_ly1 + _ly2) / 2]))
        self._executor.execute({"action_type": "click", "click_position": [cx, cy]})
        self._current_tab_idx = next_idx
        return True

    def _merge(
        self,
        t_pred:     Dict[str, Any],
        t_conf:     float,
        llm_action: Dict[str, Any],
        state:      Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine transformer + LLM outputs into one prediction.

        Rule:
          - LLM decides the action type (it has the reasoning).
          - For CLICK: if transformer also says click with good confidence,
            use its learned coordinates instead of label-resolution — this
            reflects where the user actually clicks, not just element centres.
          - For KEYBOARD/TYPE: LLM supplies the text value; transformer's
            source_elem_idx acts as a fallback if LLM text is empty.
          - For HOTKEY / SCROLL: LLM wins entirely.
        """
        l_type = llm_action.get("action_type", "wait")

        # If transformer is confident it should click but LLM says type,
        # trust the transformer — it learned from real demos and knows which
        # elements are clickable vs typeable (e.g. comboboxes need click).
        # Exception: if transformer's click lands on a tab element, do NOT override —
        # the LLM is still trying to fill fields on the current tab.
        #
        # Exception #2, found live 2026-08-07 ("Still not fixed btw, new loop"),
        # right after the fill-click-anchoring fix started landing clicks EXACTLY
        # on the focused field instead of drifting elsewhere: 'Down Payment ($)'
        # and 'Payment Due Date' (both plain editcontrol) looped forever with
        # "Validator -> no_change" every time. Root cause was always here, just
        # invisible before -- this override never checked the FOCUSED FIELD'S
        # OWN type, only the transformer's action-type head, which is collapsed
        # (documented elsewhere: constant near-certain confidence regardless of
        # what's actually focused) and therefore ALWAYS says "click". A checkbox
        # or combobox genuinely needs a click before it can be filled; a plain
        # text field does not -- clicking one that's already focused does
        # nothing at all, so the "type" decision (which had the real value)
        # never got executed. The anchoring fix made this land precisely on the
        # already-focused text field every time instead of missing it, which is
        # exactly why the loop only became visible after that fix, not before.
        _tp_fid = state.get("focused_element_id")
        _tp_fel = next((e for e in state.get("elements", []) if e.get("element_id") == _tp_fid), None)
        _tp_fel_ty = (_tp_fel.get("type") or "").lower() if _tp_fel else ""
        _needs_click_first = _tp_fel_ty in ("checkboxcontrol", "checkbox", "comboboxcontrol", "combobox")
        _TRANSFORMER_TYPE_OVERRIDE_THRESHOLD = 0.92
        if (l_type == "type"
                and t_pred.get("action_type") == "click"
                and t_conf >= _TRANSFORMER_TYPE_OVERRIDE_THRESHOLD
                and t_pred.get("click_position")
                and _needs_click_first):
            pos = t_pred["click_position"]
            _tab_elems = [e for e in state.get("elements", [])
                          if e.get("type") in ("tabitem", "tabitemcontrol")
                          and e.get("window_role") != "background"
                          and e.get("bbox")]
            _hits_tab = any(
                e["bbox"][0] <= pos[0] <= e["bbox"][2] and e["bbox"][1] <= pos[1] <= e["bbox"][3]
                for e in _tab_elems
            )
            _pos_key = (round(pos[0] / 10) * 10, round(pos[1] / 10) * 10)
            _pos_blacklisted = _pos_key in self._nochange_click_pos
            if _hits_tab:
                logger.info("[MERGE] TRANSFORMER wants tab advance but LLM says type — deferring to LLM")
            elif _pos_blacklisted:
                logger.info("[MERGE] TRANSFORMER click @ (%.0f,%.0f) blacklisted (prev no_change) — LLM types",
                            pos[0], pos[1])
            else:
                snapped = self._snap(pos, state)
                coords  = snapped or pos
                logger.info("[MERGE] TRANSFORMER overrides LLM type→click  conf=%.2f  @ (%.0f,%.0f)",
                            t_conf, coords[0], coords[1])
                return {"action_type": "click", "click_position": coords}

        # Hotkey / scroll — pure LLM reasoning, transformer can't help
        if l_type in ("hotkey", "scroll"):
            return self._llm_action_to_prediction(llm_action, state)

        # Type — LLM provides the value; transformer source_elem_idx as backup
        if l_type == "type":
            text = llm_action.get("text", "")
            if not text:
                src_idx = t_pred.get("source_elem_idx", -1)
                text = self._text_resolver.resolve(state, source_elem_idx=src_idx)
                if text:
                    logger.info("[MERGE] type: LLM had no text — TRANSFORMER source resolved %r", text[:40])
            if not text:
                return {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
            logger.info("[MERGE] type: value=%r", text[:40])
            # direct_fill_eligible: cheap, no I/O -- scoped to exactly the
            # one control type live-tested 2026-08-14 (plain EditControl).
            # Live-tested: WM_SETTEXT is a silent no-op on ComboBoxControl
            # (_tp_fel_ty already excludes those from this branch's own
            # click-override check above), so this deliberately stays a
            # strict equality, not an "in (...)" membership check -- see
            # _keyboard_direct's own docstring in executor.py for the full
            # story. The live HWND itself is resolved later, only when
            # this flag is set, at the focus/clear call site.
            return {"action_type": "keyboard", "key_count": len(text),
                    "keystrokes": list(text), "text": text,
                    "direct_fill_eligible": _tp_fel_ty == "editcontrol"}

        # Click — prefer LLM's named target (resolved to element bbox center),
        # fall back to transformer coords only when no resolvable label is available.
        if l_type == "click":
            # LLM decided it's a click (WHEN to move). Transformer decides WHICH
            # field (WHERE) — its navigation pointer is always exposed now and is
            # its real strength (click_acc ~0.76). Prefer it; fall back to LLM's
            # named target only if the transformer pointer is invalid.
            _click_conf = t_pred.get("_click_conf", 0.0)
            _t_pos      = t_pred.get("click_position")
            # In pure mode the transformer ALWAYS owns WHERE (LLM only supplies
            # values). Drop the confidence gate so the LLM can't hijack the click
            # target when the pointer is merely low-confidence.
            _conf_gate = 0.0 if self._no_autohandlers else 0.30
            if _t_pos and (_t_pos[0] > 1 or _t_pos[1] > 1) and _click_conf >= _conf_gate:
                snapped = self._snap(_t_pos, state)
                coords  = snapped or _t_pos
                logger.info("[MERGE] TRANSFORMER navigates — click @ (%.0f,%.0f)  ptr_conf=%.2f",
                            coords[0], coords[1], _click_conf)
                return {"action_type": "click", "click_position": coords}

            target = llm_action.get("target", "")
            if target:
                coords = _resolve_target(target, state)
                if coords is not None:
                    logger.info("[MERGE] LLM target (transformer ptr weak) — click=%r @ (%.0f,%.0f)",
                                target, coords[0], coords[1])
                    return {"action_type": "click", "click_position": coords}
            logger.info("[MERGE] LLM wins (no transformer click) — fallback to LLM position")
            return self._llm_action_to_prediction(llm_action, state)

        # Fallback
        logger.info("[MERGE] LLM wins (hotkey/scroll/fallback)")
        return self._llm_action_to_prediction(llm_action, state)

    def _value_for_focused(self, state: Dict[str, Any]) -> str:
        """Return the background-record value for the currently focused field, or ''."""
        elements   = state.get("elements", [])
        focused_id = state.get("focused_element_id")
        focused    = next((e for e in elements if e.get("element_id") == focused_id), None)
        if not focused:
            return ""
        field_name = (focused.get("label") or focused.get("text") or "").strip()
        if not field_name:
            return ""
        bg_elems = [e for e in elements if e.get("window_role") == "background"]
        bg_blobs = [(e.get("value") or "").strip() for e in bg_elems]
        bg_blobs = [b for b in bg_blobs if b]
        if not bg_blobs:
            return ""
        records = {}
        for blob in sorted(bg_blobs, key=len, reverse=True):
            r = _parse_records(blob)
            if r:
                records = r
                break
        if not records:
            return ""
        rec = records.get(self._record_num, records.get(min(records), {}))
        fl = field_name.lower()
        for k, v in rec.items():
            if k.lower() == fl:
                _skip_other = {"no", "yes (check)"}
                return "" if (_is_leave_blank_value(v) or v.lower().strip("()") in _skip_other) else v
        return ""

    # ── LLM dispatch ─────────────────────────────────────────────────────────

    def _source_knows(self, field_name: str) -> bool:
        """Can the data source even identify this field?

        Optional on the adapter: a source without can_answer() keeps the old
        behaviour exactly, because that is what every existing source already
        assumed of itself.
        """
        asker = getattr(self._source, "can_answer", None)
        if not callable(asker):
            return True
        try:
            return bool(asker(field_name))
        except Exception:
            return True

    def _resolve_field_value_with_escalation(self, state: Dict[str, Any],
                                              field_name: str, section: str = "") -> str:
        """Three-attempt escalation used to answer "does the record have
        this field's value" with total confidence, INCLUDING a genuinely
        blank answer -- not just a lookup miss. Tries a plain lookup, then
        a full cache refresh + retry, then a targeted Notepad peek + retry.
        Returns the found value, or "" only once all three attempts agree
        there's nothing there.

        Extracted 2026-08-14 from _ask_llm's own blank fast path (found
        live 2026-08-08, "there seems to be a loop of some kind"): a naive
        single lookup miss isn't trustworthy enough to conclude "blank" --
        'Account Type' got hallucinated by the LLM three times in a row
        before a fourth attempt, using this same escalation, finally got
        it right. Reused by both _ask_llm (the LLM decision path, unchanged
        behavior) and OPT2 batch fast-fill (a new no-model blank-skip path)
        so a genuinely-blank field is recognized with the same confidence
        either way -- one escalation, not two copies that could disagree.
        """
        if not field_name:
            return ""
        expected = self._lookup_field(field_name, section=section)
        if not expected:
            self._refresh_record_cache(state)
            expected = self._lookup_field(field_name, section=section)
        if not expected:
            self._peek_notepad(state, field_name)
            expected = self._lookup_field(field_name, section=section)
        return expected

    def _ask_llm(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer _cached_record (Win32 Notepad read, fully parsed) over visual_cache
        # (which is empty when VLM pre-scan is disabled). This prevents the LLM from
        # reading a truncated UIA text blob and hallucinating field values.
        _vc_for_llm = self._cached_record if self._cached_record else (self._visual_cache or None)
        screen_text = _state_to_text(
            state,
            record_num=self._record_num,
            visual_cache=_vc_for_llm,
            filled_labels=self._filled_this_tab,
        )
        logger.debug("LLM screen context:\n%s", screen_text)

        # Build focused-field banner — put it FIRST so the LLM sees it immediately
        focused_banner = ""
        focused_id = state.get("focused_element_id")
        if focused_id:
            focused_el = next(
                (e for e in state.get("elements", []) if e.get("element_id") == focused_id), None
            )
            if focused_el:
                _fn  = (focused_el.get("label") or focused_el.get("text") or "?").strip()[:200]
                _fv  = (focused_el.get("value") or "").strip()[:200]
                _ft  = focused_el.get("type", "?")
                _fsec = self._detect_section(state, focused_el)
                # Look up the expected value — section-qualified so Driver 3 First Name
                # returns "Tyler" not the policyholder's "James". Three-attempt
                # escalation (plain lookup, cache refresh + retry, Notepad peek +
                # retry) extracted into _resolve_field_value_with_escalation so
                # OPT2 batch fast-fill can reuse the exact same "confirmed blank"
                # trust bar this file already earned live on 2026-08-08.
                _expected = (self._resolve_field_value_with_escalation(state, _fn, section=_fsec)
                             if _fn != "?" else "")
                logger.info("LLM focused-field lookup: field=%r  expected=%r  cache_size=%d",
                            _fn, _expected[:40] if _expected else "", len(self._cached_record))

                # Fast path found 2026-08-07: the record lookup above already
                # HAS the answer — every logged run tonight showed 100% value
                # accuracy, meaning this direct/constant lookup was already
                # correct every single time. The code used to hand that exact
                # value to the LLM as a hint ("use EXACTLY this string... do
                # NOT invent") and then pay a full ~5s network round-trip just
                # to have it echoed back. Skip the call entirely when the
                # lookup is confident; only fall through to the LLM for
                # fields the record genuinely doesn't answer (blank/derived/
                # ambiguous ones — the case it's actually needed for).
                if _expected:
                    logger.info("LLM call skipped — direct lookup already answered %r -> %r",
                                _fn, _expected[:40])
                    return {"action_type": "type", "text": _expected, "_fast_path": "lookup"}

                # Blank fast path, found live 2026-08-08 ("there seems to be a
                # loop of some kind"): three separate, independent lookup
                # attempts above (cache, refresh+retry, direct peek+retry) all
                # came back empty for 'Account Type' — a genuinely, confidently
                # blank field (an ACH bank-account field on a Credit Card
                # payment record). The code used to treat "still nothing" as
                # "ambiguous, ask the LLM" — but asking turned out to be
                # UNRELIABLE specifically for blank fields: the same field, same
                # step loop, got asked 4 times in a row and answered
                # inconsistently — 'Full Coverage' (hallucinated, bled over
                # from Policy Type) three times, then correctly 'blank' only on
                # the fourth attempt. ~15s and 3 wrong answers wasted on a
                # field the deterministic lookup had already exhausted three
                # ways to check. Same trust level as the truthy fast path above
                # (this project's own lookup-as-validator principle, already
                # tracked as the fix for execution_llm_value_errors) — if three
                # independent lookup attempts agree there's no data, there's
                # nothing for the LLM to add by being asked a fourth time.
                if _fn != "?":
                    logger.info("LLM call skipped — direct lookup confirms %r has no record value "
                                "(leave blank)", _fn)
                    return {"action_type": "type", "text": "", "_fast_path": "lookup_blank"}

                _expect_hint = (
                    f"\n  → EXPECTED VALUE FROM DATA SOURCES: {_expected!r}"
                    f"\n  → Use EXACTLY this string as 'text'. Do NOT modify or invent."
                    if _expected else ""
                )
                focused_banner = (
                    f"⚠ CURRENTLY FOCUSED FIELD: [{_ft}] \"{_fn}\""
                    + (f" — current value: {_fv!r}" if _fv else " — EMPTY")
                    + _expect_hint
                    + "\nYou MUST act on THIS field only. Do NOT click other fields.\n\n"
                )

        # Build a compact already-filled summary so the LLM doesn't revisit fields
        filled_fields = []
        for e in state.get("elements", []):
            if e.get("window_role") == "background":
                continue
            val = (e.get("value") or "").strip()
            lbl = (e.get("label") or e.get("text") or "").strip()
            if val and lbl and len(lbl) <= 200 and len(val) <= 200:
                filled_fields.append(f"  {lbl} = {val!r}")
        filled_summary = ""
        if filled_fields:
            filled_summary = "\nAlready filled (DO NOT retype these):\n" + "\n".join(filled_fields[:20])

        user_msg = (
            f"{focused_banner}"
            f"Task goal: {self.goal}\n\n"
            f"Current screen:\n{screen_text}"
            f"{filled_summary}\n\n"
            f"Recent actions:\n{_history_to_text(self._history)}"
        )

        logger.info("LLM prompt size: user_msg=%d chars  screen_text=%d chars",
                    len(user_msg), len(screen_text))

        # Hard safety cap — LMStudio default context is 4096 tokens (~16k chars)
        _MAX_USER_CHARS = 12000
        if len(user_msg) > _MAX_USER_CHARS:
            logger.warning("user_msg too large (%d chars) — truncating screen_text to fit",
                           len(user_msg))
            overhead   = len(user_msg) - len(screen_text)
            keep_chars = max(2000, _MAX_USER_CHARS - overhead)
            screen_text = screen_text[:keep_chars]
            user_msg = (
                f"{focused_banner}"
                f"Task goal: {self.goal}\n\n"
                f"Current screen:\n{screen_text}"
                f"{filled_summary}\n\n"
                f"Recent actions:\n{_history_to_text(self._history)}"
            )
            logger.info("After truncation: user_msg=%d chars", len(user_msg))

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(user_msg)
            elif self.provider == "groq":
                return self._call_openai_compat(user_msg)
            elif self.provider == "gemini":
                return self._call_gemini(user_msg)
            elif self.provider == "lmstudio":
                return self._call_openai_compat(user_msg)
        except json.JSONDecodeError as e:
            logger.warning("LLM non-JSON response: %s", e)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
        return {"action_type": "wait", "reason": "llm unavailable"}

    def _llm_action_to_prediction(
        self, llm_action: Dict[str, Any], state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert an LLM action dict into the executor's prediction format."""
        action_type = llm_action.get("action_type", "wait")

        if action_type == "click":
            target = llm_action.get("target", "")
            coords = _resolve_target(target, state)
            if coords is None:
                logger.warning("LLM target %r not found in element tree — skipping.", target)
                return {"action_type": "no_op"}
            logger.info("Resolved %r → (%.0f, %.0f)", target, coords[0], coords[1])
            return {"action_type": "click", "click_position": coords}

        elif action_type == "type":
            text = llm_action.get("text", "")
            if not text:
                logger.warning("LLM 'type' action has no text — pressing Tab to skip field.")
                return {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}
            return {"action_type": "keyboard", "key_count": len(text),
                    "keystrokes": list(text), "text": text}

        elif action_type == "tab":
            # LLM is signalling "advance to next field/tab" — press Tab key
            return {"action_type": "keyboard", "key_count": 1, "keystrokes": ["tab"]}

        elif action_type == "hotkey":
            keys = llm_action.get("keys", [])
            if not keys:
                return {"action_type": "no_op"}
            return {"action_type": "keyboard", "key_count": len(keys),
                    "keystrokes": keys}

        elif action_type == "scroll":
            target    = llm_action.get("target", "")
            direction = llm_action.get("direction", "down")
            clicks    = int(llm_action.get("clicks", 3))
            # Resolve target to coordinates — scroll at that position
            coords = _resolve_target(target, state) if target else None
            if coords is None:
                # Fall back to centre of screen if no target
                res    = state.get("screen_resolution", [1920, 1080])
                coords = [res[0] / 2, res[1] / 2]
            return {"action_type": "scroll", "click_position": coords,
                    "direction": direction, "clicks": clicks}

        return {"action_type": "no_op"}

    def _call_anthropic(self, user_msg: str) -> Dict[str, Any]:
        resp = self._llm_client.messages.create(
            model=self._llm_model,
            max_tokens=256,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        return _parse_llm_response(resp.content[0].text)

    def _call_openai_compat(self, user_msg: str) -> Dict[str, Any]:
        """Handles both Groq and LM Studio — both use the OpenAI client format."""
        # A random per-call [sid:...] tag used to be appended here specifically
        # to defeat LM Studio's KV-cache reuse ("breaks server-side KV-cache
        # accumulation"). Traced 2026-08-07: that line was bundled into an
        # unrelated commit (OCR-cache instance scoping) with no diagnosed bug,
        # no test, no explanation — not a proven fix. Meanwhile it forced the
        # full ~750-token system prompt to be reprocessed from scratch on
        # every call, measured as the dominant per-step latency cost (~5s/call
        # regardless of max_tokens). Removed so the identical system-prompt
        # prefix can actually be cached across calls within a run. Each call
        # still sends a complete, self-contained message list (no threaded
        # conversation history), so there's nothing else here that could
        # legitimately bleed between calls even if the cache activates.
        system_msg = (
            self._system_prompt
            + "\n\nBefore choosing an action, reason briefly (1-2 short sentences) inside"
            + " <think>...</think> tags. Then output ONLY a JSON object on the last line."
        )
        # max_tokens found 2026-08-07: this task is a short field-value decision,
        # not open-ended generation — 2048 was a generous, unused ceiling that
        # only mattered if the model rambled. Paired with the tightened "1-2
        # short sentences" instruction above so the cap doesn't truncate a
        # verbose <think> block before the JSON line ever gets emitted.
        resp = self._llm_client.chat.completions.create(
            model=self._llm_model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        )
        return _parse_llm_response(resp.choices[0].message.content)

    def _call_gemini(self, user_msg: str) -> Dict[str, Any]:
        resp = self._llm_client.models.generate_content(
            model=self._llm_model,
            contents=user_msg,
        )
        return _parse_llm_response(resp.text)

    # ── observer / transformer helpers ───────────────────────────────────────

    def _observe(self) -> Dict[str, Any]:
        try:
            state = self._observer.snapshot()
            # Validate the perception adapter ONCE against the schema contract.
            # Fail LOUD (log) instead of the agent silently seeing a blank screen
            # because an adapter speaks a different dialect (e.g. Excel type=cell).
            if not self._schema_checked:
                self._schema_checked = True
                try:
                    from components.observers.schema import validate_state
                except ImportError:
                    from observers.schema import validate_state
                for _issue in validate_state(state):
                    _lvl = logger.error if _issue.startswith("ERROR") else logger.warning
                    _lvl("Perception schema: %s", _issue)
            if state and state.get("elements") is not None:
                # EXPERIMENTAL: OCR background-window overlay.
                # Only fires when use_ocr=True AND UIA background data is thin
                # (≤1 char total), meaning the background app (Notepad, PDF viewer,
                # etc.) isn't exposing its content via UIA.  Tesseract screenshots
                # the background window, detects text boxes, and injects them as
                # background elements so the LLM can read label→value pairs.
                if self._use_ocr:
                    bg_text_total = sum(
                        len((e.get("value") or "").strip())
                        for e in state.get("elements", [])
                        if e.get("window_role") == "background"
                    )
                    if bg_text_total <= 1:
                        ocr_elems = self._ocr_background_window(state)
                        if ocr_elems:
                            state["elements"] = state["elements"] + ocr_elems
                            logger.info("OCR overlay: injected %d element(s) from background window",
                                        len(ocr_elems))
                return state
        except Exception as exc:
            logger.warning("Observer error: %s", exc)
        return {"elements": [], "screen_resolution": [1920, 1080]}

    @staticmethod
    def _observed_state_fingerprint(state: Dict[str, Any]) -> tuple:
        """Lightweight signature of an observed state, for _adaptive_settle_wait
        below. Deliberately broader than navigation_protocol.visible_field_
        signature (which is scoped to viewport-visible fields, for scroll-
        movement detection specifically) -- this one catches ANY kind of
        change the UI might still be settling into (a value finishing typing,
        a dialog opening, a delayed repaint), not just fields moving in/out
        of view."""
        els = state.get("elements", [])
        return (
            len(els),
            state.get("focused_element_id"),
            tuple((e.get("element_id"), e.get("value")) for e in els),
        )

    def _adaptive_settle_wait(self, max_wait: float, poll: float = 0.15) -> None:
        """Replaces a blind time.sleep(max_wait) with a real settle-check:
        poll the observed state at short intervals and return as soon as two
        consecutive reads agree the UI has stopped changing, instead of
        always waiting the full budget. Bounded by real wall-clock time, not
        an assumed per-poll cost -- can overshoot max_wait by at most one
        poll interval (a slow _observe() call included), never by more.

        Found live 2026-08-14, direct request ("fix Speed without
        sacrificing reliability"): STEP_DELAY (1.0s, fixed, unconditional on
        every single step) measured as roughly HALF of total per-step wall-
        clock time across a real run. Most steps don't need the full second
        -- the UI has already settled well before it elapses -- but a real
        minority (tab switches, scrolls) genuinely do, per this project's own
        prior incident ("a bug came from checking the screen too soon after
        a tab switch"). A blind constant can't tell the difference; this can,
        using the same verify-don't-assume principle already used everywhere
        else in this file (verify-at-fill, redirect-then-confirm) instead of
        a new mechanism. Worst case (state never stabilizes) still waits the
        same max_wait this replaces, just spent polling instead of sleeping
        blind -- this can only be faster than or equal to today's behavior,
        never slower by more than one poll interval.
        """
        if max_wait <= 0:
            return
        _t0 = time.time()
        try:
            prev = self._observed_state_fingerprint(self._observe())
        except Exception:
            time.sleep(max_wait)
            return
        while time.time() - _t0 < max_wait:
            time.sleep(poll)
            try:
                cur = self._observed_state_fingerprint(self._observe())
            except Exception:
                return
            if cur == prev:
                return
            prev = cur

    @staticmethod
    def _slim_for_model(state: Dict[str, Any]) -> Dict[str, Any]:
        """Match the recorder's slim EXACTLY so the transformer sees the same
        element shape it trained on (else inputs are out-of-distribution)."""
        _KEEP = ("element_id", "type", "label", "text", "value", "bbox",
                 "window_role", "window_title", "app", "focused", "confidence",
                 "attempted")
        out = dict(state)
        slim = []
        for e in state.get("elements", []):
            ne = {k: e[k] for k in _KEEP if k in e}
            lb = ne.get("label")
            if lb and len(lb) > 200:
                ne["label"] = lb[:200]
            for fld in ("value", "text"):
                v = ne.get(fld)
                if v and len(v) > 6000:
                    ne[fld] = v[:6000]
            slim.append(ne)
        out["elements"] = slim
        return out

    def _attempt_key(self, elem: Dict[str, Any], elements: Optional[list] = None):
        """Match transformer._attempt_key exactly (label-primary, scroll-stable,
        disambiguated by rank among same-labeled elements when `elements` — the
        full elements list from elem's own state — is given). Repeated-section
        forms (Driver 1/2/3, Vehicle 1/2, ...) give every instance of a field
        the same label; without disambiguation, filling Driver 1's First Name
        silently marked Driver 2's identically-labeled, still-empty First Name
        as already-attempted too. See transformer._attempt_key for the full
        rationale — keep both copies in sync."""
        lbl = (elem.get("label") or elem.get("text") or "").strip().lower()
        if not lbl:
            b = elem.get("bbox") or [0, 0, 0, 0]
            return ("@", round((b[0] + b[2]) / 2 / 20) * 20, round((b[1] + b[3]) / 2 / 20) * 20)
        if elements:
            same_label_ids = [id(e) for e in elements
                               if (e.get("label") or e.get("text") or "").strip().lower() == lbl]
            if len(same_label_ids) > 1:
                try:
                    return (lbl, same_label_ids.index(id(elem)))
                except ValueError:
                    pass
        return lbl

    def _mark_attempted(self, elem: Dict[str, Any], elements: Optional[list] = None) -> None:
        """Record that a field has been acted on this session (attempted feature)."""
        if isinstance(elem, dict):
            self._attempted_keys.add(self._attempt_key(elem, elements=elements))

    def _elem_at(self, state: Dict[str, Any], pos) -> Optional[Dict[str, Any]]:
        """Element whose bbox contains pos (nearest center on ties)."""
        if not pos or len(pos) < 2:
            return None
        px, py = pos[0], pos[1]
        best, best_d = None, 1e18
        for e in state.get("elements", []):
            b = e.get("bbox")
            if not b or len(b) != 4:
                continue
            if b[0] - 2 <= px <= b[2] + 2 and b[1] - 2 <= py <= b[3] + 2:
                d = ((b[0] + b[2]) / 2 - px) ** 2 + ((b[1] + b[3]) / 2 - py) ** 2
                if d < best_d:
                    best, best_d = e, d
        return best

    def _record_attempt(self, state: Dict[str, Any], prediction: Dict[str, Any],
                         state_after: Optional[Dict[str, Any]] = None) -> None:
        """Mark the element this step acted on — keyboard→focused, click→element
        under the cursor — so the transformer sees it as attempted next frame.

        Found 2026-08-09, live, direct user report ("skipped most fields in
        Vehicle after Current Mileage"), confirmed on a second run via the
        new [FOCUS-DIAG] diagnostic: a plain navigate click aimed at
        'Annual Miles Est.' (its own bbox matched the click position in the
        PRE-click `state`) resulted in real focus landing on the 'Submit'
        button instead — yet this function was always called with that
        stale pre-click `state`, so the position-based lookup below still
        matched 'Annual Miles Est.' and marked THAT field attempted,
        permanently excluding it from ever being offered as a target again
        even though it was never actually focused or typed into. The
        caller already takes a fresh `state_after` observation for
        validation a few lines above this call — it just was never passed
        through. For CLICK actions only (keyboard/type intentionally left
        alone — see below), prefer `state_after`'s own focused_element_id,
        the verified post-action reality, over trusting the click landed
        where it was aimed. Same "verify what actually happened, don't
        assume the intended target was hit" principle already proven
        elsewhere in this file (verify-at-fill, the redirect-verify fix).

        Deliberately NOT applied to the keyboard branch: a type action can
        legitimately trail a Tab/commit keystroke within the same step
        (e.g. combobox selection), which could leave state_after focused on
        the NEXT field rather than the one just typed into — the pre-action
        `state`'s focused_element_id (captured before typing began) stays
        the more reliable signal there.
        """
        at = prediction.get("action_type")
        elements = state.get("elements", [])
        elem = None
        if at == "keyboard":
            fid = state.get("focused_element_id")
            elem = next((e for e in elements if e.get("element_id") == fid), None)
        elif at == "click":
            _after_fid = state_after.get("focused_element_id") if state_after else None
            if _after_fid:
                _after_elements = state_after.get("elements", [])
                elem = next((e for e in _after_elements if e.get("element_id") == _after_fid), None)
                if elem is not None:
                    elements = _after_elements
            if elem is None:
                elem = self._elem_at(state, prediction.get("click_position") or [])
        if elem is not None:
            self._mark_attempted(elem, elements=elements)
            # Real typed text (not a bare "tab"/"ctrl+a" housekeeping
            # keystroke) is the only signal strong enough to trust over a
            # live "is it empty" read for edit/combobox fields — see
            # self._typed_keys' own comment.
            if at == "keyboard" and (prediction.get("text") or "").strip():
                self._typed_keys.add(self._attempt_key(elem, elements=elements))

    def _stamp_attempted_live(self, state: Dict[str, Any]) -> None:
        """Stamp attempted=1 on observed elements acted on earlier this session."""
        if not self._attempted_keys:
            return
        elements = state.get("elements", [])
        for e in elements:
            if self._attempt_key(e, elements=elements) in self._attempted_keys:
                e["attempted"] = 1.0

    def _predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._disable_transformer:
            # Ablation mode, added 2026-08-14, direct request ("I have to
            # verify that if we don't have the model we're using right
            # now the performance would drop"). Skips the real model call
            # entirely and returns a synthetic zero-confidence result
            # instead -- NOT a new fallback invented for this test.
            # Every key here already has a caller-side `.get(key,
            # default)` (confirmed by reading every t_pred access in this
            # file before writing this), so this result routes through
            # the SAME low-confidence Tab-fallback path the real model
            # already triggers whenever ITS OWN confidence is low
            # (_gate_low_confidence_click, "[OPT2] pointer low-confidence
            # ... Tab fallback") -- proven, existing behavior, just
            # forced to fire on every step instead of rarely. Lets a
            # live comparison run (model on vs. model off, same record)
            # answer directly: does the run still complete correctly,
            # and how much slower/faster, with the model never consulted.
            self._ablation_transformer_calls_skipped += 1
            logger.info("[ABLATION] transformer call #%d skipped (disable_transformer=True) "
                        "-- would have asked the model here",
                        self._ablation_transformer_calls_skipped)
            return {"action_type": "no_op", "confidence": 0.0, "_scores": {},
                    "click_position": None, "_click_conf": 0.0}
        try:
            try:
                from components.intelligence.model.transformer import predict
            except ImportError:
                from intelligence.model.transformer import predict
            self._stamp_attempted_live(state)
            return predict(
                state=self._slim_for_model(state),
                history=self._history[-3:],
                model_path=self.model_path,
                device_str=self.device_str,
            )
        except Exception as exc:
            logger.error("Transformer predict failed: %s", exc)
            raise
