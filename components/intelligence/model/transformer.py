"""
transformer.py — TransformerAgentNetwork (all-in-one)
=======================================================
Causal Transformer agent trained via Behavioral Cloning on trace JSON files.

Sequence contract
-----------------
    (state_1, action_1, state_2, action_2, ..., state_t) -> next_action

Sections
--------
  1. Dataset   — encode_state, TrajectoryDataset
  2. Model     — StateEncoder, ActionEncoder, TransformerAgentNetwork
  3. Training  — train()
  4. Inference — predict(), predict_folder()
  5. CLI       — __main__

Quick-start
-----------
  # Train
  python -m components.intelligence.model.transformer \\
      --mode train --data_dir data/output/traces/live --epochs 20

  # Predict
  python -m components.intelligence.model.transformer \\
      --mode predict --trace_path data/output/traces/live/live_step_0006.json

Anti-overfitting tips (small datasets)
---------------------------------------
  - Collect more traces: `python record_trace.py`
  - Shrink the model:    --d_model 64 --num_layers 2
  - Label smoothing:     --label_smoothing 0.1  (built-in, default 0.1)
  - Element dropout:     --aug_drop_prob 0.15   (randomly zeros UI rows)
  - Regularisation:      --dropout 0.3 --weight_decay 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ── sentence embeddings — fully lazy (no import at module level) ───────────────
_SENT_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBED_DIM       = 384              # matches trained checkpoint (elem_features=392 = 8+384)
_sent_model: Optional[Any] = None  # loaded on first predict call
_embed_cache: Dict[str, List[float]] = {}
_SENTENCE_TRANSFORMERS_AVAILABLE: Optional[bool] = None  # None = not yet checked
_EMBED_CACHE_PATH: Optional[str] = None   # set by train() to save alongside model
_loaded_embed_caches: set = set()          # tracks which pkl files were loaded this process


# ═══════════════════════════════════════════════════════════
#  SECTION 1 — DATASET
# ═══════════════════════════════════════════════════════════

ACTION_NOOP     = 0
ACTION_CLICK    = 1
ACTION_KEYBOARD = 2
ACTION_HOTKEY   = 3
ACTION_SCROLL   = 4
ACTION_DCLICK   = 5
ACTION_DRAG     = 6
NUM_ACTIONS     = 7

# ── Universal Semantic Action Space (opt-in: action_space="semantic") ──────────
# Alternative, richer action-type vocabulary sourced from
# components/recorder/action_labeler.py instead of _decode_actions(). Kept
# fully separate from the legacy ACTION_* scheme above — default behavior
# (action_space="legacy") is completely unchanged. Ported 2026-08-12 from the
# origin/verb-loop-rewrite branch (diverged from master 2026-07-10, never
# merged) alongside its own semantic_action.py/action_labeler.py — see
# semantic_action.Verb for what each class means. That branch's own reported
# result: click_acc 0.957 vs its legacy baseline's 0.878, on ITS OWN dataset —
# unverified on ours until the isolated A/B this option enables actually runs.
try:
    from agent.semantic_action import Verb as _Verb, SemanticAction
    from recorder.action_labeler import translate_step as _translate_step
except ImportError:
    from components.agent.semantic_action import Verb as _Verb, SemanticAction
    from components.recorder.action_labeler import translate_step as _translate_step
SEMANTIC_VERBS       = list(_Verb)
NUM_SEMANTIC_ACTIONS = len(SEMANTIC_VERBS)
VERB_TO_ID           = {v: i for i, v in enumerate(SEMANTIC_VERBS)}
ID_TO_VERB           = {i: v for i, v in enumerate(SEMANTIC_VERBS)}

# Hotkey vocabulary — each gets a unique index for the hotkey_head classifier
HOTKEYS = [
    "tab", "shift+tab", "enter", "escape", "backspace", "delete",
    "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z", "ctrl+y",
    "ctrl+tab", "ctrl+shift+tab",
    "home", "end", "page_up", "page_down",
    "arrow_up", "arrow_down", "arrow_left", "arrow_right",
    "f1", "f2", "f3", "f4", "f5", "f6",
]
NUM_HOTKEYS = len(HOTKEYS)
_HOTKEY_IDX = {h: i for i, h in enumerate(HOTKEYS)}

# is_real(1) + bbox(4) + confidence(1) + window_role(1) + is_focused(1) + ctrl_type(1)
#   + is_filled(1) + attempted(1) + text_embedding(384)
# is_filled is PERCEPTION (does this field currently hold a value), read straight
# from the observed element — NOT a hand-tracked progress signal. Without it the
# model only saw field labels and was blind to which fields were already done,
# so it looped / re-entered filled fields.
# attempted is INTERACTION HISTORY: has the operator already acted on this field
# earlier in this session (click or type), regardless of whether it ended up
# filled. is_filled handles "field has a value"; attempted handles the case
# is_filled cannot — an EMPTY optional field (e.g. Suffix with no record value)
# that the model would otherwise re-target forever (is_filled never flips). Once
# acted on, attempted=1 → the model learns to move on. Derived from demo step
# order at train time; tracked by the agent at inference. (This is the principled
# successor to the old is_visited feature — keyed on "acted on", not nav-order.)
ELEM_FEATURES = 11 + _EMBED_DIM
_REAL_ELEM_FLAG_IDX = 0  # feature index for the is_real flag

# Numeric encoding for control types — focused/interactive types get distinct values
# New-style names (from UIAutomationObserver) take priority; legacy wx names kept as fallback.
_CTRL_TYPE_MAP = {
    # new-style UIA names (ui_observer.py _CTRL_TYPE_MAP output)
    "input": 0.1, "combobox": 0.2, "checkbox": 0.3,
    "button": 0.4, "label": 0.5, "window": 0.6,
    "document": 0.7, "list": 0.8, "tabitem": 0.9,
    # legacy wx fallback names
    "editcontrol": 0.1, "comboboxcontrol": 0.2, "checkboxcontrol": 0.3,
    "buttoncontrol": 0.4, "textcontrol": 0.5, "windowcontrol": 0.6,
    "documentcontrol": 0.7, "listcontrol": 0.8,
}
VOCAB_SIZE    = 10_000
DEFAULT_W     = 1920
DEFAULT_H     = 1200


def _get_sent_model():
    """Lazily import sentence-transformers and load the model (once per process)."""
    global _sent_model, _SENTENCE_TRANSFORMERS_AVAILABLE
    import os as _os2
    if _os2.environ.get("NO_SENT_TRANSFORMERS") or _SENTENCE_TRANSFORMERS_AVAILABLE is False:
        return None
    if _sent_model is None:
        _os2.environ["TQDM_DISABLE"] = "1"
        _os2.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from sentence_transformers import SentenceTransformer as _ST
            _sent_model = _ST(_SENT_MODEL_NAME)
            _SENTENCE_TRANSFORMERS_AVAILABLE = True
        except Exception:
            _SENTENCE_TRANSFORMERS_AVAILABLE = False
            return None
    return _sent_model


def _embed_text(text: str) -> List[float]:
    """
    384-dim embedding for a UI element label.
    Inference path: fast deterministic hash (no ML model, no download).
    Training path: sentence-transformers used only when _prime_embed_cache() is called.
    """
    text = (text or "").strip()
    if text not in _embed_cache:
        if not text:
            _embed_cache[text] = [0.0] * _EMBED_DIM
        else:
            import hashlib
            raw = hashlib.sha256(text.encode()).digest()  # 32 bytes
            repeated = (raw * 12)[:_EMBED_DIM]           # repeat to fill 384 bytes
            _embed_cache[text] = [((b / 127.5) - 1.0) for b in repeated]
    return _embed_cache[text]


def _save_embed_cache(path: str) -> None:
    """Persist the current _embed_cache dict to a pickle file."""
    import pickle
    try:
        with open(path, "wb") as _f:
            pickle.dump(dict(_embed_cache), _f)
        print(f"[Encoder] Embed cache saved ({len(_embed_cache)} entries) -> {path}", flush=True)
    except Exception as _e:
        print(f"[Encoder] WARNING: could not save embed cache: {_e}", flush=True)


def _load_embed_cache(path: str) -> None:
    """Load a pickled embed cache into _embed_cache (merged, not replaced)."""
    import pickle, os as _os2
    global _loaded_embed_caches
    if path in _loaded_embed_caches or not _os2.path.exists(path):
        return
    try:
        with open(path, "rb") as _f:
            loaded = pickle.load(_f)
        _embed_cache.update(loaded)
        _loaded_embed_caches.add(path)
        print(f"[Encoder] Embed cache loaded ({len(loaded)} entries) from {path}", flush=True)
    except Exception as _e:
        print(f"[Encoder] WARNING: could not load embed cache: {_e}", flush=True)


def _prime_embed_cache(texts: List[str]) -> None:
    """
    Batch-encode all unique *texts* at once (much faster than one-by-one).
    Call this before building the dataset to pre-populate the cache.
    After encoding, saves the cache to _EMBED_CACHE_PATH if set.
    """
    unseen = [t for t in set(texts) if t.strip() and t not in _embed_cache]
    if not unseen:
        return
    model = _get_sent_model()
    if model is None:
        return
    print(f"[Encoder] Batch-encoding {len(unseen)} unique text strings …", flush=True)
    vecs = model.encode(unseen, show_progress_bar=True,
                        batch_size=64, convert_to_numpy=True)
    for t, v in zip(unseen, vecs):
        _embed_cache[t] = v.tolist()
    print(f"[Encoder] Cache primed ({len(_embed_cache)} entries).", flush=True)
    if _EMBED_CACHE_PATH:
        _save_embed_cache(_EMBED_CACHE_PATH)


def _elem_label(elem: dict) -> str:
    """Stable identity for a UI element across frames."""
    return ((elem.get("label") or elem.get("text") or "").strip())


def _detect_section(elem: dict, elements: Optional[list], section_pattern: Optional[str],
                     section_prefix: str = "section_") -> str:
    """
    Offline mirror of components/agent/agent.py's LLMAgent._detect_section,
    operating on a raw elements snapshot from a trace file instead of live
    agent state (no ScopeConfig instance available at dataset-build time —
    section_pattern/section_prefix are passed in directly instead). Keep the
    geometry rule in sync with agent.py's copy: the section is the lowest
    section-pane whose top edge is at/above elem's vertical centre.

    Output format mirrors scope.py's _default_section_format (kind.title() +
    " " + num) — the only formatter any real ScopeConfig in this repo uses
    today. A scope with a custom section_format would need this kept in sync
    by hand; not worth importing ScopeConfig here just to avoid that, since
    doing so would couple a generic training module to one app's config type.
    """
    import re as _re
    if not section_pattern or not elem or not elem.get("bbox"):
        return ""
    fx1, fy1, fx2, fy2 = elem["bbox"]
    fy_center = (fy1 + fy2) / 2
    section_panes = sorted(
        [e for e in (elements or [])
         if (e.get("type") or "").lower() == "panecontrol"
         and e.get("window_role") != "background"
         and e.get("bbox")
         and (e.get("label") or e.get("text") or "").startswith(section_prefix)],
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
    m = _re.match(section_pattern, current_label.lower())
    if not m:
        return ""
    groups = m.groups()
    return f"{groups[0].title()} {groups[1]}" if len(groups) >= 2 else current_label


def _attempt_key(elem: dict, elements: Optional[list] = None, section: str = ""):
    """
    Session-stable identity for the 'attempted' feature. Label-primary so it
    survives scroll (positions shift, labels don't); falls back to a coarse
    bbox-center bucket for unlabeled elements. Must match between train-time
    derivation and the agent's inference-time tracking (components/agent/agent.py
    carries an identical copy — keep both in sync).

    Repeated-section forms (Driver 1/2/3, Vehicle 1/2, additional insureds, ...)
    routinely give every instance of a field the SAME label ("First Name" on
    every driver block) — found 2026-08-07 from a live report that Driver 1's
    and Driver 2's First Name fields were "indistinguishable." Label-only kept
    them literally indistinguishable: filling Driver 1's First Name marked the
    shared key "first name" as attempted, so Driver 2's First Name (a different,
    still-empty element) silently read as already-done too.

    Two disambiguation strategies, mutually exclusive (section wins if both given):
    - `section` (new, 2026-08-11): the caller has already resolved which repeated
      section (e.g. "Driver 2") this element belongs to, via real panecontrol
      bbox geometry (see _detect_section) — a stable signal that doesn't depend
      on accessibility-tree list order.
    - `elements` (original, 2026-08-07): when the full elements list from the
      SAME state as `elem` is given and more than one element shares `elem`'s
      label, disambiguate by this element's rank among same-labeled elements in
      list order. Real failure precedent: tested alone, this regressed
      val_click_acc to 46.9% (from a 68.9% baseline) — accessibility-tree list
      order isn't a stable enough signal across snapshots (see DEVELOPERS.md).
      Kept as an option, not the default, for exactly that reason.

    Neither given → old label-only behavior (the un-regressed baseline).
    """
    lbl = (elem.get("label") or elem.get("text") or "").strip().lower()
    if not lbl:
        b = elem.get("bbox") or [0, 0, 0, 0]
        return ("@", round((b[0] + b[2]) / 2 / 20) * 20, round((b[1] + b[3]) / 2 / 20) * 20)
    if section:
        return (section, lbl)
    if elements:
        same_label_ids = [id(e) for e in elements
                           if (e.get("label") or e.get("text") or "").strip().lower() == lbl]
        if len(same_label_ids) > 1:
            try:
                return (lbl, same_label_ids.index(id(elem)))
            except ValueError:
                pass
    return lbl


def _encode_element(elem: dict, W: float, H: float, focused_id=None) -> List[float]:
    x1, y1, x2, y2 = (float(v) for v in elem.get("bbox", [0, 0, 0, 0])[:4])
    conf       = float(elem.get("confidence", 0.0))
    role       = 1.0 if elem.get("window_role") == "background" else 0.0
    is_focused = 1.0 if (focused_id and elem.get("element_id") == focused_id) else 0.0
    ctrl_type  = _CTRL_TYPE_MAP.get((elem.get("type") or "").lower(), 0.0)
    # PERCEIVED fill-state: does this field currently hold a value? Read from the
    # element itself. Lets the model tell a done field from an empty one.
    _val       = (elem.get("value") or "").strip()
    is_filled  = 1.0 if _val else 0.0
    # INTERACTION HISTORY: has this field already been acted on this session?
    # Stamped onto the element dict (train: from demo order; inference: by agent).
    attempted  = 1.0 if elem.get("attempted") else 0.0
    # embed label + current value so a filled field is semantically distinct from
    # the same field empty (was embedding only the label → filled == empty).
    text_emb   = _embed_text(((elem.get("text") or "") + " " + _val).strip())
    # is_real=1.0 at index 0 ensures real elements are never zero-masked even if all
    # other features happen to be zero (e.g. top-left element with empty text, unknown type)
    return [1.0, x1 / W, y1 / H, x2 / W, y2 / H, conf, role, is_focused, ctrl_type,
            is_filled, attempted] + text_emb


def encode_state(state: dict, max_elements: int = 128) -> torch.Tensor:
    """State dict -> FloatTensor (max_elements, ELEM_FEATURES), zero-padded."""
    res        = state.get("screen_resolution", [DEFAULT_W, DEFAULT_H])
    W, H       = float(res[0]) or DEFAULT_W, float(res[1]) or DEFAULT_H
    focused_id = state.get("focused_element_id")
    rows = [_encode_element(e, W, H, focused_id)
            for e in state.get("elements", [])[:max_elements]]
    while len(rows) < max_elements:
        rows.append([0.0] * ELEM_FEATURES)
    return torch.tensor(rows, dtype=torch.float32)


def _decode_actions(
    mouse: dict, keyboard: dict, W: float, H: float
) -> Tuple[int, float, float, float, str]:
    """Return (action_type, cx_norm, cy_norm, aux_norm, hotkey_or_text).

    aux_norm meaning per action type:
      click/dclick/drag : key_count norm (unused, 0.0)
      keyboard          : key_count norm
      hotkey            : hotkey index norm (idx / NUM_HOTKEYS)
      scroll            : scroll_dy norm (clamped -1..1, positive=down)
    """
    mouse_actions   = mouse.get("actions", [])
    keyboard_groups = keyboard.get("actions", [])

    # ── collect typed text and hotkey ──────────────────────────────────────────
    parts, hotkey_name = [], None
    for g in keyboard_groups:
        hk = g.get("hotkey", "")
        if hk:
            hotkey_name = hk.lower()
        for s in g.get("strokes", []):
            pt = s.get("pasted_text", "")
            if pt:
                parts.append(pt)
            else:
                k = s.get("key", "")
                if len(k) == 1 and k.isprintable():
                    parts.append(k)
    typed_text = "".join(parts)
    key_count  = sum(len(g.get("strokes", [])) for g in keyboard_groups)

    # ── scroll ────────────────────────────────────────────────────────────────
    scrolls = [a for a in mouse_actions if a.get("type") == "scroll"]
    if scrolls:
        dy = float(scrolls[0].get("dy", 0))
        pos = scrolls[0].get("position", [0, 0])
        norm_dy = max(-1.0, min(1.0, dy / 10.0))
        return ACTION_SCROLL, float(pos[0]) / W, float(pos[1]) / H, norm_dy, ""

    # ── drag / double-click → CLICK ───────────────────────────────────────────
    # ACTION-SPACE COLLAPSE: the transformer only needs to decide NAVIGATE(click)
    # vs FILL(type). drag/double_click on this task are misclassified clicks or
    # text-highlight noise → fold into CLICK. Keeps the action-type head from
    # collapsing onto junk classes it can never predict reliably.
    drags = [a for a in mouse_actions if a.get("type") == "drag"]
    if drags:
        src = drags[0].get("position", [0, 0])
        return ACTION_CLICK, float(src[0]) / W, float(src[1]) / H, 0.0, ""

    dclicks = [a for a in mouse_actions if a.get("type") == "double_click"]
    if dclicks:
        pos = dclicks[0].get("position", [0, 0])
        return ACTION_CLICK, float(pos[0]) / W, float(pos[1]) / H, 0.0, ""

    # ── left click ────────────────────────────────────────────────────────────
    clicks = [a for a in mouse_actions if a.get("type") == "click"]
    if clicks:
        pos = clicks[0].get("position", [0, 0])
        return ACTION_CLICK, float(pos[0]) / W, float(pos[1]) / H, 0.0, ""

    # ── hotkey → KEYBOARD ─────────────────────────────────────────────────────
    # backspace/enter etc. recorded during value entry are part of TYPING (the
    # LLM produces the clean value at runtime, no corrections needed) → fold into
    # the KEYBOARD/type class instead of a separate hotkey class.
    if hotkey_name:
        return ACTION_KEYBOARD, 0.0, 0.0, min(max(key_count, 1) / 100.0, 1.0), typed_text

    # ── regular keyboard ──────────────────────────────────────────────────────
    if key_count > 0:
        return ACTION_KEYBOARD, 0.0, 0.0, min(key_count / 100.0, 1.0), typed_text

    return ACTION_NOOP, 0.0, 0.0, 0.0, ""


# UI element types that count as real click targets. Clicks that land only on
# decorative containers (pane/window/document) are labeled -1 and dropped from
# the pointer loss — diagnostics showed ~16.5% of raw click labels resolved to
# a single decorative pane, saturating training with uninformative signal.
_CLICK_TARGET_CTRL_TYPES = {
    # canonical (new-style) control type names
    "input", "button", "combobox", "checkbox", "listitem",
    "tabitem", "radiobutton", "hyperlink", "menuitem",
    "text",           # click-to-focus labels (often covers the field hit area)
    "datetime", "calendar",  # date/time pickers
    # legacy wx fallback names (ControlTypeName.lower())
    "editcontrol", "buttoncontrol", "comboboxcontrol", "checkboxcontrol",
    "listitemcontrol", "tabitemcontrol", "radiobuttoncontrol",
    "hyperlinkcontrol", "menuitemcontrol",
    "textcontrol", "datetimecontrol", "calendarcontrol",
}


def _find_click_elem_idx(mouse: dict, state: dict, max_elements: int) -> int:
    """
    For a click action, find which *interactive* UI element's bbox contains the
    click point. Returns the element index (0-based) in the same order/slice
    that ``encode_state`` sees, or -1 if no interactive element covers it.

    Only elements whose control type is in ``_CLICK_TARGET_CTRL_TYPES`` are
    considered. Decorative containers (panes, windows, documents, static text)
    are never returned, even if they're the only element whose bbox contains
    the click — returning -1 sends the sample to the ignore-index branch of
    the click pointer loss rather than teaching the model to click banners.

    Among qualifying elements, the one with the smallest bbox area wins, so
    nested child controls beat their parent.
    """
    clicks = [a for a in mouse.get("actions", []) if a.get("type") in ("click", "double_click")]
    if not clicks:
        return -1
    pos = clicks[0].get("position", [0, 0])
    px, py = float(pos[0]), float(pos[1])

    elements = state.get("elements", [])[:max_elements]
    best_idx  = -1
    best_area = float("inf")
    for idx, elem in enumerate(elements):
        ctype = (elem.get("type") or "").lower()
        if ctype not in _CLICK_TARGET_CTRL_TYPES:
            continue
        bbox = elem.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 <= px <= x2 and y1 <= py <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best_area = area
                best_idx  = idx
    return best_idx


def _find_source_elem_idx(
    typed_text: str,
    state: dict,
    max_elements: int,
    bg_manifest: Optional[Dict[str, Any]] = None,
) -> int:
    """
    For a keyboard action, find which background element's text contains
    what was typed.  Returns the element index (0-based) or -1 if not found.
    -1 tells the loss function to ignore this sample.

    bg_manifest: dict loaded from session_manifest.json["background"].
    Keys are "window_title|app". When an element's text was stripped to save
    space, the manifest provides the full text for substring matching so no
    training signal is lost.
    """
    if not typed_text:
        return -1
    typed_lower = typed_text.lower().strip()
    elements = state.get("elements", [])[:max_elements]
    for idx, elem in enumerate(elements):
        if elem.get("window_role") != "background":
            continue
        t = (elem.get("text") or "").lower().strip()
        v = (elem.get("value") or "").lower().strip()
        # If element text was externalised to session_manifest.json (empty in
        # step file), restore it from the manifest for substring matching.
        if bg_manifest and len(t) < 200:
            _key = (elem.get("window_title") or "") + "|" + (elem.get("app") or "")
            _bg  = bg_manifest.get(_key, {})
            t = (_bg.get("text") or t).lower().strip()
            v = (_bg.get("value") or v).lower().strip()
        if not t and not v:
            continue
        if typed_lower in t or typed_lower in v or t in typed_lower:
            return idx
    return -1


def _load_trace(fpath: Path) -> dict | None:
    try:
        with fpath.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[Dataset] Skipping {fpath.name}: {exc}")
        return None


class TrajectoryDataset(Dataset):
    """
    Sliding-window trajectory samples from consecutive trace JSON files.

    Each sample (window of H traces):
      states           : FloatTensor (H, max_elements, ELEM_FEATURES)
      past_act_types   : LongTensor  (H-1,)
      past_act_cont    : FloatTensor (H-1, 3)  [cx, cy, key_norm]
      target_type      : LongTensor  scalar
      target_click_idx : LongTensor  scalar   — element index to click, -1 = ignore
      target_key       : FloatTensor scalar
      target_src_idx   : LongTensor  scalar   — source element index, -1 = ignore
    """

    def __init__(
        self,
        data_dir: str | Path | list[str | Path],
        max_elements: int = 128,
        hist_len: int = 4,
        glob: str = "*.json",
        aug_drop_prob: float = 0.0,   # probability of zeroing a UI element row
        disambiguate_attempted: str = "none",
        rare_weight_basis: str = "type",
        section_pattern: Optional[str] = None,
        section_prefix: str = "section_",
        action_space: str = "legacy",  # "legacy" | "semantic" — see module docstring
    ):
        """
        action_space : "legacy" | "semantic" (default "legacy")
            Ported 2026-08-12 from origin/verb-loop-rewrite. "legacy" — the
            original _decode_actions()-based ACTION_* scheme, fully unchanged.
            "semantic" — labels come from action_labeler.translate_step()'s
            richer Verb vocabulary instead (FOCUS/SET_VALUE/SELECT_OPTION/
            TOGGLE/INVOKE/SCROLL_TO/HOTKEY/WAIT/VERIFY/DONE). Deliberately
            scoped: disambiguate_attempted/rare_weight_basis-by-field are NOT
            yet made semantic-aware — semantic mode gets a correct, simple
            baseline first; that reconciliation is real follow-up work once
            an isolated A/B shows this is worth pursuing further.
        disambiguate_attempted : "none" | "rank" | "section" (default "none")
            Controls how _attempt_key disambiguates repeated-label elements
            (Driver 1/2/3 "First Name" etc.) for the 'attempted' input feature.
            "none"    — old label-only behavior (the un-regressed baseline).
                        Same-labeled elements share one attempted-key, bug and
                        all (filling Driver 1's First Name silently marks
                        Driver 2's still-empty First Name as attempted too).
            "rank"    — disambiguate by this element's rank among same-labeled
                        elements in accessibility-tree list order. Introduced
                        in 7999efc7 alongside field-level rare-action weighting;
                        tested ALONE (weighting reverted to type-level) it still
                        regressed val_click_acc to 46.9% (from a 68.9% baseline)
                        — list order isn't stable enough across snapshots to be
                        a clean training signal (see DEVELOPERS.md).
            "section" — disambiguate by real panecontrol bbox geometry via
                        _detect_section (needs section_pattern set, or this
                        silently behaves like "none" for every element with no
                        section). A differentiated, more stable signal than
                        list order — found 2026-08-11, not yet validated by a
                        real A/B run; treat as an experiment, not an assumed win.
            The live agent (agent.py) keeps its own always-rank-disambiguated
            copy for real-time tracking — this flag only affects TRAINING data
            derivation.
        rare_weight_basis : "none" | "type" | "field" (default "type")
            What _wmap's rare-action loss up-weighting groups clicks by:
            "type"  — clicked element's control type (editcontrol, etc.) —
                      the original, pre-7999efc7 basis.
            "field" — clicked element's specific identity (_attempt_key) —
                      finer-grained; 7999efc7 made this the ONLY basis and
                      bundled it with disambiguate_attempted="rank", so it was
                      never tested in isolation. Uses disambiguate_attempted
                      to decide whether same-labeled repeated fields (Driver
                      1/2/3) are counted as one identity or split apart.
            "none"  — uniform weight 1.0, no reweighting.
        section_pattern / section_prefix : mirrors components/agent/scope.py's
            ScopeConfig fields of the same name (e.g. r"section_(driver|
            vehicle)_(\\d+)$" / "section_"). Only consulted when
            disambiguate_attempted="section". Not importing ScopeConfig
            directly keeps this module scope-agnostic — the caller (train.py's
            CLI) injects the real app's pattern, same as agent.py does.
        """
        self.max_elements  = max_elements
        self.hist_len      = hist_len
        self.aug_drop_prob = aug_drop_prob
        if action_space not in ("legacy", "semantic"):
            raise ValueError(f"action_space must be 'legacy'/'semantic', got {action_space!r}")
        self.action_space = action_space
        if rare_weight_basis not in ("none", "type", "field"):
            raise ValueError(f"rare_weight_basis must be 'none'/'type'/'field', got {rare_weight_basis!r}")
        if disambiguate_attempted not in ("none", "rank", "section"):
            raise ValueError(f"disambiguate_attempted must be 'none'/'rank'/'section', got {disambiguate_attempted!r}")
        self._disambiguate_attempted = disambiguate_attempted
        self._rare_weight_basis      = rare_weight_basis
        self._section_pattern        = section_pattern
        self._section_prefix         = section_prefix
        # Toggled by train() around the val_loader pass so validation is scored on
        # Toggled by train() around the val_loader pass so validation is scored on
        # clean, unaugmented samples — element dropout makes the click choice easier
        # (fewer confusable candidates on screen), which inflates val_click_acc if
        # left on. Train and val share this one Dataset instance via random_split,
        # so this is a global switch, not per-sample; safe because train()/val() run
        # sequentially per epoch (num_workers=0, no concurrent access).
        self._eval_mode    = False

        # (defined once, called at every _attempt_key call site below)
        self._key_for = self._make_key_fn()

        # Collect trace files as *groups* — each group is a contiguous recording
        # session whose traces are temporally adjacent.  The flat files directly
        # under a root form one group; each `session_*` subfolder forms its own
        # group.  Sliding windows are later built *within* each group so history
        # never crosses a session boundary (which would stitch together unrelated
        # demonstrations). data_dir may be a single directory or a list of
        # directories (e.g. combining the curated tasks/form_filling/traces set
        # with a freshly-recorded data/demos/<name> batch) — each is scanned the
        # same way and their groups are pooled together.
        # Resolved to absolute immediately: the on-disk cache below pickles
        # whatever Path objects glob() produces under these roots, and a
        # relative root makes every cached path implicitly cwd-relative.
        # A later run from a different working directory (or a background
        # process with a different cwd than the one that first built the
        # cache) then loads those paths, resolves them against the WRONG
        # cwd, and every file "not found" — _load_trace() catches that and
        # returns None, which silently becomes an empty {} state rather than
        # a crash. Found 2026-08-08: a from-components/ sanity check and a
        # from-repo-root background training run shared one cache (same
        # resolved hash) but disagreed on what the relative paths inside it
        # meant, corrupting ~100% of a training run into empty-state noise
        # with no error, no traceback — just silently wrong data.
        roots = [Path(d).resolve() for d in data_dir] if isinstance(data_dir, (list, tuple)) \
            else [Path(data_dir).resolve()]
        print(f"[Dataset] Scanning {', '.join(str(r) for r in roots)} for trace files...", flush=True)
        file_groups: List[List[Path]] = []
        for root in roots:
            flat_files = sorted(root.glob(glob))
            if flat_files:
                file_groups.append(flat_files)
            for session_dir in sorted(root.glob("session_*")):
                if session_dir.is_dir():
                    session_files = sorted(session_dir.glob(glob))
                    if session_files:
                        file_groups.append(session_files)
        if not file_groups:
            raise FileNotFoundError(f"No trace JSONs in {data_dir!r} (including session subfolders)")
        print(f"[Dataset] Found {sum(len(g) for g in file_groups)} files in {len(file_groups)} session(s).", flush=True)

        # ── Dataset init cache ──────────────────────────────────────────────────
        # Reading 10k+ JSONs at init takes 2-10 min. Cache stores filtered file
        # paths + action metadata; invalidated when any session is newer than cache.
        # Cache lives next to the first root, keyed by how many roots + their
        # resolved paths so a multi-dir combo never collides with a single-dir
        # cache (or a different combo) sitting in the same first directory.
        import pickle as _pickle
        import hashlib as _hashlib
        _roots_key  = "|".join(str(r.resolve()) for r in roots)
        _roots_hash = _hashlib.sha1(_roots_key.encode("utf-8")).hexdigest()[:10]
        _cache_name = ".dataset_cache.pkl" if len(roots) == 1 else f".dataset_cache_combined_{_roots_hash}.pkl"
        _cache_path = roots[0] / _cache_name
        # ELEM_FEATURES in the key → a change in the feature layout invalidates
        # any stale cache built with a different one.
        # Counted during the scan, carried through the cache, reported by
        # _warn_if_truncated() - a cache hit skips the scan entirely, and a
        # warning you only ever see once is a warning you will miss.
        self._truncated_files = 0
        self._max_seen        = 0

        _cache_key  = (max_elements, hist_len, aug_drop_prob, ELEM_FEATURES,
                       "v11_backspace_setvalue_fix", disambiguate_attempted, rare_weight_basis,
                       section_pattern, section_prefix, action_space)

        def _cache_valid() -> bool:
            if not _cache_path.exists():
                return False
            cache_mtime = _cache_path.stat().st_mtime
            # Check session dirs (not individual files) — avoids statting 10k files
            checked = set()
            for group in file_groups:
                d = group[0].parent
                if d not in checked:
                    checked.add(d)
                    if d.stat().st_mtime > cache_mtime:
                        return False
                mp = d / "session_manifest.json"
                if mp.exists() and mp.stat().st_mtime > cache_mtime:
                    return False
            return True

        if _cache_valid():
            try:
                with _cache_path.open("rb") as _cf:
                    _cached = _pickle.load(_cf)
                if _cached.get("key") == _cache_key:
                    self._grouped_files     = _cached["grouped_files"]
                    self._grouped_manifests = _cached["grouped_manifests"]
                    self._samples           = _cached["samples"]
                    self._sample_weights    = _cached.get("sample_weights")
                    self._attempted_by_file = _cached.get("attempted_by_file", {})
                    self._truncated_files   = _cached.get("truncated_files", 0)
                    self._max_seen          = _cached.get("max_seen", 0)
                    self._warn_if_truncated(max_elements)
                    print(f"[Dataset] Loaded from cache: {len(self._samples)} samples "
                          f"from {len(self._grouped_files)} session(s).")
                    self._prime_all_embeddings()
                    self._preload_tensors()
                    return
            except Exception as _ce:
                print(f"[Dataset] Cache load failed ({_ce}), rebuilding.")

        # Load raw traces per group — skip traces with no active-window interactive
        # controls (e.g. old Tkinter sessions where UIA saw 0 form elements).
        # Type names must match _CTRL_TYPE_MAP in ui_observer.py:
        #   "Edit" -> "input", "Button" -> "button", "ComboBox" -> "combobox", etc.
        # wxPython controls not in _CTRL_TYPE_MAP fall through as lowercased
        # ControlTypeName, e.g. "TabItemControl" -> "tabitemcontrol".
        _INTERACTIVE = {
            "input",          # Edit / text field  (was "editcontrol" — wrong)
            "combobox",       # ComboBox           (was "comboboxcontrol" — wrong)
            "checkbox",       # CheckBox           (was "checkboxcontrol" — wrong)
            "button",         # Button             (was "buttoncontrol" — wrong)
            "listitem",       # ListItem           (was "listitemcontrol" — wrong)
            "tabitem",        # tab strip items (user clicks tabs to navigate)
            "radiobutton",    # radio buttons
            # wx fallback names (ControlTypeName.lower() when not in map)
            "editcontrol", "comboboxcontrol", "checkboxcontrol",
            "buttoncontrol", "listitemcontrol", "tabitemcontrol",
            "radiobuttoncontrol",
        }

        # Load session_manifest.json for each group (background text externalised
        # from step files to save disk space; restored here for _find_source_elem_idx
        # so keyboard training labels remain intact).
        def _load_manifest(group_files: List[Path]) -> Optional[Dict[str, Any]]:
            if not group_files:
                return None
            mp = group_files[0].parent / "session_manifest.json"
            if not mp.exists():
                return None
            try:
                with mp.open(encoding="utf-8") as _mf:
                    return json.load(_mf).get("background", {})
            except Exception:
                return None

        # ── Phase 1: filter traces, extract lightweight metadata only (no tensors) ──
        # States are NOT encoded here — done lazily in __getitem__ to avoid OOM
        # on large datasets. Only action metadata (ints/floats) stored at init.
        self._grouped_files:    List[List[Path]]                    = []
        self._grouped_manifests: List[Optional[Dict[str, Any]]]     = []
        # fpath -> frozenset of attempt-keys acted on EARLIER in the same session.
        # Stamped onto each state's elements before encoding (the 'attempted' feat).
        self._attempted_by_file: Dict[str, frozenset]               = {}
        skipped = 0
        grouped_action_meta: List[List[Tuple[int, float, float, float, str]]] = []
        grouped_src_idx:     List[List[int]] = []
        grouped_click_idx:   List[List[int]] = []
        grouped_click_type:  List[List[str]] = []   # clicked element's type (rare-action weighting)
        grouped_click_key:   List[List[str]] = []   # clicked element's identity (rare-FIELD weighting)

        import gc as _gc
        total_files = sum(len(g) for g in file_groups)
        files_done  = 0

        for group_files in file_groups:
            valid_files:  List[Path]  = []
            g_actions:    List[Tuple] = []
            g_src_idx:    List[int]   = []
            g_click_idx:  List[int]   = []
            g_click_type: List[str]   = []
            g_click_key:  List[str]   = []
            g_acted: set              = set()   # attempt-keys acted on so far (this session)
            manifest = _load_manifest(group_files)

            for fpath in group_files:
                t = _load_trace(fpath)
                files_done += 1
                if files_done % 500 == 0:
                    print(f"[Dataset] Scanning... {files_done}/{total_files}", flush=True)
                if t is None:
                    continue
                state = t.get("state", {})
                elems = state.get("elements", [])
                # Everything past max_elements is invisible to training: both
                # encode_state and _find_click_elem_idx slice to it, so a click
                # on a later element resolves to -1 and its pointer loss is
                # ignored. The step still counts as a click for the action-type
                # head, so nothing looks wrong - the model simply never learns
                # where those clicks went. A 50-row grid blows past the default
                # of 128 routinely (the scope #2 portal is 303 elements, and 29
                # of its 50 grade cells sit beyond 128), which is exactly the
                # failure WebObserver's own element cap had one layer up.
                if len(elems) > max_elements:
                    self._truncated_files += 1
                    self._max_seen = max(self._max_seen, len(elems))
                active_interactive = sum(
                    1 for e in elems
                    if e.get("window_role") == "active"
                    and (e.get("type") or "").lower() in _INTERACTIVE
                )
                if active_interactive == 0:
                    skipped += 1
                    t = None  # explicit release before next iteration
                    continue
                res  = state.get("screen_resolution", [DEFAULT_W, DEFAULT_H])
                W    = float(res[0]) or DEFAULT_W
                H_px = float(res[1]) or DEFAULT_H
                if self.action_space == "semantic":
                    sem = _translate_step(t)
                    _pos = sem.position or (0.0, 0.0)
                    if sem.verb == _Verb.SET_VALUE:
                        _aux = min(len(sem.value) / 100.0, 1.0)
                    elif sem.verb == _Verb.SCROLL_TO:
                        _sign = 1.0 if sem.direction == "down" else -1.0
                        _aux  = max(-1.0, min(1.0, (sem.clicks / 10.0) * _sign))
                    else:
                        _aux = 0.0
                    action = (VERB_TO_ID[sem.verb], float(_pos[0]) / W, float(_pos[1]) / H_px,
                              _aux, sem.value)
                    _tgt = sem.target_idx if sem.target_idx is not None else -1
                    # Two separate pointer jobs, same split as the legacy scheme
                    # (click_elem vs source_elem) — just re-purposed by VERB instead
                    # of by action-type (ported from origin/verb-loop-rewrite's own
                    # documented finding: mixing both jobs onto one head measurably
                    # hurt click_acc). FOCUS/SET_VALUE ("where to type") reuse the
                    # legacy source_elem head as a typing-target pointer; click_idx
                    # stays scoped to click-like verbs only.
                    if sem.verb in (_Verb.FOCUS, _Verb.SET_VALUE):
                        click_idx, src_idx = -1, _tgt
                    else:
                        click_idx, src_idx = _tgt, -1
                else:
                    action = _decode_actions(t.get("mouse", {}), t.get("keyboard", {}), W, H_px)
                    src_idx = (
                        _find_source_elem_idx(action[4], state, max_elements, manifest)
                        if action[0] == ACTION_KEYBOARD else -1
                    )
                    click_idx = (
                        _find_click_elem_idx(t.get("mouse", {}), state, max_elements)
                        if action[0] == ACTION_CLICK else -1
                    )
                # Remember the clicked element's TYPE and specific IDENTITY → used
                # for rare-action loss weighting (general inverse-frequency; rare
                # target types like tabs AND rare individual fields within a common
                # type both get up-weighted, no hardcoded class/field name).
                click_type = ((elems[click_idx].get("type") or "").lower()
                              if 0 <= click_idx < len(elems) else "")
                click_key  = (self._key_for(elems[click_idx], elems)
                              if 0 <= click_idx < len(elems) else "")
                # 'attempted' derivation: this step SEES every field acted on in
                # PRIOR steps (snapshot before adding the current target). Then
                # record this step's target so later steps see it as attempted.
                self._attempted_by_file[str(fpath)] = frozenset(g_acted)
                # click_idx/src_idx are mutually exclusive by construction in BOTH
                # modes (exactly one is ever >=0 for a real target) -- checking them
                # directly instead of gating on action[0]'s legacy ACTION_CLICK/
                # ACTION_KEYBOARD constants keeps this correct for semantic mode too,
                # whose action[0] is a VERB_TO_ID value, not a legacy action id.
                _tgt_elem = None
                if 0 <= click_idx < len(elems):
                    _tgt_elem = elems[click_idx]
                elif 0 <= src_idx < len(elems):
                    _tgt_elem = elems[src_idx]
                if _tgt_elem is not None:
                    g_acted.add(self._key_for(_tgt_elem, elems))
                valid_files.append(fpath)
                g_actions.append(action)
                g_src_idx.append(src_idx)
                g_click_idx.append(click_idx)
                g_click_type.append(click_type)
                g_click_key.append(click_key)
                t = None  # release JSON dict immediately — prevents memory accumulation

            _gc.collect()  # force reclaim after each session group

            if valid_files:
                self._grouped_files.append(valid_files)
                self._grouped_manifests.append(manifest)
                grouped_action_meta.append(g_actions)
                grouped_src_idx.append(g_src_idx)
                grouped_click_idx.append(g_click_idx)
                grouped_click_type.append(g_click_type)
                grouped_click_key.append(g_click_key)

        if skipped:
            print(f"[Dataset] Skipped {skipped} traces with no active form controls.")
        self._warn_if_truncated(max_elements)
        if not self._grouped_files:
            raise ValueError("No usable traces after filtering.")

        # ── Phase 2: build sample index (lightweight — just ints, no tensors) ──
        # Each sample: (group_idx, window_start, past_types, past_cont,
        #               tgt_type, tgt_click_idx, tgt_key, src_idx)
        self._samples = []
        total_traces = 0
        dropped_short_groups = 0
        _noop_id = VERB_TO_ID[_Verb.WAIT] if self.action_space == "semantic" else ACTION_NOOP
        for gi, g_actions in enumerate(grouped_action_meta):
            N = len(g_actions)
            total_traces += N
            if N < hist_len:
                dropped_short_groups += 1
                continue
            for i in range(N - hist_len + 1):
                ctx       = g_actions[i : i + hist_len - 1]
                tgt       = g_actions[i + hist_len - 1]
                src_idx    = grouped_src_idx[gi][i + hist_len - 1]
                click_idx  = grouped_click_idx[gi][i + hist_len - 1]
                click_type = grouped_click_type[gi][i + hist_len - 1]
                click_key  = grouped_click_key[gi][i + hist_len - 1]
                if tgt[0] == _noop_id:
                    continue
                self._samples.append((
                    gi,                                # group index
                    i,                                 # window start
                    [a[0] for a in ctx],               # past action types
                    [[a[1], a[2], a[3]] for a in ctx], # past cont (cx, cy, key_norm)
                    tgt[0], click_idx, tgt[3], src_idx,
                    click_key,                          # 9th (temp): clicked target's specific identity
                    click_type,                         # 10th (temp): clicked target's control type
                ))

        # ── Rare-action loss weighting (general, inverse-frequency) ──────────────
        # Up-weight samples whose CLICK target is a RARE *specific field* (not just
        # a rare control TYPE), so the model learns them without distorting the data
        # (no recording tricks, no duplication). weight ∝ 1/freq(field), normalized
        # so mean click-weight ≈ 1, capped to avoid blow-ups. Non-click samples → 1.0.
        # Auto-detects whatever is rare — NO hardcoded class/field name.
        #
        # Field-identity weighting subsumes type-level weighting (a rare control
        # type's instances are also individually rare) and additionally fixes the
        # common-neighbor-wins-when-uncertain failure mode: two fields of the SAME
        # type (e.g. "First Name" appears on nearly every record, "Underwriter"
        # rarely) used to get equal weight under type-only weighting, so the model
        # had no loss-level incentive to learn the rarer one instead of defaulting
        # to its more common same-type/same-tab neighbor — exactly the confusion
        # pattern seen across every per-tab accuracy breakdown this session ("DL
        # Expiration" -> "DL Issuing State", "Comprehensive Deductible" -> "Collision
        # Deductible"). (DEVELOPERS.md -> Concepts: rare-action handling.)
        from collections import Counter as _Counter
        _RARE_W_CAP = 8.0
        _basis_idx = {"field": 8, "type": 9}.get(self._rare_weight_basis)
        if _basis_idx is None:
            self._sample_weights: List[float] = [1.0] * len(self._samples)
            print("[Dataset] rare-action weighting disabled (rare_weight_basis='none').")
        else:
            _tc = _Counter(s[_basis_idx] for s in self._samples if s[_basis_idx])
            _n_click, _n_types = sum(_tc.values()), max(len(_tc), 1)
            _wmap = {ty: min(_n_click / (_n_types * c), _RARE_W_CAP) for ty, c in _tc.items()}
            self._sample_weights = [_wmap.get(s[_basis_idx], 1.0) for s in self._samples]
            if _wmap:
                _top = sorted(_wmap.items(), key=lambda kv: -kv[1])[:4]
                print(f"[Dataset] rare-action weights (basis={self._rare_weight_basis!r}, top): "
                      f"{[(t, round(w, 1)) for t, w in _top]}")
        self._samples = [s[:8] for s in self._samples]   # strip the temp fields

        if total_traces < hist_len:
            raise ValueError(
                f"Need >= {hist_len} traces (hist_len={hist_len}), found {total_traces}."
            )
        if dropped_short_groups:
            print(f"[Dataset] Skipped {dropped_short_groups} session(s) shorter than hist_len={hist_len}.")
        n_groups = len(self._grouped_files)
        print(f"[Dataset] Built {len(self._samples)} samples from {n_groups} session(s), {total_traces} traces.")

        # Save cache for next run
        try:
            with _cache_path.open("wb") as _cf:
                _pickle.dump({
                    "key":               _cache_key,
                    "grouped_files":     self._grouped_files,
                    "grouped_manifests": self._grouped_manifests,
                    "samples":           self._samples,
                    "sample_weights":    getattr(self, "_sample_weights", None),
                    "attempted_by_file": self._attempted_by_file,
                    "truncated_files":   self._truncated_files,
                    "max_seen":          self._max_seen,
                }, _cf, protocol=4)
            print(f"[Dataset] Cache saved -> {_cache_path}")
        except Exception as _se:
            print(f"[Dataset] Cache save failed ({_se}), continuing without cache.")

        self._prime_all_embeddings()
        self._preload_tensors()

    def _warn_if_truncated(self, max_elements: int) -> None:
        """Say when states are bigger than the model is allowed to see.

        encode_state and _find_click_elem_idx both slice to max_elements, so a
        click on a later element resolves to -1 and its pointer loss is ignored.
        The step still counts as a click for the action-type head, so nothing
        looks wrong - the model simply never learns where those clicks went.
        A 50-row grid passes the default of 128 routinely: the scope #2 grade
        portal is 303 elements, with 29 of its 50 grade cells beyond the cap.
        """
        if not self._truncated_files:
            return
        print(f"[Dataset] WARNING: {self._truncated_files} trace(s) hold more than "
              f"max_elements={max_elements} elements (largest seen: {self._max_seen}). "
              f"Everything past the cap is invisible to training - a click landing "
              f"there teaches nothing. Raise it with --max_elements.", flush=True)

    def _prime_all_embeddings(self) -> None:
        """
        Batch-embed every unique element label in the dataset via the real
        sentence-transformer BEFORE tensor encoding starts.

        _embed_text() silently falls back to a content hash (no semantic
        structure at all) for any text not already in _embed_cache — and
        nothing called _prime_embed_cache() anywhere in the pipeline, so
        every training run to date learned click-targeting from 384 of its
        395 input features being meaningless per-label noise. Found
        2026-08-07 while investigating why val_click_acc plateaued at
        ~29-31% across three differently-sized datasets in one session —
        dataset size was never going to fix a representation problem.
        """
        labels: set = set()
        unique_files: list = []
        for group in self._grouped_files:
            unique_files.extend(group)
        unique_files = list(dict.fromkeys(unique_files))
        for fpath in unique_files:
            t = _load_trace(fpath)
            if not t:
                continue
            for e in t.get("state", {}).get("elements", []):
                labels.add(_elem_label(e))
        _prime_embed_cache(list(labels))

    def _preload_tensors(self) -> None:
        """
        For small datasets (<=5000 files): load all tensors into RAM upfront.
        For large datasets: use an LRU cache capped at 5000 entries — avoids
        OOM when augmented traces push total files into the tens of thousands.
        """
        unique_files: list = []
        for group in self._grouped_files:
            unique_files.extend(group)
        unique_files = list(dict.fromkeys(unique_files))  # deduplicate, preserve order

        self._zero_tensor = torch.zeros(self.max_elements, ELEM_FEATURES)
        _PRELOAD_LIMIT = 50_000

        if len(unique_files) <= _PRELOAD_LIMIT:
            self._tensor_cache: dict = {}
            self._lru_cache = None
            total = len(unique_files)
            print(f"[Dataset] Preloading {total} trace tensors into RAM...", flush=True)
            for i, fpath in enumerate(unique_files):
                t = _load_trace(fpath)
                state = t.get("state", {}) if t else {}
                self._stamp_attempted(fpath, state)
                self._tensor_cache[fpath] = encode_state(state, self.max_elements)
                if (i + 1) % 1000 == 0 or (i + 1) == total:
                    pct = (i + 1) / total * 100
                    bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
                    print(f"\r  [{bar}] {i+1}/{total} ({pct:.0f}%)", end="", flush=True)
            print(f"\n[Dataset] Preload done.", flush=True)
        else:
            # LRU cache: keeps last N tensors in RAM, evicts oldest on overflow.
            # Trades RAM for disk I/O — each cache miss re-reads one JSON file.
            from collections import OrderedDict
            self._tensor_cache = None
            self._lru_cache: "OrderedDict" = OrderedDict()
            self._lru_maxsize = _PRELOAD_LIMIT
            print(f"[Dataset] Large dataset ({len(unique_files)} files) — "
                  f"using LRU tensor cache (max={_PRELOAD_LIMIT}).", flush=True)

    def _make_key_fn(self):
        """Build the (elem, elements) -> attempt-key function matching this
        dataset's disambiguate_attempted mode. One closure per Dataset
        instance instead of branching on a string at every call site."""
        mode = self._disambiguate_attempted
        if mode == "rank":
            return lambda elem, elements: _attempt_key(elem, elements=elements)
        if mode == "section":
            pat, pfx = self._section_pattern, self._section_prefix
            return lambda elem, elements: _attempt_key(
                elem, section=_detect_section(elem, elements, pat, pfx))
        return lambda elem, elements: _attempt_key(elem)

    def _stamp_attempted(self, fpath, state: dict) -> None:
        """Mark elements that were acted on EARLIER in this session (the
        'attempted' feature) by mutating the state dict before encoding."""
        keys = self._attempted_by_file.get(str(fpath))
        if not keys:
            return
        elements = state.get("elements", [])
        for e in elements:
            if self._key_for(e, elements) in keys:
                e["attempted"] = 1.0

    def _get_tensor(self, fpath: "Path") -> torch.Tensor:
        """Fetch encoded state tensor, using preload dict or LRU cache."""
        if self._tensor_cache is not None:
            return self._tensor_cache.get(fpath, self._zero_tensor)
        # LRU path
        if fpath in self._lru_cache:
            self._lru_cache.move_to_end(fpath)
            return self._lru_cache[fpath]
        t = _load_trace(fpath)
        state = t.get("state", {}) if t else {}
        self._stamp_attempted(fpath, state)
        tensor = encode_state(state, self.max_elements)
        self._lru_cache[fpath] = tensor
        if len(self._lru_cache) > self._lru_maxsize:
            self._lru_cache.popitem(last=False)
        return tensor

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        gi, win_start, p_types, p_cont, tgt_type, tgt_click_idx, tgt_key, src_idx = self._samples[idx]

        files = self._grouped_files[gi][win_start : win_start + self.hist_len]
        state_tensors = [self._get_tensor(fpath) for fpath in files]
        while len(state_tensors) < self.hist_len:
            state_tensors.insert(0, self._zero_tensor)
        states = torch.stack(state_tensors)  # (H, T, F)

        # Augmentation block — element dropout + order shuffle
        if self.aug_drop_prob > 0.0 and not self._eval_mode:
            states = states.clone()

            # Element dropout: randomly zero out UI rows across all history steps.
            # Protect the click/source label elements in the current (last) state —
            # if the target gets zeroed its is_real flag goes to 0, the forward pass
            # masks its logit to -1e9, and CE loss becomes ~1e9, exploding training.
            mask = torch.rand(states.shape[0], states.shape[1]) < self.aug_drop_prob
            if tgt_click_idx >= 0:
                mask[-1, tgt_click_idx] = False
            if src_idx >= 0:
                mask[-1, src_idx] = False
            states[mask] = 0.0

            # Element order shuffle on the current (last) state only.
            # Historical states are mean-pooled in the forward pass so their
            # order doesn't affect the pointer heads — shuffling them is a free
            # diversity bonus but doesn't change pointer label semantics.
            # The current state order DOES matter: pointer heads select by index,
            # so we remap tgt_click_idx / src_idx through the permutation.
            T = states.shape[1]
            perm = torch.randperm(T)
            states[-1] = states[-1][perm]
            # inv_perm[old_idx] = new_idx (where the element moved to after shuffle)
            inv_perm = torch.empty(T, dtype=torch.long)
            inv_perm[perm] = torch.arange(T)
            if tgt_click_idx >= 0:
                tgt_click_idx = int(inv_perm[tgt_click_idx].item())
            if src_idx >= 0:
                src_idx = int(inv_perm[src_idx].item())

        H = self.hist_len
        if H > 1:
            past_types = torch.tensor(p_types, dtype=torch.long)
            past_cont  = torch.tensor(p_cont,  dtype=torch.float32)
        else:
            past_types = torch.zeros(0, dtype=torch.long)
            past_cont  = torch.zeros(0, 3,     dtype=torch.float32)

        _w = getattr(self, "_sample_weights", None)
        sample_weight = float(_w[idx]) if _w is not None else 1.0
        return (
            states,
            past_types,
            past_cont,
            torch.tensor(tgt_type,      dtype=torch.long),
            torch.tensor(tgt_click_idx, dtype=torch.long),
            torch.tensor(tgt_key,       dtype=torch.float32),
            torch.tensor(src_idx,       dtype=torch.long),
            torch.tensor(sample_weight, dtype=torch.float32),
        )

    def class_counts(self) -> dict:
        from collections import Counter
        if self.action_space == "semantic":
            # Verb IDs numerically collide with the legacy ACTION_* scheme
            # (e.g. FOCUS=0 == ACTION_NOOP=0) -- a shared names dict would
            # silently mislabel this purely-diagnostic printout.
            names = {i: v.value for i, v in ID_TO_VERB.items()}
        else:
            names = {ACTION_NOOP: "no_op", ACTION_CLICK: "click", ACTION_KEYBOARD: "keyboard",
                     ACTION_HOTKEY: "hotkey", ACTION_SCROLL: "scroll",
                     ACTION_DCLICK: "double_click", ACTION_DRAG: "drag"}
        return {names.get(k, f"action_{k}"): v
                for k, v in Counter(s[4] for s in self._samples).items()}

    def __repr__(self) -> str:
        return (
            f"TrajectoryDataset(samples={len(self)}, hist_len={self.hist_len}, "
            f"max_elements={self.max_elements}, class_counts={self.class_counts()})"
        )


# ═══════════════════════════════════════════════════════════
#  SECTION 2 — MODEL
# ═══════════════════════════════════════════════════════════

class PolicyOutput(NamedTuple):
    type_logits:  torch.Tensor   # (B, num_actions)
    click_elem:   torch.Tensor   # (B, max_elements) — which element to click (raw logits)
    key_count:    torch.Tensor   # (B, 1)
    source_elem:  torch.Tensor   # (B, max_elements) — which bg element to copy text from


class StateEncoder(nn.Module):
    """
    Project each UI element to d_model *independently*, returning per-element
    embeddings and a boolean padding mask. The caller decides whether to mean-
    pool the elements (for sequence tokenization) or keep them per-element
    (for pointer heads that must identify a specific element).

    2026-08-07: tried splitting this into separate structured-feature/embedding
    projections (hypothesis: the 384-dim text embedding was drowning out the
    11-dim positional signal by sheer dimensionality). Tested in isolation
    against the exact dataset the 68.9%-click_acc baseline used, same other
    hyperparameters: regressed to 45.0%, never approached the baseline through
    50 epochs. Reverted. Root cause of the "right neighborhood, wrong field"
    confusion pattern turned out to be elsewhere — see learning_fixation_solved
    / _attempt_key's repeated-label disambiguation and the rare-field loss
    weighting fix, both landed the same day.

    Returns:
        elem_emb : FloatTensor (B, T, d_model) — per-element embeddings
        mask     : BoolTensor  (B, T)          — True for real elements, False for padding
    """
    def __init__(self, elem_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(elem_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = state[..., _REAL_ELEM_FLAG_IDX] > 0.5        # (B, T) bool — index 0 is is_real flag
        elem_emb = self.norm(self.proj(state))               # (B, T, d_model)
        # Zero out padding rows so the pooled sequence token (computed upstream)
        # doesn't mix padding noise into the mean.
        elem_emb = elem_emb * mask.unsqueeze(-1).float()
        return elem_emb, mask


class ActionEncoder(nn.Module):
    """Encode (type_id, cx, cy, key_norm) -> d_model."""
    def __init__(self, num_actions: int, d_model: int):
        super().__init__()
        half = d_model // 2
        self.type_emb  = nn.Embedding(num_actions + 1, half, padding_idx=num_actions)
        self.cont_proj = nn.Linear(3, d_model - half)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, types: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        return self.norm(self.out_proj(
            torch.cat([self.type_emb(types), self.cont_proj(cont)], dim=-1)
        ))


class TransformerAgentNetwork(nn.Module):
    """
    Causal Transformer Agent.

    Interleaves state and action tokens [s1,a1,s2,a2,...,sH] and uses
    causal self-attention to predict the next action from the last state.
    """

    def __init__(
        self,
        elem_features:   int   = ELEM_FEATURES,
        max_elements:    int   = 128,
        d_model:         int   = 128,
        nhead:           int   = 4,
        num_layers:      int   = 4,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
        num_actions:     int   = NUM_ACTIONS,
        hist_len:        int   = 4,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_elements = max_elements
        self.hist_len    = hist_len
        self.num_actions = num_actions
        # Past-action dropout. For CLONING A FIXED ORDER, history ("after field A,
        # click field B") is the primary signal — dropout suppresses exactly what
        # we want. 0.8 killed it, 0.3 still muffled it. 0.1 keeps the sequence
        # signal strong (slight dropout to avoid pure copy-the-last-token).
        self.action_dropout = 0.1

        self.state_enc  = StateEncoder(elem_features, d_model)
        self.action_enc = ActionEncoder(num_actions, d_model)
        self.pos_enc    = nn.Embedding(2 * hist_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                               enable_nested_tensor=False)
        self.out_norm       = nn.LayerNorm(d_model)
        # Cross-attention decoder: the last transformer token (derived from
        # pooled state summaries) queries the per-element embeddings of the
        # CURRENT state, so all downstream heads can read specific elements
        # rather than reasoning from the pooled summary alone. This addresses
        # the observed ceiling where per-element pointer heads could not form
        # discriminative queries from a pooled sequence token.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.type_head   = nn.Linear(d_model, num_actions)
        self.key_head    = nn.Linear(d_model, 1)
        self.hotkey_head = nn.Linear(d_model, NUM_HOTKEYS)   # which hotkey
        self.scroll_head = nn.Linear(d_model, 1)              # scroll amount/direction
        # Pointer heads — shared for click, double_click, drag_src
        self.click_q      = nn.Linear(d_model, d_model)
        self.click_k      = nn.Linear(d_model, d_model)
        self.click_q_norm = nn.LayerNorm(d_model)
        self.click_k_norm = nn.LayerNorm(d_model)
        # Drag destination pointer (separate from src)
        self.drag_dst_q      = nn.Linear(d_model, d_model)
        self.drag_dst_k      = nn.Linear(d_model, d_model)
        self.drag_dst_q_norm = nn.LayerNorm(d_model)
        self.drag_dst_k_norm = nn.LayerNorm(d_model)
        # Source element pointer (keyboard text lookup)
        self.src_q        = nn.Linear(d_model, d_model)
        self.src_k        = nn.Linear(d_model, d_model)
        self.src_q_norm   = nn.LayerNorm(d_model)
        self.src_k_norm   = nn.LayerNorm(d_model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        states:     torch.Tensor,   # (B, H, T, F)
        past_types: torch.Tensor,   # (B, H-1)
        past_cont:  torch.Tensor,   # (B, H-1, 3)
    ) -> PolicyOutput:
        B, H, T, _ = states.shape

        # Per-element encoding: (B*H, T, d), mask (B*H, T)
        flat_states = states.view(B * H, T, -1)
        elem_emb, elem_mask = self.state_enc(flat_states)
        elem_emb  = elem_emb.view(B, H, T, -1)               # (B, H, T, d)
        elem_mask = elem_mask.view(B, H, T)                  # (B, H, T)

        # Pool each state into a single sequence token via masked mean.
        # Pointer heads will use the *unpooled* per-element embeddings below.
        denom = elem_mask.sum(-1, keepdim=True).clamp(min=1).float()  # (B, H, 1)
        s = elem_emb.sum(-2) / denom                                  # (B, H, d)

        # Build sequence [s1, a1, s2, a2, ..., sH]
        seq_len = 2 * H - 1
        tokens  = torch.zeros(B, seq_len, self.d_model, device=states.device)
        tokens[:, 0::2] = s
        if H > 1:
            act_tokens = self.action_enc(past_types, past_cont)
            if self.training and self.action_dropout > 0:
                # zero whole action tokens at random → model must use state
                keep = (torch.rand(B, H - 1, 1, device=states.device) >= self.action_dropout).float()
                act_tokens = act_tokens * keep
            tokens[:, 1::2] = act_tokens

        # Positional encoding
        pos    = torch.arange(seq_len, device=states.device).unsqueeze(0)
        tokens = tokens + self.pos_enc(pos)

        # Causal attention
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=states.device)
        out  = self.encoder(tokens, mask=attn_mask, is_causal=True)
        last = self.out_norm(out[:, -1])                              # (B, d)

        cur_elem_emb  = elem_emb[:, -1]                               # (B, T, d)
        cur_elem_mask = elem_mask[:, -1]                              # (B, T)

        # Cross-attention decoder: let `last` read specific elements of the
        # current state. key_padding_mask expects True = IGNORE, so invert our
        # mask (where True = real element).
        q_in   = last.unsqueeze(1)                                    # (B, 1, d)
        ctx, _ = self.cross_attn(
            query=q_in, key=cur_elem_emb, value=cur_elem_emb,
            key_padding_mask=~cur_elem_mask,
            need_weights=False,
        )
        # Residual + norm: preserve the transformer's temporal reasoning while
        # adding the element-specific information it retrieved.
        ctx = self.cross_norm(last + ctx.squeeze(1))                  # (B, d)

        # Pointer heads: dot-product between query (from cross-attended ctx)
        # and per-element keys (from the CURRENT state's per-element embeddings).
        scale = self.d_model ** 0.5
        # LayerNorm bounds the dot product to O(sqrt(d_model)) — prevents the
        # bilinear product of two learned projections from diverging during training.
        click_q = self.click_q_norm(self.click_q(ctx))                # (B, d)
        click_k = self.click_k_norm(self.click_k(cur_elem_emb))       # (B, T, d)
        click_logits = torch.einsum('bd,btd->bt', click_q, click_k) / scale
        click_logits = click_logits.masked_fill(~cur_elem_mask, -1e9)

        src_q = self.src_q_norm(self.src_q(ctx))                      # (B, d)
        src_k = self.src_k_norm(self.src_k(cur_elem_emb))             # (B, T, d)
        src_logits = torch.einsum('bd,btd->bt', src_q, src_k) / scale
        src_logits = src_logits.masked_fill(~cur_elem_mask, -1e9)

        return PolicyOutput(
            self.type_head(ctx),
            click_logits,                       # (B, T=max_elements)
            torch.sigmoid(self.key_head(ctx)),
            src_logits,                         # (B, T=max_elements)
        )

    def make_empty_history(self, B: int, device: torch.device):
        H = self.hist_len
        return (
            torch.full((B, H - 1), self.num_actions, dtype=torch.long,  device=device),
            torch.zeros(B, H - 1, 3,                  dtype=torch.float32, device=device),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TransformerAgentNetwork(d_model={self.d_model}, "
            f"hist_len={self.hist_len}, params={self.count_parameters():,})"
        )


# Backward compat alias
TransformerPolicyNetwork = TransformerAgentNetwork


# ═══════════════════════════════════════════════════════════
#  SECTION 3 — TRAINING
# ═══════════════════════════════════════════════════════════

def _masked_mse(pred, target, mask) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.sum() * 0.0  # zero loss that stays connected to the graph
    return nn.functional.mse_loss(pred[mask], target[mask])


def _run_epoch(model, loader, optimizer, device, lambda_click, lambda_key, label_smoothing,
                class_weights=None, key_verb_id=ACTION_KEYBOARD, lambda_src=0.5):
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    totals = dict(loss=0.0, type=0.0, click=0.0, key=0.0,
                  correct=0, click_correct=0, click_total=0,
                  src_correct=0, src_total=0,
                  samples=0, batches=0)

    ce_click = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")  # per-sample → rare-action weighting
    ce_src   = nn.CrossEntropyLoss(ignore_index=-1)  # -1 = no source label for this step
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for states, p_types, p_cont, tgt_types, tgt_click_idx, tgt_keys, tgt_src, sample_w in loader:
            states, p_types, p_cont = states.to(device), p_types.to(device), p_cont.to(device)
            tgt_types, tgt_click_idx, tgt_keys, tgt_src = (
                tgt_types.to(device), tgt_click_idx.to(device),
                tgt_keys.to(device),  tgt_src.to(device)
            )
            sample_w = sample_w.to(device)
            out    = model(states, p_types, p_cont)
            l_type = ce(out.type_logits, tgt_types)

            # Click pointer loss — ignore non-click steps (tgt=-1) and unmatched clicks.
            # Per-sample CE × rare-action weight (up-weights rare click targets like
            # tabs), then mean over the valid clicks. Common targets weight ~1.
            valid_click = (tgt_click_idx != -1)
            if valid_click.any():
                _per_click = ce_click(out.click_elem, tgt_click_idx)   # 0 where ignored
                l_click = (_per_click * sample_w).sum() / valid_click.sum().clamp(min=1)
            else:
                l_click = out.click_elem.sum() * 0.0

            l_key = _masked_mse(out.key_count.squeeze(-1), tgt_keys, tgt_types == key_verb_id)

            # Source-element pointer loss — only keyboard steps with a resolved source.
            # In semantic mode this same head is repurposed as the FOCUS/SET_VALUE
            # typing-target pointer (see the dataset-loading loop's own comment) —
            # structurally identical loss, just a different meaning for what tgt_src
            # points at. lambda_src defaults to 0.5 (matching this project's
            # long-standing legacy weight); semantic mode's train() call raises it to
            # lambda_click, since that pointer now carries a full "which field" job
            # instead of a narrower "which background source" one.
            valid_src = (tgt_src != -1)
            l_src = (ce_src(out.source_elem, tgt_src)
                     if valid_src.any() else out.source_elem.sum() * 0.0)

            loss = l_type + lambda_click * l_click + lambda_key * l_key + lambda_src * l_src


            if is_train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            totals["loss"]   += loss.item()
            totals["type"]   += l_type.item()
            totals["click"]  += l_click.item()
            totals["key"]    += l_key.item()
            totals["correct"] += (out.type_logits.argmax(-1) == tgt_types).sum().item()
            if valid_click.any():
                click_pred = out.click_elem[valid_click].argmax(-1)
                totals["click_correct"] += (click_pred == tgt_click_idx[valid_click]).sum().item()
                totals["click_total"]   += int(valid_click.sum().item())
            if valid_src.any():
                src_pred = out.source_elem[valid_src].argmax(-1)
                totals["src_correct"] += (src_pred == tgt_src[valid_src]).sum().item()
                totals["src_total"]   += int(valid_src.sum().item())
            totals["samples"] += tgt_types.size(0)
            totals["batches"] += 1

    n = max(totals["batches"], 1)
    return {
        "loss":       totals["loss"]  / n,
        "l_type":     totals["type"]  / n,
        "l_click":    totals["click"] / n,
        "l_key":      totals["key"]   / n,
        "accuracy":   totals["correct"] / max(totals["samples"], 1),
        "click_acc":  totals["click_correct"] / max(totals["click_total"], 1),
        "src_acc":    totals["src_correct"] / max(totals["src_total"], 1),
    }


def train(
    data_dir: str | list[str] = "data/output/traces/live",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_elements: int = 128,
    hist_len: int = 4,
    val_split: float = 0.2,
    save_path: str = "tasks/form_filling/model.pt",
    lambda_click: float = 2.0,
    lambda_key: float = 0.5,
    label_smoothing: float = 0.1,
    aug_drop_prob: float = 0.1,
    seed: int = 42,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    class_weight_mode: str = "none",          # "none" | "inverse" | "sqrt_inverse"
    balanced_sampler: bool = True,             # WeightedRandomSampler for type imbalance (default on — see Run 6)
    disambiguate_attempted: str = "none",      # "none" | "rank" | "section" — see TrajectoryDataset docstring
    rare_weight_basis: str = "type",           # "none" | "type" | "field" — see TrajectoryDataset docstring
    section_pattern: Optional[str] = None,     # only used when disambiguate_attempted="section"
    section_prefix: str = "section_",
    device_str: str = "auto",
    verbose: bool = True,
    action_space: str = "legacy",              # "legacy" | "semantic" — see TrajectoryDataset docstring
) -> TransformerAgentNetwork:
    """
    Train TransformerAgentNetwork via Behavioral Cloning.

    Anti-overfitting parameters
    ---------------------------
    label_smoothing : float (default 0.1)  — soften CrossEntropy targets
    aug_drop_prob   : float (default 0.0)  — randomly zero UI element rows
    dropout         : float (default 0.1)  — increase to 0.3+ for small data
    weight_decay    : float (default 1e-4) — increase to 1e-3 for small data
    d_model / num_layers — shrink to 64 / 2 for very small datasets
    """
    global _EMBED_CACHE_PATH
    torch.manual_seed(seed)
    if device_str == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"[train] Device: {device}")

    # Wire embed cache: load any prior saved cache, then prime will save back after encoding
    _EMBED_CACHE_PATH = str(Path(save_path).with_name("embed_cache.pkl"))
    _load_embed_cache(_EMBED_CACHE_PATH)

    dataset = TrajectoryDataset(
        data_dir, max_elements=max_elements, hist_len=hist_len,
        aug_drop_prob=aug_drop_prob,
        disambiguate_attempted=disambiguate_attempted,
        rare_weight_basis=rare_weight_basis,
        section_pattern=section_pattern,
        section_prefix=section_prefix,
        action_space=action_space,
    )
    # Semantic mode's action-type head needs NUM_SEMANTIC_ACTIONS logits instead
    # of the legacy NUM_ACTIONS; key_verb_id/lambda_src repurpose the existing
    # key-length and source-pointer losses onto SET_VALUE/FOCUS respectively —
    # see _run_epoch's own comment for why lambda_src is raised to lambda_click
    # in semantic mode instead of staying at the legacy 0.5.
    _num_actions = NUM_SEMANTIC_ACTIONS if action_space == "semantic" else NUM_ACTIONS
    _key_verb_id = VERB_TO_ID[_Verb.SET_VALUE] if action_space == "semantic" else ACTION_KEYBOARD
    _lambda_src  = lambda_click if action_space == "semantic" else 0.5
    if verbose:
        print(f"[train] {dataset}")

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    g       = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # Optional WeightedRandomSampler to balance click vs keyboard in each batch.
    # Each sample's sampling weight = 1 / (class_count) for its target type, so
    # each class is seen roughly equally often per epoch regardless of raw size.
    # Mechanism is orthogonal to class_weight_mode and tends to be more stable
    # on small imbalanced datasets than loss-weight balancing.
    train_sampler = None
    if balanced_sampler:
        from torch.utils.data import WeightedRandomSampler
        from collections import Counter as _Counter
        tgt_types_in_train = [dataset._samples[i][4] for i in train_ds.indices]
        type_counts = _Counter(tgt_types_in_train)
        sample_weights = [1.0 / max(type_counts[t], 1) for t in tgt_types_in_train]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(tgt_types_in_train),
            replacement=True,
        )
        if verbose:
            print(f"[train] balanced sampler ON — per-class counts in train: "
                  f"{dict(type_counts)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler, drop_last=False,
    )
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TransformerAgentNetwork(
        elem_features=ELEM_FEATURES, max_elements=max_elements, d_model=d_model,
        nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=dropout, hist_len=hist_len, num_actions=_num_actions,
    ).to(device)

    if verbose:
        print(f"[train] {model}")

    # Class weights for the type head. Diagnostics showed inverse-frequency
    # weighting caused the keyboard weight to dominate (2.14 vs 0.47 for click)
    # and produced wild val accuracy oscillations (0.15 ↔ 0.85 across epochs)
    # on the small val set. Default is now "none" — let label_smoothing handle
    # the imbalance. "inverse" and "sqrt_inverse" are kept for experimentation.
    _cc = dataset.class_counts()
    _total = sum(_cc.values()) or 1
    if class_weight_mode == "none":
        _class_weights = None
        if verbose:
            print("[train] class weights: none (uniform)")
    elif action_space == "semantic":
        # class_weight_mode's ACTION_*-name mapping below is legacy-only —
        # raise clearly rather than silently build a wrong-sized/wrong-order
        # tensor against NUM_SEMANTIC_ACTIONS classes. Not built out because
        # this project's own default ("none") is already the recommended
        # setting (see the comment above), and this session's own A/B is
        # run with the default — a real semantic-mode class-weight mapping
        # is separate follow-up work, not needed for this comparison.
        raise NotImplementedError(
            "class_weight_mode != 'none' is not yet supported with action_space='semantic' "
            "-- the class-name mapping below is legacy-only. Use class_weight_mode='none'.")
    else:
        # General over ALL action classes by index (was hardcoded to 3). Inverse-
        # frequency weights punish collapsing onto one class — fixes the observed
        # action-head collapse (always-scroll, then always-hotkey).
        _name_by_idx = {
            ACTION_NOOP: "no_op", ACTION_CLICK: "click", ACTION_KEYBOARD: "keyboard",
            ACTION_HOTKEY: "hotkey", ACTION_SCROLL: "scroll",
            ACTION_DCLICK: "double_click", ACTION_DRAG: "drag",
        }
        raw = []
        for i in range(NUM_ACTIONS):
            cnt = max(_cc.get(_name_by_idx.get(i, ""), 0), 1)
            w = _total / cnt
            raw.append(w ** 0.5 if class_weight_mode == "sqrt_inverse" else w)
        _class_weights = torch.tensor(raw, dtype=torch.float32, device=device)
        _class_weights = _class_weights / _class_weights.sum() * len(_class_weights)
        if verbose:
            _wstr = "  ".join(f"{_name_by_idx.get(i,i)}={_class_weights[i]:.2f}"
                              for i in range(NUM_ACTIONS))
            print(f"[train] class weights ({class_weight_mode}): {_wstr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    best_val_loss      = math.inf
    best_val_acc       = 0.0
    best_val_click_acc = 0.0
    save_path_p        = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)

    _train_start   = time.time()
    _epoch_times: list[float] = []
    for epoch in range(1, epochs + 1):
        _epoch_t0 = time.time()
        train_m = _run_epoch(model, train_loader, optimizer, device, lambda_click, lambda_key, label_smoothing,
                             _class_weights, key_verb_id=_key_verb_id, lambda_src=_lambda_src)
        dataset._eval_mode = True
        val_m   = _run_epoch(model, val_loader,   None,      device, lambda_click, lambda_key, label_smoothing,
                             _class_weights, key_verb_id=_key_verb_id, lambda_src=_lambda_src)
        dataset._eval_mode = False
        scheduler.step()
        _epoch_time = time.time() - _epoch_t0
        _epoch_times.append(_epoch_time)

        if verbose:
            print(
                f"Epoch {epoch:>3}/{epochs}  |  "
                f"train_loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.3f}  click_acc={train_m['click_acc']:.3f}  src_acc={train_m['src_acc']:.3f}  |  "
                f"val_loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.3f}  click_acc={val_m['click_acc']:.3f}  src_acc={val_m['src_acc']:.3f}  "
                f"[type={train_m['l_type']:.3f} click={train_m['l_click']:.3f} key={train_m['l_key']:.4f}]  "
                f"({_epoch_time:.1f}s)"
            )

        # Checkpoint on best click_acc alone -- the metric actually deployed on.
        # History: pure val_LOSS picked an early, underfit, scroll-biased model
        # even though val_acc kept climbing, so that was replaced with combined
        # acc (val_acc + click_acc). That in turn has its OWN failure mode,
        # found 2026-08-08 comparing two A/B runs: ~95% of actions are type
        # "click", so val_acc (action-TYPE accuracy) saturates near-ceiling by
        # epoch 1 just from "always guess click" -- it's noise after that, not
        # signal. Summing it with click_acc let a lucky early-epoch val_acc
        # spike outvote every later epoch's real click_acc improvement (a run
        # saved epoch 1, val_acc=0.914 click_acc=0.171, over epoch 44's
        # val_acc=0.747 click_acc=0.396 -- the WORST click-targeting epoch beat
        # the best one). click_acc alone can't be gamed by the dominant class.
        save_loss   = val_m["loss"] if not math.isnan(val_m["loss"]) else train_m["loss"]
        _val_score  = val_m["click_acc"]
        _best_score = best_val_click_acc
        if _val_score > _best_score:
            best_val_loss      = save_loss
            best_val_acc       = val_m["accuracy"]
            best_val_click_acc = val_m["click_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "val_click_acc": best_val_click_acc,
                "hyperparams": {
                    "elem_features": ELEM_FEATURES, "max_elements": max_elements,
                    "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward, "dropout": dropout,
                    "hist_len": hist_len, "num_actions": _num_actions,
                    "action_space": action_space,
                },
            }, save_path_p)
            if verbose:
                print(f"           -> Saved checkpoint (val_acc={best_val_acc:.3f}  click_acc={best_val_click_acc:.3f}  "
                      f"src_acc={val_m['src_acc']:.3f})")

    _total_train_time = time.time() - _train_start
    if verbose:
        print(f"[train] Done.  Best val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.3f}  click_acc={best_val_click_acc:.3f} -> {save_path_p}")
        print(f"[train] Total time: {_total_train_time:.1f}s  ({_total_train_time/epochs:.1f}s/epoch avg)")

    # Persist training metrics for trend tracking across runs
    try:
        import json as _json, datetime as _dt
        # Anchored on this FILE's own location, not save_path_p's -- save_path
        # isn't always exactly 2 levels under repo root (e.g. an experiments/
        # subdir for an A/B run), and parents[2] silently pointed the log at
        # the wrong place when it wasn't. Found 2026-08-12 running an A/B into
        # tasks/form_filling/experiments/*.pt: logged to tasks/data/output/
        # instead of data/output/.
        _repo_root = Path(__file__).resolve().parents[3]
        _log_path = _repo_root / "data" / "output" / "transformer_training_log.jsonl"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _n_train = len(train_loader.dataset)
        _row = {
            "timestamp":            _dt.datetime.now().isoformat(),
            "trace_dir":            str(data_dir),
            "n_train":              _n_train,
            "n_val":                len(val_loader.dataset),
            "epochs":               epochs,
            "best_val_loss":        round(best_val_loss, 4),
            "best_val_acc":         round(best_val_acc, 4),
            "best_val_click_acc":   round(best_val_click_acc, 4),
            # Timing — objectives 3/5 (training pipeline) and 7 (scaling: does
            # training time grow reasonably with n_train? see scripts/scaling_report.py)
            "total_train_time_sec": round(_total_train_time, 2),
            "avg_epoch_time_sec":   round(sum(_epoch_times) / len(_epoch_times), 3) if _epoch_times else None,
            "samples_per_sec":      round(_n_train * epochs / _total_train_time, 1) if _total_train_time > 0 else None,
        }
        with open(_log_path, "a", encoding="utf-8") as _lf:
            _lf.write(_json.dumps(_row) + "\n")
    except Exception:
        pass

    if save_path_p.exists():
        ckpt = torch.load(save_path_p, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # No checkpoint was saved (e.g. all losses were nan) — save current weights
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_loss": best_val_loss,
            "hyperparams": {
                "elem_features": ELEM_FEATURES, "max_elements": max_elements,
                "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                "dim_feedforward": dim_feedforward, "dropout": dropout,
                "hist_len": hist_len, "num_actions": _num_actions,
                "action_space": action_space,
            },
        }, save_path_p)
        if verbose:
            print(f"           -> Saved final checkpoint (no improvement detected)")
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
#  SECTION 4 — INFERENCE
# ═══════════════════════════════════════════════════════════

_model_cache: Dict[str, TransformerAgentNetwork] = {}
_ACTION_LABELS = {ACTION_NOOP: "no_op", ACTION_CLICK: "click", ACTION_KEYBOARD: "keyboard",
                  ACTION_HOTKEY: "hotkey", ACTION_SCROLL: "scroll",
                  ACTION_DCLICK: "double_click", ACTION_DRAG: "drag"}
_ACTION_IDS    = {v: k for k, v in _ACTION_LABELS.items()}


def _load_model(model_path: str, device: torch.device) -> TransformerAgentNetwork:
    key = f"{model_path}:{device}"
    if key not in _model_cache:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        hp   = ckpt.get("hyperparams", {})
        m    = TransformerAgentNetwork(
            elem_features=hp.get("elem_features", ELEM_FEATURES), max_elements=hp.get("max_elements", 128),
            d_model=hp.get("d_model", 128), nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 4), dim_feedforward=hp.get("dim_feedforward", 256),
            dropout=hp.get("dropout", 0.1), hist_len=hp.get("hist_len", 4),
            num_actions=hp.get("num_actions", NUM_ACTIONS),
        ).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.action_space = hp.get("action_space", "legacy")   # stamped for predict() to decode correctly
        m.eval()
        _model_cache[key] = m
    return _model_cache[key]


def predict(
    state: dict,
    history: Optional[List[Dict[str, Any]]] = None,
    model_path: str = "tasks/form_filling/model.pt",
    device_str: str = "auto",
    clear_cache: bool = False,
) -> Dict[str, Any]:
    """
    Predict the next GUI action.

    Parameters
    ----------
    state   : current state dict from a trace JSON.
    history : list of previous step dicts (oldest first), each containing:
              {"state": <dict>, "action_type": <int|str>,
               "click_xy": [x, y], "key_count": <int>}
              Up to hist_len-1 entries used; zero-padded if fewer.
    """
    if device_str == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(device_str)

    if clear_cache:
        _model_cache.pop(f"{model_path}:{device}", None)

    # Load semantic embed cache saved alongside this model (once per process per path)
    _pkl_path = str(Path(model_path).with_name("embed_cache.pkl"))
    _load_embed_cache(_pkl_path)

    model        = _load_model(model_path, device)
    H            = model.hist_len
    max_elements = model.max_elements
    num_actions  = model.num_actions
    # Computed here (not just below, near the output decode) so the history
    # encoding below can branch on it too — see the p_types loop's comment.
    _is_semantic = getattr(model, "action_space", "legacy") == "semantic"

    ctx = (history or [])[-(H - 1):]  # last H-1 items

    # Build state tensor list (H-1 context + 1 current)
    ctx_tensors = [encode_state(item["state"], max_elements) for item in ctx]
    while len(ctx_tensors) < H - 1:
        ctx_tensors.insert(0, torch.zeros(max_elements, ELEM_FEATURES))
    all_states = torch.stack(
        ctx_tensors + [encode_state(state, max_elements)])  # (H, T, F)

    # Build action tensors
    p_types_list, p_cont_list = [], []
    for item in ctx:
        if _is_semantic:
            # agent.py stamps a raw "verb" string onto each history entry
            # alongside the always-legacy "action_type" (added 2026-08-13) --
            # use it directly so a semantic model reads its own history back
            # in the same verb space it was trained on. Older history entries
            # (or a mixed-mode run) won't have "verb" yet; fall back to a
            # lossy reconstruction from the legacy action_type rather than
            # feeding a legacy id (0-6) straight into a verb (0-9) embedding,
            # where e.g. legacy CLICK=1 would silently collide with verb
            # SET_VALUE=1.
            _verb_str = item.get("verb")
            if _verb_str is not None:
                at = VERB_TO_ID.get(_Verb(_verb_str), VERB_TO_ID[_Verb.WAIT])
            else:
                at = VERB_TO_ID[SemanticAction.from_legacy_dict(item).verb]
        else:
            at  = item.get("action_type", 0)
            at  = _ACTION_IDS.get(at, at) if isinstance(at, str) else at
        cxy = item.get("click_xy", [0.0, 0.0])
        kn  = min(float(item.get("key_count", 0)) / 100.0, 1.0)
        res = item.get("state", {}).get("screen_resolution", [DEFAULT_W, DEFAULT_H])
        W   = float(res[0]) or DEFAULT_W
        H_px = float(res[1]) or DEFAULT_H
        cx   = float(cxy[0]) / W   if cxy[0] > 1 else float(cxy[0])
        cy   = float(cxy[1]) / H_px if cxy[1] > 1 else float(cxy[1])
        p_types_list.append(at)
        p_cont_list.append([cx, cy, kn])
    while len(p_types_list) < H - 1:
        p_types_list.insert(0, num_actions)
        p_cont_list.insert(0, [0.0, 0.0, 0.0])

    p_types = torch.tensor(p_types_list, dtype=torch.long).unsqueeze(0).to(device)
    p_cont  = torch.tensor(p_cont_list,  dtype=torch.float32).unsqueeze(0).to(device)
    s_batch = all_states.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(s_batch, p_types, p_cont)

    idx = out.type_logits.argmax(-1).item()
    res = state.get("screen_resolution", [DEFAULT_W, DEFAULT_H])
    W   = float(res[0]) or DEFAULT_W
    H_px = float(res[1]) or DEFAULT_H

    import torch.nn.functional as F
    probs = F.softmax(out.type_logits[0], dim=-1).tolist()
    # Ported 2026-08-12 from origin/verb-loop-rewrite. In semantic mode, decode
    # the predicted verb straight back to the legacy action_type string via
    # SemanticAction.to_legacy_dict() — this is the ONE seam that makes the
    # rest of the pipeline (agent.py's merge/execution) fully unaware of which
    # action_space trained the model at all. No agent.py changes needed for
    # either mode; both always speak the same legacy dict shape at this boundary.
    # (_is_semantic computed earlier, above the p_types history-encoding loop.)
    if _is_semantic:
        _verb = ID_TO_VERB.get(idx, _Verb.WAIT)
        result: Dict[str, Any] = {
            "action_type": SemanticAction(verb=_verb).to_legacy_dict()["action_type"],
            "verb":        _verb.value,
            "confidence":  round(max(probs), 4),
            "_scores": {ID_TO_VERB.get(i, _Verb.WAIT).value: round(p, 3) for i, p in enumerate(probs)},
        }
    else:
        result: Dict[str, Any] = {
            "action_type": _ACTION_LABELS.get(idx, "no_op"),
            "confidence":  round(max(probs), 4),
            "_scores": {_ACTION_LABELS.get(i, str(i)): round(p, 3) for i, p in enumerate(probs)},
        }

    # ALWAYS expose the click/where pointer (which field) — this is the
    # transformer's navigation signal, decoupled from the action-type head
    # (which collapses), so the merge can use "which field" even when the
    # action-type head misfires. In semantic mode there are TWO separate
    # pointer heads (click_elem for click-like verbs, source_elem repurposed
    # as the typing-target pointer for FOCUS/SET_VALUE) — pick whichever one
    # matches the predicted verb.
    if _is_semantic and ID_TO_VERB.get(idx) in (_Verb.FOCUS, _Verb.SET_VALUE):
        _where_logits = out.source_elem[0]
    else:
        _where_logits = out.click_elem[0]
    click_idx  = int(_where_logits.argmax(-1).item())
    click_conf = F.softmax(_where_logits, dim=-1).max().item()
    result["click_elem_idx"] = click_idx
    result["_click_conf"]    = round(click_conf, 3)
    elems = state.get("elements", [])[:max_elements]
    if 0 <= click_idx < len(elems):
        bbox = elems[click_idx].get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            cx_px = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy_px = (float(bbox[1]) + float(bbox[3])) / 2.0
            result["click_position"] = [round(cx_px, 1), round(cy_px, 1)]
        else:
            result["click_position"] = [0.0, 0.0]
    else:
        result["click_position"] = [0.0, 0.0]

    if _is_semantic:
        if ID_TO_VERB.get(idx) == _Verb.SET_VALUE:
            result["key_count"] = max(1, round(out.key_count[0, 0].item() * 100))
            # source_elem is repurposed as the WHERE pointer above in semantic
            # mode — WHAT to type comes from the LLM, not a copied background
            # span, so no separate source_elem_idx is exposed here.
    elif idx == ACTION_KEYBOARD:
        result["key_count"]       = max(1, round(out.key_count[0, 0].item() * 100))
        result["source_elem_idx"] = int(out.source_elem[0].argmax(-1).item())
        result["_source_conf"]    = round(F.softmax(out.source_elem[0], dim=-1).max().item(), 3)
    return result


# ═══════════════════════════════════════════════════════════
#  SECTION 5 — CLI
# ═══════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TransformerAgentNetwork — train or predict")
    p.add_argument("--mode",            choices=["train", "predict"], default="train")
    # Shared
    p.add_argument("--model_path",      default="tasks/form_filling/model.pt")
    p.add_argument("--device",          default="auto", dest="device_str")
    # Train args
    p.add_argument("--data_dir",        default="data/output/traces/live")
    p.add_argument("--epochs",          default=50,   type=int)
    p.add_argument("--batch_size",      default=16,   type=int)
    p.add_argument("--lr",              default=1e-3, type=float)
    p.add_argument("--weight_decay",    default=1e-4, type=float)
    p.add_argument("--max_elements",    default=128,  type=int)
    p.add_argument("--hist_len",        default=4,    type=int)
    p.add_argument("--val_split",       default=0.2,  type=float)
    p.add_argument("--label_smoothing", default=0.1,  type=float)
    p.add_argument("--aug_drop_prob",   default=0.0,  type=float)
    p.add_argument("--d_model",         default=128,  type=int)
    p.add_argument("--nhead",           default=4,    type=int)
    p.add_argument("--num_layers",      default=4,    type=int)
    p.add_argument("--dim_feedforward", default=256,  type=int)
    p.add_argument("--dropout",         default=0.1,  type=float)
    p.add_argument("--class_weight_mode", default="none",
                   choices=["none", "inverse", "sqrt_inverse"],
                   help="Type-head class weighting strategy (default: none; balanced_sampler handles imbalance)")
    p.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True,
                   help="Use WeightedRandomSampler to balance click vs keyboard per epoch "
                        "(default: on — disable with --no-balanced_sampler)")
    p.add_argument("--disambiguate_attempted", default="none",
                   choices=["none", "rank", "section"],
                   help="Repeated-label disambiguation (Driver 1/2/3) for the 'attempted' feature. "
                        "'none' (default, un-regressed baseline). 'rank' — list-order based; tested "
                        "alone it regressed val_click_acc to 46.9%% from a 68.9%% baseline (unstable "
                        "across snapshots). 'section' — panecontrol bbox-geometry based (needs "
                        "--section_pattern set); a more stable signal, not yet validated by a real A/B.")
    p.add_argument("--section_pattern", default=None,
                   help="Regex (2 groups: kind, number) identifying a section-pane's label, e.g. "
                        "r'section_(driver|vehicle)_(\\d+)$' (mirrors ScopeConfig.section_pattern). "
                        "Only used when --disambiguate_attempted=section.")
    p.add_argument("--section_prefix", default="section_",
                   help="Prefix identifying section-pane elements by label (mirrors "
                        "ScopeConfig.section_prefix). Only used when --disambiguate_attempted=section.")
    p.add_argument("--rare_weight_basis", default="type",
                   choices=["none", "type", "field"],
                   help="Rare-action loss up-weighting basis: 'type' (control type, original "
                        "pre-regression basis, default), 'field' (specific field identity, "
                        "never tested in isolation before), 'none' (disabled).")
    p.add_argument("--action_space", default="legacy",
                   choices=["legacy", "semantic"],
                   help="'legacy' (default) — the original ACTION_* scheme, unchanged. "
                        "'semantic' — Universal Semantic Action Space (ported 2026-08-12 from "
                        "origin/verb-loop-rewrite): richer verb vocabulary from "
                        "action_labeler.translate_step(), split click/typing-target pointer "
                        "heads. Unverified on this project's own data until an isolated A/B runs.")
    # Predict args
    p.add_argument("--trace_path",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "train":
        train(
            data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, max_elements=args.max_elements,
            hist_len=args.hist_len, val_split=args.val_split, save_path=args.model_path,
            label_smoothing=args.label_smoothing, aug_drop_prob=args.aug_drop_prob,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, dropout=args.dropout,
            class_weight_mode=args.class_weight_mode,
            balanced_sampler=args.balanced_sampler,
            disambiguate_attempted=args.disambiguate_attempted,
            rare_weight_basis=args.rare_weight_basis,
            section_pattern=args.section_pattern,
            section_prefix=args.section_prefix,
            action_space=args.action_space,
            device_str=args.device_str,
        )
    else:
        if not args.trace_path:
            raise SystemExit("--trace_path required for predict mode")
        with open(args.trace_path, encoding="utf-8") as f:
            trace = json.load(f)
        result = predict(trace.get("state", {}), model_path=args.model_path,
                         device_str=args.device_str)
        print(json.dumps(result, indent=2))
