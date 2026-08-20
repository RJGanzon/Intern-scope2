#!/usr/bin/env python3
"""
generate_portal_demos.py — synthesise Scope #2 demonstrations from the real portal.

Approved by the thesis panel as a way to accelerate training. What that approval
does NOT do is make generated traces interchangeable with human ones, so every
session this writes is marked (`generated: true`, plus the arguments that
produced it) and lands under its own `session_gen_*` directory. Nothing
downstream has to trust a folder name to tell the two apart, and a clone score
measured against generated data is not evidence about human behaviour.

WHAT MAKES THIS DATA WORTH TRAINING ON
---------------------------------------
The states are not invented. The script drives a real browser through the real
portal and snapshots it through the same WebObserver the recorder uses, so every
step carries a genuine 303-element DOM: real labels resolved from
aria-labelledby, real geometry, real derived Remarks appearing when a grade
lands.

It snapshots BEFORE and AFTER each action, which is the part that decides
whether any of this is useful. A first attempt at generated data reused one
frozen state for every step, and a dataset like that teaches a model nothing
about progress - `is_filled` never changes, so it cannot learn that a filled
field is done. DEVELOPERS.md records that exact failure from Scope #1's early
data, where the model looped because state carried labels but never values.

Values come from the grade sheet, through GradeSheetSource, so a generated row
says the same thing a human copying that row would have said.

WHAT IT DOES NOT REPLACE
-------------------------
Behaviour. The fill order here is whatever --order says, and the model will
clone that order faithfully - which proves the pipeline works, not that it
learned from a person. Keep at least one human session recorded and held out,
and score against that.

Usage
-----
    python scripts/generate_portal_demos.py --students 50
    python scripts/generate_portal_demos.py --students 50 --sessions 3 --headed
    python scripts/generate_portal_demos.py --order grade,year,course --variant v1_reordered
"""

from __future__ import annotations


# Before any other import - see components/dpi.py. Generated bboxes have to
# land in the same coordinate space a human recording produces, and that
# space is decided by whichever ruler this process ends up with.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "..", "components"))
import dpi as _dpi
_dpi.ensure_per_monitor()

import argparse
import json
import os
import random
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "components"),
           os.path.join(_ROOT, "components", "scope2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_SHEET = os.path.join("components", "scope2", "data", "sheets", "grade_sheet.xlsx")

# Portal column -> the grade sheet column it is copied from. Same three the
# guide asks a human to fill: Remarks derives itself from the grade, and
# Recommendations is optional and left blank.
COLUMN_SOURCE = {
    "course": "PROGRAM",
    "year":   "YEAR LEVEL",
    "grade":  "FINAL GRADE",
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--students", type=int, default=50,
                    help="Rows to fill per session (default: 50, the whole roster).")
    ap.add_argument("--sessions", type=int, default=1,
                    help="How many sessions to write.")
    ap.add_argument("--variant", default="v0_base",
                    help="Which mock portal to drive.")
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--out", default=os.path.join("data", "demos", "generated"),
                    help="Kept separate from data/demos/human by default. Mixing "
                         "them is a decision to make on the train command line, "
                         "where it is visible, not by writing into the same folder.")
    ap.add_argument("--order", default="course,year,grade",
                    help="Column fill order within a row. The model clones this.")
    ap.add_argument("--row-order", default="top-down",
                    choices=["top-down", "bottom-up", "shuffled"],
                    help="Order the rows are visited in.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Only used by --row-order shuffled; recorded in the session.")
    ap.add_argument("--headed", action="store_true",
                    help="Drive a visible browser. Slower, but the window has a real "
                         "screen position, so bboxes come out in the same coordinate "
                         "space a human recording produces. Headless has no window "
                         "and falls back to viewport coordinates - self-consistent, "
                         "but a different space than the live run uses.")
    return ap.parse_args(argv)


def cell_handles(page, row_index):
    """The editable controls of one row, keyed by column."""
    return page.evaluate(
        """(i) => {
            const tr = document.querySelector(
                `#records-body tr[data-row='${i}']`);
            if (!tr) return null;
            const out = {};
            tr.querySelectorAll("input[data-key], select[data-key], textarea[data-key]")
              .forEach(el => { out[el.dataset.key] = el.id || ""; });
            return out;
        }""", row_index)


# The observer's element_id is positional ("web_41"), not the DOM id - it moves
# when the page does. The accessible name is the stable handle, and it is the
# one the agent itself addresses cells by, so resolving on it here means the
# generated trace points at elements the same way a real step does.
_ACCESSIBLE_NAME_JS = r"""
(id) => {
  const el = document.getElementById(id);
  if (!el) return "";
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const ids = (el.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean);
  if (ids.length) {
    return ids.map((i) => { const n = document.getElementById(i);
                            return n ? clean(n.textContent) : ""; })
              .filter(Boolean).join(" ");
  }
  return clean(el.getAttribute("aria-label"));
}
"""


def element_for(state, label):
    """The state element carrying this accessible name."""
    if not label:
        return None
    for el in state.get("elements") or []:
        if (el.get("label") or "") == label:
            return el
    return None


def centre(bbox):
    return [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]


class SessionWriter:
    """Writes steps in the recorder's own format, one file per step."""

    def __init__(self, out_dir, meta):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(out_dir, f"session_gen_{stamp}_{meta['index']}")
        os.makedirs(self.dir, exist_ok=True)
        self.meta = meta
        self.n = 0

    def step(self, state, next_state, mouse=None, keyboard=None):
        step = {
            "trace_id":  f"live_step_{self.n:04d}",
            "timestamp": datetime.now().isoformat(),
            "duration":  1.0,
            "type":      "form_filling",
            "state":      state,
            "mouse":      mouse or {"actions": []},
            "keyboard":   keyboard or {"actions": []},
            "next_state": next_state,
            # Carried on every step, not just a manifest: a step copied out of
            # here into another folder takes its provenance with it.
            "generated": True,
            "generator": self.meta,
        }
        path = os.path.join(self.dir, f"live_step_{self.n:04d}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(step, fh, ensure_ascii=False)
        self.n += 1


def generate_session(observer, page, source, args, index, students):
    from observers.web_observer import WebObserver  # noqa: F401  (documented dep)

    meta = {
        "index": index,
        "variant": args.variant,
        "order": args.order,
        "row_order": args.row_order,
        "seed": args.seed,
        "sheet": os.path.basename(args.sheet),
        "script": "scripts/generate_portal_demos.py",
    }
    writer = SessionWriter(args.out, meta)
    columns = [c.strip() for c in args.order.split(",") if c.strip()]

    # A fresh page per session, so row one starts empty the way a human's would.
    page.reload()
    page.wait_for_selector("#records-body tr")

    state = observer.snapshot()
    for row in students:
        ids = cell_handles(page, row)
        if not ids:
            continue
        source.refresh(row)

        for col in columns:
            element_id = ids.get(col)
            value = source.lookup(COLUMN_SOURCE[col])
            if not element_id or value is None:
                continue

            handle = page.locator(f"#{element_id}")
            label = page.evaluate(_ACCESSIBLE_NAME_JS, element_id)
            # A row below the fold has to be brought into view first, exactly as
            # a person would scroll to it - and the scroll is a step of its own,
            # because the model has to learn that a target it cannot see needs
            # one before the click.
            before_scroll = state
            handle.scroll_into_view_if_needed()
            state = observer.snapshot()
            if state != before_scroll:
                el = element_for(before_scroll, label)
                pos = centre(el["bbox"]) if el and el.get("bbox") else [0, 0]
                writer.step(before_scroll, state,
                            mouse={"actions": [{"position": pos, "type": "scroll",
                                                "dy": 3.0,
                                                "timestamp": datetime.now().isoformat()}]})

            el = element_for(state, label)
            if not el or not el.get("bbox"):
                continue

            # CLICK: focus the cell.
            handle.click()
            after_click = observer.snapshot()
            writer.step(state, after_click,
                        mouse={"actions": [{"position": centre(el["bbox"]),
                                            "type": "click",
                                            "timestamp": datetime.now().isoformat()}]})
            state = after_click

            # TYPE: the value, keystroke by keystroke, so the page reacts the
            # way it does for a person - which is what makes Remarks appear.
            handle.type(str(value), delay=1)
            after_type = observer.snapshot()
            writer.step(state, after_type,
                        keyboard={"actions": [{"strokes": [
                            {"pasted_text": str(value), "key": ""}]}]})
            state = after_type

    return writer


def main(argv=None):
    args = parse_args(argv)

    from data_sources.grade_sheet_source import GradeSheetSource
    from executor.scanner import variant_url
    from observers.web_observer import WebObserver

    sheet = args.sheet if os.path.isabs(args.sheet) else os.path.join(_ROOT, args.sheet)
    if not os.path.exists(sheet):
        raise SystemExit(f"grade sheet not found: {sheet}")
    source = GradeSheetSource(sheet)

    rows = list(range(args.students))
    if args.row_order == "bottom-up":
        rows.reverse()
    elif args.row_order == "shuffled":
        random.Random(args.seed).shuffle(rows)

    observer = WebObserver(headless=not args.headed)
    if not observer.connect(variant_url(args.variant)):
        raise SystemExit("could not open the portal")

    try:
        page = observer._page
        page.wait_for_selector("#records-body tr")
        for i in range(args.sessions):
            writer = generate_session(observer, page, source, args, i, rows)
            print(f"session {i + 1}/{args.sessions}: {writer.n} steps -> {writer.dir}")
    finally:
        observer.disconnect()

    print("\nGenerated data is marked as such on every step. Keep a human session "
          "held out and score against it - a clone score measured on this is a "
          "measure of the generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
