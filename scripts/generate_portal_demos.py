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
    ap.add_argument("--from-demo", dest="from_demo", default=None,
                    help="A recorded human session to continue. The column order "
                         "is read off that demonstration instead of --order, and "
                         "the students it already covers are skipped, so the "
                         "generated rows extend the same pattern rather than "
                         "inventing one. This is the honest way to turn ten "
                         "recorded rows into a trainable dataset.")
    ap.add_argument("--skip-demonstrated", action="store_true",
                    help="With --from-demo, leave out the rows the human already "
                         "covered. Off by default, and read the note it prints "
                         "before turning it on: excluding them means the sheet "
                         "never starts empty in training, and the human session "
                         "you would score against is exactly the sheet starting "
                         "empty.")
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


def cell_names(page):
    """{accessible name: column key} for every editable cell on the page.

    Built from the live page, so it is exact rather than parsed out of a label:
    a variant that renames Grade to "Final Rating 0-100" renames the names here
    too, and a demonstration recorded against it still resolves.
    """
    return page.evaluate(
        r"""() => {
            const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
            const name = (el) => {
              const ids = (el.getAttribute("aria-labelledby") || "")
                            .split(/\s+/).filter(Boolean);
              if (ids.length) {
                return ids.map((i) => { const n = document.getElementById(i);
                                        return n ? clean(n.textContent) : ""; })
                          .filter(Boolean).join(" ");
              }
              return clean(el.getAttribute("aria-label"));
            };
            const out = {};
            document.querySelectorAll(
                "#records-body input[data-key], #records-body select[data-key], "
                + "#records-body textarea[data-key]")
              .forEach(el => { const n = name(el); if (n) out[n] = el.dataset.key; });
            return out;
        }""")


def infer_pattern(session_dir, names_to_keys):
    """What order did the human actually fill a row in, and how far did they get?

    Read off the demonstration rather than assumed, because the whole reason to
    do this is that the order is theirs. Each click is resolved to the cell that
    was under it - the same resolution the cleaner and the trainer use - and the
    order columns first appear within a student is the pattern to continue.

    Returns (column order, student names already demonstrated).
    """
    import glob as _glob

    order, seen_students = [], []
    for path in sorted(_glob.glob(os.path.join(session_dir, "live_step_*.json"))):
        with open(path, encoding="utf-8") as fh:
            step = json.load(fh)
        actions = step.get("mouse", {}).get("actions", [])
        if not actions or actions[0].get("type") not in ("click", "double_click"):
            continue
        target = _under(step.get("state", {}), actions[0].get("position") or [0, 0])
        label = (target or {}).get("label") or ""
        key = names_to_keys.get(label)
        if not key:
            continue
        if key not in order:
            order.append(key)
        student = label[len(_column_prefix(label, names_to_keys)):].strip()
        if student and student not in seen_students:
            seen_students.append(student)
    return order, seen_students


def _column_prefix(label, names_to_keys):
    """The column half of "Grade 0-100 Abad, Andrea A.".

    Taken as the prefix shared by every cell of that column, then cut back to
    the last word boundary. The cut is the part that matters: "Course Abad..."
    and "Course Aguilar..." share "Course A", so the raw common prefix eats the
    first letter of the name and leaves "bad, Andrea A." as the student.
    """
    key = names_to_keys.get(label)
    same_column = [n for n, k in names_to_keys.items() if k == key]
    if len(same_column) < 2:
        return ""
    prefix = os.path.commonprefix(same_column)
    cut = prefix.rfind(" ")
    return prefix[:cut + 1] if cut >= 0 else prefix


def _under(state, pos):
    best, area = None, float("inf")
    for el in state.get("elements") or []:
        box = el.get("bbox") or []
        if len(box) == 4 and box[0] <= pos[0] <= box[2] and box[1] <= pos[1] <= box[3]:
            a = (box[2] - box[0]) * (box[3] - box[1])
            if a < area:
                best, area = el, a
    return best


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
        "derived_from": (os.path.basename(args.from_demo.rstrip("/\\"))
                         if args.from_demo else None),
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

        if args.from_demo:
            demo = args.from_demo
            if not os.path.isabs(demo):
                demo = os.path.join(_ROOT, demo)
            if not os.path.isdir(demo):
                raise SystemExit(f"no such session: {demo}")

            order, demonstrated = infer_pattern(demo, cell_names(page))
            if not order:
                raise SystemExit(
                    f"could not read a fill order out of {os.path.basename(demo)} - "
                    "no click in it resolved to a cell. That session is the one "
                    "with the coordinate problem; check it with the clicks-landing "
                    "count before building on it.")
            args.order = ",".join(order)
            print(f"read from {os.path.basename(demo)}:")
            print(f"  column order   {' -> '.join(order)}")
            print(f"  already shown  {len(demonstrated)} student(s): "
                  f"{', '.join(demonstrated[:3])}"
                  f"{' ...' if len(demonstrated) > 3 else ''}")
            # The demonstrated rows are generated too, and skipping them was a
            # real mistake worth recording. Starting generation at row 4 meant
            # every training state had rows 0-3 empty while filling began
            # further down - and the human session is precisely rows 0-3 filling
            # from an empty sheet, a situation the model then never saw. Scored
            # against that session it collapsed to one constant prediction,
            # 0/12, despite 0.93 click accuracy on its own validation split.
            # A demonstration is a pattern to continue, not a range to avoid.
            if args.skip_demonstrated:
                rows[:] = [r for r in rows if r >= len(demonstrated)]
                print("  NOTE: --skip-demonstrated leaves the demonstrated rows "
                      "out of training. The sheet then never starts empty, so a "
                      "session that does is out of distribution.")
            print(f"  generating     rows {rows[0]}-{rows[-1]} in that same order")
            print()

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
