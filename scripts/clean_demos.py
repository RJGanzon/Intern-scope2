"""
clean_demos.py — produce clean NAVIGATION training data from raw recordings.

Removes the noise that corrupts the recorded field order:
  1. Dropdown-SELECTION clicks — a click made while a combobox dropdown is open
     (listitems present in the state) lands on the option, which visually sits
     OVER a lower field, so it gets mis-logged as a click on that field
     (e.g. a phantom "Expiration Date" between Type and Term). These are value
     selection, not navigation — dropped.
  2. Non-form-window clicks (terminal, recorder GUI, Notepad).
  3. Clicks on panes / non-interactive chrome.
  4. Consecutive duplicate clicks on the same field (combobox open + reopen).

Typing frames are KEPT — they fill the form, which is the navigation signal.

Which window counts as "the form" comes from ScopeConfig.window_markers, not
from a literal in this file. It was `"insurance" in title` until 2026-08-20,
which meant any other scope's recording had every one of its clicks counted as
junk — a scope #2 browser session cleaned down to typing frames only, with the
navigation signal (the whole point of this script) silently gone.

Usage:
    python scripts/clean_demos.py <src_dir> <dst_dir>
    python scripts/clean_demos.py data/demos/policy_nav data/demos/policy_clean
    python scripts/clean_demos.py <src> <dst> --scope grade_portal
    python scripts/clean_demos.py <src> <dst> --window-match "grade portal,roster"
"""
from __future__ import annotations
import sys, os, glob, json, shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "components")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.scope import GRADE_PORTAL_SCOPE, INSURANCE_SCOPE, ScopeConfig

# Named on the command line as --scope <name>. Default stays "insurance" so
# every existing scope #1 invocation cleans exactly as it did before.
SCOPES = {
    "insurance":    INSURANCE_SCOPE,
    "grade_portal": GRADE_PORTAL_SCOPE,
    "generic":      ScopeConfig(),      # no markers → every window accepted
}

FIELD = {"editcontrol", "input", "comboboxcontrol", "combobox",
         "checkboxcontrol", "checkbox", "buttoncontrol", "button"}


def elem_at(state, pos, role="active"):
    if not state or not pos:
        return None
    px, py = pos
    best, ba = None, 1e18
    for e in state.get("elements", []):
        if role and e.get("window_role") != role:
            continue
        b = e.get("bbox")
        if not b or len(b) != 4:
            continue
        if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
            a = (b[2] - b[0]) * (b[3] - b[1])
            if a < ba:
                best, ba = e, a
    if best is None and role == "active":
        return elem_at(state, pos, None)
    return best


def is_form(st, scope=INSURANCE_SCOPE):
    """Is this observation from the demonstrated window? Delegates to the scope."""
    return scope.is_target_window(st or {})


def n_listitems(st):
    return sum(1 for e in st.get("elements", [])
               if "listitem" in (e.get("type") or "").lower())


def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scope", default="insurance", choices=sorted(SCOPES),
                    help="Which scope's window markers identify the demonstrated "
                         "window (default: insurance).")
    ap.add_argument("--window-match", default=None,
                    help="Comma-separated title/app substrings, overriding --scope. "
                         "Use for a one-off app with no ScopeConfig of its own.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    src, dst = args.src, args.dst

    if args.window_match:
        markers = [m.strip().lower() for m in args.window_match.split(",") if m.strip()]
        scope = ScopeConfig(window_markers=markers)
    else:
        scope = SCOPES[args.scope]
    print(f"window markers: {scope.window_markers or '(none — every window accepted)'}")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    kept = drop_sel = drop_junk = drop_dupe = 0
    kept_generated = 0
    for sess in sorted(glob.glob(os.path.join(src, "session_*"))):
        files = sorted(glob.glob(os.path.join(sess, "live_step_*.json")))
        out = os.path.join(dst, os.path.basename(sess))
        os.makedirs(out, exist_ok=True)
        oi = 0
        prev_lbl = None
        for f in files:
            t = json.load(open(f, encoding="utf-8"))
            st = t.get("state", {})
            ns = t.get("next_state", {})
            m = t.get("mouse", {}).get("actions", [])
            k = t.get("keyboard", {}).get("actions", [])

            if m:
                # must be on the form window
                if not is_form(ns, scope) and not is_form(st, scope):
                    drop_junk += 1
                    continue
                # DROPDOWN SELECTION: dropdown was open when this click happened →
                # it's a value-pick, not navigation. THE KEY FIX.
                if n_listitems(st) > 0:
                    drop_sel += 1
                    continue
                tgt = elem_at(ns, m[0].get("position")) or {}
                ty = (tgt.get("type") or "").lower()
                lbl = tgt.get("label") or tgt.get("text") or ""
                if ty not in FIELD:
                    drop_junk += 1
                    continue
                if lbl and lbl == prev_lbl:   # combobox open + reopen on same field
                    drop_dupe += 1
                    continue
                prev_lbl = lbl
            elif k:
                pass   # keep typing (fills the form)
            else:
                drop_junk += 1
                continue

            json.dump(t, open(os.path.join(out, f"live_step_{oi:04d}.json"),
                              "w", encoding="utf-8"), ensure_ascii=False)
            oi += 1
            kept += 1
            if t.get("generated"):
                kept_generated += 1

    print(f"kept {kept}  |  dropped: dropdown-select={drop_sel}, "
          f"junk={drop_junk}, dupes={drop_dupe}  ->  {dst}")
    # Provenance, always, not only when mixed. A number that reads "0 generated"
    # is the evidence that a run was human-only; silence is not.
    print(f"provenance: {kept - kept_generated} human, {kept_generated} generated")
    if kept_generated and kept - kept_generated:
        print("  NOTE: this directory mixes both. A clone score measured on it "
              "is partly a measure of the generator, so hold a human session out "
              "and score against that separately.")


if __name__ == "__main__":
    main()
