"""
test_clone.py — does the transformer reproduce the user's navigation?

Offline, pure transformer. No LLM, no agent crutches, no live form.
For each recorded frame in a session: feed the SAME state the user saw,
let the model predict a click, resolve it to a field, compare to the field
the user actually clicked. Reports per-step match + overall click-clone rate.

Usage:
    python scripts/test_clone.py                       # newest policy_nav session
    python scripts/test_clone.py <session_dir>
    python scripts/test_clone.py <session_dir> --model tasks/grade_portal/model.pt
"""
from __future__ import annotations
import os, sys, glob, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "components"))
sys.path.insert(0, _ROOT)

from intelligence.model.transformer import predict, _find_click_elem_idx  # noqa: E402

DEFAULT_MODEL = os.path.join(_ROOT, "tasks", "form_filling", "model.pt")


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


def label_of(e):
    return ((e.get("label") or e.get("text") or "").strip()) if e else "?"


def main():
    # Which checkpoint. Scope #1's was the only one that existed when this was
    # written, and scoring scope #2 against it would have produced a real number
    # for the wrong model rather than an error.
    args = [a for a in sys.argv[1:]]
    model = DEFAULT_MODEL
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
        if not os.path.isabs(model):
            model = os.path.join(_ROOT, model)
    if not os.path.exists(model):
        print(f"no checkpoint at {model}")
        return
    print(f"  model: {os.path.relpath(model, _ROOT)}")

    if args:
        sess = args[0]
    else:
        cands = glob.glob(os.path.join(_ROOT, "data", "demos", "policy_nav", "session_*"))
        sess = max(cands, key=os.path.getmtime) if cands else None
    if not sess or not os.path.isdir(sess):
        print(f"no session: {sess}")
        return

    files = sorted(glob.glob(os.path.join(sess, "live_step_*.json")))
    print(f"\n  session: {os.path.basename(sess)}  ({len(files)} frames)\n")
    print(f"  {'#':>4}  {'YOU clicked':24}  {'MODEL predicts':24}  match")
    print("  " + "-" * 64)

    history = []
    visited, cycle_first = set(), None
    match = total = 0
    _nav = [0, 0]   # [field-nav total, field-nav match]
    shown = 0

    for i, f in enumerate(files):
        t = json.load(open(f, encoding="utf-8"))
        state = t.get("state", {})
        nstate = t.get("next_state", {})
        m = t.get("mouse", {}).get("actions", [])
        if not m:
            continue
        pos = m[0].get("position")
        els = state.get("elements", [])
        # ACTUAL clicked element index — exactly how training resolves the target
        actual_idx = _find_click_elem_idx(t.get("mouse", {}), state, len(els) or 128)
        actual = label_of(els[actual_idx]) if 0 <= actual_idx < len(els) \
            else label_of(elem_at(nstate, pos) or elem_at(state, pos))

        # model prediction (pure: state + short history, no hand-fed signals)
        pred = predict(state=state, history=history[-3:], model_path=model)
        ei = pred.get("click_elem_idx", -1)
        predicted = label_of(els[ei]) if 0 <= ei < len(els) else "?"
        # index-level match (matches training's click_acc), with label fallback
        idx_match = (actual_idx >= 0 and ei == actual_idx)

        # is the actual click a NAVIGABLE FIELD, or a value-pick (dropdown item)?
        ae = elem_at(nstate, pos) or elem_at(state, pos)
        atype = (ae.get("type") or "").lower() if ae else ""
        is_field = atype in ("editcontrol", "input", "comboboxcontrol", "combobox",
                              "checkboxcontrol", "checkbox", "buttoncontrol", "button")

        ok = idx_match or (actual == predicted and actual not in ("", "?"))
        if actual not in ("", "?"):
            total += 1
            match += 1 if ok else 0
            if is_field:
                _nav[0] += 1
                _nav[1] += 1 if ok else 0
        if shown < 45:
            tag = "" if is_field else "  (value-pick, not nav)"
            print(f"  {i:>4}  {actual[:22]:22}  {predicted[:22]:22}  {'OK' if ok else 'x'}{tag}")
            shown += 1

        # advance history + visited (same cycle logic as the dataset)
        res = state.get("screen_resolution", [1920, 1080])
        W = float(res[0]) or 1920.0
        H = float(res[1]) or 1080.0
        history.append({
            "state": state,
            "action_type": "click",
            "click_xy": [pos[0] / W, pos[1] / H],
            "key_count": 0,
        })
        if actual and actual not in ("", "?"):
            if actual == cycle_first and len(visited) > 1:
                visited, cycle_first = set(), None
            if cycle_first is None:
                cycle_first = actual
            visited.add(actual)

    print("  " + "-" * 64)
    rate = (match / total * 100) if total else 0.0
    nav_rate = (_nav[1] / _nav[0] * 100) if _nav[0] else 0.0
    print(f"\n  OVERALL CLONE:        {match}/{total} = {rate:.1f}%  (incl. value-picks)")
    print(f"  NAVIGATION CLONE:     {_nav[1]}/{_nav[0]} = {nav_rate:.1f}%  "
          f"(field clicks only — the real question)\n")


if __name__ == "__main__":
    main()
