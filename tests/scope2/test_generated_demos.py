"""Generated demonstrations are real states, and they are labelled as generated.

The panel approved synthesising training data to accelerate training. That makes
generated traces legitimate; it does not make them interchangeable with human
ones, and the difference has to survive being copied between folders. So the
marker rides on every step rather than on a manifest, and cleaning reports the
split whether or not anything is mixed.

The quality bar these tests exist to hold is progression. A first attempt reused
one frozen state for every step, which is worthless: `is_filled` never changes,
so a model cannot learn that a filled field is done - the exact failure
DEVELOPERS.md records from Scope #1's early data, where the model looped because
state carried labels but never values. Snapshotting before AND after each action
is what fixes it, and a test that only checked "files were written" would have
passed on the broken version.

Run:  python -m pytest tests/scope2/test_generated_demos.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "components" / "scope2"))
sys.path.insert(0, str(REPO / "scripts"))

pytest.importorskip("playwright.sync_api")

import generate_portal_demos as gen  # noqa: E402


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Two students, generated for real. Slow enough to build once."""
    from executor.scanner import CHROMIUM

    if not CHROMIUM.exists():
        pytest.skip(f"no chromium at {CHROMIUM}")
    out = tmp_path_factory.mktemp("generated")
    gen.main(["--students", "2", "--out", str(out)])

    dirs = sorted(out.glob("session_gen_*"))
    assert dirs, "the generator wrote no session"
    steps = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(dirs[0].glob("live_step_*.json"))]
    assert steps, "the session has no steps"
    return steps


def filled(state):
    return sum(1 for e in state.get("elements") or []
               if str(e.get("value") or "").strip())


def clicks(steps):
    return [s for s in steps if s["mouse"]["actions"]
            and s["mouse"]["actions"][0]["type"] == "click"]


def under(state, pos):
    best, area = None, float("inf")
    for el in state.get("elements") or []:
        b = el.get("bbox") or []
        if len(b) == 4 and b[0] <= pos[0] <= b[2] and b[1] <= pos[1] <= b[3]:
            a = (b[2] - b[0]) * (b[3] - b[1])
            if a < area:
                best, area = el, a
    return best


# ── the states are real ──────────────────────────────────────────────────────

def test_every_state_carries_the_whole_page(session):
    """A real portal snapshot, not a sketch of one - the grid is 303 elements."""
    for step in session:
        assert len(step["state"]["elements"]) > 250


def test_the_sheet_fills_up_as_the_session_goes_on(session):
    """The bar the first attempt failed. Without progression the dataset cannot
    teach that a filled field is finished."""
    start = filled(session[0]["state"])
    end = filled(session[-1]["next_state"])
    assert end > start, f"nothing was filled: {start} -> {end}"


def test_no_step_reuses_its_own_before_state_as_its_after_state(session):
    """Specifically what "frozen" looked like: state and next_state identical."""
    typed = [s for s in session if s["keyboard"]["actions"]]
    assert typed, "the session typed nothing"
    changed = [s for s in typed if s["state"] != s["next_state"]]
    assert len(changed) == len(typed), "a typing step left the page unchanged"


def test_every_click_lands_on_an_element(session):
    """The failure that made the first two human recordings unusable. Generated
    data can hit it just as easily, since it uses the same geometry."""
    for step in clicks(session):
        pos = step["mouse"]["actions"][0]["position"]
        assert under(step["state"], pos) is not None, f"click at {pos} hit nothing"


def test_clicks_and_the_values_that_follow_agree(session):
    """A click on Course followed by a year number would be coherent-looking
    nonsense. The pairing is the whole content of the demonstration."""
    expected = {"Course": lambda v: v.startswith("BS "),
                "Year": lambda v: len(v) == 1 and v.isdigit(),
                "Grade": lambda v: v.replace(".", "").isdigit() and len(v) > 1}
    pairs = 0
    for i, step in enumerate(session[:-1]):
        if not (step["mouse"]["actions"]
                and step["mouse"]["actions"][0]["type"] == "click"):
            continue
        target = under(step["state"], step["mouse"]["actions"][0]["position"])
        label = (target or {}).get("label", "")
        nxt = session[i + 1]
        if not nxt["keyboard"]["actions"]:
            continue
        value = "".join(s.get("pasted_text") or ""
                        for s in nxt["keyboard"]["actions"][0].get("strokes", []))
        for column, ok in expected.items():
            if label.startswith(column):
                assert ok(value), f"{label!r} was given {value!r}"
                pairs += 1
    assert pairs >= 4, f"only {pairs} click/value pairs checked"


def test_the_portal_fills_remarks_without_being_typed_into(session):
    """The derived column, captured in the data for free - and evidence the
    generator drives the real page rather than writing values into a model
    of it."""
    typed = {
        "".join(s.get("pasted_text") or "" for s in step["keyboard"]["actions"][0]["strokes"])
        for step in session if step["keyboard"]["actions"]
    }
    remarks = [el for el in session[-1]["next_state"]["elements"]
               if (el.get("label") or "").startswith("Remarks")
               and str(el.get("value") or "").strip()]
    assert remarks, "no Remarks was derived"
    for el in remarks:
        assert el["value"] not in typed, "Remarks was typed, not derived"


# ── the data says what it is ─────────────────────────────────────────────────

def test_every_single_step_is_marked_generated(session):
    """On the step, not a manifest: a step copied into another folder takes its
    provenance with it."""
    assert all(step.get("generated") is True for step in session)


def test_the_marker_records_how_it_was_made(session):
    meta = session[0]["generator"]
    assert meta["order"] == "course,year,grade"
    assert meta["variant"] == "v0_base"
    assert meta["script"].endswith("generate_portal_demos.py")


def test_generated_sessions_have_their_own_directory_name(tmp_path):
    writer = gen.SessionWriter(str(tmp_path), {"index": 0})
    assert Path(writer.dir).name.startswith("session_gen_")


def test_the_default_output_is_not_the_human_folder():
    """Mixing is a decision to make visibly, on the train command line - not by
    writing into the folder human recordings live in."""
    args = gen.parse_args([])
    assert "human" not in args.out
    assert "generated" in args.out


def test_cleaning_reports_the_split(tmp_path, capsys):
    """Reported even when nothing is mixed: "0 generated" is the evidence that a
    run was human-only, and silence is not evidence of anything."""
    import clean_demos

    session_dir = tmp_path / "raw" / "session_20260821_000000"
    session_dir.mkdir(parents=True)
    # Two DIFFERENT cells: consecutive clicks on the same field are dropped as
    # a duplicate, which would leave one step and nothing to report a split on.
    state = {
        "window_title": "Grade Encoding Portal", "application": "browser",
        "elements": [
            {"type": "editcontrol", "label": "Grade 0-100 A", "bbox": [0, 0, 20, 20],
             "window_role": "active"},
            {"type": "editcontrol", "label": "Grade 0-100 B", "bbox": [40, 0, 60, 20],
             "window_role": "active"},
        ],
    }
    for i, (generated, x) in enumerate(((True, 10), (False, 50))):
        step = {"state": state, "next_state": state,
                "mouse": {"actions": [{"position": [x, 10]}]},
                "keyboard": {"actions": []}}
        if generated:
            step["generated"] = True
        (session_dir / f"live_step_{i:04d}.json").write_text(json.dumps(step), encoding="utf-8")

    clean_demos.main([str(tmp_path / "raw"), str(tmp_path / "clean"),
                      "--scope", "grade_portal"])
    out = capsys.readouterr().out
    assert "provenance: 1 human, 1 generated" in out
    assert "mixes both" in out
