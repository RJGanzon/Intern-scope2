"""Cleaning keeps the demonstrated window's clicks, whichever scope recorded them.

clean_demos.py drops clicks that did not happen on the task's own window - the
terminal, the recorder GUI, the source document. It decided that with a literal:

    return "insurance" in t or "insurance" in a or "data entry" in t

which is true of exactly one scope. A scope #2 recording (Chrome, titled "Grade
Encoding Portal") had every single click counted as junk, leaving a "clean"
directory of typing frames with no navigation signal in it at all - and the
script reports that as a large junk count, not as an error.

The window test now comes from ScopeConfig.window_markers, which is where
per-application knobs already live (tab names, section patterns, the record
delimiter). Scope #1 behaviour is unchanged, and is pinned here.

Run:  python -m pytest tests/scope2/test_clean_demos_scope.py -q
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))
sys.path.insert(0, str(REPO / "scripts"))

import clean_demos  # noqa: E402
from agent.scope import GRADE_PORTAL_SCOPE, INSURANCE_SCOPE, ScopeConfig  # noqa: E402


def click_step(window_title, label, pos=(50, 50), app=""):
    """One recorded click on an editable field in the named window."""
    state = {
        "window_title": window_title,
        "application": app,
        "elements": [{
            "type": "editcontrol", "label": label, "window_role": "active",
            "bbox": [pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10],
        }],
    }
    return {
        "state": state,
        "next_state": json.loads(json.dumps(state)),
        "mouse": {"actions": [{"position": list(pos)}]},
        "keyboard": {"actions": []},
    }


def write_session(root, steps):
    session = root / "session_20260820_120000"
    session.mkdir(parents=True, exist_ok=True)
    for i, step in enumerate(steps):
        (session / f"live_step_{i:04d}.json").write_text(
            json.dumps(step), encoding="utf-8")
    return session


def clean(tmp_path, steps, *args):
    src, dst = tmp_path / "raw", tmp_path / "clean"
    write_session(src, steps)
    clean_demos.main([str(src), str(dst), *args])
    return sorted((dst / "session_20260820_120000").glob("live_step_*.json"))


# ── the scope #1 contract, unchanged ─────────────────────────────────────────

def test_insurance_titles_still_pass():
    for title in ("Car Insurance Entry", "Data Entry Form"):
        assert INSURANCE_SCOPE.is_target_window({"window_title": title})


def test_the_scope1_junk_it_was_written_to_drop_is_still_dropped():
    for title in ("Untitled - Notepad", "Command Prompt", "Intern Recorder"):
        assert not INSURANCE_SCOPE.is_target_window({"window_title": title})


def test_default_cli_still_cleans_as_scope_1(tmp_path):
    """No --scope given: the insurance form's clicks survive, Notepad's do not."""
    kept = clean(tmp_path, [
        click_step("Car Insurance Entry", "First Name", (50, 50)),
        click_step("Untitled - Notepad", "source text", (60, 60)),
        click_step("Car Insurance Entry", "Last Name", (70, 70)),
    ])
    labels = [json.loads(f.read_text())["state"]["elements"][0]["label"] for f in kept]
    assert labels == ["First Name", "Last Name"]


# ── the scope #2 bug this fixes ──────────────────────────────────────────────

def test_a_browser_recording_survives_cleaning(tmp_path):
    """The whole defect in one test: without --scope these three clicks are all
    junk, and the navigation signal the script exists to preserve is gone."""
    steps = [
        click_step("Grade Encoding Portal - V0 Base", "Grade 0-100 Abad, Andrea A.",
                   (50, 50), app="browser"),
        click_step("Grade Encoding Portal - V0 Base", "Course Abad, Andrea A.",
                   (60, 60), app="browser"),
        click_step("Grade Encoding Portal - V0 Base", "Grade 0-100 Bautista, Ben B.",
                   (70, 70), app="browser"),
    ]
    assert clean(tmp_path, steps) == []              # what used to happen
    assert len(clean(tmp_path, steps, "--scope", "grade_portal")) == 3


def test_the_relabeled_variant_is_still_recognised():
    """v2_relabeled renames the visible heading to "Student Rating Sheet" to
    break anything keyed on it. The markers read the title, which still says
    Grade Encoding Portal, and cover the renamed heading besides."""
    assert GRADE_PORTAL_SCOPE.is_target_window(
        {"window_title": "Grade Encoding Portal - V2 Relabeled", "application": "browser"})
    assert GRADE_PORTAL_SCOPE.is_target_window({"window_title": "Student Rating Sheet"})


def test_scopes_do_not_accept_each_others_windows():
    portal = {"window_title": "Grade Encoding Portal - V0 Base", "application": "browser"}
    form = {"window_title": "Car Insurance Entry"}
    assert not INSURANCE_SCOPE.is_target_window(portal)
    assert not GRADE_PORTAL_SCOPE.is_target_window(form)


# ── the generic default ──────────────────────────────────────────────────────

def test_a_scope_with_no_markers_accepts_every_window():
    """An app nobody has configured has no marker to test. Accepting everything
    keeps its demos; dropping everything loses them silently, which is worse."""
    generic = ScopeConfig()
    assert generic.window_markers == []
    assert generic.is_target_window({"window_title": "Some New App"})
    assert generic.is_target_window({})


def test_window_match_overrides_the_scope_for_a_one_off_app(tmp_path):
    kept = clean(tmp_path,
                 [click_step("Payroll Console", "Net Pay", (50, 50))],
                 "--window-match", "payroll")
    assert len(kept) == 1


def test_window_match_is_case_insensitive_and_ignores_blanks(tmp_path):
    kept = clean(tmp_path,
                 [click_step("PAYROLL Console", "Net Pay", (50, 50))],
                 "--window-match", " Payroll , ")
    assert len(kept) == 1
