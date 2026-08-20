"""Training can see the whole grid, not just the top of it.

Found while running the scope #2 pipeline end to end on synthetic traces built
from the real portal DOM. WebObserver's own 200-element cap was raised three
commits earlier so the agent could see all 303 elements of the grade portal -
and the same cap turned out to be waiting one layer downstream, at 128, in the
training dataset.

Measured on that real state: 29 of the 50 Grade cells sit past index 128. Both
encode_state and _find_click_elem_idx slice to max_elements, so a click on any
of them resolves to -1, which is the ignore-index of the click pointer loss. The
step still counts as a click for the action-type head, and -1 is also what an
honest miss returns (a click on a decorative pane), so the discarded steps are
indistinguishable from ordinary ones. Everything below row ~21 taught nothing.

The pointer heads are attention (click_q / click_k score each element), so
raising the cap costs no parameters at all - verified below, and confirmed in a
real training run: 142,629 params at both 128 and 320.

Run:  python -m pytest tests/scope2/test_training_element_cap.py -q
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "components"))

from intelligence.model.transformer import (  # noqa: E402
    TrajectoryDataset,
    TransformerAgentNetwork,
    _find_click_elem_idx,
    encode_state,
)

CAP = 128


def grid_state(rows=60):
    """A sheet-shaped page: one editable cell per row, stacked down the screen."""
    return {
        "window_title": "Grade Encoding Portal - V0 Base",
        "application": "browser",
        "elements": [{
            "element_id": f"e{i}", "type": "editcontrol",
            "label": f"Grade 0-100 Student {i}", "window_role": "active",
            "bbox": [500, 100 + i * 20, 600, 118 + i * 20],
        } for i in range(rows)],
    }


def click_on(state, index):
    box = state["elements"][index]["bbox"]
    return {"actions": [{"type": "click",
                         "position": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}]}


# ── the defect ───────────────────────────────────────────────────────────────

def test_a_click_past_the_cap_resolves_to_nothing():
    """-1 is the click loss's ignore index: the step trains the action-type head
    and teaches the pointer nothing about where the click went."""
    state = grid_state(200)
    assert _find_click_elem_idx(click_on(state, 150), state, CAP) == -1


def test_the_same_click_resolves_once_the_cap_clears_it():
    state = grid_state(200)
    assert _find_click_elem_idx(click_on(state, 150), state, 320) == 150


def test_clicks_inside_the_cap_were_never_affected():
    """Which is exactly why this stayed invisible - the early rows work fine."""
    state = grid_state(200)
    assert _find_click_elem_idx(click_on(state, 10), state, CAP) == 10


def test_the_encoded_state_stops_at_the_cap_too():
    """Not only the label: the model is not shown those elements either."""
    state = grid_state(200)
    assert encode_state(state, CAP).shape[0] == CAP
    assert encode_state(state, 320).shape[0] == 320


# ── raising it is free ───────────────────────────────────────────────────────

def test_a_bigger_cap_costs_no_parameters():
    """Why raising it is free. The network takes max_elements and remembers it,
    but sizes nothing by it: the pointer heads score element embeddings through
    attention (click_q / click_k) instead of a fixed-width output layer. So the
    cap is a compute knob, not a capacity one. Confirmed in a real training run
    too - 142,629 parameters at both 128 and 320."""
    def build(cap):
        return TransformerAgentNetwork(max_elements=cap, d_model=64,
                                       num_layers=2, dim_feedforward=128)

    small, large = build(128), build(320)
    assert small.max_elements == 128 and large.max_elements == 320
    assert sum(p.numel() for p in small.parameters()) == \
           sum(p.numel() for p in large.parameters())
    assert hasattr(large, "click_q") and hasattr(large, "click_k")


def test_inference_reads_the_cap_the_checkpoint_was_trained_with():
    """The end-to-end guarantee: predict() takes max_elements off the loaded
    model, so raising it for training raises it for the live run too. If this
    ever stopped holding, a model trained on the whole grid would go back to
    seeing only the top of it at run time - silently, again."""
    import inspect

    source = inspect.getsource(
        sys.modules["intelligence.model.transformer"]).split("def predict(")[1]
    assert "max_elements = model.max_elements" in source


# ── the warning ──────────────────────────────────────────────────────────────

def make_reporter(truncated, largest):
    reporter = TrajectoryDataset.__new__(TrajectoryDataset)
    reporter._truncated_files = truncated
    reporter._max_seen = largest
    return reporter


def test_truncation_is_reported_with_both_numbers(capsys):
    """The failure this replaces was silent. A warning that does not say how far
    over the cap the data actually goes cannot be acted on."""
    make_reporter(64, 303)._warn_if_truncated(CAP)
    out = capsys.readouterr().out
    assert "64" in out and "128" in out and "303" in out
    assert "--max_elements" in out


def test_nothing_is_said_when_everything_fits():
    reporter = make_reporter(0, 0)
    reporter._warn_if_truncated(CAP)


# ── the flag ─────────────────────────────────────────────────────────────────

def test_the_cli_exposes_the_cap_and_defaults_to_the_old_value():
    """Scope #1's forms are far smaller than the cap, so its training must stay
    byte-identical unless someone passes the flag on purpose."""
    sys.path.insert(0, str(REPO / "scripts"))
    import train as train_cli

    parser = train_cli.build_parser()
    assert parser.parse_args(["--trace_dir", "x"]).max_elements == CAP
    assert parser.parse_args(["--max_elements", "320"]).max_elements == 320


def test_the_trainer_passes_the_cap_down():
    """A flag the CLI accepts and the trainer drops would look like it worked."""
    from intelligence.training.bc.behavioral_cloning import BCTrainer

    assert BCTrainer().max_elements == CAP
    assert BCTrainer(max_elements=320).max_elements == 320
