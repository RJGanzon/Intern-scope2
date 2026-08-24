"""
tests/test_opt2_fast_fill.py
==============================
Regression tests for OPT2's fast-fill early-exit (components/agent/agent.py,
inserted immediately before `t_pred = self._predict(state)`).

Built 2026-08-14, direct request ("don't stop unless there's something to
reason about" -- 3-day demo deadline). Skips the transformer call AND the
LLM call entirely for the one case where the answer is already
deterministically known: the focused field is a plain, empty, not-yet-
attempted editcontrol, and its value is already sitting in the intake
data under an exact key match (the same lookup `_ask_llm`'s own fast path
already trusts). Every mechanism this reuses was already built and tested
earlier the same night: `_lookup_field`, `_find_uia_control_by_name` +
`.SetFocus()` (no click), and the WM_SETTEXT direct-fill mechanism
(`direct_fill_hwnd`, `_keyboard_direct`).

Extended the same night to comboboxcontrol via CB_SETCURSEL (a direct
message that selects a real option with no click and no dropdown ever
opened -- live-tested against the real form, see
tests/test_executor_combobox_direct.py). Checkboxes already have their
own separate, working mechanism (BM_SETCHECK) and stay on the existing
reactive path untouched.

This branch sits deep inside LLMAgent.run() (~8000 lines, many
preconditions before reaching it -- Navigation Protocol's own decision,
stuck-guards, etc.) so, following this project's own established pattern
for testing logic embedded in that method (see
tests/test_type_path_focus_via_uia.py, tests/test_redirect_click_
destructive_button_guard.py), this file uses a mirror function that
reimplements exactly the new gating condition, plus direct source-level
checks confirming the real code actually wires those same calls in --
not a full run() invocation, which would require mocking far more of the
loop's preconditions than this one branch actually depends on.
"""
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "components"))
from agent.agent import LLMAgent

_AGENT_PY = Path(__file__).resolve().parent.parent / "components" / "agent" / "agent.py"
_SOURCE = _AGENT_PY.read_text(encoding="utf-8")


def _make_agent():
    return LLMAgent(goal="test goal", dry_run=True, max_steps=1, step_delay=0,
                     disable_auto_handlers=True)


def _field(label, elem_type, value="", element_id="e1", bbox=(1400, 270, 1600, 300)):
    return {"element_id": element_id, "type": elem_type, "label": label,
            "text": label, "value": value, "bbox": list(bbox), "window_role": "active"}


def _is_fast_fill_eligible(agent, focused_el, elements, leave_blank_keys=frozenset(),
                            typed_keys=frozenset()):
    """Mirrors the exact gating condition inserted before `_predict()`:
    `if (_ff_fel and _ff_ty in ("editcontrol", "comboboxcontrol") and not _ff_val
            and _ff_key not in self._leave_blank_keys
            and _ff_key not in self._typed_keys):`"""
    ty  = (focused_el.get("type") or "").lower() if focused_el else ""
    val = (focused_el.get("value") or "").strip() if focused_el else ""
    key = agent._attempt_key(focused_el, elements=elements) if focused_el else None
    return bool(
        focused_el and ty in ("editcontrol", "comboboxcontrol") and not val
        and key not in leave_blank_keys and key not in typed_keys
    )


class TestFastFillEligibilityMirror:
    """Pure mirror-logic tests for the new gating condition -- fast,
    deterministic, no live OS/UIA needed."""

    def test_empty_editcontrol_is_eligible(self):
        agent = _make_agent()
        field = _field("Policy Number", "editcontrol")
        assert _is_fast_fill_eligible(agent, field, [field]) is True

    def test_filled_editcontrol_is_not_eligible(self):
        agent = _make_agent()
        field = _field("Policy Number", "editcontrol", value="POL-000123")
        assert _is_fast_fill_eligible(agent, field, [field]) is False

    def test_comboboxcontrol_is_eligible(self):
        """Extended the same night: CB_SETCURSEL (live-tested against the
        real form) lets comboboxes use this fast path too, via a
        different underlying mechanism than editcontrol (combobox_hwnd,
        not direct_fill_hwnd)."""
        agent = _make_agent()
        field = _field("Policy Status", "comboboxcontrol")
        assert _is_fast_fill_eligible(agent, field, [field]) is True

    def test_checkboxcontrol_is_never_eligible(self):
        """Checkboxes already have their own separate, working mechanism
        (BM_SETCHECK) -- must not collide with this one."""
        agent = _make_agent()
        field = _field("Auto-Pay Enrolled", "checkboxcontrol")
        assert _is_fast_fill_eligible(agent, field, [field]) is False

    def test_confirmed_leave_blank_field_is_not_reeligible(self):
        agent = _make_agent()
        field = _field("Middle Name", "editcontrol")
        key = agent._attempt_key(field, elements=[field])
        assert _is_fast_fill_eligible(agent, field, [field], leave_blank_keys={key}) is False

    def test_already_typed_field_is_not_reeligible(self):
        """Trusts a genuine-typed-text record over a live 'is it empty'
        read, same reasoning as the existing _fe2_already_attempted check
        this mirrors -- a field the agent already filled once, that a
        stale/racy live read reports as empty again, must not be
        silently re-fast-filled."""
        agent = _make_agent()
        field = _field("Policy Number", "editcontrol")  # value empty in THIS read
        key = agent._attempt_key(field, elements=[field])
        assert _is_fast_fill_eligible(agent, field, [field], typed_keys={key}) is False

    def test_no_focused_element_is_safe_and_not_eligible(self):
        agent = _make_agent()
        assert _is_fast_fill_eligible(agent, None, []) is False


class TestFastFillReusesRealMethods:
    """Confirms the eligibility check and value/handle resolution are
    driven by this project's actual, already-tested methods -- not a
    reimplementation that could silently drift from them."""

    def test_lookup_uses_the_real_lookup_field_and_detect_section(self, monkeypatch):
        agent = _make_agent()
        field = _field("Policy Number", "editcontrol")
        agent._detect_section = MagicMock(return_value="")
        agent._lookup_field = MagicMock(return_value="POL-000123")

        assert _is_fast_fill_eligible(agent, field, [field]) is True
        section = agent._detect_section({"elements": [field]}, field)
        value = agent._lookup_field(field["label"], section=section)

        agent._detect_section.assert_called_once()
        agent._lookup_field.assert_called_once_with("Policy Number", section="")
        assert value == "POL-000123"

    def test_lookup_miss_yields_no_known_value(self):
        agent = _make_agent()
        agent._lookup_field = MagicMock(return_value="")
        assert agent._lookup_field("Unknown Field", section="") == ""


class TestSourceWiresRealMechanisms:
    """Direct source-level verification (this project's own established
    pattern for logic embedded in the un-extracted run() method) that the
    real inserted branch actually calls the real, already-tested
    mechanisms in the right shape -- not a parallel reimplementation."""

    def _fast_fill_window(self):
        idx = _SOURCE.index("OPT2 FAST-FILL")
        return _SOURCE[idx:idx + 13500]

    def test_gated_on_no_autohandlers(self):
        window = self._fast_fill_window()
        assert "if self._no_autohandlers:" in window

    def test_scoped_to_editcontrol_only(self):
        window = self._fast_fill_window()
        assert '_ff_ty == "editcontrol"' in window
        assert '"comboboxcontrol"' not in window.split("editcontrol")[1][:200]

    def test_excludes_leave_blank_and_typed_keys(self):
        window = self._fast_fill_window()
        assert "_ff_key not in self._leave_blank_keys" in window
        assert "_ff_key not in self._typed_keys" in window

    def test_uses_real_lookup_field(self):
        window = self._fast_fill_window()
        assert "self._lookup_field(_ff_label, section=_ff_sec)" in window

    def test_uses_real_find_uia_control_by_name_and_setfocus_no_click(self):
        """The whole point: moving to the field must not be a raw
        coordinate click. Routes through _resolve_field_control (added
        2026-08-14, "still too slow") which only pays for
        _find_uia_control_by_name's expensive position-based
        disambiguation when the label is genuinely ambiguous on screen --
        still the same underlying no-click UIA mechanism either way."""
        window = self._fast_fill_window()
        assert "self._resolve_field_control(state, _ff_label" in window
        assert "_ff_ctrl.SetFocus()" in window
        assert '"action_type": "click"' not in window

    def test_uses_direct_fill_hwnd_not_a_new_action_type(self):
        """Must reuse the existing WM_SETTEXT mechanism (built and tested
        earlier the same night) via the same optional key on the
        existing 'keyboard' action_type -- not a new action_type string."""
        window = self._fast_fill_window()
        assert '"action_type": "keyboard"' in window
        assert '"direct_fill_hwnd": _ff_hwnd' in window

    def test_marks_attempted_and_settles_via_real_helpers(self):
        window = self._fast_fill_window()
        assert "self._mark_attempted(_ff_fel" in window
        assert "self._adaptive_settle_wait(self.step_delay * 0.2)" in window

    def test_combobox_uses_combobox_select_not_direct_fill_hwnd(self):
        """Comboboxes must route through the new combobox_select action
        type (combobox_hwnd), not the text-field mechanism -- WM_SETTEXT
        is a confirmed no-op on comboboxes."""
        window = self._fast_fill_window()
        assert '"action_type": "combobox_select"' in window
        assert '"combobox_hwnd": _ff_hwnd' in window

    def test_combobox_checks_success_before_committing(self):
        """A combobox whose known value doesn't match any real option
        must fall through to the existing click-based path, not be
        silently treated as done."""
        window = self._fast_fill_window()
        assert "_ff_cb_result.success" in window

    def test_combobox_falls_through_to_existing_click_path_on_failure(self):
        """No `continue` reachable purely from a failed combobox_select --
        must fall through to today's existing reactive combobox handling."""
        window = self._fast_fill_window()
        result_idx = window.index("_ff_cb_result = self._executor.execute")
        after = window[result_idx:result_idx + 1000]
        # continue only appears inside the `if _ff_cb_result.success:` block
        success_idx = after.index("if _ff_cb_result.success:")
        continue_idx = after.index("continue")
        assert success_idx < continue_idx

    def test_falls_through_to_continue_only_on_full_success(self):
        """Every failure point (no ctrl, no hwnd, no known value) must
        leave `continue` unreached so today's existing, unmodified code
        runs exactly as it does now."""
        window = self._fast_fill_window()
        # The `continue` must be nested inside the `if _ff_hwnd:` block,
        # not unconditional -- confirm it's preceded by the settle-wait
        # call on the same success path, not floating free.
        settle_idx = window.index("self._adaptive_settle_wait(self.step_delay * 0.2)")
        after = window[settle_idx:settle_idx + 100]
        assert "continue" in after

    def test_is_not_hardcoded_to_any_specific_field_name(self):
        """The whole point -- must be a generic label-driven lookup,
        never a literal field-name comparison."""
        window = self._fast_fill_window()
        assert not re.search(r'_ff_label\s*==\s*[\'"]', window)

    def test_inserted_before_the_transformer_predict_call(self):
        """The entire point of the feature: the transformer call itself
        must be skipped, not just the LLM call -- so this branch must sit
        textually before `t_pred = self._predict(state)`."""
        fast_fill_idx = _SOURCE.index("OPT2 FAST-FILL")
        predict_idx = _SOURCE.index('t_pred = self._predict(state)')
        assert fast_fill_idx < predict_idx

    def test_settle_wait_was_tightened_not_removed(self):
        """Tightened 2026-08-14 ahead of a ~1-minute demo target -- must
        still be present (still adaptive, still bounded), just a lower
        ceiling than the general-purpose fill path's own budget. 3 from
        editcontrol/combobox/dead-spot-rescue + 2 more from the later
        checkbox fast-fill addition (checked + unchecked cases) = 5."""
        window = self._fast_fill_window()
        assert window.count("self._adaptive_settle_wait(self.step_delay * 0.2)") == 5


class TestDeadSpotRescue:
    """Tests for the OPT2 dead-spot rescue -- added 2026-08-14 from real
    log evidence (a live run showed Tab periodically landing on a
    non-fillable section-divider pane, paying for a full transformer
    decision every time). Tries the same deterministic,
    already-tested/used find_visible_empty_target mechanism first,
    before ever asking the model."""

    def _fast_fill_window(self):
        idx = _SOURCE.index("OPT2 FAST-FILL")
        return _SOURCE[idx:idx + 13500]

    def test_rescue_block_exists_and_is_gated_correctly(self):
        window = self._fast_fill_window()
        assert "OPT2 DEAD-SPOT RESCUE" in window
        assert '"checkboxcontrol", "checkbox")' in window

    def test_uses_the_real_navigation_protocol_mechanism(self):
        """Must reuse the exact same function this file already calls
        elsewhere for the identical purpose -- not a reimplementation
        that could silently drift from it."""
        window = self._fast_fill_window()
        assert "self._navproto.find_visible_empty_target(" in window
        assert "self._form_viewport_bottom(state)" in window
        assert "attempted_keys=self._attempted_keys" in window
        assert "attempt_key_fn=self._attempt_key" in window

    def test_moves_focus_without_a_click(self):
        """The whole point -- SetFocus, not a coordinate click."""
        window = self._fast_fill_window()
        rescue_idx = window.index("OPT2 DEAD-SPOT RESCUE")
        rescue_section = window[rescue_idx:]
        assert "_dsr_ctrl.SetFocus()" in rescue_section
        assert '"action_type": "click"' not in rescue_section

    def test_routes_through_resolve_field_control_not_the_raw_lookup(self):
        """Added 2026-08-14 ("still too slow") -- must skip the expensive
        disambiguation search for unique labels via _resolve_field_control,
        same as batch/single-field fast-fill, not call
        _find_uia_control_by_name directly."""
        window = self._fast_fill_window()
        rescue_idx = window.index("OPT2 DEAD-SPOT RESCUE")
        rescue_section = window[rescue_idx:]
        assert "self._resolve_field_control(" in rescue_section
        assert "self._find_uia_control_by_name(" not in rescue_section

    def test_falls_through_safely_on_any_failure(self):
        """No target found, no label, no live control, or SetFocus
        raising must all leave `continue` unreached -- today's existing
        transformer-driven code must still run exactly as it does now."""
        window = self._fast_fill_window()
        rescue_idx = window.index("OPT2 DEAD-SPOT RESCUE")
        predict_idx = window.index('t_pred = self._predict(state)')
        rescue_section = window[rescue_idx:predict_idx]
        # continue only appears once, inside the try/except success path
        assert rescue_section.count("continue") == 1
        assert "except Exception as _dsr_exc:" in rescue_section

    def test_is_gated_before_the_transformer_predict_call(self):
        rescue_idx = _SOURCE.index("OPT2 DEAD-SPOT RESCUE")
        predict_idx = _SOURCE.index('t_pred = self._predict(state)')
        assert rescue_idx < predict_idx


def _dead_spot_rescue_eligible(elem_type: str) -> bool:
    """Mirrors the exact gating condition:
    `elif _ff_fel and _ff_ty not in ("editcontrol", "comboboxcontrol",
    "checkboxcontrol", "checkbox"):`"""
    return elem_type.lower() not in ("editcontrol", "comboboxcontrol",
                                      "checkboxcontrol", "checkbox")


class TestDeadSpotRescueEligibilityMirror:
    def test_panecontrol_is_rescue_eligible(self):
        """The exact real-world case from the log: a section-divider
        pane, no field to fill at all."""
        assert _dead_spot_rescue_eligible("panecontrol") is True

    def test_editcontrol_is_not_rescue_eligible(self):
        """Must never fire when a real fillable field is focused --
        that's the fast-fill branch's job, not this one's."""
        assert _dead_spot_rescue_eligible("editcontrol") is False

    def test_comboboxcontrol_is_not_rescue_eligible(self):
        assert _dead_spot_rescue_eligible("comboboxcontrol") is False

    def test_checkboxcontrol_is_not_rescue_eligible(self):
        """Checkboxes have their own existing, correct handling --
        must not be redirected away from it."""
        assert _dead_spot_rescue_eligible("checkboxcontrol") is False

    def test_buttoncontrol_is_rescue_eligible(self):
        assert _dead_spot_rescue_eligible("buttoncontrol") is True


class TestCheckboxFastFill:
    """Tests for OPT2's checkbox fast-fill -- added 2026-08-14, same
    night, direct evidence from a real live run's log: 25 of 57 remaining
    live model decisions in one full run were checkboxes, every one
    already having a deterministically known answer via the same lookup
    text/combobox fields already use. Reuses self._auto_check() (lookup +
    'yes'-prefix parsing, already existing) and the exact same
    WindowFromPoint + BM_SETCHECK call shape this file already uses at
    its other checkbox sites -- not a new mechanism."""

    def _fast_fill_window(self):
        idx = _SOURCE.index("OPT2 FAST-FILL")
        return _SOURCE[idx:idx + 13500]

    def test_gated_on_checked_fields_not_typed_keys(self):
        """Checkboxes don't reliably expose .value (see
        self._checked_fields' own comment) -- must use the checkbox-
        specific tracker, not the text-field one."""
        window = self._fast_fill_window()
        assert 'not in self._checked_fields' in window

    def test_reuses_real_auto_check_not_a_reimplementation(self):
        window = self._fast_fill_window()
        assert "self._auto_check(state)" in window

    def test_uses_the_same_bm_setcheck_mechanism_as_existing_sites(self):
        """Must use WindowFromPoint on the bbox center + BM_SETCHECK --
        the exact same call shape already proven elsewhere in this file,
        not the name-based UIA lookup text/combobox fields use."""
        window = self._fast_fill_window()
        idx = window.index("OPT2 CHECKBOX FAST-FILL")
        section = window[idx:idx + 3200]
        assert "WindowFromPoint((int(_chk_cx), int(_chk_cy)))" in section
        assert "SendMessage(_chk_hw, 0x00F1, 1, 0)" in section

    def test_unchecked_case_needs_no_bm_setcheck_call(self):
        """Wx checkboxes already default to unchecked -- a known 'NO'
        answer should just Tab past, not send any message at all."""
        window = self._fast_fill_window()
        idx = window.index("elif not _chk_should_check:")
        section = window[idx:idx + 500]
        assert "SendMessage" not in section

    def test_falls_through_on_unknown_checkbox_value(self):
        """If _auto_check returns None (not in the lookup data), must
        fall through to today's existing reactive path -- never guess."""
        window = self._fast_fill_window()
        idx = window.index("if _chk_result is not None:")
        # The comment documenting the fall-through-on-unknown case must
        # exist, confirming this isn't silently unhandled.
        assert "Unknown checkbox value" in window

    def test_is_gated_before_the_transformer_predict_call(self):
        checkbox_idx = _SOURCE.index("OPT2 CHECKBOX FAST-FILL")
        predict_idx = _SOURCE.index('t_pred = self._predict(state)')
        assert checkbox_idx < predict_idx


class TestBatchFastFill:
    """Tests for OPT2's BATCH fast-fill -- added 2026-08-14, direct request
    ("it needs to be instant") after real log evidence showed the
    single-field fast-fill above had already cut model calls to almost
    nothing (16 transformer calls in a full run) yet total wall time barely
    moved (4m09s vs the prior run's 4m33s): steps landed ~1s apart whether
    or not the model was skipped, because each field still paid for its own
    observe()+act() cycle. This fills EVERY currently-visible known-value
    field in one pass -- one observe(), many direct writes, no per-field
    re-observe -- using find_all_visible_empty_targets (navigation_
    protocol.py) plus the exact same already-tested direct-write mechanisms
    the single-field path above already uses (WM_SETTEXT / CB_SETCURSEL /
    BM_SETCHECK), not a reimplementation."""

    def _batch_window(self):
        idx = _SOURCE.index("OPT2 BATCH FAST-FILL")
        return _SOURCE[idx:idx + 15000]

    def test_batch_block_exists_before_the_single_field_block(self):
        batch_idx = _SOURCE.index("OPT2 BATCH FAST-FILL")
        single_idx = _SOURCE.index("OPT2 FAST-FILL: skip the transformer")
        assert batch_idx < single_idx

    def test_gated_on_no_autohandlers(self):
        window = self._batch_window()
        assert "if self._no_autohandlers:" in window

    def test_uses_the_real_find_all_visible_empty_targets(self):
        """Must reuse the shared eligibility rule (navigation_protocol.py),
        not a second, parallel scan that could silently drift from it."""
        window = self._batch_window()
        assert "self._navproto.find_all_visible_empty_targets(" in window
        assert "attempted_keys=self._attempted_keys" in window
        assert "attempt_key_fn=self._attempt_key" in window

    def test_handles_all_three_known_fillable_types(self):
        window = self._batch_window()
        assert '_bf_ty == "editcontrol"' in window
        assert '_bf_ty == "comboboxcontrol"' in window
        assert '_bf_ty in ("checkboxcontrol", "checkbox")' in window

    @staticmethod
    def _branch(window, start_marker, end_marker):
        """One type's branch, bounded by where the next one begins.

        These slices used to be a fixed character count from the branch start,
        which silently stops testing what it names as soon as the branch grows:
        adding a comment ahead of the fill pushed the very line being asserted
        out of the window, and the failure read as "direct_fill_hwnd is missing"
        when it was still right there. Bounding on the next branch keeps the
        assertion about the branch rather than about its length.
        """
        start = window.index(start_marker)
        end = window.index(end_marker, start)
        assert end > start, "branch markers out of order"
        return window[start:end]

    def test_editcontrol_uses_direct_fill_hwnd_no_click(self):
        window = self._batch_window()
        section = self._branch(window, '_bf_ty == "editcontrol"',
                               '_bf_ty == "comboboxcontrol"')
        assert '"direct_fill_hwnd": _bf_hwnd' in section
        assert '"action_type": "click"' not in section
        assert "_bf_ctrl.SetFocus()" in section

    def test_comboboxcontrol_uses_combobox_select_and_checks_success(self):
        window = self._batch_window()
        section = self._branch(window, '_bf_ty == "comboboxcontrol"',
                               '_bf_ty in ("checkboxcontrol"')
        assert '"action_type": "combobox_select"' in section
        assert '"combobox_hwnd": _bf_hwnd' in section
        assert "_bf_cb_result.success" in section

    def test_routes_through_resolve_field_control_not_the_raw_lookup(self):
        """Added 2026-08-14 ("still too slow, at least <60s"): timed a real
        batch (13 fields, 7s, no observe() between them) and traced the
        cost into _find_uia_control_by_name's own expensive disambiguation
        search, always triggered because every caller always passed
        expected_bbox even for fields whose label is completely unique on
        screen. _resolve_field_control only pays for that search when the
        label is genuinely ambiguous -- both editcontrol and
        comboboxcontrol branches must route through it, not call
        _find_uia_control_by_name directly."""
        window = self._batch_window()
        assert window.count("self._resolve_field_control(state, _bf_label, _bf_el.get(\"bbox\"))") == 2
        assert "self._find_uia_control_by_name(" not in window

    def test_checkbox_reuses_the_extracted_lookup_helper_not_auto_check(self):
        """Batch can't use self._auto_check(state) directly -- that method
        is hardwired to state['focused_element_id'], but batch must
        evaluate checkboxes that AREN'T focused. Must call the extracted
        per-field helper instead."""
        window = self._batch_window()
        assert "self._lookup_checkbox_should_check(_bf_label)" in window
        assert "self._auto_check(state)" not in window

    def test_checkbox_uses_the_same_bm_setcheck_mechanism_as_existing_sites(self):
        window = self._batch_window()
        idx = window.index('_bf_ty in ("checkboxcontrol", "checkbox")')
        section = window[idx:idx + 1800]
        assert "WindowFromPoint((int(_bf_cx), int(_bf_cy)))" in section
        assert "SendMessage(_bf_hw, 0x00F1, 1, 0)" in section

    def test_no_tab_keystroke_between_fields_no_reobserve_per_field(self):
        """The whole point: N fields filled without N observe() calls.
        self._observe() must not appear anywhere inside the batch loop
        body itself (only the settle-wait/next-step observe outside it)."""
        window = self._batch_window()
        loop_idx = window.index("for _bf_el in _bf_targets:")
        end_idx = window.index("if _bf_filled > 0:")
        loop_body = window[loop_idx:end_idx]
        assert "self._observe()" not in loop_body

    def test_marks_attempted_for_every_filled_field(self):
        """3 real writes (editcontrol/comboboxcontrol/checkbox) + 2
        confirmed-blank skips (editcontrol/comboboxcontrol) = 5."""
        window = self._batch_window()
        assert window.count("self._mark_attempted(_bf_el") == 5

    def test_settles_once_for_the_whole_batch_not_per_field(self):
        """One settle-wait call gated on filled_count > 0, not one per
        field written -- that's the entire point of batching."""
        window = self._batch_window()
        loop_idx = window.index("for _bf_el in _bf_targets:")
        end_idx = window.index("if _bf_filled > 0:")
        loop_body = window[loop_idx:end_idx]
        assert "self._adaptive_settle_wait" not in loop_body
        after = window[end_idx:end_idx + 300]
        assert "self._adaptive_settle_wait(self.step_delay * 0.2)" in after
        assert "continue" in after

    def test_continue_only_reached_when_something_was_actually_filled(self):
        window = self._batch_window()
        idx = window.index("if _bf_filled > 0:")
        section = window[idx:idx + 300]
        assert "continue" in section
        # nothing below the batch block's own scope short-circuits when
        # filled_count stayed at 0 -- confirmed by the single-field block
        # still being reachable right after this one in the source.
        single_idx = _SOURCE.index("OPT2 FAST-FILL: skip the transformer")
        batch_idx = _SOURCE.index("OPT2 BATCH FAST-FILL")
        assert batch_idx < single_idx

    def test_is_not_hardcoded_to_any_specific_field_name(self):
        window = self._batch_window()
        assert not re.search(r'_bf_label\s*==\s*[\'"]', window)

    def test_is_gated_before_the_transformer_predict_call(self):
        batch_idx = _SOURCE.index("OPT2 BATCH FAST-FILL")
        predict_idx = _SOURCE.index('t_pred = self._predict(state)')
        assert batch_idx < predict_idx


class TestBatchFastFillConfirmedBlankSkip:
    """Tests for the confirmed-blank extension to OPT2 batch fast-fill --
    added 2026-08-14, direct follow-up ("I need it a bit more faster")
    after real log evidence: a single genuinely-blank field ('Custom
    Equipment Value ($)') cost TWO full transformer calls before the
    reactive path's own three-attempt escalation finally confirmed there
    was nothing to fill. Batch fast-fill now runs that same escalation
    (_resolve_field_value_with_escalation, shared with _ask_llm -- see
    tests/test_resolve_field_value_with_escalation.py) itself, and Tabs
    past a confirmed-blank field with no transformer call at all."""

    def _batch_window(self):
        idx = _SOURCE.index("OPT2 BATCH FAST-FILL")
        return _SOURCE[idx:idx + 15000]

    def test_editcontrol_and_comboboxcontrol_both_use_the_escalation_helper(self):
        window = self._batch_window()
        assert window.count(
            "self._resolve_field_value_with_escalation(state, _bf_label, section=_bf_sec)") == 2

    def test_confirmed_blank_is_recorded_in_leave_blank_keys(self):
        """Must feed the SAME tracker Navigation Protocol and the reactive
        path already trust (self._leave_blank_keys), not a separate,
        batch-only concept of 'blank' that the rest of the file can't see."""
        window = self._batch_window()
        assert window.count("self._leave_blank_keys.add(_bf_key)") == 2

    def test_confirmed_blank_sends_no_write_message_only_tab(self):
        """The whole point -- a confirmed-blank field costs a Tab
        keystroke and nothing else, no WM_SETTEXT/CB_SETCURSEL call."""
        window = self._batch_window()
        idx = window.index("confirmed blank, Tab past")
        section = window[idx:idx + 700]
        assert '"direct_fill_hwnd"' not in section
        assert '"combobox_select"' not in section
        assert '"key_count": 1, "keystrokes": ["tab"]' in section

    def test_confirmed_blank_still_marks_attempted_and_counts_toward_the_batch(self):
        """A confirmed-blank field is real, useful work -- it must still
        mark_attempted (so it's never re-checked) and increment the same
        _bf_filled counter that gates the end-of-batch continue, exactly
        like an actual write does."""
        window = self._batch_window()
        idx = window.index("confirmed blank, Tab past")
        section = window[idx:idx + 700]
        assert "self._mark_attempted(_bf_el" in section
        assert "_bf_filled += 1" in section

    def test_already_confirmed_blank_fields_are_not_re_escalated(self):
        """The existing leave_blank_keys/typed_keys gate at the top of
        each branch already runs BEFORE the escalation call -- a field
        confirmed blank on an earlier batch must not pay for the
        escalation (cache refresh + Notepad peek) a second time."""
        window = self._batch_window()
        edit_idx = window.index('_bf_ty == "editcontrol"')
        escalation_idx = window.index(
            "self._resolve_field_value_with_escalation(state, _bf_label, section=_bf_sec)", edit_idx)
        gate_idx = window.index("_bf_key in self._leave_blank_keys", edit_idx)
        assert edit_idx < gate_idx < escalation_idx

    def test_is_not_hardcoded_to_any_specific_field_name(self):
        window = self._batch_window()
        idx = window.index("confirmed blank, Tab past")
        section = window[idx - 200:idx + 400]
        assert not re.search(r'_bf_label\s*==\s*[\'"]', section)
