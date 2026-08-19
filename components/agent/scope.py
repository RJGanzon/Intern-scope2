"""
ScopeConfig — per-application configuration injected into LLMAgent.

Everything app-specific (which form, which tabs, how sections/records are named)
lives here instead of being hardcoded in agent.py. The DEFAULT is fully generic:
no tabs, no sections, no record delimiter — so a brand-new GUI gets an agent that
makes ZERO assumptions. Each scope (insurance form, Excel, triage, …) passes its
own ScopeConfig; the agent code stays application-blind.

This is the seam that turns "an insurance-form agent" into "scope #1 of a
scope-agnostic engine."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set


def _default_section_format(kind: str, num: str) -> str:
    return f"{kind.title()} {num}"


@dataclass
class ScopeConfig:
    """App-specific knobs. All default to generic / none."""

    # Tab navigation (forms with tab strips). Empty → tab logic never fires.
    tab_names:      Set[str]  = field(default_factory=set)   # was _KNOWN_TABS
    tab_pane_names: List[str] = field(default_factory=list)  # was _TAB_PANE_NAMES

    # Repeated sections (e.g. Driver 1..N, Vehicle 1..N). section_pattern=None →
    # _detect_section is a no-op (returns ""), which is correct for non-sectioned
    # apps. The pattern is a regex with two groups (kind, number).
    section_prefix:  str             = "section_"
    section_pattern: Optional[str]   = None   # was r"section_(driver|vehicle)_(\d+)$"
    section_format:  Callable[[str, str], str] = _default_section_format

    # Multi-record source delimiter (moves to the DataSource, kept here for ref).
    record_delimiter: Optional[str] = None    # was "RECORD N OF M"

    # Substrings (lowercase) identifying the task's OWN window, tested against a
    # state's window_title / application. Used to tell demonstration clicks from
    # clicks on the terminal, the recorder GUI, or the source document — see
    # scripts/clean_demos.py, where this used to be a literal `"insurance" in t`
    # that silently dropped every click of any other scope's recording.
    # Empty (the generic default) → accept every window, since an unknown app
    # has no marker to test and dropping everything is the worse failure.
    window_markers: List[str] = field(default_factory=list)

    def is_target_window(self, state: dict) -> bool:
        """Does this observation come from the window being demonstrated?"""
        if not self.window_markers:
            return True
        title = (state.get("window_title") or "").lower()
        app   = (state.get("application")  or "").lower()
        return any(m in title or m in app for m in self.window_markers)


# ── Prebuilt scope: the car-insurance data-entry form (dev fixture) ───────────
INSURANCE_SCOPE = ScopeConfig(
    tab_names={"policy", "policyholder", "vehicle", "coverage",
               "drivers", "history", "claims", "payment"},
    tab_pane_names=["tab_policy", "tab_policyholder", "tab_vehicle", "tab_coverage",
                    "tab_drivers", "tab_history", "tab_claims", "tab_payment"],
    section_prefix="section_",
    section_pattern=r"section_(driver|vehicle)_(\d+)$",
    section_format=_default_section_format,
    record_delimiter="RECORD N OF M",
    window_markers=["insurance", "data entry"],
)


# ── Prebuilt scope: the scope #2 grade portal (a web page, not a desktop app) ─
# No tabs and no sections: the portal is one long table, so the tab/section
# machinery correctly never fires. The markers match the mock portal's own
# <title>, which WebObserver reports as window_title — every variant keeps
# "Grade" there, including v2_relabeled, whose visible heading is renamed to
# "Student Rating Sheet" precisely to break anything keyed on the heading.
GRADE_PORTAL_SCOPE = ScopeConfig(
    window_markers=["grade encoding portal", "grade portal", "student rating"],
)
