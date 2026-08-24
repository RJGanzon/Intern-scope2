"""
components/data_sources/matched_source.py
==========================================
MatchedFieldSource — a DataSource that answers in the TARGET's vocabulary.

The bridge Scope #2 needs and Scope #1 never did. A Notepad record's keys are
the form's own labels, so the agent can ask the source using the name on screen
and get an answer. A grade book has PROGRAM and FINAL GRADE while the portal
asks for "Course Abad, Andrea A.", and every such question comes back empty.

Three ways to answer it, measured on this machine rather than argued about:

  1. Hand-write the mapping. That is GradePortalPlugin.COLUMN_MAP, whose own
     docstring calls itself scaffolding: it is exactly what Scope #2 exists to
     learn, and it goes blind the moment a variant renames a column.

  2. Ask the LLM. Tried three ways against google/gemma-3-4b, the model
     available here. Asked to copy a value it answered 85 for the Course, the
     Year and the Grade of one row - the first number in the record. Told the
     row half of a grid label must be ignored, it answered 85 for all three.
     Asked to pick from a numbered list, it picked the student's NAME every
     time, matching on precisely the half it had been told to ignore.

  3. The matcher that was ported for this. 17 hand-engineered features per
     (column, field) pair, a 1,121-parameter network, Hungarian assignment, and
     an abstention threshold. On this sheet and this portal:

         Course       <- PROGRAM       1.000  auto
         Year 1-5     <- YEAR LEVEL    1.000  auto
         Grade 0-100  <- FINAL GRADE   1.000  auto
         Remarks                       0.005  abstain
         Recommendations               0.000  abstain

     Three for three, and it declines the two fields that genuinely have no
     column - Remarks is derived by the portal from the grade, Recommendations
     is optional and left blank. An abstention is the right answer there, and a
     confident wrong guess would have been worse than none.

This wraps a source rather than modifying one, so the agent is untouched: it
still receives one object with lookup/get_all/can_answer and still cannot tell
which application it is driving.

WHAT THE MATCHER IS FED MATTERS
--------------------------------
The first run of the same model mapped Course to "No." at 0.833, because every
column was declared "text" and no field carried its placeholder. The portal's
own header declares placeholder="BS Computer Science" and the sheet's PROGRAM
column holds "BS Information Systems"; with those present the same pair scores
1.000. Feeding it what the page actually says is not a detail.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from .base import DataSource

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMP = os.path.dirname(_HERE)
for _p in (_COMP, os.path.join(_COMP, "scope2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_MATCHER = os.path.join(_COMP, "scope2", "data", "models", "matcher.pt")


def infer_type(samples: List[str]) -> str:
    """"number" when every sample is one, else "text"."""
    values = [str(v).strip() for v in samples if str(v).strip()]
    if values and all(re.fullmatch(r"-?\d+(\.\d+)?", v) for v in values):
        return "number"
    return "text"


class MatchedFieldSource(DataSource):
    """Answers questions phrased in the target's words, using a matched mapping.

    Parameters
    ----------
    inner  : the real source (GradeSheetSource, or anything with
             lookup/get_all/headers/samples).
    fields : the target's own field descriptions - label plus whatever the page
             declares about it (placeholder, input type, options, min/max).
             See the note above on why these matter.
    matcher_path : trained matcher checkpoint.
    """

    def __init__(self, inner: Any, fields: List[Dict[str, Any]],
                 matcher_path: str = DEFAULT_MATCHER):
        self._inner = inner
        self._field_specs = list(fields)
        self._matcher_path = matcher_path
        self._mapping: Optional[Dict[str, str]] = None      # field label -> column
        self._column_names = [str(f.get("label", "")) for f in self._field_specs]

    # ── the mapping ──────────────────────────────────────────────────────────

    def mapping(self) -> Dict[str, str]:
        """Field label -> source column, computed once.

        Abstentions are absent from the result rather than present-and-empty:
        a field the matcher declined is one nothing should be typed into, and
        that is different from a field mapped to a column that happens to be
        blank.
        """
        if self._mapping is not None:
            return self._mapping

        self._mapping = {}
        try:
            import torch
            from descriptors import FieldDescriptor, SourceColumn
            from features.extractor import candidates, extract
            from model import matcher as matcher_module
            from resolver.assign import resolve
        except Exception as exc:
            logger.warning("MatchedFieldSource: matcher unavailable (%s); every "
                           "field will read as unknown.", exc)
            return self._mapping

        try:
            columns = []
            for index, header in enumerate(self._inner.headers()):
                header = str(header)
                if header.lower().startswith("unnamed"):
                    # pandas' name for the columns under a merged header; they
                    # carry the rest of a student's name and are not fields.
                    continue
                samples = [str(s) for s in self._inner.samples(header, 5)]
                columns.append(SourceColumn(header=header, index=index,
                                            inferred_type=infer_type(samples),
                                            samples=samples,
                                            non_null=len(samples), total=len(samples)))

            fields = []
            for order, spec in enumerate(self._field_specs):
                label = str(spec.get("label", ""))
                fields.append(FieldDescriptor(
                    label=label, label_rule=1, kind="input",
                    input_type=str(spec.get("input_type", "text")),
                    column_key=label.lower().split()[0] if label else "",
                    column_index=order, header_text=label,
                    name=label.lower().split()[0] if label else "",
                    placeholder=str(spec.get("placeholder", "")),
                    options=spec.get("options"), maxlength=spec.get("maxlength"),
                    min=spec.get("min"), max=spec.get("max"), step=spec.get("step"),
                    required=bool(spec.get("required", False)), dom_order=order))

            if not columns or not fields:
                return self._mapping

            model, _ = matcher_module.load(self._matcher_path)
            pairs = candidates(columns, fields)
            scores = [model.probability(
                torch.tensor([extract(c)], dtype=torch.float32)).item() for c in pairs]
            matrix = [[scores[i * len(fields) + j] for j in range(len(fields))]
                      for i in range(len(columns))]

            for assignment in resolve(columns, fields, matrix).assignments:
                if assignment.status == "auto":
                    self._mapping[assignment.target_label] = assignment.source_header
                    logger.info("Matched %r -> %r (%.3f)", assignment.target_label,
                                assignment.source_header, assignment.score)
                else:
                    logger.info("Abstained on %r (best %r, %.3f) - nothing will be "
                                "typed into it", assignment.target_label,
                                assignment.source_header, assignment.score)
        except Exception as exc:
            logger.warning("MatchedFieldSource: matching failed (%s)", exc)
        return self._mapping

    def column_for(self, field_name: str) -> Optional[str]:
        """The source column a screen field wants, or None.

        A grid cell is named "<column> <which row>", so the column is found by
        the longest field label that starts the name - longest, because "Year"
        and "Year 1-5" would otherwise both match and the shorter one would win
        on iteration order alone.
        """
        if not field_name:
            return None
        name = field_name.strip()
        mapping = self.mapping()
        best = None
        for label in mapping:
            if name == label or name.startswith(label + " "):
                if best is None or len(label) > len(best):
                    best = label
        return mapping.get(best) if best else None

    # ── the DataSource contract ──────────────────────────────────────────────

    def lookup(self, field_name: str, section: str = "") -> Optional[str]:
        column = self.column_for(field_name)
        if column is None:
            # Not a rename of the inner source's own vocabulary: a caller asking
            # in the source's words still gets its answer.
            return self._inner.lookup(field_name, section=section)
        return self._inner.lookup(column, section=section)

    def is_known_field(self, field_name: str) -> bool:
        """Is this one of the target's own columns, mapped or not?"""
        name = (field_name or "").strip()
        return any(name == label or name.startswith(label + " ")
                   for label in self._column_names if label)

    def can_answer(self, field_name: str) -> bool:
        """True for a mapped field, and ALSO true for one that was abstained on.

        An abstention is an answer: the matcher looked at every column and found
        none that feeds this field. Remarks is derived by the portal from the
        grade; Recommendations is optional. The right value for both is nothing.

        Reporting them as unanswerable sent them to the LLM instead, which is
        the one place they must not go - asked what belonged in Remarks,
        gemma-3-4b replied 85, the grade, into a field the page fills itself.
        Claiming to answer, with lookup() returning None, is what makes the
        agent treat them as confirmed blank and leave them alone.
        """
        if self.column_for(field_name) is not None:
            return True
        if self.is_known_field(field_name):
            return True
        asker = getattr(self._inner, "can_answer", None)
        return bool(asker(field_name)) if callable(asker) else True

    def get_all(self) -> Dict[str, str]:
        return self._inner.get_all()

    def refresh(self, record_num: int) -> None:
        # The mapping is between two schemas and does not change per row, so it
        # is deliberately not cleared here: re-matching every student would cost
        # a sentence-transformer pass each time and could only produce a
        # different answer if the page changed shape mid-run.
        self._inner.refresh(record_num)

    # ── passthrough for everything else the agent may reach for ──────────────

    def __getattr__(self, item):
        return getattr(self._inner, item)
