"""
components/data_sources/grade_sheet_source.py
==============================================
GradeSheetSource — reads field values for one student at a time out of an
institutional grade book (.xlsx).

This is the Scope #2 data source: where NotepadDataSource reads a record out of
a text window, this reads one row out of a spreadsheet. Both speak the same
DataSource contract, so the agent does not care which is behind it:

    src = GradeSheetSource("data/sheets/grade_sheet.xlsx")
    src.refresh(0)                    # first student
    src.lookup("Program")             # -> "BS Information Systems"
    src.lookup("Final Grade")         # -> "85"
    src.get_all()                     # every field of that student

Two things about real grade books that this has to handle, and which a naive
`read_excel` gets wrong:

  * **The header row is not row 1.** These workbooks carry a merged title block
    - institution, department, subject, instructor - above the table. Headers
      sit around row 12 and data starts a few rows below. `header_row` is that
      override; `find_header_row` proposes it when it is not given.

  * **Lookups arrive in the form's vocabulary, not the sheet's.** The agent asks
    for the field it is looking at, which is rarely spelled the way the column
    is. Matching is normalised and falls back to token overlap, so "Year Level"
    finds "YEAR LEVEL" and "Final Grade" finds "FINAL GRADE".

    It deliberately stops there. "Course" does NOT resolve to "PROGRAM", and
    "Yr Level" does not resolve to "YEAR LEVEL" - a synonym and an abbreviation
    are exactly what the Scope #2 matcher exists to decide, and guessing them
    here would put an unaudited mapping underneath it. This class is forgiving
    about spelling, not about meaning; when it cannot tell, it returns None and
    lets the matcher answer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DataSource

logger = logging.getLogger(__name__)

DEFAULT_SHEET = "SUMMARY"

# Columns that identify a student rather than carrying encodable data. They are
# still readable through lookup() - the executor needs the student number to
# find the right row on the portal - but get_all() marks them so a caller can
# tell identity from payload.
IDENTITY_COLUMNS = {"student number", "stud no", "name of student"}

# Columns that carry neither identity nor encodable data: the sheet's own row
# counter, and the continuation columns a merged header leaves unnamed (a
# "NAME OF STUDENT" header spanning LASTNAME/FIRSTNAME/MI names only the first).
NOISE_COLUMN = re.compile(r"^(no|unnamed \d+)$")


def _normalise(text: Any) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _tokens(text: Any) -> List[str]:
    return [t for t in _normalise(text).split() if t]


def find_header_row(path: Path, sheet_name: str = DEFAULT_SHEET,
                    max_scan: int = 30) -> int:
    """Propose the 0-based header row: the scanned row with the most labels.

    A grade book's title block is one merged cell per row, so the header row is
    the first one that is genuinely wide.
    """
    import pandas as pd

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan)
    best, best_score = 0, -1
    for i in range(len(raw)):
        labels = [v for v in raw.iloc[i] if isinstance(v, str) and v.strip()]
        if len(labels) > best_score:
            best, best_score = i, len(labels)
    return best


class GradeSheetSource(DataSource):
    """One student per record, read from a grade book worksheet."""

    def __init__(self, path: str | Path, sheet_name: str = DEFAULT_SHEET,
                 header_row: Optional[int] = None,
                 key_column: str = "STUDENT NUMBER"):
        self.path = Path(path)
        self.sheet_name = sheet_name
        self.key_column = key_column
        self._header_row = header_row
        self._frame = None
        self._headers: List[str] = []
        self._record: Dict[str, str] = {}
        self._record_num: Optional[int] = None

    # ── loading ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._frame is not None:
            return
        import pandas as pd

        if not self.path.exists():
            raise FileNotFoundError(f"grade sheet not found: {self.path}")

        if self._header_row is None:
            self._header_row = find_header_row(self.path, self.sheet_name)
            logger.info("GradeSheetSource: header row inferred as %d",
                        self._header_row + 1)

        frame = pd.read_excel(self.path, sheet_name=self.sheet_name,
                              header=self._header_row)
        frame = frame.dropna(how="all")

        # Drop the footer rows a grade book carries below the class list.
        key = self._resolve_column(self.key_column, list(frame.columns))
        if key is not None:
            frame = frame[frame[key].notna()]

        self._frame = frame.reset_index(drop=True)
        self._headers = [str(c) for c in self._frame.columns]
        logger.info("GradeSheetSource: %d records, %d columns from %s",
                    len(self._frame), len(self._headers), self.path.name)

    @staticmethod
    def _resolve_column(wanted: str, columns: List[Any]) -> Optional[Any]:
        """Find the column a caller means, or None.

        Exact, then normalised ("Final Grade" -> "FINAL GRADE"), then token
        overlap with at least half the words shared. Below that it returns None
        rather than picking the nearest: "Course" against "PROGRAM" is a
        synonym and "Yr Level" against "YEAR LEVEL" is an abbreviation, and
        both are the matcher's decision to make, not a lookup's.
        """
        if wanted in columns:
            return wanted

        target = _normalise(wanted)
        if not target:
            return None

        for column in columns:
            if _normalise(column) == target:
                return column

        wanted_tokens = set(_tokens(wanted))
        if not wanted_tokens:
            return None

        best, best_score = None, 0.0
        for column in columns:
            column_tokens = set(_tokens(column))
            if not column_tokens:
                continue
            overlap = len(wanted_tokens & column_tokens)
            if not overlap:
                continue
            score = overlap / len(wanted_tokens | column_tokens)
            if score > best_score:
                best, best_score = column, score

        # Half the words in common, or it is a guess rather than a match.
        return best if best_score >= 0.5 else None

    @staticmethod
    def _as_text(value: Any) -> str:
        """A cell as a form should receive it: no trailing .0 on whole numbers."""
        if value is None:
            return ""
        try:
            import pandas as pd

            if pd.isna(value):
                return ""
        except Exception:  # noqa: BLE001 - non-scalar values are fine as-is
            pass
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    # ── DataSource contract ───────────────────────────────────────────────

    def refresh(self, record_num: int) -> None:
        """Load the student at `record_num` (0-based)."""
        self._load()

        if not 0 <= record_num < len(self._frame):
            raise IndexError(
                f"record {record_num} out of range - the sheet has "
                f"{len(self._frame)} students"
            )

        row = self._frame.iloc[record_num]
        self._record = {str(h): self._as_text(row[h]) for h in self._frame.columns}
        self._record_num = record_num

    def lookup(self, field_name: str, section: str = "") -> Optional[str]:
        """The value of `field_name` for the current student, or None.

        `section` is unused: a grade book row has no sections. It stays in the
        signature because the agent calls every data source the same way.
        """
        if self._record_num is None:
            self.refresh(0)

        column = self._resolve_column(field_name, list(self._record.keys()))
        if column is None:
            return None
        value = self._record.get(column, "")
        return value if value != "" else None

    def get_all(self) -> Dict[str, str]:
        """Every field of the current student."""
        if self._record_num is None:
            self.refresh(0)
        return dict(self._record)

    # ── extras the executor needs ─────────────────────────────────────────

    def record_count(self) -> int:
        self._load()
        return len(self._frame)

    def headers(self) -> List[str]:
        self._load()
        return list(self._headers)

    def identity(self) -> Dict[str, str]:
        """The columns that say *which* student this is, rather than what to
        encode. The executor matches a portal row on these instead of typing
        them in."""
        return {k: v for k, v in self.get_all().items()
                if _normalise(k) in IDENTITY_COLUMNS}

    def payload(self) -> Dict[str, str]:
        """Everything that is neither identity nor sheet bookkeeping - the
        columns that are actually candidates for encoding."""
        out = {}
        for key, value in self.get_all().items():
            folded = _normalise(key)
            if not key or not value:
                continue
            if folded in IDENTITY_COLUMNS or NOISE_COLUMN.match(folded):
                continue
            out[key] = value
        return out

    def samples(self, header: str, limit: int = 5) -> List[str]:
        """Up to `limit` values from a column, for value-shape matching."""
        self._load()
        column = self._resolve_column(header, list(self._frame.columns))
        if column is None:
            return []
        values = [self._as_text(v) for v in self._frame[column].dropna().tolist()]
        return [v for v in values if v][:limit]
