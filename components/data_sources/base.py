"""
components/data_sources/base.py
================================
Abstract base class for data sources.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict


class DataSource(ABC):
    """Abstract data source for reading field values."""

    @abstractmethod
    def lookup(self, field_name: str, section: str = "") -> Optional[str]:
        """Return the value for field_name, or None if not found."""
        ...

    @abstractmethod
    def get_all(self) -> Dict[str, str]:
        """Return all cached field values."""
        ...

    def refresh(self, record_num: int) -> None:
        """Reload data for the given record number. Optional."""

    def can_answer(self, field_name: str) -> bool:
        """Could this source have a value for a field with this name?

        The distinction between "the record says this is blank" and "I have no
        idea what this field is" - which lookup() collapses into None, and which
        the agent then reads as the former.

        That collapse is invisible in scope #1, where the source's own keys ARE
        the form's labels, so a miss really does mean blank. It is fatal in
        scope #2: the sheet has PROGRAM and the portal asks for "Course Abad,
        Andrea A.", so every field missed, every field was declared confirmed
        blank, and the agent tabbed through 109 of them without filling one.

        Default True keeps every existing source behaving exactly as before: a
        source that cannot tell the two apart should keep claiming it can
        answer, because that is what the agent already assumes of it.
        """
        return True
