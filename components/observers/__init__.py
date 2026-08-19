"""Observer implementations, each imported independently.

These used to be four eager imports, which made the package all-or-nothing: the
VLM observers need pyautogui, so a machine without it lost UIAutomationObserver
and WebObserver as well - neither of which has anything to do with pyautogui.
Worse, the failure was reported against the wrong dependency, because callers
guard `from observers import X` with `except ImportError` and then print
"UIAutomation unavailable".

Guarding each import separately means a missing optional dependency removes
only the observer that actually needs it. Importing a name that failed raises
the usual ImportError from the package, and `__all__` lists what is really
available.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

__all__ = []

for _name, _module in (
    ("UIAutomationObserver", ".ui_observer"),
    ("ExcelObserver",        ".excel_observer"),
    ("WebObserver",          ".web_observer"),
    ("VisionObserver",       ".vlm.vision_observer"),
    ("VisualDataReader",     ".vlm.visual_data_reader"),
):
    try:
        _mod = __import__(f"{__name__}{_module}", fromlist=[_name])
        globals()[_name] = getattr(_mod, _name)
        __all__.append(_name)
    except Exception as _exc:  # noqa: BLE001 - an optional dep can fail any way
        _logger.debug("observers: %s unavailable (%s)", _name, _exc)

del _name, _module
