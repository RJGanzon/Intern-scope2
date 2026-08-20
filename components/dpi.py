"""Make a process see real pixels, before anything else has an opinion.

Windows scales geometry for a process that has not declared its DPI awareness:
on a 125% display GetWindowRect and GetSystemMetrics come back divided by 1.25,
so that software written before high-DPI displays still lays out sensibly. Mouse
input is not scaled - pynput and pyautogui deal in raw physical pixels - so a
process reading geometry in one ruler and clicks in the other is comparing two
different coordinate systems and getting plausible, wrong answers.

Measured on a real recording session: the operator's laptop is the secondary
display at 125%, and every recorded click resolved to the last column of the
grid instead of the one being filled. One click sat at x=3567 on a desktop the
process believed was 3456 wide. Declaring per-monitor awareness first put 9 of 9
clicks on the correct column and the correct student.

WHY THIS LIVES IN ITS OWN MODULE, WITH NO IMPORTS
--------------------------------------------------
Awareness can only be set once per process, and the first claim wins. Something
in the usual import chain (pywin32, pyautogui, mss - not pinned down, and it
does not matter which) already claims SYSTEM awareness at import time, after
which SetProcessDpiAwareness returns E_ACCESSDENIED and the process is stuck
being told about a 3456-pixel desktop that is really 3840.

So this module imports nothing but ctypes, and entry points call it as their
first executable line. Importing anything heavier here would reintroduce exactly
the race it exists to win.

    import dpi
    dpi.ensure_per_monitor()      # before importing anything else
"""

from __future__ import annotations

import ctypes

UNAWARE = "UNAWARE"
SYSTEM = "SYSTEM"
PER_MONITOR = "PER_MONITOR"

_NAMES = {0: UNAWARE, 1: SYSTEM, 2: PER_MONITOR}


def current() -> str:
    """This process's DPI awareness, or UNAWARE if Windows cannot say."""
    try:
        value = ctypes.c_int()
        ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(value))
        return _NAMES.get(value.value, UNAWARE)
    except Exception:
        return UNAWARE


def ensure_per_monitor() -> str:
    """Claim per-monitor awareness. Returns what the process ended up with.

    Per-monitor rather than system awareness, so a window dragged between
    displays of different scaling stays correct rather than only the primary
    being right.

    Never raises. A process that already claimed awareness cannot change it, and
    the caller is expected to check the returned value rather than assume.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()   # pre-8.1: system-aware
        except Exception:
            pass
    return current()


def virtual_desktop() -> tuple:
    """(width, height) of every monitor together, in this process's ruler."""
    try:
        user32 = ctypes.windll.user32
        return (int(user32.GetSystemMetrics(78)), int(user32.GetSystemMetrics(79)))
    except Exception:
        return (0, 0)
