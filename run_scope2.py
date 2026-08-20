"""
run_scope2.py
=============
Scope #2 on the same pipeline as Scope #1: a transformer decides WHERE to act
and WHAT kind of action it is, an LLM supplies a value when the sheet cannot,
and a plugin handles the task-shaped parts. run_task.py does this for a desktop
form fed by Notepad; this does it for a web grade portal fed by a spreadsheet.

The point is that almost nothing here is new. Every difference between the two
scopes goes through a seam LLMAgent already had:

    Scope #1                        Scope #2
    ----------------------------    ----------------------------------
    observer  = UIA tree            observer    = WebObserver (DOM, over CDP)
    source    = NotepadDataSource   data_source = GradeSheetSource (.xlsx)
    scope     = INSURANCE_SCOPE     scope       = GRADE_PORTAL_SCOPE
    plugin    = FormFillerPlugin    task_plugin = GradePortalPlugin

The agent itself is unchanged and unaware. That is the thesis claim this file
exists to make concrete: swapping the application should cost configuration,
not agent edits.

Prerequisites (all of them the operator's to start):
    python practice_apps/mocksite/serve.py
    chrome.exe --remote-debugging-port=9222 --user-data-dir=%TEMP%\\chrome-scope2
        http://127.0.0.1:8765/v0_base/index.html

Then:
    python run_scope2.py                       # first student, v0_base
    python run_scope2.py --records 0-4         # five students
    python run_scope2.py --model tasks/grade_portal/model.pt
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# DPI awareness must be claimed before the import chain does it for us -
# see components/dpi.py. sys.path is not set up yet, so this reaches
# components/ directly rather than importing anything of ours.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "components"))
import dpi as _dpi
_dpi.ensure_per_monitor()

import argparse
import datetime as _datetime
import json
import logging
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "components")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_env_path = os.path.join(_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, f"run_scope2_{_datetime.datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("run_scope2")

# ── config ───────────────────────────────────────────────────────────────────
GOAL = "Encode each student's grades into the grade portal using the grade sheet"
PROVIDER = "lmstudio"        # anthropic | groq | gemini | lmstudio | none
MAX_STEPS = 1000
STEP_DELAY = 0.5
DEFAULT_SHEET = os.path.join("components", "scope2", "data", "sheets", "grade_sheet.xlsx")
DEFAULT_MODEL = os.path.join("tasks", "grade_portal", "model.pt")
# DAgger's correction window blocks in real time waiting for a human to fix a
# failed step. Same value run_task.py settled on, for the same reason: nobody is
# watching most runs.
CORRECTION_WATCH_SECONDS = 0.5


def _flush_safe_print(text: str) -> None:
    """print(), then attempt-and-ignore the flush.

    Copied deliberately from run_task.py rather than imported: this script is
    spawned through the same windowsHide Electron chain, where a flush on a
    piped stdout can raise OSError even though the write succeeded, and
    importing run_task.py would execute its module-level config and logging
    setup as a side effect.
    """
    print(text)
    try:
        sys.stdout.flush()
    except OSError:
        pass


def print_countdown(seconds: int = 5, sleep_fn=None, print_fn=None) -> None:
    """The pre-run countdown, in the sentinel format the Play panel parses.

    The line after COUNTDOWN_BEGIN becomes the widget's hint text, so it says
    what THIS workflow needs. Scope #1 asks the operator to click the target
    window; here the browser is driven over CDP and clicking into it is exactly
    what must not happen mid-run.
    """
    sleep_fn = sleep_fn or time.sleep
    print_fn = print_fn or _flush_safe_print
    print_fn("COUNTDOWN_BEGIN")
    print_fn("Bring the portal window to the front, then keep your hands off it.")
    for i in range(seconds, 0, -1):
        print_fn(f"COUNTDOWN {i}")
        sleep_fn(1)
    print_fn("COUNTDOWN_END")


def parse_records(spec: str) -> range:
    """"3" -> just student 3. "0-4" -> five students. 0-based, like the sheet."""
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return range(int(lo), int(hi) + 1)
    n = int(spec)
    return range(n, n + 1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="0",
                    help="Which students, 0-based: \"0\" or \"0-4\" (default: 0).")
    ap.add_argument("--sheet", default=DEFAULT_SHEET,
                    help="Grade sheet .xlsx to read values from.")
    ap.add_argument("--sheet-name", default=None,
                    help="Worksheet to read (default: the source's own choice).")
    ap.add_argument("--browser-url", default="http://localhost:9222",
                    help="CDP endpoint of the browser already showing the portal.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Transformer checkpoint. Missing is not fatal - the agent "
                         "falls back the same way it does for scope #1.")
    ap.add_argument("--provider", default=PROVIDER,
                    choices=["anthropic", "groq", "gemini", "lmstudio", "none"])
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--step-delay", type=float, default=STEP_DELAY)
    ap.add_argument("--max-elements", type=int, default=1000,
                    help="Cap on elements read per snapshot (the portal has 303).")
    ap.add_argument("--no-plugin", action="store_true",
                    help="Pure transformer + LLM, no GradePortalPlugin - the shape "
                         "run_task.py uses. Needs a trained checkpoint to be useful.")
    ap.add_argument("--countdown", type=int, default=5)
    return ap.parse_args(argv)


def build_observer(args):
    """Attach to the operator's browser. Loud failure, not a blank screen.

    connect() returning False here is the difference between "no portal" and a
    run that observes an empty page and reports every field missing, so it is
    checked rather than assumed.
    """
    from observers.web_observer import WebObserver

    observer = WebObserver(browser_url=args.browser_url, max_elements=args.max_elements)
    if not observer.available:
        raise SystemExit("playwright is not installed - pip install playwright")
    if not observer.connect():
        raise SystemExit(
            f"No browser answering at {args.browser_url}.\n"
            "Start one with:  chrome.exe --remote-debugging-port=9222 "
            "--user-data-dir=%TEMP%\\chrome-scope2 <portal url>")
    return observer


def main(argv=None) -> int:
    args = parse_args(argv)
    records = parse_records(args.records)

    sheet = args.sheet if os.path.isabs(args.sheet) else os.path.join(_ROOT, args.sheet)
    if not os.path.exists(sheet):
        raise SystemExit(f"grade sheet not found: {sheet}")

    from agent.agent import LLMAgent
    from agent.scope import GRADE_PORTAL_SCOPE
    from agent.task_plugins.grade_portal_plugin import GradePortalPlugin
    from data_sources.grade_sheet_source import GradeSheetSource

    observer = build_observer(args)
    logger.info("Perception: WebObserver over CDP at %s", args.browser_url)

    api_key = (os.environ.get("ANTHROPIC_API_KEY", "")
               or os.environ.get("GROQ_API_KEY", "")
               or os.environ.get("GEMINI_API_KEY", ""))

    print_countdown(args.countdown)

    all_results = []
    agent = None
    try:
        for record_num in records:
            logger.info("=" * 60)
            logger.info("STUDENT %d", record_num)
            logger.info("=" * 60)

            # A fresh source per student, mirroring run_agent.py's per-record
            # construction: refresh(n) is what selects the row, and a source
            # shared across students is one cache away from filling every row
            # with the first student's grades.
            source = GradeSheetSource(sheet, sheet_name=args.sheet_name) \
                if args.sheet_name else GradeSheetSource(sheet)

            plugin = None
            if not args.no_plugin:
                plugin = GradePortalPlugin(
                    executor=None,          # LLMAgent wires this in
                    data_source=source,
                    record_num=record_num,
                    step_delay=args.step_delay,
                )

            agent = LLMAgent(
                goal=GOAL,
                provider=args.provider,
                api_key=api_key,
                task_plugin=plugin,
                pure_transformer=False,
                disable_auto_handlers=True,
                observer=observer,           # the seam: DOM instead of UIA
                data_source=source,          # the seam: sheet instead of Notepad
                scope=GRADE_PORTAL_SCOPE,    # the seam: no tabs, no sections
                record_num=record_num,
                max_steps=args.max_steps,
                step_delay=args.step_delay,
                model_path=args.model,
                route_capsule=False,
                correction_watch_seconds=CORRECTION_WATCH_SECONDS,
            )
            results = agent.run(max_steps=args.max_steps, task_name="grade_portal")
            all_results.extend(results or [])
    except KeyboardInterrupt:
        logger.info("Run interrupted by user after %d step(s).", len(all_results))
    except Exception:
        logger.error("Run crashed after %d step(s):", len(all_results), exc_info=True)
    finally:
        try:
            observer.disconnect()
        except Exception:
            pass

        logger.info("Run ended — %d steps", len(all_results))
        _report(all_results, args)

    return 0


def _report(results, args):
    """Same evaluation Scope #1 runs, so the two are comparable at all."""
    try:
        sys.path.insert(0, os.path.join(_ROOT, "scripts"))
        from eval_metrics import evaluate_run

        metrics = evaluate_run(results, goal=GOAL)
        row = {
            "timestamp": _datetime.datetime.now().isoformat(),
            "scope": "grade_portal",
            "goal": GOAL,
            "provider": args.provider,
            **{k: v for k, v in metrics.items() if k != "summary"},
        }
        path = os.path.join(_ROOT, "data", "output", "run_metrics.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        logger.info("Metrics appended to %s", path)
    except Exception as exc:
        logger.warning("Evaluation skipped: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
