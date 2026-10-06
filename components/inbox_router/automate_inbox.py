"""
components/inbox_router/automate_inbox.py
=============================================
Scope #3, end to end -- mirrors components/scope2/automate.py's shape and
its whole point: the earlier local_server.py + local_ui pair could compute
a real decision and display it, but nothing ever CLICKED anything. This
script is what actually operates Inbox Dispatch, the same way a human
triaging their own inbox would -- open the local server's real page in a
real browser, read each email off the real DOM, and click the real button
the decision calls for (Confirm, Override, Archive, Reply). No shortcuts
through the HTTP API from here: every read is a DOM read, every action is
a real Playwright click on the same page a human uses.

    python automate_inbox.py                # dry run -- reads and decides, clicks nothing
    python automate_inbox.py --commit        # actually clicks Confirm for each email
    python automate_inbox.py --commit --limit 5

Stages:

    1  make sure the local server is up (starts it if not)
    2  open the real page in a real (visible by default) browser
    3  for each pending email: read it off the DOM, print the real
       decision + rationale the pipeline already computed, pause so a
       human can follow it, then click Confirm for real (--commit only)
    4  write a run log, same as automate.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# A real Windows console (not a piped/redirected stream, which Python
# already treats as UTF-8) defaults to the system codepage -- cp1252 on
# most US/Windows setups -- which can't encode characters like curly
# quotes or em dashes that a real LLM response routinely contains.
# Found live: printing an AI-drafted reply crashed main() mid-run with
# UnicodeEncodeError the moment this ran in a real console instead of a
# captured one. errors="replace" only affects the rare unencodable
# character (swapped for "?"), never the reply text that actually gets
# typed into the page -- that's read straight from inbox_reply_llm, not
# from anything printed here.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from pointer import Pointer
from workspace import open_page
from schedule_extract import extract_event_time
# 127.0.0.1, not "localhost" -- local_server.py's HTTPServer binds only
# IPv4 (127.0.0.1). Found live: Chromium's own resolution of "localhost"
# can try IPv6 (::1) first depending on the environment, which nothing
# is listening on -- page.goto() then hangs until it times out (30s)
# instead of falling back quickly, even though the server is genuinely
# up and curl reaches it instantly. Using the literal IP removes the
# resolution step, and therefore the ambiguity, entirely.
SERVER_URL = "http://127.0.0.1:8765/"
INBOX_LOAD_TIMEOUT_MS = 180_000
RULE = "-" * 74

# os.system("") is a well-known, dependency-free trick that forces the
# classic Windows console into ANSI/VT100 processing mode -- modern
# Windows Terminal and PowerShell already support these codes directly,
# but this makes color work even in an older conhost window, at zero
# cost either way.
if sys.platform == "win32":
    os.system("")

_GREEN, _YELLOW, _RED, _BLUE, _BOLD, _DIM, _RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[1m", "\033[2m", "\033[0m",
)


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _outcome_color(outcome: str) -> str:
    """Green for a real completed action, yellow for correctly-left-
    pending, red for a real failure (LM Studio unavailable) -- the same
    three-way read a person gets from the plain text, just faster to
    scan at a glance."""
    if outcome.startswith("confirmed") or outcome == "drafted":
        return _GREEN
    if "unavailable" in outcome:
        return _RED
    return _YELLOW


def banner(number, title):
    print(f"\n{_color(RULE, _DIM)}\n {_color(f'{number}. {title}', _BOLD)}\n{_color(RULE, _DIM)}")


def _flush_safe_print(text: str) -> None:
    """Same guard as automate.py's -- an explicit flush can raise OSError
    on Windows when this is spawned with no console window (Electron's
    Play button, windowsHide=True), even though the write itself already
    succeeded."""
    print(text)
    try:
        sys.stdout.flush()
    except OSError:
        pass


def print_countdown(seconds: int = 5,
                     message: str = "Starting Inbox Dispatch -- opening a real browser to click through it.") -> None:
    # COUNTDOWN_BEGIN / COUNTDOWN N / COUNTDOWN_END are a real, exact-match
    # sentinel contract -- app_electron/renderer/renderer.js's
    # handleCapsuleProgressLine() parses these precise lines to drive its
    # own countdown widget. The progress-bar line below is purely
    # additive -- a new line, never altering those three -- so it's
    # invisible to that parser and just logs normally there, while a
    # person watching a direct terminal (PowerShell) sees an actual bar
    # shrink instead of a bare number.
    _flush_safe_print("COUNTDOWN_BEGIN")
    _flush_safe_print(message)
    for i in range(seconds, 0, -1):
        _flush_safe_print(f"COUNTDOWN {i}")
        filled = "█" * i
        empty = "░" * (seconds - i)
        _flush_safe_print(f"  {_color(filled, _BLUE)}{_color(empty, _DIM)} {i}s")
        time.sleep(1)
    _flush_safe_print("COUNTDOWN_END")


def ensure_server_running(timeout_s: float = 45.0) -> subprocess.Popen | None:
    """Starts local_server.py if nothing is answering on SERVER_URL yet.
    Was 20.0 -- found live, twice in a row: under heavy system load
    (torch's own import chain plus everything else running at the same
    time), the server can genuinely take longer than 20s to come up
    even though it starts fine and answers moments later. Bumped to
    give it real room instead of failing a healthy startup.
    Returns the Popen handle if this call started it (so main() can leave
    it running for --show, same as the Electron app already does), or
    None if a server was already up."""
    try:
        urllib.request.urlopen(SERVER_URL, timeout=1)
        return None
    except (urllib.error.URLError, ConnectionError, OSError):
        pass

    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "local_server.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(SERVER_URL, timeout=1)
            return proc
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    raise SystemExit(f"local_server.py didn't come up within {timeout_s:.0f}s")


MODE_LABELS = {
    "habits": "Habits only -- your learned patterns; unsure emails are left for you",
    "hybrid": "Habits + reasoning -- habits first, the LLM thinks about the rest",
    "reasoning": "Reasoning only -- the LLM reads and decides every email",
}


def _redecide_progress() -> dict | None:
    """The server's {"done", "total", "active"} for the running mode switch,
    or None when it can't say (an older server, or a hiccup)."""
    try:
        with urllib.request.urlopen(SERVER_URL + "api/mode/progress", timeout=2) as resp:
            return json.loads(resp.read() or b"{}")
    except Exception:
        return None


def set_decision_mode(mode: str, poll_s: float = 1.0) -> int:
    """Tell the inbox server how to decide, and have it re-decide every email
    still pending that was decided some other way. Returns how many were
    re-decided.

    The request can take a while -- each re-decided email may be an LLM
    call -- so it runs on a worker thread while this one polls the server's
    progress and prints "re-deciding N/M" as it moves. Before, the run sat
    silently on "deciding ..." for a minute or more and read as a hang.

    A server left running from before decision modes existed has no
    /api/mode endpoint; that is reported plainly rather than letting the run
    carry on silently in the wrong mode."""
    req = urllib.request.Request(
        SERVER_URL + "api/mode", data=json.dumps({"mode": mode}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    outcome: dict = {}

    def _post():
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                outcome["redecided"] = int(json.loads(resp.read() or b"{}").get("redecided", 0))
        except BaseException as exc:          # re-raised on the calling thread
            outcome["error"] = exc

    worker = threading.Thread(target=_post, daemon=True)
    worker.start()
    shown = None
    while worker.is_alive():
        worker.join(poll_s)
        progress = _redecide_progress() if worker.is_alive() else None
        if progress and progress.get("active") and progress.get("total"):
            line = f"  re-deciding  {progress.get('done', 0)}/{progress['total']} emails"
            if line != shown:
                _flush_safe_print(line)
                shown = line

    error = outcome.get("error")
    if isinstance(error, urllib.error.HTTPError) and error.code == 404:
        raise SystemExit(
            "The inbox server running on 127.0.0.1:8765 predates decision modes. "
            "Close it (or restart the Intern app) and press Play again.")
    if error is not None:
        raise error
    return outcome.get("redecided", 0)


# Decisions that need real human-typed content (a reply, a forward, a
# schedule note) can never be auto-confirmed here -- this script has no
# real text to type, and confirming with none would either create an
# empty Gmail draft or an empty schedule note, defeating the whole point
# of the text box a human types into on the real page. cold_email has no
# control on this page at all -- it's not a reaction to an existing
# email, it lives on its own separate page. Only leave_alone can be
# confirmed with a single real click, since it needs no typed content --
# clicking the same real icon a human would click for it.
NEEDS_HUMAN_TEXT = {"reply", "forward", "schedule", "cold_email"}
IMMEDIATE_ICON = {"leave_alone": "#archiveBtn"}


def process_one(page, commit: bool, index: int, skipped: int = 0, dwell_ms: int = 0,
                 auto_draft_reply: bool = False, pointer: "Pointer | None" = None):
    """Reads one pending row off the real DOM, opens it, prints the real
    decision + rationale, then clicks the real icon for that decision
    (Archive for leave_alone) if --commit. Returns a result dict, or
    None once the inbox is empty.

    dwell_ms: how long to leave the opened email visible on screen before
    acting on it. Found live: with dwell_ms=0, main()'s own --pace only
    delayed BETWEEN emails, so a person watching the real browser window
    saw each email open and close in the same instant -- a flash, not
    something followable. Real emails need to actually stay on screen
    long enough to read before the next action happens.

    Confirming removes a row from the list, so committed runs read row
    `skipped` -- every row already skipped-in-place (see below) still
    sits above it, and every confirmed row is gone, so `skipped` always
    points at the next row actually needing a decision. A dry run never
    removes or skips anything, so it reads row `index` instead, advancing
    through the list without ever changing it.

    Forward/schedule/cold_email are not auto-confirmed by default -- see
    NEEDS_HUMAN_TEXT above. Those are left pending, in place, for a human
    to actually open and answer themselves. One exception for schedule:
    when committing and the email itself states both a date and a time
    ("September 3rd at 2pm"), that time is read from the email's own words
    by schedule_extract.py and the schedule is completed through the page
    like a person would. A date with no time, or no date at all, still
    stays pending -- a time is never invented.

    auto_draft_reply=True is a deliberate, explicitly authorized
    exception (see inbox_reply_llm.py's own docstring) for "reply" and
    "forward": a real AI-generated body is typed and sent for real,
    same shape as automate_cold_email.py's existing --commit exception.
    Forward's recipient is never LLM-invented -- forward_recipient()
    derives a synthetic team alias from the sender's own real domain,
    so only the note text is AI-generated, not the address itself.
    Schedule is still untouched by this flag -- a fabricated date would
    create a genuinely wrong calendar event, a materially worse mistake
    than an unreviewed forward note on mock data. (Schedule's own,
    separate exception above only ever uses a time the email states.)"""
    # Every click below goes through the pointer: with a real Pointer the
    # mouse visibly glides to the element and clicks it; with none, the
    # disabled Pointer falls back to Playwright's own invisible click.
    pointer = pointer or Pointer(page, enabled=False)
    row_index = skipped if commit else index
    row = page.locator("#rowList .row-item").nth(row_index)
    if row.count() == 0:
        return None

    # A visible beat before acting -- highlights the exact row about to
    # be opened, in Gmail's own blue, so a person watching can see WHICH
    # email the Agent is about to act on, not just the result after the
    # fact. Purely cosmetic: adds a CSS class local_ui/style.css already
    # defines an animation for, no behavior change.
    row.evaluate("el => el.classList.add('row-agent-active')")
    page.wait_for_timeout(500)

    pointer.click(row)
    page.wait_for_selector("#detailView:not([hidden])")
    # openMessage() holds the real rationale text back behind a ~0.5s
    # "thinking" shimmer (a CSS class, not a text change) before
    # revealing it -- reading #detailRationale before that class clears
    # would capture the shimmer's placeholder state, not the real text.
    page.wait_for_selector("#detailRationale:not(.rationale-thinking)")

    subject = page.locator("#detailSubject").inner_text()
    sender = page.locator("#detailSender").inner_text()
    # text_content(), not inner_text() -- #detailDecision is now a hidden
    # element (the "Suggested: ..." text was removed from what a human
    # sees), and inner_text() reads rendered text, returning "" for
    # anything hidden. text_content() reads the raw DOM text regardless
    # of visibility, which is what this script actually needs here.
    decision = page.locator("#detailDecision").text_content()
    rationale = page.locator("#detailRationale").inner_text()

    print(f"\n  {_color(sender, _BOLD)}")
    print(f"  {subject!r}")
    print(f"    decided: {_color(decision, _BLUE)}")
    print(f"    {_color('because:', _DIM)} {rationale}")

    if dwell_ms:
        page.wait_for_timeout(dwell_ms)

    if decision == "reply" and commit and auto_draft_reply:
        body_text = page.locator("#detailBody").inner_text()
        from inbox_reply_llm import generate_reply
        reply_text = generate_reply(sender, subject, body_text)
        if reply_text:
            pointer.click("#replyPillBtn")
            page.wait_for_selector("#replyBoxWrap:not([hidden])")
            # press_sequentially(), not fill() -- fill() sets the value in
            # one instant DOM write, nothing visible actually happens on
            # screen. press_sequentially() sends one real keystroke at a
            # time with a delay between each, so a person watching the
            # real browser window actually sees it being typed.
            page.locator("#replyBody").press_sequentially(reply_text, delay=35)
            print(f"    AI-drafted reply: {reply_text!r}")
            pointer.click("#sendBtn")
            page.wait_for_selector("#listView:not([hidden])")
            outcome = "confirmed (AI-drafted reply -- draft created, not sent)"
        else:
            pointer.click("#backBtn")
            outcome = "left pending -- LM Studio unavailable for auto-draft"
    elif decision == "forward" and commit and auto_draft_reply:
        body_text = page.locator("#detailBody").inner_text()
        from inbox_reply_llm import generate_forward_note, forward_recipient
        # sender is "Display Name <address@domain>" -- pull the address
        # out for forward_recipient() rather than fabricating one.
        match = re.search(r"<([^>]+)>", sender)
        sender_email = match.group(1) if match else sender
        note = generate_forward_note(sender, subject, body_text)
        recipient = forward_recipient(sender_email)
        if note:
            pointer.click("#forwardPillBtn")
            page.wait_for_selector("#replyBoxWrap:not([hidden])")
            # press_sequentially(), not fill() -- see the reply branch
            # above for why: a person watching the real browser window
            # needs to actually see this get typed, not just appear.
            page.locator("#forwardTo").press_sequentially(recipient, delay=35)
            page.locator("#replyBody").press_sequentially(note, delay=35)
            print(f"    AI-drafted forward to {recipient!r}: {note!r}")
            pointer.click("#sendBtn")
            page.wait_for_selector("#listView:not([hidden])")
            outcome = "confirmed (AI-drafted forward -- draft created, not sent)"
        else:
            pointer.click("#backBtn")
            outcome = "left pending -- LM Studio unavailable for auto-draft"
    elif decision == "schedule" and commit and (
            when := extract_event_time(subject, page.locator("#detailBody").inner_text())):
        # The email itself states when -- "September 3rd at 2pm" -- so the
        # schedule can be completed the way a person would: open Schedule,
        # write a note, set When, Send. The time is read from the email's
        # own words by schedule_extract.py, never generated: an email that
        # gives only a date, or no time at all, returns nothing and falls
        # through to the left-pending branch below, unchanged.
        print(f"    when: {_color(when.describe(), _BLUE)}  "
              f"{_color('read from the email:', _DIM)} {when.quote!r}")
        pointer.click("#scheduleBtn")
        page.wait_for_selector("#replyBoxWrap:not([hidden])")
        note = f"{subject} ({when.quote})"
        page.locator("#replyBody").press_sequentially(note, delay=35)
        pointer.click("#eventWhen")
        page.locator("#eventWhen").fill(when.as_input_value())
        page.wait_for_timeout(400)   # let the filled-in time be seen before Send
        pointer.click("#sendBtn")
        page.wait_for_selector("#listView:not([hidden])")
        outcome = f"confirmed (scheduled for {when.describe()}, from the email's own words)"
    elif decision == "undecided":
        # Habits-only mode, with no habit confident enough to act on. Left
        # for a person, exactly as a pending email would be -- never guessed.
        pointer.click("#backBtn")
        outcome = "left pending -- no confident habit, left for you to decide"
    elif decision in NEEDS_HUMAN_TEXT:
        pointer.click("#backBtn")
        outcome = ("left pending -- needs a real reply typed by a human" if decision in ("reply", "forward")
                    else "left pending -- needs real content typed by a human")
    elif commit:
        pointer.click(IMMEDIATE_ICON[decision])
        page.wait_for_selector("#listView:not([hidden])")
        outcome = "confirmed"
    else:
        pointer.click("#backBtn")
        outcome = "skipped (dry run)"
    print(f"    -> {_color(outcome, _outcome_color(outcome))}")

    return {"sender": sender, "subject": subject, "decision": decision,
            "rationale": rationale, "outcome": outcome}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", action="store_true",
                    help="actually click Confirm for each email; the default is a dry run")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N pending emails")
    ap.add_argument("--pace", type=float, default=1.5,
                    help="seconds to pause on each email so a human can follow it (default: 1.5)")
    ap.add_argument("--headless", action="store_true",
                    help="run without a visible browser window (default: visible)")
    ap.add_argument("--mode", choices=list(MODE_LABELS), default="hybrid",
                    help="how each email is decided: habits (learned patterns only), hybrid "
                         "(habits, then the LLM for the rest -- the default) or reasoning "
                         "(the LLM decides every email)")
    ap.add_argument("--no-pointer", action="store_true",
                    help="click through the page invisibly instead of moving the real mouse pointer "
                         "(the pointer is always off when --headless)")
    ap.add_argument("--log", type=Path, default=None,
                    help="where to write the run log (default: data/runs/)")
    ap.add_argument("--auto-draft-reply", action="store_true",
                    help="explicit, deliberate exception: for 'reply' and 'forward' decisions, generate "
                         "a real body via LM Studio and send it for real, instead of leaving it pending "
                         "for a human to type -- requires --commit, has no effect otherwise. Schedule "
                         "still always needs a real human-picked date.")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright, Error as PlaywrightError

    print(f"\n  mode  {'COMMIT' if args.commit else 'dry run'}")
    print_countdown()

    started_server = ensure_server_running()
    print(f"  deciding  {MODE_LABELS[args.mode]}")
    redecided = set_decision_mode(args.mode)
    if redecided:
        print(f"            re-decided {redecided} pending email(s) under this mode")

    results = []
    with sync_playwright() as p:
        # The window Launch opened when there is one (workspace.py), else a
        # fresh maximised browser as before.
        browser, page, _attached = open_page(p, SERVER_URL, args.headless)
        page.goto(SERVER_URL)
        pointer = Pointer(page, enabled=not (args.headless or args.no_pointer))
        # A fixed short wait here used to be enough when /api/inbox only
        # ever read already-cached decisions, but a real LLM classify()
        # call can take a couple of seconds per email and this endpoint
        # can be classifying many at once on first poll -- a fixed 600ms
        # wait raced ahead of the real response and read an empty list.
        # Wait for the actual response the click causes instead of
        # guessing how long it takes.
        # 3 minutes, not 1: the server sorts every new email before it
        # answers, and right after Launch's reset that is all 30 (~45 s
        # measured, more with a slow LLM) -- a 60 s wait could time out and
        # skip the whole inbox phase.
        with page.expect_response(lambda r: "/api/inbox" in r.url and r.request.method == "GET",
                                   timeout=INBOX_LOAD_TIMEOUT_MS):
            pointer.click("#toolbarRefreshBtn")
        page.wait_for_timeout(200)  # let the synchronous DOM render after the fetch settle

        banner(1, "Working through the inbox")
        skipped = 0
        try:
            while args.limit is None or len(results) < args.limit:
                result = process_one(page, args.commit, len(results), skipped,
                                      dwell_ms=int(args.pace * 1000),
                                      auto_draft_reply=args.auto_draft_reply,
                                      pointer=pointer)
                if result is None:
                    break
                results.append(result)
                # startswith(), not ==: creating a real AI-drafted reply
                # also removes the row, same as a plain confirm, but
                # reports a more specific outcome string ("confirmed
                # (AI-drafted reply -- draft created, not sent)") --
                # an exact-match check here would wrongly
                # treat that row as still-pending and skip past whatever
                # row replaces it next.
                if not result["outcome"].startswith("confirmed"):
                    skipped += 1
                time.sleep(args.pace)
        except PlaywrightError as exc:
            # The browser/page can close out from under this loop for
            # reasons entirely outside this script's own control (the OS
            # reclaiming it, another heavy process crowding it out) --
            # nothing here ever closes it itself. Whatever was already
            # confirmed before that point is real and already recorded in
            # `results`; report that honestly instead of dying with a raw
            # traceback and losing the summary of real work already done.
            print(f"\n  Browser closed unexpectedly mid-run ({exc.__class__.__name__}) -- "
                  f"stopping here with what was already completed.")

        try:
            if not args.headless:
                page.wait_for_timeout(1500)
            browser.close()
        except PlaywrightError:
            pass  # already gone -- nothing left to close

    if started_server is not None:
        started_server.terminate()

    banner(2, "Result")
    print(f"  emails processed  {_color(str(len(results)), _BOLD)}")
    for r in results:
        # Pad the plain text to a fixed width FIRST, then wrap it in color
        # codes -- coloring first would pad the invisible ANSI bytes along
        # with it, under-filling the visible column width.
        padded_decision = f"{r['decision']:<14}"
        print(f"    {_color(padded_decision, _outcome_color(r['outcome']))} {r['subject']}")
    needing_reply = [r for r in results
                      if r["decision"] in ("reply", "forward") and not r["outcome"].startswith("confirmed")]
    if needing_reply:
        print(f"\n  {len(needing_reply)} email(s) need a real reply typed by a human -- "
              f"open http://localhost:8765/ yourself to answer them.")
    if not args.commit:
        print("\n  Nothing was clicked for real. Re-run with --commit to actually confirm each one.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log or (REPO / "data" / "runs" / f"automate_inbox_{stamp}.json")
    if not log_path.is_absolute():
        log_path = REPO / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "commit": args.commit, "processed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"  run log           {log_path.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
