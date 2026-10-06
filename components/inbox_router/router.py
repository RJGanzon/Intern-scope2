"""
components/inbox_router/router.py
====================================
Standalone entrypoint, spawned as a child process by
app/recorder_bridge.py's start_inbox_router() -- the same shape as
run_task.py being spawned for a capsule Play run. Reads {"cmd": ...} lines
from stdin, emits {"event": ...} lines on stdout. Runs from repo root:

    python components/inbox_router/router.py

Commands (stdin, one per line)
-------------------------------
  {"cmd": "confirm", "message_id": "...", "decision": "..."}
  {"cmd": "override", "message_id": "...", "new_decision": "...", "reason": ""}
  {"cmd": "shutdown"}

Events (stdout, one per line)
-------------------------------
  {"event": "ready"}
  {"event": "inbox_poll_started", "provider": "mock"|"real", "poll_interval_s": 30}
  {"event": "inbox_routed", "message_id":..., "subject":..., "sender":...,
   "decision":..., "confidence":..., "rationale":..., "layer": "rule"|"llm", ...}
  {"event": "inbox_draft_created", "message_id":..., "draft_id":..., "decision":...}
  {"event": "inbox_confirm_applied", "message_id":..., "decision":..., "draft_id":...}
  {"event": "inbox_override_applied", "message_id":..., "old_decision":..., "new_decision":..., "reason":...}
  {"event": "inbox_log", "line":"...", "level":"ok"|"err"|"dim"}
  {"event": "inbox_error", "message":"..."}
  {"event": "inbox_stopped"}
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_COMP = os.path.join(_ROOT, "components")
for _p in (_ROOT, _COMP, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Same hand-rolled .env loader as run_task.py -- no python-dotenv anywhere
# in this project.
_env_path = os.path.join(_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from gmail_client import EmailMessage, GmailClientBase, RealGmailClient, get_gmail_client
from llm_classifier import LLMClassifier
from pattern_profile import PatternProfile
from routing_rules import RuleLayer
from inbox_agent import DEFAULT_CHECKPOINT_PATH, InboxAgent
from decision_recorder import DEFAULT_EXAMPLES_PATH, record_example
from reply_recorder import DEFAULT_REPLY_EXAMPLES_PATH, record_reply_example
from schedule_recorder import DEFAULT_SCHEDULE_LOG_PATH, record_schedule_entry
from calendar_client import CalendarClientBase, MockCalendarClient, get_calendar_client

HISTORY_PATH = os.path.join(_THIS_DIR, "data", "routed_history.json")
SENT_LOOKBACK_DAYS = 90
DEFAULT_POLL_INTERVAL_S = 30.0

# Follows Scope #1's OWN architecture as it exists on master (run_task.py's
# finally: block -- a local metrics computation + a plain inline try/except
# JSONL append), not the shared components/shared/ recorder from the
# (unmerged) unification branch -- deliberately no dependency on that
# branch. Scope #1 records once per run; Scope #3 is a forever-polling
# daemon with no such natural end, so it accumulates counters across the
# whole session and records once, on shutdown.
_SESSION_METRICS_PATH = os.path.join(_ROOT, "data", "output", "run_metrics.jsonl")


def emit(event: str, **fields) -> None:
    # write() then a separately-guarded flush() -- same reasoning as
    # recorder_bridge.py's emit(): this process is spawned with no console,
    # and an unguarded stdout.flush() can raise OSError there on Windows.
    print(json.dumps({"event": event, **fields}))
    try:
        sys.stdout.flush()
    except OSError:
        pass


def _pick_provider() -> tuple[str, str]:
    """Same preference order as this project's other entry points: prefer
    whichever real API key is actually set, else local LM Studio, else no
    LLM at all (RuleLayer + "leave alone everything the rules can't
    resolve" still works with zero LLM configured)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("GROQ_API_KEY"):
        return "groq", os.environ["GROQ_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini", os.environ["GEMINI_API_KEY"]
    return "lmstudio", ""


class InboxRouter:
    def __init__(self, gmail_client: GmailClientBase, profile: PatternProfile,
                 rule_layer: RuleLayer, llm_classifier: LLMClassifier,
                 history_path: str = HISTORY_PATH,
                 poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
                 inbox_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                 examples_path: str = DEFAULT_EXAMPLES_PATH,
                 reply_examples_path: str = DEFAULT_REPLY_EXAMPLES_PATH,
                 schedule_log_path: str = DEFAULT_SCHEDULE_LOG_PATH,
                 calendar_client: Optional[CalendarClientBase] = None) -> None:
        self._gmail = gmail_client
        self._profile = profile
        self._rules = rule_layer
        self._llm = llm_classifier
        self._agent = InboxAgent(profile, rule_layer, llm_classifier,
                                  checkpoint_path=inbox_checkpoint_path)
        self._history_path = history_path
        self._poll_interval_s = poll_interval_s
        self._examples_path = examples_path
        self._reply_examples_path = reply_examples_path
        self._schedule_log_path = schedule_log_path
        self._calendar = calendar_client if calendar_client is not None else get_calendar_client()
        self._stop = False
        # In-memory cache of what this process has routed, so confirm/
        # override don't need a disk round-trip in the common case --
        # _find_history_entry() below is still the fallback for a command
        # referencing a message routed in an earlier process lifetime.
        self._pending: dict[str, dict] = {}

        # ── session metrics accumulators (one row written on shutdown) ──
        self._session_start_ts = time.time()
        self._routed_count = 0
        self._decision_counts: dict[str, int] = {}
        self._layer_counts = {"rule": 0, "llm": 0, "fast_fill": 0}
        self._confirmed_count = 0
        self._overridden_count = 0
        self._confidence_sum = 0.0
        self._confidence_count = 0

    def bootstrap(self) -> None:
        """The "learned passively, no manual labeling" step -- runs once at
        startup, before the first poll."""
        since_dt = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() - SENT_LOOKBACK_DAYS * 86400, tz=timezone.utc,
        )
        since_iso = since_dt.isoformat()
        sent = self._gmail.list_sent(since_iso)
        inbox = self._gmail.list_recent_inbox(since_iso)
        self._profile.observe_sent_history(sent, inbox)
        emit("inbox_log",
             line=f"Pattern profile bootstrapped from {len(sent)} sent + {len(inbox)} inbox messages.",
             level="dim")

    def poll_once(self) -> list:
        routed = []
        for message in self._gmail.list_inbox_unprocessed():
            entry = self._classify_and_record(message)
            self._gmail.mark_processed(message.id)
            emit("inbox_routed", **entry)
            routed.append(entry)
        return routed

    def _classify_and_record(self, message: EmailMessage) -> dict:
        result = self._agent.decide(message)
        decision, confidence, rationale = result.decision, result.confidence, result.rationale
        capsule_name, forward_to, layer = result.capsule_name, result.forward_to, result.layer

        entry = {
            "message_id": message.id, "thread_id": message.thread_id,
            "subject": message.subject, "sender": message.sender,
            "sender_email": message.sender_email, "received_at": message.received_at,
            "body_text": message.body_text,
            "decision": decision, "capsule_name": capsule_name, "confidence": confidence,
            "rationale": rationale, "layer": layer, "forward_to": forward_to,
            # Which decision mode produced this row, so a later mode switch
            # can tell which pending emails are already decided its way.
            "mode": self._agent.mode,
            "status": "pending", "draft_id": "",
            "routed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._pending[message.id] = entry
        self._append_history(entry)

        self._routed_count += 1
        self._decision_counts[decision] = self._decision_counts.get(decision, 0) + 1
        self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1
        if isinstance(confidence, (int, float)):
            self._confidence_sum += confidence
            self._confidence_count += 1

        return entry

    # ── confirm / override -- the only place create_draft() is ever called ──
    def confirm_suggestion(self, message_id: str, decision: str, reply_body: str = "",
                            event_start: str = "", event_end: str = "", forward_to: str = "") -> None:
        """reply_body is real text a human actually typed -- never
        generated. When decision is "reply"/"forward" and reply_body is
        given, that's the draft's real content, and it's saved as a real
        example for the reply model to learn from later. When no
        reply_body is given, the draft is created empty rather than
        asking the LLM to invent content -- this project doesn't put
        AI-authored words in front of anyone as if they were the user's
        own.

        forward_to: the human's own typed recipient for a "forward"
        decision. Takes priority over entry["forward_to"] (the AI
        pipeline's own guess, from a sender-pattern match) when given --
        a human explicitly typing a recipient is a stronger signal than a
        pattern guess. Falls back to the AI's guess only when the human
        left this blank, e.g. confirming an AI-suggested forward as-is."""
        entry = self._pending.get(message_id) or self._find_history_entry(message_id)
        if entry is None:
            emit("inbox_error", message=f"Unknown message id: {message_id}")
            return
        message = self._gmail.get_message(message_id)
        draft_id = ""
        if decision in ("reply", "forward") and message is not None:
            to = (forward_to.strip() or entry.get("forward_to", "")) if decision == "forward" else message.sender_email
            subject = ("Fwd: " if decision == "forward" else "Re: ") + message.subject
            draft_id = self._gmail.create_draft(to=to, subject=subject, body=reply_body, thread_id=message.thread_id)
            emit("inbox_draft_created", message_id=message_id, draft_id=draft_id, decision=decision)
            if reply_body.strip():
                try:
                    record_reply_example(message, reply_body, source="live", path=self._reply_examples_path)
                except Exception as exc:
                    emit("inbox_log", line=f"Failed to record reply example: {exc}", level="err")
        elif decision == "schedule" and message is not None:
            if reply_body.strip():
                try:
                    record_schedule_entry(message, reply_body, path=self._schedule_log_path)
                except Exception as exc:
                    emit("inbox_log", line=f"Failed to record schedule entry: {exc}", level="err")
            if event_start.strip() and event_end.strip():
                try:
                    self._calendar.create_event(summary=message.subject, description=reply_body,
                                                 start_iso=event_start, end_iso=event_end)
                except Exception as exc:
                    emit("inbox_log", line=f"Failed to create calendar event: {exc}", level="err")
        elif decision == "cold_email":
            emit("inbox_log", line="Cold Email is handled on its own page -- no action taken here.", level="info")
        # leave_alone needs no Gmail-side action at all.
        entry["status"] = "confirmed"
        entry["decision"] = decision
        entry["draft_id"] = draft_id
        self._update_history_entry(entry)
        if message is not None:
            self._profile.record_confirmed_decision(message, decision)
            try:
                record_example(message, decision, source="live", path=self._examples_path)
            except Exception as exc:
                emit("inbox_log", line=f"Failed to record training example: {exc}", level="err")
        self._confirmed_count += 1
        emit("inbox_confirm_applied", message_id=message_id, decision=decision, draft_id=draft_id)

    def override_decision(self, message_id: str, new_decision: str, reason: str = "",
                           reply_body: str = "", event_start: str = "", event_end: str = "",
                           forward_to: str = "") -> None:
        """reply_body: same contract as confirm_suggestion() -- real
        human-typed text only. Overriding TO "reply"/"forward" now
        creates a real draft (it didn't before -- there was no way to
        override into a reply and actually get a draft out of it), and
        saves reply_body as a real example when it's given.

        forward_to: same contract as confirm_suggestion()'s -- the
        human's own typed recipient, preferred over the AI's guess."""
        entry = self._pending.get(message_id) or self._find_history_entry(message_id)
        if entry is None:
            emit("inbox_error", message=f"Unknown message id: {message_id}")
            return
        old_decision = entry.get("decision", "")
        message = self._gmail.get_message(message_id)
        draft_id = ""
        if new_decision in ("reply", "forward") and message is not None:
            to = (forward_to.strip() or entry.get("forward_to", "")) if new_decision == "forward" else message.sender_email
            subject = ("Fwd: " if new_decision == "forward" else "Re: ") + message.subject
            draft_id = self._gmail.create_draft(to=to, subject=subject, body=reply_body, thread_id=message.thread_id)
            emit("inbox_draft_created", message_id=message_id, draft_id=draft_id, decision=new_decision)
            if reply_body.strip():
                try:
                    record_reply_example(message, reply_body, source="live", path=self._reply_examples_path)
                except Exception as exc:
                    emit("inbox_log", line=f"Failed to record reply example: {exc}", level="err")
        elif new_decision == "schedule" and message is not None:
            if reply_body.strip():
                try:
                    record_schedule_entry(message, reply_body, path=self._schedule_log_path)
                except Exception as exc:
                    emit("inbox_log", line=f"Failed to record schedule entry: {exc}", level="err")
            if event_start.strip() and event_end.strip():
                try:
                    self._calendar.create_event(summary=message.subject, description=reply_body,
                                                 start_iso=event_start, end_iso=event_end)
                except Exception as exc:
                    emit("inbox_log", line=f"Failed to create calendar event: {exc}", level="err")
        elif new_decision == "cold_email":
            emit("inbox_log", line="Cold Email is handled on its own page -- no action taken here.", level="info")
        entry["decision"] = new_decision
        entry["status"] = "overridden"
        entry["override_reason"] = reason
        entry["draft_id"] = draft_id
        self._update_history_entry(entry)
        if message is not None:
            self._profile.record_override(message, old_decision, new_decision)
            try:
                record_example(message, new_decision, source="live", path=self._examples_path)
            except Exception as exc:
                emit("inbox_log", line=f"Failed to record training example: {exc}", level="err")
        self._overridden_count += 1
        emit("inbox_override_applied", message_id=message_id, old_decision=old_decision,
             new_decision=new_decision, reason=reason)

    # ── routed_history.json -- single writer (this process), main.js only reads it
    def _append_history(self, entry: dict) -> None:
        history = self._load_history()
        history.append(entry)
        self._save_history(history)

    def _update_history_entry(self, entry: dict) -> None:
        """Updates the LATEST history row for this message_id -- matching
        pending_entries()'s own "latest wins" dedup exactly. A message can
        end up with more than one history row for the same id (routed,
        never confirmed, then re-polled and routed again on a later
        restart); updating only the FIRST match (the old behavior) meant
        confirming/overriding silently patched a stale, possibly
        already-confirmed row while the actually-current one -- the one
        pending_entries() reports -- never changed, so a message could
        never actually be confirmed once duplicate rows existed. Found
        live: confirming mock-001 kept "succeeding" while it stayed
        pending forever, traced to 6 history rows for that one id."""
        history = self._load_history()
        last_match_index = None
        for i, existing in enumerate(history):
            if existing.get("message_id") == entry.get("message_id"):
                last_match_index = i
        if last_match_index is not None:
            history[last_match_index] = entry
            self._save_history(history)
            return
        history.append(entry)
        self._save_history(history)

    def _find_history_entry(self, message_id: str) -> Optional[dict]:
        """Returns the LATEST history row for this message_id, same
        "latest wins" rule as _update_history_entry()/pending_entries() --
        see _update_history_entry()'s docstring for why returning the
        first match (the old behavior) was wrong whenever duplicate rows
        exist for one message."""
        match = None
        for entry in self._load_history():
            if entry.get("message_id") == message_id:
                match = entry
        return match

    @property
    def decision_mode(self) -> str:
        return self._agent.mode

    def set_decision_mode(self, mode: str) -> int:
        """Switch how decisions are reached (see inbox_agent.DECISION_MODES)
        and re-decide every pending email not already decided this way.

        Decisions are made once per email and saved, so without the
        re-decide step a mode chosen at Play would only ever apply to mail
        that arrived afterwards, and the run would show the old mode's
        answers. History is append-only and pending_entries() keeps only
        the newest row per message, so each re-decided email simply gains a
        newer pending row; nothing already confirmed is touched. Returns how
        many emails were re-decided.

        Only rows decided under a different mode -- or with no mode recorded,
        i.e. saved before rows carried one -- are redone. Play always sends
        a mode, usually the default it already runs in, and re-deciding all
        30 emails then cost ~30 sequential LLM calls (a minute or more of
        silence that read as a hang) to arrive at the answers already on
        screen. Progress is published through redecide_progress() so the
        run can show it while this blocks."""
        self._agent.mode = mode          # validates; raises on an unknown mode
        stale = [e for e in self.pending_entries() if e.get("mode") != mode]
        self._redecide_progress = {"mode": mode, "done": 0, "total": len(stale), "active": True}
        redecided = 0
        try:
            for entry in stale:
                message = self._gmail.get_message(entry["message_id"])
                if message is not None:
                    self._classify_and_record(message)
                    redecided += 1
                self._redecide_progress["done"] += 1
        finally:
            self._redecide_progress["active"] = False
        emit("inbox_log", line=f"Decision mode set to {mode!r}; re-decided {redecided} pending email(s).",
             level="info")
        return redecided

    def redecide_progress(self) -> dict:
        """How far the current (or last) set_decision_mode() has got:
        {"mode", "done", "total", "active"}. A copy, safe to read from
        another thread while the re-decide loop runs."""
        return dict(getattr(self, "_redecide_progress",
                            {"mode": self._agent.mode, "done": 0, "total": 0, "active": False}))

    def pending_entries(self) -> list:
        """Every history entry still awaiting a Confirm/Override -- exposed
        as a real public method (rather than reaching into the private
        _load_history()) for local_server.py, a second driver of this same
        class outside router.py's own stdin/stdout protocol.

        Deduplicated by message_id, keeping only the latest history row per
        message -- a message can end up with more than one "pending" row on
        disk (e.g. it was routed, never confirmed, then later re-polled and
        routed again), and it should only ever appear once here. The most
        recent row for a given id also correctly wins over an older one:
        if the latest row is "confirmed"/"overridden", the message is gone
        from this list even if an earlier abandoned "pending" row for the
        same id is still sitting in history."""
        latest_by_id: dict = {}
        for e in self._load_history():
            latest_by_id[e.get("message_id")] = e
        return [e for e in latest_by_id.values() if e.get("status") == "pending"]

    def list_unprocessed_stubs(self) -> list:
        """What's waiting to be triaged, before any reasoning has happened --
        sender/subject only, no decision. A non-destructive peek: unlike
        poll_once()/process_next_unprocessed(), this never marks anything
        processed. Lets a UI show "N emails waiting" the way a human would
        glance at an inbox before reading anything in it."""
        return [
            {"message_id": m.id, "subject": m.subject, "sender": m.sender,
             "sender_email": m.sender_email}
            for m in self._gmail.list_inbox_unprocessed()
        ]

    def process_next_unprocessed(self) -> Optional[dict]:
        """Classify exactly one waiting message through the real pipeline
        (rule layer -> trained agent -> LLM fallback -- the same
        _classify_and_record() poll_once() already calls per message) and
        mark it processed. Returns None once nothing is left. Exists so a
        UI can drive the pipeline one visible step at a time instead of
        poll_once()'s all-at-once loop, without changing what the pipeline
        actually decides or why."""
        unprocessed = self._gmail.list_inbox_unprocessed()
        if not unprocessed:
            return None
        message = unprocessed[0]
        entry = self._classify_and_record(message)
        self._gmail.mark_processed(message.id)
        emit("inbox_routed", **entry)
        return entry

    def list_practice_inbox(self) -> list:
        """Every mock inbox message available to practice-demonstrate on,
        unfiltered by processed state -- unlike poll_once()'s
        list_inbox_unprocessed(), practice mode is meant to be repeatable,
        not a one-shot triage queue. Wraps the same list_recent_inbox()
        bootstrap() already uses for a wide lookback window."""
        since_iso = "2020-01-01T00:00:00+00:00"  # effectively "everything" for the mock fixture
        return self._gmail.list_recent_inbox(since_iso)

    def record_practice_decision(self, message_id: str, decision: str, reply_body: str = "") -> None:
        """A raw human demonstration -- no AI suggestion involved anywhere,
        the opposite of confirm_suggestion()/override_decision(). Fetches
        the real message and records it exactly like every other recorded
        example, via the same decision_recorder.record_example() call.
        Also folds into the sender-pattern profile the same way a real
        confirm does, since a genuine demonstration is at least as strong
        a signal as a confirm.

        reply_body: same contract as confirm_suggestion()'s -- real
        human-typed text only, never invented. When given for a
        reply/forward/schedule decision, it's recorded as real content
        the same way confirm_suggestion()/override_decision() do, via the
        same record_reply_example()/record_schedule_entry() calls. A
        blank reply_body records nothing extra -- the decision label
        alone still gets recorded either way."""
        message = self._gmail.get_message(message_id)
        if message is None:
            emit("inbox_error", message=f"Unknown message id: {message_id}")
            return
        if reply_body.strip():
            try:
                if decision in ("reply", "forward"):
                    record_reply_example(message, reply_body, source="live", path=self._reply_examples_path)
                elif decision == "schedule":
                    record_schedule_entry(message, reply_body, path=self._schedule_log_path)
            except Exception as exc:
                emit("inbox_log", line=f"Failed to record practice reply/schedule content: {exc}", level="err")
        record_example(message, decision, source="live", path=self._examples_path)
        self._profile.record_confirmed_decision(message, decision)

    def _load_history(self) -> list:
        if not os.path.exists(self._history_path):
            return []
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                return json.load(f).get("messages", [])
        except Exception:
            return []

    def _save_history(self, history: list) -> None:
        os.makedirs(os.path.dirname(self._history_path), exist_ok=True)
        with open(self._history_path, "w", encoding="utf-8") as f:
            json.dump({"messages": history}, f, indent=2)

    # ── session metrics -- one row per session, written on shutdown ────────
    def _record_session_metrics(self, path: str = None) -> None:
        """Persists this session's accumulated counters as one JSONL row,
        following Scope #1's OWN architecture on master (run_task.py's
        finally: block) rather than importing the shared recorder from the
        unification branch. Never raises -- recording a session's metrics
        must never crash the shutdown path."""
        target = path if path is not None else _SESSION_METRICS_PATH
        avg_confidence = (
            self._confidence_sum / self._confidence_count
            if self._confidence_count else None
        )
        row = {
            "scope": "scope3",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_duration_sec": time.time() - self._session_start_ts,
            "messages_routed": self._routed_count,
            "decisions": dict(self._decision_counts),
            "layer": dict(self._layer_counts),
            "confirmed": self._confirmed_count,
            "overridden": self._overridden_count,
            "avg_confidence": avg_confidence,
        }
        try:
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as _me:
            emit("inbox_log", line=f"Session metrics persist failed: {_me}", level="err")

    # ── main loop ─────────────────────────────────────────────────────────
    def run_forever(self) -> None:
        threading.Thread(target=self._read_stdin_commands, daemon=True).start()
        provider_kind = "real" if isinstance(self._gmail, RealGmailClient) else "mock"
        emit("inbox_poll_started", provider=provider_kind, poll_interval_s=self._poll_interval_s)
        self.bootstrap()
        while not self._stop:
            try:
                self.poll_once()
            except Exception as exc:
                emit("inbox_error", message=f"Poll failed: {exc}")
            # Sleep in short slices so "shutdown" doesn't have to wait out
            # a full poll interval to take effect.
            for _ in range(int(self._poll_interval_s * 10)):
                if self._stop:
                    break
                time.sleep(0.1)
        self._record_session_metrics()
        emit("inbox_stopped")

    def _read_stdin_commands(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                emit("inbox_error", message=f"Bad JSON: {line!r}")
                continue
            cmd = msg.get("cmd")
            try:
                if cmd == "confirm":
                    self.confirm_suggestion(msg.get("message_id", ""), msg.get("decision", ""),
                                             reply_body=msg.get("reply_body", ""))
                elif cmd == "override":
                    self.override_decision(msg.get("message_id", ""), msg.get("new_decision", ""),
                                            msg.get("reason", ""), reply_body=msg.get("reply_body", ""))
                elif cmd == "shutdown":
                    self._stop = True
                    break
                else:
                    emit("inbox_error", message=f"Unknown command: {cmd!r}")
            except Exception as exc:
                # One bad command can't be allowed to kill this thread --
                # it's the only thing still reading confirm/override
                # commands for the rest of the process's life.
                emit("inbox_error", message=f"Command failed: {exc}")


def main() -> None:
    emit("ready")
    gmail_client = get_gmail_client()
    profile = PatternProfile()
    rule_layer = RuleLayer(profile)
    provider, api_key = _pick_provider()
    classifier = LLMClassifier(provider=provider, api_key=api_key)
    router = InboxRouter(gmail_client, profile, rule_layer, classifier)
    router.run_forever()


if __name__ == "__main__":
    main()
