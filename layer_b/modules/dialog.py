#!/usr/bin/env python3
# layer_b/modules/dialog.py
"""
Dialog broker (Layer B) - the single owner of "what question is the robot
waiting on an answer to, and does this utterance answer it?".

WHY THIS EXISTS
---------------
Asking the human a question and treating "the next thing they say" as the
answer used to be done independently in two modules, each racing on the raw
picarx/audio/heard stream:

  - curiosity.py  ("is that a chair or a speaker?") captured the next
    utterance as a label, guarded after the fact by looks_like_label_answer;
  - companion.py  ("what did you want me to do?") captured the next utterance
    as a correction.

Nothing coordinated them, so with two questions open the same utterance could
be grabbed by the wrong one, and a command or a web-console button published on
the same heard topic could be swallowed as an answer (the bug curiosity's guard
was bolted on to patch).

This broker replaces "capture the next utterance" with an explicit protocol and
ONE global open-question registry (one question at a time, exactly as each
module tried to enforce locally). It is a pure TURN-TAKING interceptor: it
watches heard, and when the live question is genuinely answered it routes the
answer to whoever asked. Anything that ISN'T an answer it leaves completely
alone, so the normal command/chat pipeline (field_agent, companion,
tools_registry, ...) is untouched.

The addressing half of attention ("is this even for me?" - wake word,
conversation window, command shape) lives in the shared attention.py model and
is still applied by field_agent's forwarding path; this module only owns the
"is this an answer to my open question?" half.

PROTOCOL
--------
  in   picarx/dialog/ask     {asker, question_id, kind, options?, ttl?, prompt?}
         A module announces it just asked something and wants the next real
         answer routed back. Registering a new question replaces any question
         still open (and emits a 'cleared' for the old one), mirroring the
         old single-pending behaviour.
  in   picarx/audio/heard    the raw transcript stream (repairs/echoes skipped)
  out  picarx/dialog/answer  {question_id, asker, text, confidence, kind, options}
         The routed answer - the asker consumes only its own question_id.
  out  picarx/dialog/cleared {question_id, asker, reason}
         The question ended without being answered: 'expired' (ttl passed) or
         'replaced' (a newer question took its place). Lets the asker fall back
         (e.g. curiosity's escalate-to-LLM).

Fail-soft and stdlib-only. A sweeper thread expires stale questions even in a
silent room so an unanswered question always resolves.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config
import attention

import threading
import time

HEARD_TOPIC = "picarx/audio/heard"
ASK_TOPIC = "picarx/dialog/ask"
ANSWER_TOPIC = "picarx/dialog/answer"
CLEARED_TOPIC = "picarx/dialog/cleared"

# Wake phrases and the no-wake-word follow-up window are shared with
# field_agent's forwarding path; both read them here so there's one source of
# truth. (field_agent still owns the forwarding decision; the broker keeps the
# window only so it never mistakes a wake-addressed fresh command for an answer.)
CONVERSATION_WINDOW_SEC = float(robot_config.get(
    "dialog", "conversation_window_sec", 45.0, env="DIALOG_CONVERSATION_WINDOW_SEC"))
EXPIRY_SWEEP_SEC = 1.0   # how often the background sweeper checks the live deadline


WAKE_PHRASES = attention.normalize_wake_phrases(robot_config.get(
    "dialog", "wake_phrases", "robot,hey robot,computer",
    env="DIALOG_WAKE_PHRASES"))


class DialogBroker:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.question = None            # the single live attention.Question, or None
        self.last_directed_at = 0.0     # last wake/command-shaped utterance (window base)

    # ---------- question registry ----------

    def on_ask(self, payload):
        """A module registers the question it just asked. Replaces any still-open
        question (one at a time), emitting a 'cleared' for the displaced one."""
        asker = payload.get("asker")
        qid = payload.get("question_id")
        if not asker or not qid:
            print(f"Dialog: ignoring malformed ask {payload}")
            return
        now = time.time()
        ttl = payload.get("ttl")
        deadline = now + float(ttl) if ttl else 0.0
        q = attention.Question(
            id=qid, asker=asker,
            kind=payload.get("kind", attention.FREEFORM),
            options=payload.get("options") or [],
            deadline=deadline, prompt=payload.get("prompt"))
        with self.lock:
            displaced = self.question
            self.question = q
        if displaced is not None and not displaced.expired(now):
            self._emit_cleared(displaced, "replaced")
        print(f"Dialog: {asker} asked (id={qid}, kind={q.kind}, "
              f"options={q.options}, ttl={ttl})")

    def _emit_cleared(self, question, reason):
        self.bus.publish(CLEARED_TOPIC, {
            "question_id": question.id, "asker": question.asker,
            "reason": reason, "ts": time.time()})
        print(f"Dialog: question {question.id} ({question.asker}) cleared - {reason}")

    # ---------- utterance routing ----------

    def on_heard(self, payload):
        """Route one heard utterance IF it answers the live question; otherwise
        leave it alone for the normal command/chat pipeline. Repaired text and
        console-correction echoes are never answers (same guard the old capture
        paths used)."""
        if payload.get("source") in ("intent_repair", "user_correction"):
            return
        text = (payload.get("text") or "").strip()
        if not text:
            return
        now = time.time()

        # Drop a stale question before considering this utterance as its answer.
        expired = None
        with self.lock:
            q = self.question
            if q is not None and q.expired(now):
                self.question = None
                expired, q = q, None
        if expired is not None:
            self._emit_cleared(expired, "expired")

        # An explicitly wake-addressed utterance ("robot, stop") is a fresh
        # command to the robot, never the answer to a pending question - so it
        # doesn't get swallowed as one. Everything else is judged by kind.
        wake_addressed = attention.strip_wake_phrase(text, WAKE_PHRASES) is not None
        if wake_addressed:
            self.last_directed_at = now
        elif attention.classify(
                text, wake_phrases=WAKE_PHRASES,
                in_conversation=(now - self.last_directed_at) < CONVERSATION_WINDOW_SEC
             ).reason == attention.COMMAND_SHAPE:
            # Track command-shaped speech too, so the window reflects real
            # addressing the same way field_agent's does.
            self.last_directed_at = now

        if q is None or wake_addressed or not attention.answers_question(text, q):
            return   # not an answer to anything we're waiting on - hands off

        with self.lock:
            if self.question is q:
                self.question = None
        self.bus.publish(ANSWER_TOPIC, {
            "question_id": q.id, "asker": q.asker, "text": text,
            "confidence": payload.get("confidence"),
            "kind": q.kind, "options": q.options, "ts": now})
        print(f"Dialog: routed answer to {q.asker} (id={q.id}): '{text}'")

    # ---------- expiry sweeper ----------

    def _sweep_once(self, now):
        expired = None
        with self.lock:
            q = self.question
            if q is not None and q.expired(now):
                self.question = None
                expired = q
        if expired is not None:
            self._emit_cleared(expired, "expired")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(ASK_TOPIC, self.on_ask)
        self.bus.subscribe(HEARD_TOPIC, self.on_heard)
        print(f"Dialog broker active - one open question at a time "
              f"(wake={list(WAKE_PHRASES)}, window {CONVERSATION_WINDOW_SEC:.0f}s)")
        while True:
            time.sleep(EXPIRY_SWEEP_SEC)
            self._sweep_once(time.time())


if __name__ == "__main__":
    DialogBroker().run()
