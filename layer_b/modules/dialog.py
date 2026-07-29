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

This broker owns BOTH halves of turn-taking now. The addressing half ("is this
even for me?" - wake word, conversation window, command shape, all from the
shared attention.py model) used to be applied by field_agent's own forwarding
path; it lives here now (on_directed), so field_agent just forwards its
command-misses and the wake phrases / conversation window are defined in exactly
one place. The answer half ("is this an answer to my open question?") is
on_heard, unchanged.

PROTOCOL
--------
  in   picarx/dialog/ask     {asker, question_id, kind, options?, ttl?, prompt?}
         A module announces it just asked something and wants the next real
         answer routed back. Registering a new question replaces any question
         still open (and emits a 'cleared' for the old one), mirroring the
         old single-pending behaviour.
  in   picarx/audio/heard    the raw transcript stream (repairs/echoes skipped);
         used only for answer-capture.
  in   picarx/audio/directed {text, confidence?, from_repair?} | {handled: true}
         field_agent's command-misses, for the addressing decision; a
         {handled: true} ping means it DID match a command (holds the window).
  out  picarx/dialog/answer  {question_id, asker, text, confidence, kind, options}
         The routed answer - the asker consumes only its own question_id.
  out  picarx/dialog/cleared {question_id, asker, reason}
         The question ended without being answered: 'expired' (ttl passed) or
         'replaced' (a newer question took its place). Lets the asker fall back
         (e.g. curiosity's escalate-to-LLM).
  out  picarx/audio/unhandled {text, confidence}   addressed free-form -> chat.
  out  picarx/audio/uncertain {text, confidence, from}  command-shaped -> the
         LLM intent arbiter.

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
import capabilities
import identity
import speech_match

import threading
import time

HEARD_TOPIC = "picarx/audio/heard"
ASK_TOPIC = "picarx/dialog/ask"
ANSWER_TOPIC = "picarx/dialog/answer"
CLEARED_TOPIC = "picarx/dialog/cleared"
# The OTHER half of turn-taking, consolidated here from field_agent: it forwards
# every utterance it couldn't handle as a local command on DIRECTED_TOPIC, and
# this broker decides whether it was addressed to the robot and routes it onward.
DIRECTED_TOPIC = "picarx/audio/directed"      # field_agent's command-misses, in
UNHANDLED_TOPIC = "picarx/audio/unhandled"    # -> companion free-form chat
UNCERTAIN_TOPIC = "picarx/audio/uncertain"    # -> companion LLM intent arbiter

# Wake phrases and the no-wake-word follow-up window are shared with
# field_agent's forwarding path; both read them here so there's one source of
# truth. (field_agent still owns the forwarding decision; the broker keeps the
# window only so it never mistakes a wake-addressed fresh command for an answer.)
CONVERSATION_WINDOW_SEC = float(robot_config.get(
    "dialog", "conversation_window_sec", 45.0, env="DIALOG_CONVERSATION_WINDOW_SEC"))
EXPIRY_SWEEP_SEC = 1.0   # how often the background sweeper checks the live deadline
# When an utterance is routed as an answer, field_agent (which also sees it on
# the raw heard stream) forwards its own copy on the directed stream a beat
# later. Suppress that echo for this long so an answer is never ALSO re-forwarded
# to chat - the dedup the single-authority broker makes possible.
ANSWER_SUPPRESS_SEC = 3.0


# The robot's own name addresses it, alongside the generic wake phrases: we
# tell it (in companion's personality prompt) that it answers to its name, so
# the addressing layer has to actually recognize it, or "Marco, come here"
# would be heard by everyone except Marco. The name is appended to the
# configured phrases and de-duplicated, so setting identity.name is enough -
# no need to also edit dialog.wake_phrases.
_CONFIGURED_WAKE = attention.normalize_wake_phrases(robot_config.get(
    "dialog", "wake_phrases", "robot,hey robot,computer",
    env="DIALOG_WAKE_PHRASES"))
WAKE_PHRASES = tuple(dict.fromkeys(
    _CONFIGURED_WAKE + attention.normalize_wake_phrases([identity.name()])))


class DialogBroker:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.question = None            # the single live attention.Question, or None
        self.last_directed_at = 0.0     # last wake/command-shaped utterance (window base)
        self._answered = None           # (text_lower, ts) of the last routed answer
        self.meeting_recording = False  # notes daemon owns heard speech while active

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

    def on_notes_state(self, payload):
        active = payload.get("active_meeting") or {}
        with self.lock:
            self.meeting_recording = active.get("state") == "recording"

    def on_heard(self, payload):
        """Route one heard utterance IF it answers the live question; otherwise
        leave it alone for the normal command/chat pipeline. Repaired text and
        console-correction echoes are never answers (same guard the old capture
        paths used)."""
        with self.lock:
            meeting_recording = self.meeting_recording
        if meeting_recording:
            return
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
        # doesn't get swallowed as one. Everything else is judged by kind. (The
        # conversation window is maintained on the directed stream now - see
        # on_directed - so answer-capture only needs the wake test here.)
        wake_addressed = attention.strip_wake_phrase(text, WAKE_PHRASES) is not None

        if q is None or wake_addressed or not attention.answers_question(text, q):
            return   # not an answer to anything we're waiting on - hands off

        with self.lock:
            if self.question is q:
                self.question = None
            # Remember it so field_agent's directed echo of the same utterance
            # isn't ALSO forwarded to chat (see on_directed).
            self._answered = (text.lower(), now)
        self.bus.publish(ANSWER_TOPIC, {
            "question_id": q.id, "asker": q.asker, "text": text,
            "confidence": payload.get("confidence"),
            "kind": q.kind, "options": q.options, "ts": now})
        print(f"Dialog: routed answer to {q.asker} (id={q.id}): '{text}'")

    # ---------- addressing: is this for me, and as what? ----------

    def _is_talk_about_a_capability(self, text):
        """True when an utterance only LOOKS like a command because it mentions
        a capability's vocabulary, while actually being talk aimed at the robot
        ("do you like music", "what do you think of the radio").

        This is the capability-ownership question the roadmap's router
        unification calls for, asked in the one place that decides where an
        utterance goes. attention.classify() can only judge SHAPE, and shape
        alone can't tell "play some music" from "do you like music" - both
        carry radio vocabulary, so both were command-shaped and both went to
        the LLM intent arbiter. The arbiter has a chat verdict for the second
        case, but it is throttled by INTENT_REPAIR_COOLDOWN as a *command
        repair* budget, so ordinary conversation was being dropped by a
        cost control meant for misheard orders.

        Three conditions, all required, so the reroute stays narrow:
          - no capability actually PARSES it (a real MATCH is a real command,
            and belongs on the command path regardless of how it's phrased);
          - it isn't shaped like an instruction (looks_directed_command, so
            "play me some music" and "turn off the radio" are untouched);
          - it addresses the robot in the second person - the signal that this
            is someone talking TO it rather than ordering it. A bare question
            opener is deliberately not enough: "is the radio on" is a status
            question the arbiter can still repair into a real command.
        Nothing here can suppress a safety word: "stop"/"halt" utterances are
        directed commands, and field_agent acts on them before this point."""
        if speech_match.looks_directed_command(text):
            return False
        if not attention.addresses_the_robot(text):
            return False
        canon = speech_match.canonicalize(text)
        return not capabilities.ROUTER.route(canon, text).matched

    def on_directed(self, payload):
        """Route ONE utterance field_agent could not handle as a local command.
        This is the addressing half of turn-taking, moved here so the broker is
        the single source of truth: field_agent forwards its misses instead of
        classifying them itself.

          {"handled": true}  - field_agent DID act on the utterance (a matched
              command / report / feedback). Nothing to forward, but a matched
              command is the human addressing the robot, so it holds the
              no-wake-word conversation window open.
          otherwise          - classify the utterance and route it onward:
              wake phrase        -> chat (UNHANDLED_TOPIC), stripped remainder,
                                    and the window is (re)opened;
              open conversation  -> chat (UNHANDLED_TOPIC), full text, window NOT
                                    re-extended (only wake / matched commands do,
                                    so a chatty TV can't hold it open forever);
              bare command shape -> the LLM intent arbiter (UNCERTAIN_TOPIC),
                                    UNLESS it is already a repair (loop guard),
                                    or no capability actually claims it as a
                                    command and it is plainly talk aimed at the
                                    robot (see _is_talk_about_a_capability),
                                    in which case it is chat;
              chat shape         -> chat (UNHANDLED_TOPIC): talk aimed at the
                                    robot ("do you like music") with no wake
                                    word and no open window; window NOT opened;
              not addressed      -> dropped.
        An utterance just routed as an ANSWER to an open question (on_heard) is
        also dropped here, so it reaches its asker only and isn't re-forwarded to
        chat. Fail-soft and stdlib-only, same as the rest of the broker."""
        with self.lock:
            if self.meeting_recording:
                return
        now = time.time()
        if payload.get("handled"):
            with self.lock:
                self.last_directed_at = now
            return
        text = (payload.get("text") or "").strip()
        if not text:
            return
        with self.lock:
            in_conversation = (now - self.last_directed_at) < CONVERSATION_WINDOW_SEC
            answered = self._answered
        # Already routed as an answer a beat ago -> it reached its asker; don't
        # also forward it to chat (the dedup only the single-authority broker can
        # do - the old split forwarder never saw the answer routing).
        if answered and answered[0] == text.lower() \
                and (now - answered[1]) < ANSWER_SUPPRESS_SEC:
            print(f"Dialog: '{text}' already routed as an answer, not re-forwarding")
            return
        confidence = payload.get("confidence")
        from_repair = bool(payload.get("from_repair"))
        addressing = attention.classify(
            text, wake_phrases=WAKE_PHRASES, in_conversation=in_conversation)

        if addressing.reason == attention.WAKE:
            with self.lock:
                self.last_directed_at = now      # an explicit wake opens the window
            self.bus.publish(UNHANDLED_TOPIC,
                             {"text": addressing.remainder, "confidence": confidence})
            print(f"Dialog: wake chat -> companion: '{addressing.remainder}'")
        elif addressing.reason == attention.COMMAND_SHAPE:
            if self._is_talk_about_a_capability(text):
                self.bus.publish(UNHANDLED_TOPIC,
                                 {"text": text, "confidence": confidence})
                print(f"Dialog: talk about a capability -> companion: '{text}'")
                return
            # Loop guard: text the intent arbiter already repaired never
            # re-escalates - if its best repair still matched nothing, it dies.
            if from_repair:
                print(f"Dialog: repaired but still unmatched, dropping: '{text}'")
                return
            self.bus.publish(UNCERTAIN_TOPIC, {
                "text": text, "confidence": confidence, "from": "field_agent"})
            print(f"Dialog: command-shaped -> intent arbiter: '{text}'")
        elif addressing.reason == attention.CONVERSATION:
            # A wake-less follow-up inside the window: chat, but don't re-extend.
            self.bus.publish(UNHANDLED_TOPIC, {"text": text, "confidence": confidence})
            print(f"Dialog: in-conversation chat -> companion: '{text}'")
        elif addressing.reason == attention.CHAT_SHAPE:
            # Talk aimed at the robot with no wake word and no open window
            # ("do you like music"). Reaches chat, where companion's quality
            # gate decides whether it's worth answering; the window is NOT
            # opened, so overhearing one question doesn't hand the next 45
            # seconds of room noise to the LLM.
            self.bus.publish(UNHANDLED_TOPIC, {"text": text, "confidence": confidence})
            print(f"Dialog: chat-shaped -> companion: '{text}'")
        else:
            print(f"Dialog: not addressed, dropping: '{text}'")

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
        self.bus.subscribe("picarx/tools/notes/state", self.on_notes_state)
        self.bus.subscribe(DIRECTED_TOPIC, self.on_directed)
        print(f"Dialog broker active - one open question at a time, and the sole "
              f"turn-taking router (wake={list(WAKE_PHRASES)}, "
              f"window {CONVERSATION_WINDOW_SEC:.0f}s)")
        while True:
            time.sleep(EXPIRY_SWEEP_SEC)
            self._sweep_once(time.time())


if __name__ == "__main__":
    DialogBroker().run()
