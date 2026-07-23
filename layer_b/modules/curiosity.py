#!/usr/bin/env python3
# layer_b/modules/curiosity.py
"""
Curiosity (Layer B) - uncertainty-driven perception questions.

When the vision detector is genuinely UNSURE what it is looking at, the
robot should ask the human rather than silently guessing wrong: "Is that
a chair or a speaker?" The answer is a cheap, high-value label - a person
confirming an identity in one word is worth far more than the offline
reflection loop slowly inferring it - so it is fed straight back into the
semantic store (via picarx/perception/label, which reflection.py writes).
That is the whole point of this module: turn the robot's own uncertainty
into fast, human-labeled fact accumulation.

Two uncertainty signals from vision_basic.py's picarx/vision/objects:
  - alt_label present  -> a real two-way vote tie (contested_label there):
                          "Is that a <label> or a <alt_label>?"
  - low confidence     -> a shaky single guess just over the publish bar:
                          "I think I see a <label>, but I'm not sure. What
                          is that?"

Discipline (this shares one speaker with everything else, and asking too
often is worse than staying quiet):
  - one open question at a time,
  - a global ASK_COOLDOWN between questions,
  - each object id is asked about at most once.

Answers arrive through the central dialog broker (dialog.py), not by
racing on the raw heard stream. When it asks, this module registers the
question on picarx/dialog/ask (kind "label", its options, an
ANSWER_WINDOW_SEC ttl); the broker decides whether a later utterance is
genuinely an answer (vs. a command or a web-console button on the same
heard topic - the bug the old looks_like_label_answer guard patched) and
routes real answers back on picarx/dialog/answer. If the question expires
unanswered the broker says so on picarx/dialog/cleared, which is what
triggers the cloud-LLM escalation below. The answer is parsed into a label
- one of the offered options, an affirmation of the guess, or a fresh noun
- and published on picarx/perception/label.

The web console feeds the SAME picarx/perception/label topic when someone
relabels a sighting with the check / X buttons, so voice and console
corrections converge on one path. Fail-soft and stdlib-only; no LLM call.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import speech_match
import attention
# Answer classification + label parsing live in the shared attention model now;
# re-exported here so curiosity.parse_label_answer / looks_like_label_answer
# (and their unit tests) keep working against this module.
from attention import parse_label_answer, looks_like_label_answer  # noqa: F401

import threading
import time
import uuid

OBJECTS_TOPIC = "picarx/vision/objects"
SPEAK_TOPIC = "picarx/audio/speak"
LABEL_TOPIC = "picarx/perception/label"
MOVE_TOPIC = "picarx/intent/move"
# Dialog broker protocol (dialog.py) - questions and their routed answers.
ASK_TOPIC = "picarx/dialog/ask"
ANSWER_TOPIC = "picarx/dialog/answer"
CLEARED_TOPIC = "picarx/dialog/cleared"
ASKER = "curiosity"
# LAST-resort tier: when a spoken question goes unanswered, hand the object
# to companion.py to identify with the cloud LLM. companion feeds the answer
# back on LABEL_TOPIC, which trains the on-board memory - so the cloud is
# needed at most once per object kind, then never again for that look.
IDENTIFY_TOPIC = "picarx/perception/identify_request"

# Below this reported confidence a single (uncontested) guess is shaky
# enough to be worth a "what is that?" - above the OBJECT_CONFIDENCE_THRESHOLD
# (0.5) vision needs to publish it at all, but not by much.
LOW_CONF_THRESHOLD = 0.6
ASK_COOLDOWN = 30.0          # min seconds between spoken questions
ANSWER_WINDOW_SEC = 12.0     # how long the next utterance counts as the answer
ASKED_MEMORY = 300           # cap on remembered "already asked" object ids
LLM_COOLDOWN = 60.0          # min seconds between cloud identify escalations

# When it asks, the robot briefly holds still so it visibly WAITS for an answer
# instead of driving on. A modest priority + short TTL stop: wander (5) and a
# passive watch (6) yield to it, but obstacle evasion (8), coach maneuvers (9)
# and the safety daemon all still preempt - the pause never overrides a reflex.
ATTENTION_PRIORITY = 7
ATTENTION_PAUSE_SEC = 2.5

class Curiosity:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.asked = set()          # object ids already asked about
        # Local record of the question the dialog broker is currently holding
        # for us, or None: {question_id, object_id, guess, options}. The broker
        # owns the answer window and expiry; we keep this only to resolve the
        # routed answer (or escalate when the broker says it expired).
        self.pending = None
        self.last_ask_at = 0.0
        self.last_llm_at = 0.0      # last cloud identify escalation

    # ---------- speaking ----------

    def _say(self, text, kind=None, label=None, object_id=None):
        msg = {"text": text, "ts": time.time()}
        if kind:                       # tag a relabelable claim; plain speech otherwise
            msg["kind"] = kind
            if label is not None:
                msg["objects"] = [{"label": label, "id": object_id}]
        self.bus.publish(SPEAK_TOPIC, msg)

    # ---------- uncertainty detection ----------

    def _pick_uncertain(self, items):
        """First object worth asking about: a contested vote (alt_label) or
        a shaky low-confidence single guess, skipping ids already asked."""
        with self.lock:
            asked = set(self.asked)
        for obj in items:
            oid = obj.get("id")
            label = obj.get("label")
            if not oid or not label or oid in asked:
                continue
            alt = obj.get("alt_label")
            if alt and alt != label:
                return {"id": oid, "guess": label, "alt": alt}
            conf = obj.get("confidence")
            if conf is not None and conf < LOW_CONF_THRESHOLD:
                return {"id": oid, "guess": label, "alt": None}
        return None

    def on_objects(self, payload):
        now = time.time()
        with self.lock:
            if self.pending is not None:
                return  # one open question at a time (broker owns its expiry)
            if now - self.last_ask_at < ASK_COOLDOWN:
                return
        target = self._pick_uncertain(payload.get("objects") or [])
        if target:
            self._ask(target, now)

    def _escalate_to_llm(self, pending, now):
        """A spoken question went unanswered - fall through to the cloud LLM
        as the LAST resort (hard-throttled, since it costs money and network).
        companion.py does the identify call and feeds the answer back on
        LABEL_TOPIC, which trains the on-board memory for next time."""
        with self.lock:
            if now - self.last_llm_at < LLM_COOLDOWN:
                return
            self.last_llm_at = now
        self.bus.publish(IDENTIFY_TOPIC, {
            "object_id": pending["object_id"], "guess": pending["guess"],
            "options": pending["options"], "ts": now})
        print(f"Curiosity: no answer about {pending['object_id']} - "
              f"asking the LLM to identify it (last resort)")

    def _ask(self, target, now):
        guess, alt = target["guess"], target["alt"]
        if alt:
            options = [guess, alt]
            text = f"Is that a {guess} or a {alt}?"
        else:
            options = [guess]
            text = f"I think I see a {guess}, but I'm not sure. What is that?"
        question_id = uuid.uuid4().hex
        with self.lock:
            self.asked.add(target["id"])
            if len(self.asked) > ASKED_MEMORY:
                self.asked = set(list(self.asked)[-ASKED_MEMORY:])
            self.pending = {"question_id": question_id, "object_id": target["id"],
                            "guess": guess, "options": options}
            self.last_ask_at = now
        # Register the question with the dialog broker BEFORE speaking, so an
        # answer that comes back quickly always has somewhere to route.
        self.bus.publish(ASK_TOPIC, {
            "asker": ASKER, "question_id": question_id, "kind": "label",
            "options": options, "ttl": ANSWER_WINDOW_SEC, "prompt": text,
            "ts": now})
        print(f"Curiosity: asking about {target['id']} - '{text}'")
        self._say(text, kind="question", label=guess, object_id=target["id"])
        self._pause_to_listen()

    def _pause_to_listen(self):
        """Briefly hold still when asking, so the robot visibly waits for the
        answer instead of driving on to its next thought. Fail-soft - a pause
        that can't publish must never stop the question itself."""
        try:
            self.bus.publish(MOVE_TOPIC, {
                "source": "curiosity", "priority": ATTENTION_PRIORITY,
                "action": {"direction": "stop"}, "ttl": ATTENTION_PAUSE_SEC})
        except Exception as e:
            print(f"Curiosity: attention pause failed: {e}")

    # ---------- answer capture (via the dialog broker) ----------

    def on_answer(self, payload):
        """The dialog broker routed a genuine answer to our open question. It
        has already screened out commands / console buttons / stale utterances,
        so we just resolve the label. Ignore answers for anyone else's
        question, or a stale id (a newer question superseded this one)."""
        if payload.get("asker") != ASKER:
            return
        with self.lock:
            pending = self.pending
            if not pending or payload.get("question_id") != pending["question_id"]:
                return
            self.pending = None
        text = (payload.get("text") or "").strip()
        self._resolve_answer(pending, text, origin="voice")

    def on_cleared(self, payload):
        """The broker ended our question without an answer. On expiry, escalate
        to the cloud LLM (last resort); a 'replaced' clear just drops it."""
        if payload.get("asker") != ASKER:
            return
        now = time.time()
        with self.lock:
            pending = self.pending
            if not pending or payload.get("question_id") != pending["question_id"]:
                return
            self.pending = None
        if payload.get("reason") == "expired":
            self._escalate_to_llm(pending, now)

    def _resolve_answer(self, pending, text, origin):
        guess, options = pending["guess"], pending["options"]
        verdict = speech_match.parse_feedback(text)
        if verdict == "correct":
            correct = guess                      # "yes, that's right"
        elif verdict == "incorrect":
            # "no" - maybe with the real label ("no, a mug"), maybe not.
            correct = parse_label_answer(text, options)
            if correct == guess:
                correct = None                   # bare "no" names nothing new
        else:
            correct = parse_label_answer(text, options)
        if not correct:
            print(f"Curiosity: no usable label in answer '{text}' for '{guess}'")
            return
        self._publish_label(guess, correct, pending["object_id"], options, origin)

    def _publish_label(self, guess, correct, object_id, options, origin):
        self.bus.publish(LABEL_TOPIC, {
            "guess": guess, "correct_label": correct, "object_id": object_id,
            "options": options, "origin": origin, "ts": time.time()})
        print(f"Curiosity: label {object_id} '{guess}' -> '{correct}' ({origin})")
        # One terse confirmation only when we actually corrected the guess -
        # a plain "yes" needs no reply (speaker time is for useful talk).
        if correct != guess:
            self._say(f"Thanks. A {correct}, then. I'll remember that.")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(OBJECTS_TOPIC, self.on_objects)
        self.bus.subscribe(ANSWER_TOPIC, self.on_answer)
        self.bus.subscribe(CLEARED_TOPIC, self.on_cleared)
        print("Curiosity active - asking about ambiguous sightings "
              f"(cooldown {ASK_COOLDOWN:.0f}s, answer window {ANSWER_WINDOW_SEC:.0f}s)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    Curiosity().run()
