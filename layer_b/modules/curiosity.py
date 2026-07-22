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

The next thing the human says within ANSWER_WINDOW_SEC is taken as the
answer (the same "capture the next utterance" pattern companion.py uses
for spoken corrections). It is parsed into a label - one of the offered
options, an affirmation of the guess, or a fresh noun - and published on
picarx/perception/label. That utterance ALSO flows through the normal
heard pipeline on its own; here it is only LEARNED FROM, never dispatched.

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

import re
import threading
import time

OBJECTS_TOPIC = "picarx/vision/objects"
HEARD_TOPIC = "picarx/audio/heard"
SPEAK_TOPIC = "picarx/audio/speak"
LABEL_TOPIC = "picarx/perception/label"
MOVE_TOPIC = "picarx/intent/move"
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

# Words that aren't the noun in a spoken answer ("it's a speaker", "I think
# that's a chair", "no, a mug") - stripped so what's left is the label.
_ANSWER_FILLER = {
    "it's", "its", "it", "is", "a", "an", "the", "that's", "thats", "that",
    "this", "i", "think", "maybe", "looks", "look", "like", "actually", "no",
    "not", "but", "sorry", "um", "uh", "well", "you", "mean", "meant", "of",
    "course", "yeah", "yes", "nope", "just", "some", "kind", "sort",
}


# Words that mark an utterance as a COMMAND or QUESTION addressed to the robot
# rather than a label answer. If any appears, "what is that?" must not swallow
# it as a label - object labels are nouns and never contain these. Guards the
# real bug: a command issued while a question is open (spoken, or a web-console
# button, which publishes on the SAME heard topic - e.g. "who am I") was being
# mis-stored as a label ("I'll remember that's a 'who am'").
_NON_LABEL_WORDS = {
    # interrogatives
    "who", "whom", "whose", "whos", "what", "whats", "where", "when", "why",
    "how", "which",
    # question-leading auxiliaries ("are you...", "can you...", "do you...")
    "are", "am", "do", "does", "did", "can", "could", "would", "will", "should",
    # command verbs / motion the robot acts on
    "stop", "halt", "go", "come", "follow", "wander", "explore", "turn", "move",
    "drive", "spin", "find", "search", "play", "pause", "resume", "forward",
    "backward", "left", "right",
}


def parse_label_answer(text, options):
    """Extract the intended label from a spoken answer, or None.

    An explicitly offered option named anywhere in the answer wins ("it's
    the speaker" -> "speaker"); otherwise the answer's remaining non-filler
    words become a fresh label ("that's a coffee mug" -> "coffee mug"). Pure
    and hardware-free so it's unit-testable off the robot."""
    low = text.lower()
    for opt in options:
        if opt and re.search(rf"\b{re.escape(opt)}\b", low):
            return opt
    words = [w for w in re.findall(r"[a-z']+", low) if w not in _ANSWER_FILLER]
    return " ".join(words[:3]) if words else None


def looks_like_label_answer(text, options):
    """True if `text` could plausibly answer 'what is that?' - an affirmation or
    negation, one of the offered options, or a plain noun label - and NOT a
    command or question aimed at the robot. Pure/unit-testable. Keeps an
    unrelated command from being mis-stored as a label when a question happens
    to be open (see _NON_LABEL_WORDS)."""
    low = (text or "").lower().strip()
    if not low:
        return False
    if speech_match.parse_feedback(low) in ("correct", "incorrect"):
        return True                      # "yes" / "no, a mug" - a real answer
    tokens = re.findall(r"[a-z']+", low)
    if any(t in _NON_LABEL_WORDS for t in tokens):
        return False                     # a command/question, not a label
    return True


class Curiosity:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.asked = set()          # object ids already asked about
        self.pending = None         # open question, or None (see _ask)
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
        expired = None
        with self.lock:
            if self.pending and now > self.pending["until"]:
                expired = self.pending   # question timed out with no answer
                self.pending = None
        if expired:
            self._escalate_to_llm(expired, now)
        with self.lock:
            if self.pending and now <= self.pending["until"]:
                return  # one open question at a time
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
        with self.lock:
            self.asked.add(target["id"])
            if len(self.asked) > ASKED_MEMORY:
                self.asked = set(list(self.asked)[-ASKED_MEMORY:])
            self.pending = {"object_id": target["id"], "guess": guess,
                            "options": options, "until": now + ANSWER_WINDOW_SEC}
            self.last_ask_at = now
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

    # ---------- answer capture ----------

    def on_heard(self, payload):
        """The next human utterance after a question is its answer. Repaired and
        correction echoes are skipped (same guard companion uses); a stale or
        absent question just clears; and an utterance that is really a COMMAND or
        question addressed to the robot is left alone so it isn't mis-stored as a
        label - the question stays open for a genuine answer. Never dispatches."""
        if payload.get("source") in ("intent_repair", "user_correction"):
            return
        text = (payload.get("text") or "").strip()
        if not text:
            return
        now = time.time()
        with self.lock:
            pending = self.pending
        if not pending:
            return
        if now > pending["until"]:
            with self.lock:
                if self.pending is pending:
                    self.pending = None    # window closed - drop it, no answer
            return
        # Don't hijack a command/question meant for the robot (spoken, or a
        # web-console button on this same heard topic) as a label answer. Leave
        # the question open so a real answer within the window still lands.
        if not looks_like_label_answer(text, pending["options"]):
            print(f"Curiosity: '{text}' looks like a command, not a label "
                  f"- leaving the question open")
            return
        with self.lock:
            if self.pending is pending:
                self.pending = None
        self._resolve_answer(pending, text, origin="voice")

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
        self.bus.subscribe(HEARD_TOPIC, self.on_heard)
        print("Curiosity active - asking about ambiguous sightings "
              f"(cooldown {ASK_COOLDOWN:.0f}s, answer window {ANSWER_WINDOW_SEC:.0f}s)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    Curiosity().run()
