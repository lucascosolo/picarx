#!/usr/bin/env python3
# /home/picarx/layer_b/modules/companion.py
"""
Companion (Layer B) - natural conversation fallback.

field_agent.py handles a small fixed vocabulary of hard commands
("explore", "stop", "status", "objects", "history", "battery",
"hello") entirely locally, with zero network dependency, because
"stop" in particular must never wait on an LLM round-trip. Anything
that doesn't match one of those gets published to
picarx/audio/unhandled instead of silently doing nothing - that's
this module's entire job: turn it into a natural spoken reply.

This module never controls the robot. It cannot publish movement
intents at all - if someone asks it to drive somewhere in
conversation, its system prompt tells it to point them at the actual
command words instead of trying to comply itself. That split (fast
local safety-relevant commands vs. this slower, LLM-backed chat
layer) is deliberate and should not be blurred.

Each reply is grounded with a short snapshot of picarx/state/world
(face/objects/distance/battery) folded into the prompt, so it can
answer naturally ("are you doing okay?", "what's that thing you're
looking at?") without needing its own sensor access. Conversation
history is a rolling window (HISTORY_TURNS messages) persisted to
disk (COMPANION_MEMORY_PATH) after every turn, so a restart doesn't
erase who it was just talking to - it picks the same conversation
back up rather than meeting the room as a stranger every boot. If the
gap since the last turn is long enough to plausibly be a new
conversation (MEMORY_STALE_GAP), that gap is surfaced to the model as
context instead of being hidden, so it doesn't continue an hour-old
sentence as if no time passed.

This module is also the INTENT ARBITER (picarx/audio/uncertain): when
a router hears something command-shaped it can't parse, the arbiter
maps it onto a known command with one tiny LLM call and CACHES the
mapping (data/learned_intents.json), so each new phrasing is bought
from the API exactly once and handled on-board forever after. Movement
commands are excluded on principle - motion never starts from an LLM's
guess. And when someone asks what the robot is looking at (or teaches
it an object name), a live camera frame is attached to the chat call,
giving it open-vocabulary sight beyond the on-board detector's labels;
those exchanges ride picarx/audio/heard into events.db, where
reflection.py later consolidates them into durable semantic facts.

Requires ANTHROPIC_API_KEY in the environment, same as coach.py. If
it's missing, or a request fails/times out, this module just replies
with a short apology instead of raising - a quiet, unhelpful companion
is fine; a crashed process that stops handling any future messages is
not.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus
from semantic_store import SemanticStore
import speech_match

import threading
import queue
import time
import json
from collections import deque

HISTORY_TURNS = 12          # user+assistant messages kept for context
SELF_MODEL_MAX = 5          # self-model facts folded into the personality prompt
WORKER_THREADS = 2
REPLY_TIMEOUT = 8.0
REPLY_MAX_TOKENS = 150

DATA_DIR = "/home/picarx/layer_b/data"
COMPANION_MEMORY_PATH = f"{DATA_DIR}/companion_memory.json"
MEMORY_STALE_GAP = 1800      # seconds of silence before a gap is worth mentioning to the model

COMPANION_MODEL = os.environ.get("COMPANION_MODEL", "claude-sonnet-5")

# ---------- intent arbiter (picarx/audio/uncertain) ----------
# The routers escalate command-shaped utterances they couldn't parse
# ("could you put the radio on for me?", a mangled "next station").
# The arbiter maps them onto a KNOWN command via a small, cheap LLM
# call - and remembers each successful mapping in a local phrase cache,
# so a phrasing only ever costs one API call in the robot's lifetime:
# afterward it's handled on-board like a native command. That's the
# learning loop: the LLM is the teacher, the cache is what was learned.
INTENT_MODEL = os.environ.get("INTENT_MODEL", "claude-haiku-4-5-20251001")
LEARNED_INTENTS_PATH = f"{DATA_DIR}/learned_intents.json"
LEARNED_INTENTS_MAX = 300        # oldest-used entries beyond this get evicted
INTENT_REPAIR_COOLDOWN = 10.0    # min seconds between arbiter API calls
INTENT_TIMEOUT = 6.0
INTENT_MAX_TOKENS = 80

# Commands the arbiter may emit. Deliberately EXCLUDES "explore" and
# any movement: motion must only ever start from the literal spoken
# word through field_agent's strict local path, never from an LLM's
# guess about a garbled transcript.
ALLOWED_INTENTS = {
    "stop", "battery", "status", "history", "objects", "map", "why",
    "hello", "play radio", "stop radio", "next station",
    "what's playing", "list stations",
}
ALLOWED_INTENT_PREFIXES = ("tune to ", "radio find ", "station ")

INTENT_SYSTEM_PROMPT = """You repair garbled voice-command transcripts for a small robot car.
The transcript comes from an offline speech recognizer and may contain misheard words.

Known commands: stop, battery, status, history, objects, map, why, hello, play radio,
stop radio, next station, what's playing, list stations, tune to <number>,
radio find <keywords>, station <name>.

Reply with JSON only, one of:
{"command": "<one known command, with its parameter filled in if it takes one>"}
  - only if the transcript was clearly an attempt at that command
{"chat": true}   - it was speech directed at the robot, but not a command
{"ignore": true} - background noise, TV, or speech not meant for the robot

NEVER return a movement command: requests to explore, drive, turn, or go somewhere
must be answered with {"chat": true}, not a command."""

# ---------- camera-grounded chat ----------
# When someone asks the robot what it's looking at (or teaches it a new
# object: "remember this is a watering can"), attach a live camera
# frame to the LLM call so the reply is grounded in ACTUAL sight, not
# the 20 labels the on-board detector knows. Frames come from
# vision_basic.py's on-demand stream (same one the web console uses).
VISION_STREAM_CONTROL = "picarx/vision/stream_control"
VISION_FRAME_TOPIC = "picarx/vision/frame"
FRAME_FRESH_SEC = 2.0            # a frame this recent is "now" - reuse it
FRAME_WAIT_SEC = 4.0             # how long to wait for a requested frame
CAMERA_TRIGGERS = (
    "what is this", "what's this", "what is that", "what do you see",
    "what are you looking at", "look at this", "what am i holding",
    "can you see", "take a look", "remember this", "learn this",
    "what does this look like",
)

# ---------- autobiographical memory readback ----------
# reflection.py writes diary-style "episode:<YYYY-MM-DD>" facts to
# semantic.db at session boundaries. When someone asks about the robot's
# day we answer straight from that store - a pure read-only SELECT, no LLM
# call and no tokens spent - so "what did you do today" is instant even
# with no API key.
EPISODE_TRIGGERS = (
    "what did you do", "what have you done", "what happened", "summarize",
    "summarise", "recap", "tell me about your day", "how was your day",
)

SYSTEM_PROMPT = """You are the voice and personality of a small autonomous robot car (PiCar-X).
You are friendly, a little playful, and curious about the world you're rolling around in.

You are talking out loud through a text-to-speech engine, so keep every reply SHORT -
one or two sentences, plain spoken English, no markdown, no lists, no emoji.

Each message you receive starts with a bracketed snapshot of your current sensors, like
"[current status: sees a face; tracking: chair, bottle; nearest obstacle ~40cm away;
battery 7.4V]", followed by what the person actually said. Use that snapshot naturally
when it's relevant to the conversation, but don't recite it like a status report unless
asked directly what you see/sense.

You do NOT control your own motors from this conversation - a separate, instant,
safety-critical command system handles "explore", "stop", "status", "objects",
"history", and "battery". If someone asks you to move, stop, explore, or asks a
question one of those commands already answers, tell them briefly to just say that
word directly instead of trying to comply here yourself.

Some messages include a photo: that is what you see through your camera RIGHT NOW.
Use it naturally - describe what's actually in it when asked what you see or what
something is. If someone teaches you a name for a thing ("remember, this is my
watering can"), acknowledge it and use their name for it from then on.

Your conversation history survives your own restarts, so earlier messages in this
conversation may be from before you rebooted. Treat that history as a real memory
of an ongoing relationship, not a stranger's transcript. If a message starts with
"[picked back up after ...]", meaningful time passed since the last exchange - don't
awkwardly continue an old sentence, but you can naturally reference what you talked
about before if it's relevant.
"""


class Companion:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.history, self.last_turn_at = self._load_memory()
        self.latest_world = None
        self.work_queue = queue.Queue()
        self._client = None
        self._warned_no_key = False
        # Read-only view of what reflection.py has learned; fail-soft
        # (returns [] until the first reflection has ever run).
        self.semantic = SemanticStore(readonly=True)
        # Intent arbiter state
        self.learned_intents = self._load_learned_intents()
        self.last_repair_at = 0.0
        # Latest camera frame (base64 JPEG) seen on the bus, if any
        self.latest_frame_b64 = None
        self.latest_frame_at = 0.0

    # ---------- memory persistence ----------

    def _load_memory(self):
        try:
            with open(COMPANION_MEMORY_PATH) as f:
                raw = json.load(f)
            history = deque(raw.get("history", []), maxlen=HISTORY_TURNS)
            last_turn_at = raw.get("last_turn_at")
            print(f"Companion: resuming memory ({len(history)} messages, "
                  f"last turn at {last_turn_at})")
            return history, last_turn_at
        except FileNotFoundError:
            return deque(maxlen=HISTORY_TURNS), None
        except (json.JSONDecodeError, OSError) as e:
            print(f"Companion: failed to load memory, starting fresh: {e}")
            return deque(maxlen=HISTORY_TURNS), None

    def _save_memory(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with self.lock:
            snapshot = json.dumps({
                "history": list(self.history),
                "last_turn_at": self.last_turn_at,
            }, indent=2)
        tmp_path = f"{COMPANION_MEMORY_PATH}.tmp"
        with open(tmp_path, "w") as f:
            f.write(snapshot)
        os.replace(tmp_path, COMPANION_MEMORY_PATH)

    # ---------- learned intent cache ----------

    def _load_learned_intents(self):
        try:
            with open(LEARNED_INTENTS_PATH) as f:
                cache = json.load(f)
            print(f"Companion: {len(cache)} learned phrases loaded")
            return cache
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"Companion: failed to load learned intents, starting fresh: {e}")
            return {}

    def _save_learned_intents(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with self.lock:
            if len(self.learned_intents) > LEARNED_INTENTS_MAX:
                keep = sorted(self.learned_intents.items(),
                              key=lambda kv: kv[1].get("last", 0),
                              reverse=True)[:LEARNED_INTENTS_MAX]
                self.learned_intents = dict(keep)
            snapshot = json.dumps(self.learned_intents, indent=1)
        tmp_path = f"{LEARNED_INTENTS_PATH}.tmp"
        with open(tmp_path, "w") as f:
            f.write(snapshot)
        os.replace(tmp_path, LEARNED_INTENTS_PATH)

    def _dispatch_repaired(self, command, original_text, learned):
        """Re-inject a repaired command as if it had been heard cleanly.
        source=intent_repair is the routers' loop guard - repaired text
        that STILL matches nothing gets dropped, never re-escalated."""
        print(f"Companion arbiter: '{original_text}' -> '{command}'"
              f"{' (from phrase cache, no API)' if learned else ''}")
        self.bus.publish("picarx/audio/heard",
                         {"text": command, "source": "intent_repair"})
        self.bus.publish("picarx/decision", {
            "source": "companion", "kind": "intent_repair",
            "choice": {"command": command, "cached": learned},
            "reason": f"unparsed utterance: '{original_text}'", "ts": time.time()})

    @staticmethod
    def _intent_allowed(command):
        return (command in ALLOWED_INTENTS or
                any(command.startswith(p) and len(command) > len(p)
                    for p in ALLOWED_INTENT_PREFIXES))

    # ---------- inbound ----------

    def on_world_state(self, payload):
        with self.lock:
            self.latest_world = payload

    def on_unhandled(self, payload):
        text = (payload.get("text") or "").strip()
        if text:
            self.work_queue.put(("chat", text))

    def on_frame(self, payload):
        b64 = payload.get("jpeg")
        if b64:
            with self.lock:
                self.latest_frame_b64 = b64
                self.latest_frame_at = time.time()

    def on_uncertain(self, payload):
        """A router escalated a command-shaped utterance it couldn't
        parse. Cheap path first: the learned phrase cache handles it
        with zero network. Only a genuinely new phrasing costs an API
        call - and its answer feeds the cache for next time."""
        text = (payload.get("text") or "").strip()
        if not text:
            return
        key = speech_match.canonicalize(text)
        with self.lock:
            entry = self.learned_intents.get(key)
            if entry:
                entry["count"] = entry.get("count", 0) + 1
                entry["last"] = time.time()
        if entry:
            self._dispatch_repaired(entry["command"], text, learned=True)
            self._save_learned_intents()
            return
        now = time.time()
        if now - self.last_repair_at < INTENT_REPAIR_COOLDOWN:
            print(f"Companion arbiter: cooling down, dropping '{text}'")
            return
        self.last_repair_at = now
        self.work_queue.put(("repair", text))

    # ---------- Anthropic call ----------

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if not self._warned_no_key:
                print("Companion: ANTHROPIC_API_KEY not set - can't chat, will apologize instead.")
                self._warned_no_key = True
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Companion: 'anthropic' package not installed - pip install anthropic to enable chat.")
        return self._client

    def _context_blurb(self):
        with self.lock:
            snap = dict(self.latest_world) if self.latest_world else None
        if not snap:
            return "no sensor data yet"

        parts = []
        face = snap.get("face", {})
        parts.append("sees a face" if face.get("detected") and not face.get("stale", True) else "doesn't currently see a face")

        objects = snap.get("objects", {})
        if not objects.get("stale", True) and objects.get("items"):
            labels = [o.get("label", "something") for o in objects["items"]]
            parts.append(f"tracking: {', '.join(labels)}")

        distance = snap.get("distance_cm")
        if distance is not None and not snap.get("distance_stale", True):
            parts.append(f"nearest obstacle ~{distance:.0f}cm away")

        battery = snap.get("battery", {})
        if battery.get("voltage") is not None:
            low_note = " (low)" if battery.get("low") else ""
            parts.append(f"battery {battery['voltage']:.1f}V{low_note}")

        # Fold in a couple of long-term learned facts (from reflection.py's
        # semantic store) so conversation can draw on more than the last
        # few seconds of sensors. One tiny read-only SELECT per utterance.
        # The self-model (subject "self") is handled separately - it grounds
        # the PERSONALITY (system prompt), not this per-turn sensor snapshot -
        # so exclude it here to avoid reciting it twice.
        facts = [f for f in self.semantic.recent_facts(limit=4)
                 if f["subject"] != "self"][:2]
        if facts:
            remembered = "; ".join(f"{f['subject']}: {f['fact']}" for f in facts)
            parts.append(f"long-term memory notes: {remembered}")

        return "; ".join(parts)

    def _self_model_notes(self):
        """The robot's current self-model - first-person facts under
        subject "self" that reflection.py's offline self-model pass writes.
        Read-only and fail-soft: [] until reflection has ever run."""
        return [f["fact"] for f in self.semantic.facts_for("self", limit=SELF_MODEL_MAX)]

    def _compose_system_prompt(self):
        """Base personality + a DYNAMIC self-model block, so the robot's
        conversational voice is grounded in what it has actually learned
        about itself rather than only the fixed prompt string. Costs one
        tiny read-only SELECT, no API call. Falls back to the plain prompt
        before the first self-model exists."""
        notes = self._self_model_notes()
        if not notes:
            return SYSTEM_PROMPT
        block = "\n".join(f"- {n}" for n in notes)
        return (SYSTEM_PROMPT +
                "\n\nYour current self-understanding - things you have learned about "
                "your own behaviour and your home from experience. Let it colour your "
                "personality and answers naturally, speaking from it in the first "
                "person; do not just recite the list:\n" + block)

    def _gap_note(self, now):
        """Empty unless enough silence passed since the last turn (possibly
        across a restart) that the model should know it's not still mid-conversation."""
        if not self.history or self.last_turn_at is None:
            return ""
        gap = now - self.last_turn_at
        if gap < MEMORY_STALE_GAP:
            return ""
        minutes = gap / 60.0
        if minutes < 90:
            span = f"{minutes:.0f} minutes"
        else:
            span = f"{minutes / 60.0:.1f} hours"
        return f"[picked back up after {span} of silence]\n"

    def _repair_intent(self, text):
        """One strict, tiny LLM call: map a garbled utterance onto a
        known command (cached for next time), route it to chat, or
        drop it as noise. Fail-soft: any error just drops the text."""
        client = self._get_client()
        if client is None:
            return
        try:
            response = client.messages.create(
                model=INTENT_MODEL,
                max_tokens=INTENT_MAX_TOKENS,
                system=INTENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Transcript: {text}"}],
                timeout=INTENT_TIMEOUT,
            )
            raw = "".join(b.text for b in response.content
                          if getattr(b, "type", None) == "text").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            verdict = json.loads(raw)
        except Exception as e:
            print(f"Companion arbiter: repair failed ({e}), dropping '{text}'")
            return

        command = (verdict.get("command") or "").strip().lower() \
            if isinstance(verdict, dict) else ""
        if command and self._intent_allowed(command):
            with self.lock:
                self.learned_intents[speech_match.canonicalize(text)] = {
                    "command": command, "count": 1, "last": time.time()}
            self._save_learned_intents()
            self._dispatch_repaired(command, text, learned=False)
        elif isinstance(verdict, dict) and verdict.get("chat"):
            self._handle_utterance(text)
        else:
            # ignore verdict, disallowed command, or junk output - all
            # end the same way: silently not acting on garbled audio.
            print(f"Companion arbiter: no action for '{text}' ({verdict})")

    # ---------- camera ----------

    def _get_camera_frame(self):
        """Base64 JPEG of what the camera sees right now, or None.
        Reuses a fresh frame if one is already flowing (e.g. the web
        console's live view is open) so we don't fight over the stream
        control topic; otherwise asks vision for a brief burst."""
        now = time.time()
        with self.lock:
            if self.latest_frame_b64 and now - self.latest_frame_at < FRAME_FRESH_SEC:
                return self.latest_frame_b64
        self.bus.publish(VISION_STREAM_CONTROL, {"enabled": True})
        try:
            deadline = now + FRAME_WAIT_SEC
            while time.time() < deadline:
                time.sleep(0.2)
                with self.lock:
                    if self.latest_frame_at > now:
                        return self.latest_frame_b64
            print("Companion: no camera frame arrived in time")
            return None
        finally:
            self.bus.publish(VISION_STREAM_CONTROL, {"enabled": False})

    @staticmethod
    def _wants_camera(text):
        lowered = text.lower()
        return any(t in lowered for t in CAMERA_TRIGGERS)

    # ---------- autobiographical memory readback ----------

    def _episode_query_date(self, text):
        """'YYYY-MM-DD' if this utterance is asking about the robot's day
        ("what did you do today", "summarize yesterday"), else None. Date
        is formatted in local time to match reflection.py's episode keys."""
        lowered = text.lower()
        if not any(t in lowered for t in EPISODE_TRIGGERS):
            return None
        if not any(w in lowered for w in ("today", "yesterday", "day")):
            return None
        now = time.time()
        offset = -86400 if "yesterday" in lowered else 0
        return time.strftime("%Y-%m-%d", time.localtime(now + offset))

    def _maybe_answer_episode(self, text):
        """Answer 'what did you do today / summarize yesterday' straight
        from semantic.db (the episode:<date> fact). Returns True if it
        handled the utterance. No API call - works even without a key."""
        date = self._episode_query_date(text)
        if not date:
            return False
        entries = self.semantic.facts_for(f"episode:{date}", limit=1)
        if entries:
            reply = entries[0]["fact"]
        else:
            when = "yesterday" if "yesterday" in text.lower() else "today"
            reply = f"I don't have my thoughts on {when} put together yet."
        now = time.time()
        with self.lock:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            self.last_turn_at = now
        self._save_memory()
        print(f"Companion (episode {date}): {reply}")
        self.bus.publish("picarx/audio/speak", {"text": reply})
        return True

    # ---------- chat ----------

    def _handle_utterance(self, text):
        # Autobiographical readback first: a diary question is answered from
        # the semantic store directly, never spending an LLM round-trip.
        if self._maybe_answer_episode(text):
            return
        client = self._get_client()
        if client is None:
            self.bus.publish("picarx/audio/speak", {"text": "Sorry, I can't chat right now."})
            return

        now = time.time()
        with self.lock:
            messages = list(self.history)
        gap_note = self._gap_note(now)
        user_text = f"{gap_note}[current status: {self._context_blurb()}]\n{text}"
        # Ground "what is this?"-style questions (and taught objects) in
        # an actual camera frame - open-vocabulary sight via the LLM,
        # not the fixed label set of the on-board detector.
        content = user_text
        if self._wants_camera(text):
            frame_b64 = self._get_camera_frame()
            if frame_b64:
                content = [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": frame_b64}},
                    {"type": "text", "text": user_text},
                ]
        messages = messages + [{"role": "user", "content": content}]

        try:
            response = client.messages.create(
                model=COMPANION_MODEL,
                max_tokens=REPLY_MAX_TOKENS,
                system=self._compose_system_prompt(),
                messages=messages,
                timeout=REPLY_TIMEOUT,
            )
            reply = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
        except Exception as e:
            print(f"Companion: chat failed: {e}")
            reply = "Sorry, I got a little confused there."

        if not reply:
            return

        with self.lock:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            self.last_turn_at = now
        self._save_memory()

        print(f"Companion says: {reply}")
        self.bus.publish("picarx/audio/speak", {"text": reply})

    # ---------- worker pool ----------

    def _worker_loop(self):
        while True:
            kind, text = self.work_queue.get()
            try:
                if kind == "repair":
                    self._repair_intent(text)
                else:
                    self._handle_utterance(text)
            except Exception as e:
                print(f"Companion: error handling {kind} '{text}': {e}")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/unhandled", self.on_unhandled)
        self.bus.subscribe("picarx/audio/uncertain", self.on_uncertain)
        self.bus.subscribe(VISION_FRAME_TOPIC, self.on_frame)
        self.bus.subscribe("picarx/state/world", self.on_world_state)

        for _ in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, daemon=True).start()

        print("Companion active, listening on picarx/audio/unhandled")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Companion().run()
