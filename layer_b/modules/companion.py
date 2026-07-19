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

This module never publishes a movement primitive. It cannot drive the
wheels directly - if someone asks it to drive somewhere in conversation,
its system prompt tells it to point them at the actual command words
instead. That split (fast local safety-relevant commands vs. this slower,
LLM-backed chat layer) is deliberate and should not be blurred.

It does expose a small set of LLM TOOLS (see TOOLS) that let the model
ACT by TOGGLING other daemons over picarx/tools/* topics - never by
emitting motion. schedule_reminder arms reminder_daemon, share_connection
asks bluetooth_daemon to tether to a paired phone, and start/stop_following
flip follow_daemon's mode. Even start_following only sets a switch: follow_daemon
generates the actual motion deterministically from vision and every command
still flows through the safety daemon, so "motion never starts from raw LLM
output" holds - the model chooses a behaviour, not a maneuver.

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
import robot_config
from semantic_store import SemanticStore
from spatial_store import SpatialStore
import speech_match

import threading
import queue
import time
import json
from collections import deque

HISTORY_TURNS = 20          # user+assistant messages kept for context
SELF_MODEL_MAX = 5          # self-model facts folded into the personality prompt
WORKER_THREADS = 2
REPLY_TIMEOUT = 8.0
REPLY_MAX_TOKENS = 150

DATA_DIR = "/home/picarx/layer_b/data"
COMPANION_MEMORY_PATH = f"{DATA_DIR}/companion_memory.json"
MEMORY_STALE_GAP = 1800      # seconds of silence before a gap is worth mentioning to the model

COMPANION_MODEL = str(robot_config.get("companion", "model", "claude-sonnet-5",
                                       env="COMPANION_MODEL"))

# ---------- intent arbiter (picarx/audio/uncertain) ----------
# The routers escalate command-shaped utterances they couldn't parse
# ("could you put the radio on for me?", a mangled "next station").
# The arbiter maps them onto a KNOWN command via a small, cheap LLM
# call - and remembers each successful mapping in a local phrase cache,
# so a phrasing only ever costs one API call in the robot's lifetime:
# afterward it's handled on-board like a native command. That's the
# learning loop: the LLM is the teacher, the cache is what was learned.
INTENT_MODEL = str(robot_config.get("companion", "intent_model",
                                    "claude-haiku-4-5-20251001", env="INTENT_MODEL"))

# ---------- chat quality gate (noise rejection, zero-LLM) ----------
# audio_nodes already screens raw decodes, but the chat path deserves its
# own gate: during the no-wake-word conversation window field_agent
# forwards EVERYTHING here, so one real command near a chatty TV used to
# mean 45 seconds of paid LLM calls answering the television. Three tiers
# on speech_match.quality_score (deterministic word-list arithmetic, no
# models, no API):
#   < chat_noise_quality  -> almost certainly noise: SILENT drop (answering
#                            would be the robot talking to itself), posted
#                            on picarx/audio/rejected for later debugging;
#   < chat_min_quality    -> words but no discernible intent: a soft
#                            "I didn't catch that." (throttled, so a noisy
#                            room doesn't have the robot muttering it on
#                            loop) and NO LLM call;
#   otherwise             -> real speech, full chat pipeline.
CHAT_NOISE_QUALITY = float(robot_config.get(
    "companion", "chat_noise_quality", 0.2, env="CHAT_NOISE_QUALITY"))
CHAT_MIN_QUALITY = float(robot_config.get(
    "companion", "chat_min_quality", 0.45, env="CHAT_MIN_QUALITY"))
DIDNT_CATCH_COOLDOWN = 15.0   # min seconds between soft "didn't catch" replies
LEARNED_INTENTS_PATH = f"{DATA_DIR}/learned_intents.json"
LEARNED_INTENTS_MAX = 300        # oldest-used entries beyond this get evicted
INTENT_REPAIR_COOLDOWN = 10.0    # min seconds between arbiter API calls
INTENT_TIMEOUT = 6.0
INTENT_MAX_TOKENS = 80

# ---------- intent feedback (the user grading interpretations) ----------
# picarx/intent/feedback carries explicit judgments - the web console's
# check/X buttons and spoken phrases like "that's not what I meant"
# (routed by field_agent with the utterance being judged attached).
#   correct   -> reinforce the cached phrase mapping, if one produced it.
#   incorrect -> DELETE the cached mapping (it taught the wrong thing),
#                learn from a supplied correction, or - voice only - ask
#                "what did you want?" and treat the next utterance as
#                the answer. The answer executes through the normal
#                heard pipeline on its own; here it's only LEARNED FROM:
#                normalized onto a known command (allowlist first, one
#                small LLM call only if it's fuzzy) and cached against
#                the ORIGINAL phrasing, so next time it's on-board.
# Motion stays out of the cache in every path, same invariant as ever.
FEEDBACK_TOPIC = "picarx/intent/feedback"
CORRECTION_WINDOW_SEC = 45.0     # how long "what did you want?" waits for an answer

# Commands the arbiter may emit. Deliberately EXCLUDES "explore",
# "go to <place>" and any other movement: motion must only ever start
# from the literal spoken word through field_agent's strict local path,
# never from an LLM's guess about a garbled transcript. (field_agent
# additionally refuses motion commands arriving with source=
# intent_repair, so this exclusion is enforced on both ends.)
ALLOWED_INTENTS = {
    "stop", "battery", "status", "history", "objects", "map", "why",
    "hello", "who am i", "where are you",
    "play radio", "stop radio", "next station",
    "what's playing", "list stations",
}
ALLOWED_INTENT_PREFIXES = ("tune to ", "radio find ", "station ",
                           "where is ", "what's in ", "call this place ")

INTENT_SYSTEM_PROMPT = """You repair garbled voice-command transcripts for a small robot car.
The transcript comes from an offline speech recognizer and may contain misheard words.

Known commands: stop, battery, status, history, objects, map, why, hello, who am i,
where are you, play radio, stop radio, next station, what's playing, list stations,
tune to <number>, radio find <keywords>, station <name>,
where is <object>  (asks the robot's memory where it last saw an object),
what's in <place>  (asks what objects it has seen at a named place),
call this place <name>  (names the robot's current location).

Reply with JSON only, one of:
{"command": "<one known command, with its parameter filled in if it takes one>"}
  - only if the transcript was clearly an attempt at that command
{"chat": true}   - it was speech directed at the robot, but not one of the commands
{"ignore": true} - background noise, TV, or speech not meant for the robot

NEVER return a movement command: requests to explore, drive, turn, go somewhere, or
follow someone must be answered with {"chat": true}, not a command - the chat layer
has its own carefully-gated tools for those. Requests to set reminders, to be
remembered/recognized, or to share a connection are also {"chat": true}."""

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

# ---------- perception LAST resort (picarx/perception/identify_request) ----------
# When the on-board detector is unsure AND the on-board label memory can't help
# AND a spoken question to a human went unanswered, curiosity.py hands the
# object here. We identify it with one cheap camera-grounded LLM call and feed
# the answer back on picarx/perception/label - which both trains the on-board
# visual memory (vision_basic) and records a durable fact (reflection). So the
# cloud is the LAST tier and is paid at most once per object kind. Hard
# throttled; fail-soft (no key / no frame -> silently give up).
PERCEPTION_IDENTIFY_TOPIC = "picarx/perception/identify_request"
PERCEPTION_LABEL_TOPIC = "picarx/perception/label"
IDENTIFY_COOLDOWN = 45.0
IDENTIFY_MAX_TOKENS = 20
IDENTIFY_SYSTEM_PROMPT = (
    "You name what a small robot's camera is looking at. Reply with just the "
    "single most prominent physical object in the photo, in 1 to 3 words, "
    "lowercase, no punctuation and no sentence (e.g. 'watering can', 'slipper', "
    "'coffee mug'). If you cannot tell, reply exactly 'unknown'.")
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

# ---------- LLM tools (companion is the only module that runs a tool loop) ----------
# These let the model ACT, not just talk. Crucially they only ever TOGGLE
# other daemons via picarx/tools/* mode topics - companion never publishes a
# motion primitive itself, so the "motion never starts from raw LLM output"
# invariant holds: start_following just flips a switch, and follow_daemon
# generates the actual movement deterministically from vision, every command
# still gated by the safety daemon. Reminders and network-sharing issue no
# motion at all.
MAX_TOOL_ROUNDS = 3          # bound the tool<->model round-trips per utterance
REMINDER_SET_TOPIC = "picarx/tools/reminder/set"
FOLLOW_CONTROL_TOPIC = "picarx/tools/follow/set"
BLUETOOTH_CONNECT_TOPIC = "picarx/tools/bluetooth/connect"
HEALTH_STATE_TOPIC = "picarx/health/state"
LOWPOWER_REQUEST_TOPIC = "picarx/tools/lowpower/request"

TOOLS = [
    {"name": "schedule_reminder",
     "description": "Set a spoken reminder for the person for later. Use when they "
                    "ask to be reminded of something after a delay or at a time. "
                    "You know the current time from the system prompt.",
     "input_schema": {"type": "object", "properties": {
         "message": {"type": "string",
                     "description": "what to remind them about, in a few plain words"},
         "delay_minutes": {"type": "number",
                           "description": "minutes from now to fire the reminder"},
         "at": {"type": "string",
                "description": "exact local time instead of a delay, e.g. '18:30' "
                               "or '2026-07-15 18:30'"}},
         "required": ["message"]}},
    {"name": "start_following",
     "description": "Start physically following the person around, driving to keep "
                    "them centered in view. This MOVES the robot, so only call it "
                    "when you are VERY CONFIDENT the person is clearly, explicitly "
                    "asking to be followed right now (e.g. 'follow me', 'come with "
                    "me', 'walk with me'). If it's ambiguous, casual, hypothetical, "
                    "or just talk ABOUT following, do NOT call it - reply in words "
                    "and let them confirm. Movement stays under the safety system "
                    "and can be stopped anytime.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "stop_following",
     "description": "Stop following the person.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "share_connection",
     "description": "Get internet by tethering over BLUETOOTH to the person's "
                    "already-paired phone, so radio and chat keep working where "
                    "there is no wifi. Use when they offer to share their phone's "
                    "connection, or when you're offline. (Wifi networks are managed "
                    "with the system's own tools, not this.)",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string",
                  "description": "optional saved phone name to tether to"}},
         "required": []}},
    {"name": "where_is_object",
     "description": "Look up in your spatial memory where an object was last "
                    "seen while exploring (which place, how long ago). Use when "
                    "the person asks where something is or where you saw it.",
     "input_schema": {"type": "object", "properties": {
         "label": {"type": "string",
                   "description": "the object, e.g. 'bottle' or 'chair'"}},
         "required": ["label"]}},
    {"name": "recall_memory",
     "description": "Search your long-term memory of learned facts about the "
                    "home, the people in it, and your own experiences. Use when "
                    "asked what you know or remember about something.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string",
                   "description": "a word or short phrase to search for"}},
         "required": ["query"]}},
    {"name": "list_known_people",
     "description": "List the people whose faces you have learned to recognize. "
                    "Use when asked who you know or whether you'd recognize "
                    "someone.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "check_vital_stats",
     "description": "Check your own physical health: battery voltage/percentage, "
                    "CPU temperature, and free disk space. Use when the person asks "
                    "how you're doing/feeling, about your battery/power/temperature, "
                    "or before deciding whether to conserve power.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "register_low_power_intent",
     "description": "Enter low-power mode to preserve yourself when the battery is "
                    "low: this curtails high-power work (heavy vision processing) and "
                    "drops to a low-overhead monitoring state. Call it when you see "
                    "from check_vital_stats that the battery is low, or when the "
                    "person tells you to conserve power. (A safety system also does "
                    "this on its own if the battery gets critically low.)",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]

PEOPLE_DIR = f"{DATA_DIR}/people"


def _known_people():
    """Names of enrolled people (person_memory.py owns data/people/);
    fail-soft to [] when face memory isn't set up."""
    try:
        return sorted(d for d in os.listdir(PEOPLE_DIR)
                      if os.path.isdir(os.path.join(PEOPLE_DIR, d)))
    except OSError:
        return []


def _spoken_age(seconds):
    """'just now' / '5 minutes ago' / 'about 3 hours ago'."""
    if seconds < 90:
        return "just now"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} minutes ago"
    hours = minutes / 60.0
    if hours < 36:
        return f"about {hours:.0f} hour{'s' if round(hours) != 1 else ''} ago"
    return f"about {hours / 24.0:.0f} days ago"


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
"history", "battery", "go to <place>", "where is <object>", "call this place
<name>", and "who am I". If someone asks you to move, stop, explore, or asks a
question one of those commands already answers, tell them briefly to just say that
phrase directly instead of trying to comply here yourself.

If the sensor snapshot names the person you're looking at, that IS who you're
talking to - address them by name naturally, like a friend would. If someone new
wants you to remember them, tell them to face you and say "remember me, I am"
followed by their name.

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
        # Read-only view of the spatial map + object sightings
        # (location_graph owns spatial.db); fail-soft to "no map yet".
        self.spatial = SpatialStore(readonly=True)
        # Intent arbiter state
        self.learned_intents = self._load_learned_intents()
        self.last_repair_at = 0.0
        self._last_didnt_catch_at = 0.0   # throttles the soft low-quality reply
        # Latest camera frame (base64 JPEG) seen on the bus, if any
        self.latest_frame_b64 = None
        self.latest_frame_at = 0.0
        # Latest vital stats from health_daemon (for the check_vital_stats tool)
        self.latest_health = None
        # Pending "what did you want me to do?" question, or None:
        # {"utterance": <original misread phrasing>, "until": <deadline>}
        self.awaiting_correction = None
        self._last_identify_at = 0.0      # throttles the cloud identify tier

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
        if not text:
            return
        # Quality gate BEFORE anything queues toward the LLM (see the
        # CHAT_NOISE_QUALITY / CHAT_MIN_QUALITY block up top).
        quality = speech_match.quality_score(text, payload.get("confidence"))
        if quality < CHAT_NOISE_QUALITY:
            print(f"Companion: dropping probable noise '{text}' (quality {quality})")
            self.bus.publish("picarx/audio/rejected", {
                "text": text, "quality": quality,
                "stage": "companion", "ts": time.time()})
            return
        if quality < CHAT_MIN_QUALITY:
            now = time.time()
            print(f"Companion: no clear intent in '{text}' (quality {quality}), "
                  f"skipping the LLM")
            if now - self._last_didnt_catch_at > DIDNT_CATCH_COOLDOWN:
                self._last_didnt_catch_at = now
                self.bus.publish("picarx/audio/speak",
                                 {"text": "I didn't catch that.", "ts": now})
            return
        self.work_queue.put(("chat", text))

    def on_frame(self, payload):
        b64 = payload.get("jpeg")
        if b64:
            with self.lock:
                self.latest_frame_b64 = b64
                self.latest_frame_at = time.time()

    def on_identify(self, payload):
        """Last-resort identify request from curiosity.py. Throttled here too
        (belt-and-suspenders with curiosity's own cooldown), then handed to a
        worker so the camera wait/LLM call never blocks the MQTT thread."""
        now = time.time()
        with self.lock:
            if now - self._last_identify_at < IDENTIFY_COOLDOWN:
                return
            self._last_identify_at = now
        self.work_queue.put(("identify", payload))

    def on_health(self, payload):
        # Vital stats from health_daemon, cached for the check_vital_stats tool.
        with self.lock:
            self.latest_health = payload

    @staticmethod
    def _format_health(health):
        """Spoken-friendly one-liner from a cached health payload."""
        if not health:
            return "I don't have my vital stats yet."
        parts = []
        v, pct = health.get("battery_v"), health.get("battery_pct")
        if v is not None and pct is not None:
            parts.append(f"battery {v:.1f} volts, about {pct} percent")
        elif v is not None:
            parts.append(f"battery {v:.1f} volts")
        if health.get("temp_c") is not None:
            parts.append(f"CPU {health['temp_c']:.0f} degrees")
        if health.get("disk_free_gb") is not None:
            parts.append(f"{health['disk_free_gb']:.1f} gigabytes of disk free")
        if health.get("low_power"):
            parts.append("currently in low-power mode")
        if not parts:
            return "My vital stats are unavailable right now."
        return "; ".join(parts) + "."

    # ---------- intent feedback ----------

    def _say(self, text):
        self.bus.publish("picarx/audio/speak", {"text": text, "ts": time.time()})

    def _journal_feedback(self, verdict, utterance, detail):
        self.bus.publish("picarx/decision", {
            "source": "companion", "kind": "intent_feedback",
            "choice": {"verdict": verdict, **detail},
            "reason": f"user judged the interpretation of: '{utterance}'",
            "ts": time.time()})

    def on_heard(self, payload):
        """Only consumed while a 'what did you want me to do?' question
        is pending: the next human utterance is the answer, captured for
        LEARNING. It also executes through field_agent's normal pipeline
        on its own - this handler never dispatches anything."""
        if payload.get("source") in ("intent_repair", "user_correction"):
            return
        text = (payload.get("text") or "").strip()
        if not text or speech_match.parse_feedback(text):
            return
        with self.lock:
            awaiting = self.awaiting_correction
            self.awaiting_correction = None
        if not awaiting or time.time() > awaiting["until"]:
            return
        self.work_queue.put(("learn", (awaiting["utterance"], text)))

    def on_feedback(self, payload):
        verdict = payload.get("verdict")
        if verdict not in ("correct", "incorrect"):
            return
        utterance = (payload.get("utterance") or "").strip()
        correction = (payload.get("correction") or "").strip()
        origin = payload.get("origin", "web")
        key = speech_match.canonicalize(utterance) if utterance else None
        now = time.time()

        if verdict == "correct":
            reinforced = False
            with self.lock:
                entry = self.learned_intents.get(key) if key else None
                if entry:
                    entry["count"] = entry.get("count", 0) + 1
                    entry["last"] = now
                    entry["confirmed"] = True
                    reinforced = True
            if reinforced:
                self._save_learned_intents()
            print(f"Companion: feedback CORRECT on '{utterance}'"
                  f"{' (mapping reinforced)' if reinforced else ''}")
            self._journal_feedback(verdict, utterance, {"reinforced": reinforced})
            if origin == "voice":
                self._say("Good to know, thanks.")
            return

        # incorrect: a wrong mapping must not fire a second time.
        removed = None
        with self.lock:
            if key and key in self.learned_intents:
                removed = self.learned_intents.pop(key)["command"]
        if removed:
            self._save_learned_intents()
            print(f"Companion: feedback INCORRECT - unlearned '{utterance}' -> '{removed}'")
        else:
            print(f"Companion: feedback INCORRECT on '{utterance}' (nothing cached)")
        self._journal_feedback(verdict, utterance,
                               {"unlearned": removed, "correction": correction or None})
        if correction and utterance:
            self.work_queue.put(("learn", (utterance, correction)))
        elif origin == "voice":
            if utterance:
                with self.lock:
                    self.awaiting_correction = {
                        "utterance": utterance, "until": now + CORRECTION_WINDOW_SEC}
                self._say("Sorry about that. What did you want me to do?")
            else:
                self._say("Sorry about that.")

    def _learn_correction(self, original, answer):
        """Cache original-phrasing -> intended command. The answer may be
        a clean command ('battery') or free-form ('I wanted to know the
        battery level'); try the allowlist directly first, and only spend
        an LLM call to normalize a fuzzy answer. Motion commands are never
        cached (the arbiter allowlist enforces it), matching the standing
        invariant that the cache can't start the robot moving."""
        key = speech_match.canonicalize(original)
        command = answer.strip().lower()
        if not self._intent_allowed(command):
            verdict = self._arbiter_verdict(
                f"Transcript: {original}\n"
                f"The user says the robot misunderstood, and clarified they "
                f"actually wanted: {answer}") or {}
            command = (verdict.get("command") or "").strip().lower()
        if command and self._intent_allowed(command):
            with self.lock:
                self.learned_intents[key] = {
                    "command": command, "count": 1, "last": time.time(),
                    "taught": True}
            self._save_learned_intents()
            print(f"Companion: user-taught mapping '{original}' -> '{command}'")
            self._journal_feedback("correction", original, {"learned": command})
            self._say(f"Got it. When you say that, I'll take it as: {command}.")
            return True
        print(f"Companion: couldn't map correction for '{original}' "
              f"(answer: '{answer}') onto a safe known command")
        self._journal_feedback("correction", original,
                               {"learned": None, "answer": answer})
        self._say("Thanks, I'll keep that in mind.")
        return False

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
        person = snap.get("person", {})
        if person.get("name") and not person.get("stale", True):
            parts.append(f"recognizes the person in front of it: {person['name']}")
        else:
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
        """Base personality + the current local date/time + a DYNAMIC
        self-model block, so the robot's conversational voice is grounded
        in what it has actually learned about itself and knows what time it
        is (needed for time-aware replies and the schedule_reminder tool).
        Costs one tiny read-only SELECT, no API call."""
        prompt = (SYSTEM_PROMPT + "\n\nThe current local date and time is "
                  + time.strftime("%A %B %d %Y, %I:%M %p") + ".")
        notes = self._self_model_notes()
        if not notes:
            return prompt
        block = "\n".join(f"- {n}" for n in notes)
        return (prompt +
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

    def _arbiter_verdict(self, content):
        """One strict, tiny LLM call against the intent prompt. Returns
        the parsed verdict dict ({"command"} / {"chat"} / {"ignore"}) or
        None on any failure - callers fail soft."""
        client = self._get_client()
        if client is None:
            return None
        try:
            response = client.messages.create(
                model=INTENT_MODEL,
                max_tokens=INTENT_MAX_TOKENS,
                system=INTENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                timeout=INTENT_TIMEOUT,
            )
            raw = "".join(b.text for b in response.content
                          if getattr(b, "type", None) == "text").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            verdict = json.loads(raw)
            return verdict if isinstance(verdict, dict) else None
        except Exception as e:
            print(f"Companion arbiter: LLM verdict failed ({e})")
            return None

    def _repair_intent(self, text):
        """One strict, tiny LLM call: map a garbled utterance onto a
        known command (cached for next time), route it to chat, or
        drop it as noise. Fail-soft: any error just drops the text."""
        verdict = self._arbiter_verdict(f"Transcript: {text}")
        if verdict is None:
            print(f"Companion arbiter: repair failed, dropping '{text}'")
            return

        command = (verdict.get("command") or "").strip().lower()
        if command and self._intent_allowed(command):
            with self.lock:
                self.learned_intents[speech_match.canonicalize(text)] = {
                    "command": command, "count": 1, "last": time.time()}
            self._save_learned_intents()
            self._dispatch_repaired(command, text, learned=False)
        elif verdict.get("chat"):
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

    # ---------- perception last resort ----------

    @staticmethod
    def _clean_identify_label(raw):
        """A short, storable label from the identify model's reply, or None
        (unsure / junk). Keeps it to a 1-3 word noun phrase."""
        lines = (raw or "").strip().lower().splitlines()
        text = lines[0].strip().strip(".!?\"'").strip() if lines else ""
        if not text or "unknown" in text or len(text) > 40 or len(text.split()) > 3:
            return None
        return text

    def _identify_object(self, payload):
        """One camera-grounded LLM call to name an object the on-board tiers
        couldn't. The answer goes back on picarx/perception/label, training the
        on-board memory (so this cloud call is paid at most once per look) and
        recording a fact. Fail-soft: no key or no frame just gives up quietly."""
        guess = (payload.get("guess") or "").strip().lower()
        object_id = payload.get("object_id")
        client = self._get_client()
        if client is None:
            return
        frame_b64 = self._get_camera_frame()
        if not frame_b64:
            print("Companion: identify - no camera frame, giving up (last resort)")
            return
        try:
            response = client.messages.create(
                model=INTENT_MODEL,
                max_tokens=IDENTIFY_MAX_TOKENS,
                system=IDENTIFY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": frame_b64}},
                    {"type": "text", "text": "What object is this?"}]}],
                timeout=INTENT_TIMEOUT,
            )
            raw = "".join(b.text for b in response.content
                          if getattr(b, "type", None) == "text").strip()
        except Exception as e:
            print(f"Companion: identify call failed ({e})")
            return
        label = self._clean_identify_label(raw)
        if not label:
            print(f"Companion: identify unsure for {object_id} (model said '{raw}')")
            return
        print(f"Companion: identified {object_id} as '{label}' "
              f"(detector had guessed '{guess}')")
        self.bus.publish(PERCEPTION_LABEL_TOPIC, {
            "correct_label": label, "guess": guess, "object_id": object_id,
            "origin": "llm", "ts": time.time()})
        # Tagged observation carrying the object id: the user hears the robot's
        # own conclusion and can correct it from the console, which retrains
        # the on-board memory by that id.
        self.bus.publish("picarx/audio/speak", {
            "text": f"I think that's a {label}.", "ts": time.time(),
            "kind": "observation",
            "objects": [{"label": label, "id": object_id}]})

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
            # Derive the word from the date we already resolved rather than
            # re-lowercasing and re-scanning the utterance.
            when = "today" if date == time.strftime("%Y-%m-%d") else "yesterday"
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

    # ---------- LLM tool loop ----------

    def _chat_with_tools(self, client, messages):
        """One utterance, with the model allowed to call tools. Runs a
        bounded tool<->model loop: each round, execute any tool_use blocks
        (which just publish mode toggles to the daemons) and feed the
        results back so the model can produce a natural spoken reply.
        Returns the final spoken text ("" if none)."""
        convo = list(messages)
        final_text = ""
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model=COMPANION_MODEL,
                max_tokens=REPLY_MAX_TOKENS,
                system=self._compose_system_prompt(),
                tools=TOOLS,
                messages=convo,
                timeout=REPLY_TIMEOUT,
            )
            text = "".join(b.text for b in response.content
                           if getattr(b, "type", None) == "text").strip()
            if text:
                final_text = text
            tool_uses = [b for b in response.content
                         if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break
            convo.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                out = self._execute_tool(tu.name, getattr(tu, "input", None) or {})
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out})
            convo.append({"role": "user", "content": results})
        return final_text

    def _execute_tool(self, name, tool_input):
        """Run one tool call by publishing the matching mode/request topic.
        Returns a short result string fed back to the model. Never emits a
        motion primitive - follow motion is generated by follow_daemon."""
        try:
            if name == "schedule_reminder":
                message = str(tool_input.get("message") or "").strip()
                if not message:
                    return "No reminder text was provided."
                req = {"message": message}
                if tool_input.get("delay_minutes") is not None:
                    req["delay_minutes"] = tool_input["delay_minutes"]
                if tool_input.get("at"):
                    req["at"] = tool_input["at"]
                if "delay_minutes" not in req and "at" not in req:
                    return "Need either a delay in minutes or an exact time."
                self.bus.publish(REMINDER_SET_TOPIC, req)
                return "Reminder scheduled."
            if name == "start_following":
                self.bus.publish(FOLLOW_CONTROL_TOPIC, {"enabled": True})
                return "Following started; movement is safety-checked."
            if name == "stop_following":
                self.bus.publish(FOLLOW_CONTROL_TOPIC, {"enabled": False})
                return "Following stopped."
            if name == "share_connection":
                req = {}
                if tool_input.get("name"):
                    req["name"] = tool_input["name"]
                self.bus.publish(BLUETOOTH_CONNECT_TOPIC, req)
                return "Trying to tether over Bluetooth to the phone."
            if name == "where_is_object":
                label_query = str(tool_input.get("label") or "").strip().lower()
                if not label_query:
                    return "No object name was given."
                label = speech_match.best_label_match(
                    label_query, self.spatial.sighting_labels())
                places = self.spatial.object_locations(label) if label else []
                if not places:
                    return f"No memory of ever seeing a {label_query} anywhere."
                top = places[0]
                out = (f"Last saw a {label} at {top['place']}, "
                       f"{_spoken_age(time.time() - top['last_seen'])}; "
                       f"seen there {top['times_seen']} time(s).")
                if len(places) > 1:
                    out += f" Also seen at {places[1]['place']}."
                return out
            if name == "recall_memory":
                query = str(tool_input.get("query") or "").strip()
                if not query:
                    return "No search query was given."
                facts = self.semantic.search_facts(query)
                if not facts:
                    return f"Nothing in long-term memory matches '{query}'."
                return " | ".join(f"{f['subject']}: {f['fact']}" for f in facts)
            if name == "list_known_people":
                names = _known_people()
                if not names:
                    return ("No faces learned yet. A person can say 'remember "
                            "me, I am <name>' while facing the camera to be "
                            "learned.")
                return "Recognizes these people by face: " + ", ".join(names)
            if name == "check_vital_stats":
                with self.lock:
                    health = dict(self.latest_health) if self.latest_health else None
                return self._format_health(health)
            if name == "register_low_power_intent":
                self.bus.publish(LOWPOWER_REQUEST_TOPIC, {"active": True})
                return "Entering low-power mode to conserve battery."
        except Exception as e:
            print(f"Companion: tool '{name}' failed: {e}")
            return "That didn't work."
        return f"Unknown tool: {name}"

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
            reply = self._chat_with_tools(client, messages)
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
            kind, item = self.work_queue.get()
            try:
                if kind == "repair":
                    self._repair_intent(item)
                elif kind == "learn":
                    self._learn_correction(*item)
                elif kind == "identify":
                    self._identify_object(item)
                else:
                    self._handle_utterance(item)
            except Exception as e:
                print(f"Companion: error handling {kind} '{item}': {e}")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/unhandled", self.on_unhandled)
        self.bus.subscribe("picarx/audio/uncertain", self.on_uncertain)
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.bus.subscribe(FEEDBACK_TOPIC, self.on_feedback)
        self.bus.subscribe(VISION_FRAME_TOPIC, self.on_frame)
        self.bus.subscribe("picarx/state/world", self.on_world_state)
        self.bus.subscribe(HEALTH_STATE_TOPIC, self.on_health)
        self.bus.subscribe(PERCEPTION_IDENTIFY_TOPIC, self.on_identify)

        for _ in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, daemon=True).start()

        print("Companion active, listening on picarx/audio/unhandled")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Companion().run()
