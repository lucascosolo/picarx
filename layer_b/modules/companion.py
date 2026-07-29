#!/usr/bin/env python3
# layer_b/modules/companion.py
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

It exposes a typed set of non-movement LLM TOOLS (see TOOLS) that let the
model ACT by TOGGLING other daemons over picarx/tools/* topics - never by
emitting motion. schedule_reminder arms reminder_daemon, share_connection
asks bluetooth_daemon to tether to a paired phone, and the other typed tools
query or control bounded services. Movement stays on the local command path;
the thinking model cannot start or stop following. The model chooses useful
non-movement actions and questions, never a maneuver.
The remote project tools likewise only publish typed requests to the SSH
helper; they never execute host commands in the companion process, and
writes/commands still require explicit confirmation.

Each reply is grounded with a short snapshot of picarx/state/world
(face/objects/distance/battery) folded into the prompt, so it can
answer naturally ("are you doing okay?", "what's that thing you're
looking at?") without needing its own sensor access. The snapshot also
carries the robot's own recent EXPERIENCE - whether it's moving or being
lifted, a bump/pickup it just felt (picarx/sensors/imu/event), what it
did earlier today (reflection.py's episodic diary), and what it learned
from an idle self-training run (picarx/self_trainer/status) - so it can
talk about itself in the first person ("someone just picked me up", "I
got stuck in the corner earlier"). All of that is fail-soft: the IMU is
optional and currently flaky, so a stale/absent reading simply drops the
motion and bump notes (vision's scene_motion is the moving-vs-still
fallback), never blocking a reply. Its personality (system prompt) is
separately grounded in the self-model facts reflection.py writes. Conversation
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

Requires a configured LLM provider (ANTHROPIC_API_KEY is preferred and
OPENAI_API_KEY is an optional fallback). If both are missing, or a request
fails/times out, this module just replies
with a short apology instead of raising - a quiet, unhelpful companion
is fine; a crashed process that stops handling any future messages is
not.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config
from llm_gateway import LLMGateway
from semantic_store import SemanticStore
from spatial_store import SpatialStore
import speech_match
import tool_catalog

import threading
import queue
import time
import json
import uuid
import math
from collections import deque

HISTORY_TURNS = 20          # user+assistant messages kept for context
SELF_MODEL_MAX = 5          # self-model facts folded into the personality prompt
WORKER_THREADS = 2
REPLY_TIMEOUT = 8.0
REPLY_MAX_TOKENS = 150

DATA_DIR = robot_config.data_path()
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
LEARNED_INTENTS_TTL_SEC = max(0.0, float(robot_config.get(
    "companion", "learned_intent_ttl_sec", 30 * 86400,
    env="LEARNED_INTENT_TTL_SEC")))
LEARNED_INTENTS_CONTROL_TOPIC = "picarx/intent/learned/control"
LEARNED_INTENTS_STATUS_TOPIC = "picarx/intent/learned/status"
INTENT_REPAIR_COOLDOWN = 10.0    # min seconds between arbiter API calls
INTENT_TIMEOUT = 6.0
INTENT_MAX_TOKENS = 80
INTENT_REPAIR_MIN_CONFIDENCE = min(1.0, max(0.0, float(robot_config.get(
    "companion", "intent_repair_min_confidence", 0.90,
    env="INTENT_REPAIR_MIN_CONFIDENCE"))))
INTENT_RECOVERY_STATUS_TOPIC = "picarx/intent/recovery/status"

# ---------- intent feedback (the user grading interpretations) ----------
# picarx/intent/feedback carries explicit judgments - the web console's
# check/X buttons and spoken phrases like "that's not what I meant"
# (routed by field_agent with the utterance being judged attached).
#   correct   -> reinforce the cached phrase mapping, if one produced it.
#   incorrect -> DELETE the cached mapping (it taught the wrong thing),
#                learn from a supplied correction, or - voice only - ask
#                "what did you want?" and treat the next utterance as
#                the answer. The clarification is routed back by the dialog
#                broker (dialog.py) on picarx/dialog/answer - it decides what
#                counts as an answer, not a race on the raw heard stream. Here
#                it's only LEARNED FROM: normalized onto a known command
#                (allowlist first, one small LLM call only if it's fuzzy) and
#                cached against the ORIGINAL phrasing, so next time it's on-board.
# Motion stays out of the cache in every path, same invariant as ever.
FEEDBACK_TOPIC = "picarx/intent/feedback"
CORRECTION_WINDOW_SEC = 45.0     # how long "what did you want?" waits for an answer
# Dialog broker protocol (dialog.py): register the "what did you want?"
# question, and receive the routed clarification.
DIALOG_ASK_TOPIC = "picarx/dialog/ask"
DIALOG_ANSWER_TOPIC = "picarx/dialog/answer"
DIALOG_CLEARED_TOPIC = "picarx/dialog/cleared"
DIALOG_ASKER = "companion"

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
    "what's playing", "list stations", "list reminders",
    "start meeting notes", "pause meeting notes", "resume meeting notes",
    "stop meeting notes", "list notes", "disconnect remote assist",
}
ALLOWED_INTENT_PREFIXES = (
    "tune to ", "radio find ", "station ", "where is ", "what's in ",
    "call this place ", "remind me in ", "remind me at ", "take a note ",
    "delete reminders ", "cancel reminders ", "delete notes ",
    "ssh into ", "connect to ",
)

INTENT_SYSTEM_PROMPT = """You repair garbled voice-command transcripts for a small robot car.
The transcript comes from an offline speech recognizer and may contain misheard words.

Known commands: stop, battery, status, history, objects, map, why, hello, who am i,
where are you, play radio, stop radio, next station, what's playing, list stations,
list reminders, start meeting notes, pause meeting notes, resume meeting notes,
stop meeting notes, list notes, disconnect remote assist, tune to <number>,
radio find <keywords>, station <name>, remind me in <time> to <message>,
remind me at <time> to <message>, take a note <text>, delete/cancel reminders <query>,
delete notes <query>, ssh into <host>, connect to <host>,
where is <object>  (asks the robot's memory where it last saw an object),
what's in <place>  (asks what objects it has seen at a named place),
call this place <name>  (names the robot's current location).

Reply with JSON only, one of:
{"command": "<one known command, with its parameter filled in if it takes one>",
 "confidence": 0.0, "rationale": "short evidence-based reason"}
  - only if the transcript was clearly an attempt at that command
{"chat": true}   - it was speech directed at the robot, but not one of the commands
{"ignore": true} - background noise, TV, or speech not meant for the robot

For a command, confidence must be a number from 0 to 1. Do not inflate it:
use a low value when words or parameters are ambiguous. The local catalog
below marks which commands may be auto-repaired; the robot will refuse to
execute a guessed state-changing, destructive, remote, or movement command.
Catalog: """ + tool_catalog.prompt_text() + """

NEVER return a movement command: requests to explore, drive, turn, go somewhere, or
follow someone must be answered with {"chat": true}, not a command. Never return a
remote write/command/authorization operation, recording consent, or an arbitrary
tool name. For a clearly attempted non-motion tool request, repair it only to one
of the exact commands listed above; otherwise return {"chat": true}."""

# ---------- camera-grounded chat ----------
# When someone asks the robot what it's looking at (or teaches it a new
# object: "remember this is a watering can"), attach a live camera
# frame to the LLM call so the reply is grounded in ACTUAL sight, not
# the 20 labels the on-board detector knows. Frames come from the single
# camera controller's temporary subscription.
CAMERA_SUBSCRIBE_TOPIC = "picarx/camera/subscribe"
CAMERA_FRAME_TOPIC = "picarx/camera/frame"
CAMERA_SUBSCRIBER = "companion"
CAMERA_FPS = 2.0
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
MAX_TOOL_ROUNDS = 8          # bounded, natural multi-step tool conversations
MAX_TOOL_CALLS = 16          # cap parallel/repeated calls in one utterance
TOOL_RESULT_WAIT_SEC = 0.8    # short wait; long jobs remain observable as pending
REMINDER_SET_TOPIC = "picarx/tools/reminder/set"
REMINDER_CONTROL_TOPIC = "picarx/tools/reminder/control"
REMINDER_RESULT_TOPIC = "picarx/tools/reminder/result"
REMINDER_STATE_TOPIC = "picarx/tools/reminder/state"
NOTES_TOPIC = "picarx/tools/notes"
NOTES_RESULT_TOPIC = "picarx/tools/notes/result"
FOLLOW_CONTROL_TOPIC = "picarx/tools/follow/set"
BLUETOOTH_CONNECT_TOPIC = "picarx/tools/bluetooth/connect"
HEALTH_STATE_TOPIC = "picarx/health/state"
LOWPOWER_REQUEST_TOPIC = "picarx/tools/lowpower/request"
REMOTE_ASSIST_TOPIC = "picarx/tools/remote_assist"
RADIO_TOPIC = "picarx/tools/radio"
ROBOT_STATE_TOPIC = "picarx/state/current"
GESTURE_STATUS_TOPIC = "picarx/gesture/status"
REMOTE_RESULT_TOPIC = "picarx/tools/remote_assist/result"
FOLLOW_STATUS_TOPIC = "picarx/tools/follow/status"
THINKING_CONTROL_TOPIC = "picarx/companion/thinking/control"
THINKING_STATUS_TOPIC = "picarx/companion/thinking/status"

# ---------- talking about its own experience ----------
# Beyond what it sees and who it's with, the companion grounds replies in the
# robot's own recent PHYSICAL experience so it can say "I just got picked up"
# or "I felt a bump", and in what it LEARNED from an idle self-training run.
# All fail-soft: the IMU is optional (and currently flaky), so a stale/absent
# imu block just drops these notes - the fallback is vision's scene_motion.
IMU_EVENT_TOPIC = "picarx/sensors/imu/event"          # edge-triggered bump/pickup
SELF_TRAINER_STATUS_TOPIC = "picarx/self_trainer/status"
PHYSICAL_EVENT_MEMORY_SEC = 90.0    # how long a bump/pickup stays "recent" to mention
BODY_TILT_LIFTED_DEG = 25.0         # body tilt above this reads as tilted/lifted
TRAINING_REPORT_TTL_SEC = 1800.0    # how long a finished-practice result stays fresh to mention
TRAINING_ANNOUNCE_COOLDOWN = 60.0   # min seconds between spoken practice-result reports

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
    {"name": "manage_reminders",
     "description": "List or cancel pending reminders. Cancellation is a "
                    "destructive action: set confirmed=true only after the "
                    "person explicitly approves deleting the selected reminder.",
     "input_schema": {"type": "object", "properties": {
         "operation": {"type": "string", "enum": ["list", "delete"]},
         "id": {"type": "string"}, "query": {"type": "string"},
         "confirmed": {"type": "boolean"}, "plan_id": {"type": "string"}},
         "required": ["operation"]}},
    {"name": "create_note",
     "description": "Save one user-authored note in durable memory. Use for "
                    "'take a note ...', not for an ongoing meeting transcript.",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"}, "title": {"type": "string"}},
         "required": ["text"]}},
    {"name": "manage_notes",
     "description": "List, read, export, search, or delete saved notes and "
                    "meeting logs. Deletion requires explicit confirmation.",
     "input_schema": {"type": "object", "properties": {
         "operation": {"type": "string", "enum": ["list", "get", "search",
                       "export", "delete"]}, "id": {"type": "string"},
         "query": {"type": "string"}, "confirmed": {"type": "boolean"},
         "plan_id": {"type": "string"}},
         "required": ["operation"]}},
    {"name": "control_meeting_notes",
     "description": "Start, pause, resume, or stop a consented meeting-note "
                    "transcript. Starting requires confirmed=true after the "
                    "person explicitly consents; the transcript stays local.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["start", "pause", "resume", "stop"]},
         "title": {"type": "string"}, "id": {"type": "string"},
         "confirmed": {"type": "boolean"}, "plan_id": {"type": "string"}},
         "required": ["action"]}},
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
    {"name": "describe_tools",
     "description": "Explain the non-movement tools you can use in this thinking "
                    "conversation. Use when the person asks what tools or abilities "
                    "you have; do not claim access to movement controls here.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_robot_status",
     "description": "Read what the robot is currently doing and its latest state: "
                    "mode/claims, health, perception, radio, follow, gesture, and "
                    "remote-session status. Use this before deciding what to do next "
                    "or when the person asks what is happening.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "propose_plan",
     "description": "Propose a bounded non-movement plan before doing a long-running "
                    "or destructive task. This only asks the person for approval; "
                    "it never approves itself or executes a step. Include the goal "
                    "and ordered steps.",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string"},
         "steps": {"type": "array", "items": {"type": "string"}}},
         "required": ["goal", "steps"]}},
    {"name": "cancel_current_task",
     "description": "Cancel the current thinking/tool task without moving the robot. "
                    "Use when the person says to stop the current research, coding, "
                    "or multi-tool task. It is safe to call even if nothing is running.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "control_radio",
     "description": "Control internet radio without moving the robot. You can play, "
                    "stop, skip, list stations, find stations, report status, or play "
                    "a named station. Use command and optional station/query.",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string", "enum": ["play", "stop", "next",
                     "list", "find", "status"]},
         "station": {"type": "string"},
         "query": {"type": "string"}}, "required": ["command"]}},
    {"name": "connect_remote_host",
     "description": "Connect to a developer's computer over SSH using the robot's "
                    "already-provisioned key, or a password entered in the tools "
                    "page. Use only when the person explicitly gives a host/IP "
                    "and project scope; never invent a host or ask them to speak "
                    "a password. Passwords are never part of this tool input or "
                    "sent to the model. The robot copies and runs its own bounded "
                    "helper on the host, so no host installation is needed.",
     "input_schema": {"type": "object", "properties": {
         "host": {"type": "string", "description": "IPv4, IPv6, or hostname"},
         "user": {"type": "string", "description": "optional SSH username"},
         "project_root": {"type": "string",
                           "description": "host project directory to scope access to"},
         "port": {"type": "integer", "description": "optional SSH port"}},
         "required": ["host"]}},
    {"name": "remote_project_operation",
     "description": "Use the connected host helper to inspect, edit, or debug the scoped "
                    "project. Supported operations are status, list, read, search, "
                    "write_file, delete_path, preview_patch, apply_patch, rollback, "
                    "run, logs, authorize_write, "
                    "revoke_write, and disconnect. "
                    "Read/list/search/preview/logs are safe. Grant write access once "
                    "with authorize_write after the person explicitly approves it; "
                    "that grant covers file edits, apply_patch, and rollback until "
                    "disconnect. Destructive or long-running operations require a "
                    "bounded plan approved through the local control path; include "
                    "its plan_id after approval. File overwrites use expected_sha256 "
                    "when available "
                    "to avoid clobbering a concurrent edit. "
                    "For run, set confirmed=true ONLY after the person explicitly "
                    "approves that specific command; otherwise ask for approval. Commands remain "
                    "host-side allowlisted and bounded.",
     "input_schema": {"type": "object", "properties": {
         "operation": {"type": "string", "enum": ["status", "list", "read",
                       "search", "stat", "logs", "write_file", "delete_path",
                       "preview_patch", "apply_patch",
                       "rollback", "run", "authorize_write", "revoke_write",
                       "disconnect"]},
         "path": {"type": "string"},
         "pattern": {"type": "string"},
         "patch": {"type": "string"},
         "content": {"type": "string"},
         "expected_sha256": {"type": "string"},
         "command": {"type": "string"},
         "cwd": {"type": "string"},
         "confirmed": {"type": "boolean"},
         "plan_id": {"type": "string"}},
         "required": ["operation"]}},
]

# Movement stays on the instant local command path and is deliberately not
# offered to the slower thinking model. Keep the exclusion centralized so a
# newly added conversational tool cannot accidentally acquire motor authority.
MOVEMENT_TOOL_NAMES = frozenset({"start_following", "stop_following"})
THINKING_TOOLS = tuple(
    tool for tool in TOOLS if tool["name"] not in MOVEMENT_TOOL_NAMES)
THINKING_TOOL_NAMES = frozenset(tool["name"] for tool in THINKING_TOOLS)


def tools_for_utterance(text):
    """Return the complete non-movement thinking catalog.

    Tool selection used to be a lexical pre-filter. That made the robot look
    incapable of useful follow-up work: a question about a reminder could not
    discover notes or status, and a multi-step task lost the tool it needed on
    the next model round. The model now receives one stable, movement-free
    catalog and chooses among typed tools itself; bounded rounds and per-call
    validation remain the safety limits.
    """
    del text  # retained as a compatibility seam for callers/tests
    return list(THINKING_TOOLS)

PEOPLE_DIR = f"{DATA_DIR}/people"


def _known_people():
    """Names of enrolled people (person_memory.py owns data/people/);
    fail-soft to [] when face memory isn't set up."""
    try:
        return sorted(d for d in os.listdir(PEOPLE_DIR)
                      if os.path.isdir(os.path.join(PEOPLE_DIR, d)))
    except OSError:
        return []


class ThinkingPlanManager:
    """Ephemeral approval state for model-selected non-movement plans.

    A model may propose a plan, but it cannot approve its own plan. Approval
    arrives through the typed local control topic (or an equally explicit UI
    path), expires automatically, and is required again after reconnect or
    restart. Keeping this state separate from the LLM transcript prevents a
    model-generated ``confirmed=true`` from becoming human consent.
    """

    def __init__(self, clock=None, ttl_sec=600.0):
        self.clock = clock or time.time
        self.ttl_sec = max(30.0, float(ttl_sec))
        self._lock = threading.Lock()
        self._plan = None

    def propose(self, goal, steps):
        goal = str(goal or "").strip()
        if not goal or len(goal) > 500:
            raise ValueError("plan goal is empty or too long")
        if not isinstance(steps, list) or not steps or len(steps) > 12:
            raise ValueError("plan needs between 1 and 12 steps")
        normalized = []
        for step in steps:
            step = str(step or "").strip()
            if not step or len(step) > 300:
                raise ValueError("each plan step must be 1-300 characters")
            normalized.append(step)
        now = float(self.clock())
        with self._lock:
            if self._plan and self._plan["status"] in {"pending", "approved"}:
                raise ValueError("another thinking plan is already active")
            self._plan = {
                "plan_id": uuid.uuid4().hex,
                "goal": goal,
                "steps": normalized,
                "status": "pending",
                "created_at": now,
                "expires_at": now + self.ttl_sec,
                "used_tools": [],
            }
            return dict(self._plan)

    def _expire_locked(self, now):
        if self._plan and self._plan["status"] in {"pending", "approved"} and \
                now >= self._plan["expires_at"]:
            self._plan["status"] = "expired"

    def current(self):
        with self._lock:
            self._expire_locked(float(self.clock()))
            return dict(self._plan) if self._plan else None

    def approve(self, plan_id):
        with self._lock:
            self._expire_locked(float(self.clock()))
            if not self._plan or self._plan["plan_id"] != str(plan_id):
                return None
            if self._plan["status"] != "pending":
                return dict(self._plan)
            self._plan["status"] = "approved"
            self._plan["approved_at"] = float(self.clock())
            return dict(self._plan)

    def reject(self, plan_id, status="rejected"):
        with self._lock:
            if not self._plan or self._plan["plan_id"] != str(plan_id):
                return None
            self._plan["status"] = status
            return dict(self._plan)

    def allows(self, plan_id, tool_name=None):
        with self._lock:
            self._expire_locked(float(self.clock()))
            if not self._plan or self._plan["status"] != "approved" or \
                    self._plan["plan_id"] != str(plan_id):
                return False
            if tool_name and tool_name not in self._plan["used_tools"]:
                self._plan["used_tools"].append(str(tool_name))
            return True


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

The snapshot can also include your own recent EXPERIENCE: whether you're moving or
sitting still or being lifted, a bump or pickup you just felt, what you got up to
earlier today, and anything you learned from a practice session. Talk about these in
the first person, like your own memories - "someone just picked me up", "I got stuck
in the corner earlier", "I practised avoiding obstacles and picked up a new trick".
Only mention them when they fit the conversation; don't list them off.

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
        self.gateway = LLMGateway(bus=self.bus)
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
        # Read-only status mirrors used by the thinking tools. Producers own
        # these state machines; Companion only retains the latest bounded
        # payload so a question can be answered without inventing activity.
        self.latest_robot_state = None
        self.latest_gesture_status = None
        self.latest_follow_status = None
        self.latest_remote_result = None
        self.latest_radio_state = None
        self._tool_result_condition = threading.Condition()
        self._tool_results = {}
        self._thinking_runs_lock = threading.Lock()
        self._thinking_runs = {}
        self.plan_manager = ThinkingPlanManager()
        # The reminder daemon owns timers; this is only a small read-only cache
        # of its published state so follow-up questions can be answered locally
        # without asking Claude to reconstruct an asynchronous tool call.
        self.pending_reminders = {}
        # Pending "what did you want me to do?" question, or None:
        # {"question_id": <id the dialog broker holds>, "utterance": <original
        # misread phrasing>}. The broker owns the answer window/expiry; this is
        # just what we need to learn from the routed clarification.
        self.awaiting_correction = None
        self._last_identify_at = 0.0      # throttles the cloud identify tier
        # Recent physical experience (from picarx/sensors/imu/event): a short
        # rolling memory of bumps/pickups so it can mention "I just felt a bump".
        # Empty and harmless when the IMU is unavailable.
        self.recent_physical_events = deque(maxlen=8)
        # Last idle self-training result worth mentioning, or None:
        # {"scenario", "notes", "patterns", "adopted", "ts"}.
        self.latest_training = None
        self._last_training_announce_at = 0.0

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
            if not isinstance(cache, dict):
                raise ValueError("learned intent cache is not an object")
            print(f"Companion: {len(cache)} learned phrases loaded")
            return cache
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"Companion: failed to load learned intents, starting fresh: {e}")
            return {}

    def _prune_learned_intents(self, now=None):
        """Remove expired, malformed, unsafe, and over-capacity aliases.

        Learned aliases are useful only while their evidence is reasonably
        fresh.  Keep legacy entries with ``last=0`` readable, but normalize
        every newly touched entry to a real timestamp.  The allowlist is
        checked again here so a future code/config change cannot revive a
        previously unsafe cache entry.
        """
        now = time.time() if now is None else float(now)
        removed = []
        with self.lock:
            for key, entry in list(self.learned_intents.items()):
                reason = None
                if not isinstance(key, str) or not key.strip():
                    reason = "invalid_phrase"
                elif not isinstance(entry, dict):
                    reason = "invalid_entry"
                else:
                    command = str(entry.get("command") or "").strip().lower()
                    try:
                        last = float(entry.get("last", 0) or 0)
                    except (TypeError, ValueError):
                        last = 0
                        reason = "invalid_timestamp"
                    if not reason and (not math.isfinite(last) or last < 0):
                        reason = "invalid_timestamp"
                    if not reason and not self._intent_allowed(command):
                        reason = "command_not_allowed"
                    elif (not reason and last > 0 and
                          now - last > LEARNED_INTENTS_TTL_SEC):
                        reason = "expired"
                    elif not reason:
                        entry["command"] = command
                        entry["last"] = last
                if reason:
                    self.learned_intents.pop(key, None)
                    removed.append({"phrase": str(key), "reason": reason})

            if len(self.learned_intents) > LEARNED_INTENTS_MAX:
                keep = sorted(
                    self.learned_intents.items(),
                    key=lambda kv: float(kv[1].get("last", 0) or 0),
                    reverse=True)[:LEARNED_INTENTS_MAX]
                kept = {key for key, _ in keep}
                for key in list(self.learned_intents):
                    if key not in kept:
                        self.learned_intents.pop(key, None)
                        removed.append({"phrase": str(key), "reason": "capacity"})
        return removed

    def _save_learned_intents(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        # Re-validate on every write because feedback, correction, and repair
        # arrive through different asynchronous paths.
        self._prune_learned_intents()
        with self.lock:
            snapshot = json.dumps(self.learned_intents, indent=1)
        tmp_path = f"{LEARNED_INTENTS_PATH}.tmp"
        with open(tmp_path, "w") as f:
            f.write(snapshot)
        os.replace(tmp_path, LEARNED_INTENTS_PATH)

    def _learned_intent_status(self, operation="list", request_id=None,
                               removed=None, deleted=None, error=None):
        """Publish an operator-readable snapshot without executing aliases."""
        rows = []
        now = time.time()
        with self.lock:
            for phrase, entry in self.learned_intents.items():
                last = float(entry.get("last", 0) or 0)
                rows.append({
                    "phrase": phrase,
                    "command": entry.get("command"),
                    "count": int(entry.get("count", 0) or 0),
                    "confirmed": bool(entry.get("confirmed")),
                    "taught": bool(entry.get("taught")),
                    "confidence": entry.get("confidence"),
                    "source": entry.get("source"),
                    "last": last,
                    "expires_at": (last + LEARNED_INTENTS_TTL_SEC
                                   if last > 0 else None),
                })
        rows.sort(key=lambda row: (row["last"], row["phrase"]), reverse=True)
        payload = {
            "operation": operation,
            "entries": rows,
            "count": len(rows),
            "ttl_sec": LEARNED_INTENTS_TTL_SEC,
            "now": now,
            "ts": now,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        if removed:
            payload["removed"] = removed
        if deleted is not None:
            payload["deleted"] = deleted
        if error:
            payload["error"] = error
        self.bus.publish(LEARNED_INTENTS_STATUS_TOPIC, payload)

    def on_learned_intent_control(self, payload):
        """Review or delete learned aliases through an explicit local-admin path.

        Supported operations are ``list``, ``delete`` (by original phrase or
        canonical key), and confirmation-protected ``clear``.  This is kept
        separate from the voice command path so an utterance can never erase
        the robot's learned behavior by accident.
        """
        operation = str(payload.get("operation") or "list").strip().lower()
        request_id = payload.get("request_id")
        removed = self._prune_learned_intents()
        if removed:
            self._save_learned_intents()
        if operation == "list":
            self._learned_intent_status(operation, request_id, removed=removed)
            return
        if operation == "delete":
            phrase = payload.get("phrase") or payload.get("key")
            key = speech_match.canonicalize(str(phrase)) if phrase else ""
            with self.lock:
                entry = self.learned_intents.pop(key, None) if key else None
            if entry is not None:
                self._save_learned_intents()
            self._learned_intent_status(operation, request_id, removed=removed,
                                        deleted=bool(entry),
                                        error=None if phrase else "phrase is required")
            return
        if operation == "clear":
            if payload.get("confirmed") is not True:
                self._learned_intent_status(
                    operation, request_id, removed=removed, deleted=False,
                    error="confirmed=true is required")
                return
            with self.lock:
                count = len(self.learned_intents)
                self.learned_intents.clear()
            self._save_learned_intents()
            self._learned_intent_status(operation, request_id, removed=removed,
                                        deleted=count > 0)
            return
        self._learned_intent_status(
            operation, request_id, removed=removed,
            error="operation must be list, delete, or clear")

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

    def _publish_recovery_status(self, state, original_text, verdict=None,
                                 reason=None, spec=None):
        """Publish bounded, non-secret diagnostics for intent recovery."""
        verdict = verdict if isinstance(verdict, dict) else {}
        payload = {
            "state": state,
            "text": str(original_text or "")[:240],
            "candidate": str(verdict.get("command") or "")[:160] or None,
            "confidence": verdict.get("confidence"),
            "rationale": str(verdict.get("rationale") or "")[:240] or None,
            "reason": reason,
            "safety_class": (spec or {}).get("safety_class"),
            "auto_repair_min_confidence": INTENT_REPAIR_MIN_CONFIDENCE,
            "ts": time.time(),
        }
        self.bus.publish(INTENT_RECOVERY_STATUS_TOPIC, payload)

    @staticmethod
    def _intent_allowed(command):
        return (command in ALLOWED_INTENTS or
                any(command.startswith(p) and len(command) > len(p)
                    for p in ALLOWED_INTENT_PREFIXES))

    # ---------- inbound ----------

    def on_world_state(self, payload):
        with self.lock:
            self.latest_world = payload

    def on_robot_state(self, payload):
        with self.lock:
            self.latest_robot_state = dict(payload or {})

    def on_gesture_status(self, payload):
        with self.lock:
            self.latest_gesture_status = dict(payload or {})

    def on_follow_status(self, payload):
        with self.lock:
            self.latest_follow_status = dict(payload or {})

    def on_remote_result(self, payload):
        with self.lock:
            self.latest_remote_result = dict(payload or {})
        self._record_tool_result(payload)

    def on_radio_state(self, payload):
        with self.lock:
            self.latest_radio_state = dict(payload or {})

    def on_notes_result(self, payload):
        self._record_tool_result(payload)

    def _record_tool_result(self, payload):
        request_id = str((payload or {}).get("request_id") or "").strip()
        if not request_id:
            return
        condition = getattr(self, "_tool_result_condition", None)
        if condition is None:
            self._tool_result_condition = condition = threading.Condition()
        with condition:
            results = getattr(self, "_tool_results", None)
            if results is None:
                self._tool_results = results = {}
            results[request_id] = dict(payload or {})
            # Keep the correlation cache bounded if a daemon repeats an old
            # result or a caller never waits for a long-running operation.
            if len(results) > 128:
                for old in list(results)[:32]:
                    results.pop(old, None)
            condition.notify_all()

    def _dispatch_thinking_request(self, topic, payload, result_topic=None):
        """Publish a typed request and briefly correlate its daemon result."""
        request_id = uuid.uuid4().hex
        request = dict(payload or {}, request_id=request_id)
        self.bus.publish(topic, request)
        if not result_topic:
            return f"request sent (id {request_id[:8]})."
        condition = getattr(self, "_tool_result_condition", None)
        if condition is None:
            self._tool_result_condition = condition = threading.Condition()
        deadline = time.monotonic() + TOOL_RESULT_WAIT_SEC
        with condition:
            while request_id not in getattr(self, "_tool_results", {}):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                condition.wait(remaining)
            result = getattr(self, "_tool_results", {}).pop(request_id, None)
        if result is None:
            return f"request sent (id {request_id[:8]}); result is still pending."
        if not result.get("ok"):
            return "request failed: " + str(result.get("error") or "unknown error")[:400]
        value = result.get("result")
        if value is None:
            return f"request completed (id {request_id[:8]})."
        # Preserve bounded structured results so the model can use a read/list
        # result in its next tool call, while never exposing the request body.
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
        return f"request completed (id {request_id[:8]}): {rendered[:20000]}"

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

    def on_reminder_state(self, payload):
        """Cache reminder-daemon state for local, token-free follow-ups."""
        payload = dict(payload or {})
        event = str(payload.get("event") or "").lower()
        rows = payload.get("reminders")
        with self.lock:
            if not isinstance(getattr(self, "pending_reminders", None), dict):
                self.pending_reminders = {}
            if isinstance(rows, list):
                self.pending_reminders = {
                    str(row.get("id")): dict(row)
                    for row in rows if isinstance(row, dict) and row.get("id")
                }
            elif event == "set" and payload.get("id"):
                self.pending_reminders[str(payload["id"])] = payload
            elif event in {"deleted", "fired"} and payload.get("id"):
                self.pending_reminders.pop(str(payload["id"]), None)

    def on_reminder_result(self, payload):
        """Ingest list/set/delete results, including requests made by tools."""
        payload = dict(payload or {})
        self._record_tool_result(payload)
        result = payload.get("result") or {}
        rows = result.get("reminders") if isinstance(result, dict) else None
        if isinstance(rows, list):
            self.on_reminder_state({"reminders": rows})
            return
        command = str(payload.get("command") or "").lower()
        if command == "set" and isinstance(result, dict) and result.get("id"):
            self.on_reminder_state({"event": "set", **result})
        elif command == "delete" and isinstance(result, dict) and result.get("id"):
            self.on_reminder_state({"event": "deleted", **result})

    @staticmethod
    def _reminder_question(text):
        """Recognize reminder follow-ups without stealing new set requests."""
        lowered = (text or "").lower()
        if "remind" in lowered or "reminder" in lowered:
            return ("what" in lowered or "which" in lowered or "when" in lowered)
        # The common immediate follow-up omits the word reminder:
        # "what are you going to tell me in five minutes?"
        return ("what" in lowered and "minute" in lowered and
                any(word in lowered for word in ("tell", "do", "again")))

    @staticmethod
    def _spoken_duration(seconds):
        minutes = max(1, round(max(0.0, seconds) / 60.0))
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        hours = max(1, round(minutes / 60.0))
        return f"about {hours} hour{'s' if hours != 1 else ''}"

    def _maybe_answer_reminder(self, text):
        """Answer reminder-status questions from the daemon's local state."""
        if not self._reminder_question(text):
            return False
        with self.lock:
            rows = [dict(row) for row in
                    getattr(self, "pending_reminders", {}).values()]
        if not rows:
            return False
        rows.sort(key=lambda r: float(r.get("fire_at", 0)))
        now = time.time()
        if len(rows) == 1:
            row = rows[0]
            message = str(row.get("message") or "something")
            remaining = float(row.get("fire_at", now)) - now
            if remaining > 0:
                reply = (f"I'm reminding you to {message} in "
                         f"{self._spoken_duration(remaining)}.")
            else:
                reply = f"I'm reminding you to {message}."
        else:
            items = "; ".join(str(row.get("message") or "something")
                               for row in rows[:5])
            suffix = " and more" if len(rows) > 5 else ""
            reply = f"Your pending reminders are: {items}{suffix}."
        with self.lock:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            self.last_turn_at = now
        self._save_memory()
        print(f"Companion (reminder state): {reply}")
        self.bus.publish("picarx/audio/speak", {"text": reply, "ts": now})
        return True

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

    # ---------- its own physical experience ----------

    def on_imu_event(self, payload):
        """A bump or a pickup/tilt from imu.py (edge-triggered). Remembered
        briefly so the model can reference it ("someone just picked me up").
        Fail-soft: with the IMU down, no events arrive and this stays empty."""
        kind = payload.get("kind")
        if kind not in ("impact", "tilted"):
            return
        with self.lock:
            self.recent_physical_events.append(
                {"kind": kind, "ts": payload.get("ts") or time.time()})

    def on_self_trainer_status(self, payload):
        """Idle self-training lifecycle. On a 'published' result (a session
        that actually produced learning), remember it so chat can mention what
        was learned, and report it out loud once - the robot narrating its own
        practice. Other states are ignored here."""
        if payload.get("state") != "published":
            return
        now = time.time()
        result = {
            "scenario": payload.get("scenario"),
            "notes": payload.get("notes") or 0,
            "patterns": payload.get("patterns") or 0,
            "adopted": bool(payload.get("adopted")),
            "ts": now,
        }
        with self.lock:
            self.latest_training = result
            announce = now - self._last_training_announce_at > TRAINING_ANNOUNCE_COOLDOWN
            if announce:
                self._last_training_announce_at = now
        if announce:
            self._say(self._training_report(result, spoken=True))

    @staticmethod
    def _training_report(result, spoken=False):
        """A short sentence about what a practice session produced. `spoken` adds
        a friendly lead-in for the proactive announcement."""
        scenario = result.get("scenario")
        where = f" on {scenario}" if scenario else ""
        bits = []
        notes = result.get("notes") or 0
        if notes:
            bits.append(f"{notes} new note{'s' if notes != 1 else ''}")
        if result.get("adopted"):
            bits.append("a new driving tactic")
        learned = " and ".join(bits) if bits else "a little more about how I drive"
        lead = "I just finished practising" if spoken else "I recently practised"
        return f"{lead}{where}. I picked up {learned}."

    # ---------- intent feedback ----------

    def _say(self, text):
        self.bus.publish("picarx/audio/speak", {"text": text, "ts": time.time()})

    def _journal_feedback(self, verdict, utterance, detail):
        self.bus.publish("picarx/decision", {
            "source": "companion", "kind": "intent_feedback",
            "choice": {"verdict": verdict, **detail},
            "reason": f"user judged the interpretation of: '{utterance}'",
            "ts": time.time()})

    def on_dialog_answer(self, payload):
        """The dialog broker routed the clarification to our 'what did you want
        me to do?' question. The broker already screened it (a real reply, not
        a feedback verdict), so we just LEARN from it. The clarification still
        executes through field_agent's normal pipeline on its own; this handler
        never dispatches anything."""
        if payload.get("asker") != DIALOG_ASKER:
            return
        with self.lock:
            awaiting = self.awaiting_correction
            if not awaiting or payload.get("question_id") != awaiting["question_id"]:
                return
            self.awaiting_correction = None
        text = (payload.get("text") or "").strip()
        if text:
            self.work_queue.put(("learn", (awaiting["utterance"], text)))

    def on_dialog_cleared(self, payload):
        """Our correction question expired (or was replaced) with no answer -
        just drop the pending record."""
        if payload.get("asker") != DIALOG_ASKER:
            return
        with self.lock:
            awaiting = self.awaiting_correction
            if awaiting and payload.get("question_id") == awaiting["question_id"]:
                self.awaiting_correction = None

    def on_feedback(self, payload):
        verdict = payload.get("verdict")
        if verdict not in ("correct", "incorrect"):
            return
        pruned = self._prune_learned_intents()
        if pruned:
            self._save_learned_intents()
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
                question_id = uuid.uuid4().hex
                with self.lock:
                    self.awaiting_correction = {
                        "question_id": question_id, "utterance": utterance}
                # Register with the dialog broker BEFORE speaking, so the
                # clarification always has somewhere to route.
                self.bus.publish(DIALOG_ASK_TOPIC, {
                    "asker": DIALOG_ASKER, "question_id": question_id,
                    "kind": "correction", "ttl": CORRECTION_WINDOW_SEC,
                    "prompt": "what did you want me to do?", "ts": now})
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
        pruned = self._prune_learned_intents()
        if pruned:
            self._save_learned_intents()
        key = speech_match.canonicalize(text)
        with self.lock:
            entry = self.learned_intents.get(key)
            if entry:
                entry["count"] = entry.get("count", 0) + 1
                entry["last"] = time.time()
        if entry:
            spec = tool_catalog.command_spec(entry.get("command"))
            # Entries created by the old arbiter had no provenance and could
            # include state-changing tool guesses. Never replay those after
            # the confidence gate was introduced. User-taught aliases remain
            # valid because they came from an explicit correction path.
            if not entry.get("taught") and (spec is None or not spec["auto_repair"]):
                with self.lock:
                    self.learned_intents.pop(key, None)
                self._save_learned_intents()
                self._publish_recovery_status(
                    "rejected", text,
                    {"command": entry.get("command"),
                     "confidence": entry.get("confidence"),
                     "rationale": entry.get("rationale")},
                    reason="unsafe_cached_alias", spec=spec)
                return
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
        # `_client` remains a compatibility seam for off-robot tests and
        # local overlays. Production calls use the shared gateway so provider
        # selection, fallback, privacy, and telemetry stay centralized.
        client = getattr(self, "_client", None)
        if client is not None:
            return client
        gateway = getattr(self, "gateway", None)
        if gateway is None:
            gateway = self.gateway = LLMGateway(bus=getattr(self, "bus", None))
        return gateway if gateway.available() else None

    def _context_blurb(self):
        with self.lock:
            snap = dict(self.latest_world) if self.latest_world else None
            reminders = list(getattr(self, "pending_reminders", {}).values())
        if not snap:
            return self._reminder_blurb(reminders) or "no sensor data yet"

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

        reminder_note = self._reminder_blurb(reminders)
        if reminder_note:
            parts.append(reminder_note)

        plan = self._plan_manager().current()
        if plan and plan.get("status") in {"pending", "approved"}:
            parts.append("thinking plan " + str(plan.get("plan_id")) +
                         " is " + str(plan.get("status")) +
                         f" with {len(plan.get('steps') or [])} steps")

        # Its own recent EXPERIENCE: how it's moving/being handled right now,
        # a bump/pickup it just felt, what it did earlier today, and anything it
        # learned from practising. All fail-soft (see _experience_notes).
        parts.extend(self._experience_notes(time.time(), snap))

        return "; ".join(parts)

    @staticmethod
    def _reminder_blurb(reminders):
        """Compact reminder context; never include more than a few rows."""
        if not reminders:
            return ""
        rows = sorted(reminders, key=lambda r: float(r.get("fire_at", 0)))[:5]
        return "pending reminders: " + "; ".join(
            str(row.get("message") or "(unnamed reminder)")[:80]
            for row in rows)

    def _experience_notes(self, now, snap):
        """First-person notes about the robot's own recent experience, folded
        into the per-turn context so it can talk about itself naturally. Every
        source is optional and fail-soft - a broken/absent IMU just drops the
        motion and bump notes (vision's scene_motion is the fallback for
        moving-vs-still), an empty diary drops the 'earlier today' line."""
        notes = []

        # Live motion / orientation. Prefer the IMU; fall back to vision motion.
        imu = snap.get("imu") or {}
        if not imu.get("stale", True):
            body_tilt = imu.get("body_tilt_deg") or 0
            if imu.get("tilted") or body_tilt >= BODY_TILT_LIFTED_DEG:
                notes.append("I'm being tilted or lifted right now")
            elif imu.get("moving"):
                notes.append("I'm moving")
            else:
                notes.append("I'm sitting still")
        else:
            scene_motion = (snap.get("objects") or {}).get("scene_motion")
            if scene_motion is not None:
                notes.append("I seem to be moving" if scene_motion >= 3.0
                             else "I'm sitting still")

        # A bump or pickup felt recently (edge-triggered IMU events).
        with self.lock:
            events = [e for e in self.recent_physical_events
                      if now - e["ts"] <= PHYSICAL_EVENT_MEMORY_SEC]
            training = dict(self.latest_training) if self.latest_training else None
        if events:
            last = events[-1]
            what = ("was picked up or tilted" if last["kind"] == "tilted"
                    else "felt a bump")
            notes.append(f"a moment ago I {what} ({_spoken_age(now - last['ts'])})")

        # What it did earlier today (reflection.py's episodic diary).
        try:
            episodes = self.semantic.facts_for(
                f"episode:{time.strftime('%Y-%m-%d')}", limit=1)
        except Exception:
            episodes = []
        if episodes:
            diary = episodes[0]["fact"]
            if len(diary) > 200:
                diary = diary[:200].rstrip() + "..."
            notes.append(f"earlier today: {diary}")

        # What it learned from a recent idle self-training session.
        if training and now - training["ts"] <= TRAINING_REPORT_TTL_SEC:
            notes.append(self._training_report(training))

        return notes

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
            if isinstance(client, LLMGateway):
                result = client.complete(
                    request_id=f"companion-intent-{uuid.uuid4().hex}",
                    task="intent_repair", complexity="low",
                    system=INTENT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=INTENT_MAX_TOKENS, timeout=INTENT_TIMEOUT,
                    privacy="command_repair", idempotent=True)
                if not result.ok:
                    return None
                raw = result.text
            else:
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
            if not isinstance(verdict, dict):
                return None
            if "command" in verdict:
                try:
                    confidence = float(verdict.get("confidence"))
                except (TypeError, ValueError):
                    confidence = 0.0
                verdict["confidence"] = min(1.0, max(0.0, confidence))
                verdict["rationale"] = str(verdict.get("rationale") or "")[:240]
            return verdict
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
            self._publish_recovery_status("failed", text, reason="no_valid_verdict")
            return

        command = (verdict.get("command") or "").strip().lower()
        if command and self._intent_allowed(command):
            spec = tool_catalog.command_spec(command)
            confidence = float(verdict.get("confidence") or 0.0)
            if spec is None:
                self._publish_recovery_status(
                    "rejected", text, verdict,
                    reason="command_missing_from_catalog")
                print(f"Companion arbiter: catalog rejected '{command}'")
                return
            if not spec["auto_repair"]:
                self._publish_recovery_status(
                    "confirmation_required", text, verdict,
                    reason="state_changing_or_external_command", spec=spec)
                print(f"Companion arbiter: refusing guessed {spec['safety_class']} "
                      f"command '{command}'")
                return
            if confidence < INTENT_REPAIR_MIN_CONFIDENCE:
                self._publish_recovery_status(
                    "low_confidence", text, verdict,
                    reason="below_auto_repair_threshold", spec=spec)
                print(f"Companion arbiter: low confidence {confidence:.2f} for "
                      f"'{command}', dropping")
                return
            with self.lock:
                self.learned_intents[speech_match.canonicalize(text)] = {
                    "command": command, "count": 1, "last": time.time(),
                    "confidence": confidence,
                    "rationale": str(verdict.get("rationale") or "")[:240],
                    "source": "llm_repair"}
            self._save_learned_intents()
            self._dispatch_repaired(command, text, learned=False)
            self._publish_recovery_status(
                "accepted", text, verdict, reason="read_only_confident",
                spec=spec)
        elif verdict.get("chat"):
            self._publish_recovery_status("chat", text, verdict,
                                          reason="model_classified_as_chat")
            self._handle_utterance(text)
        else:
            # ignore verdict, disallowed command, or junk output - all
            # end the same way: silently not acting on garbled audio.
            print(f"Companion arbiter: no action for '{text}' ({verdict})")
            self._publish_recovery_status("rejected", text, verdict,
                                          reason="no_safe_command")

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
        self.bus.publish(CAMERA_SUBSCRIBE_TOPIC, {
            "subscriber": CAMERA_SUBSCRIBER, "enabled": True,
            "fps": CAMERA_FPS, "ttl": 2.0, "ts": time.time()})
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
            self.bus.publish(CAMERA_SUBSCRIBE_TOPIC, {
                "subscriber": CAMERA_SUBSCRIBER, "enabled": False,
                "ts": time.time()})

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
            messages = [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": frame_b64}},
                {"type": "text", "text": "What object is this?"}]}]
            if isinstance(client, LLMGateway):
                result = client.complete(
                    request_id=f"companion-identify-{uuid.uuid4().hex}",
                    task="object_identification", complexity="low",
                    system=IDENTIFY_SYSTEM_PROMPT, messages=messages,
                    max_tokens=IDENTIFY_MAX_TOKENS, timeout=INTENT_TIMEOUT,
                    privacy="user_visual_request", idempotent=True)
                if not result.ok:
                    return
                raw = result.text
            else:
                response = client.messages.create(
                    model=INTENT_MODEL,
                    max_tokens=IDENTIFY_MAX_TOKENS,
                    system=IDENTIFY_SYSTEM_PROMPT,
                    messages=messages,
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

    def _plan_manager(self):
        manager = getattr(self, "plan_manager", None)
        if manager is None:
            self.plan_manager = manager = ThinkingPlanManager()
        return manager

    def _publish_plan_event(self, plan, event, reason, plan_id=None):
        plan_id = plan_id or (plan or {}).get("plan_id")
        payload = {
            "state": "plan_" + str(event),
            "plan_id": plan_id,
            "plan_status": (plan or {}).get("status"),
            "step_count": len((plan or {}).get("steps") or []),
            "ts": time.time(),
        }
        self.bus.publish(THINKING_STATUS_TOPIC, payload)
        self.bus.publish("picarx/decision", {
            "source": "companion", "kind": "thinking_plan",
            "choice": {"event": event, "plan_status": (plan or {}).get("status"),
                        "plan_id": plan_id,
                        "step_count": len((plan or {}).get("steps") or [])},
            "reason": str(reason)[:240], "ts": time.time(),
        })

    def _approved_plan_required(self, tool_input, tool_name):
        plan_id = str((tool_input or {}).get("plan_id") or "").strip()
        if not plan_id:
            current = self._plan_manager().current()
            suffix = (f" Plan {current['plan_id']} is waiting for your approval."
                      if current and current.get("status") == "pending" else "")
            return ("This operation needs an explicitly approved thinking plan. "
                    "Propose a plan first, then approve it through the local control "
                    "path." + suffix)
        if not self._plan_manager().allows(plan_id, tool_name):
            current = self._plan_manager().current()
            state = current.get("status") if current else "missing"
            return f"Plan approval is not valid for this operation (state: {state})."
        return None

    def _thinking_registry(self):
        lock = getattr(self, "_thinking_runs_lock", None)
        if lock is None:
            self._thinking_runs_lock = lock = threading.Lock()
        runs = getattr(self, "_thinking_runs", None)
        if runs is None:
            self._thinking_runs = runs = {}
        return lock, runs

    def _start_thinking_run(self, run_id):
        lock, runs = self._thinking_registry()
        cancel = threading.Event()
        with lock:
            runs[run_id] = {"cancel": cancel, "started_at": time.time()}
        self.bus.publish(THINKING_STATUS_TOPIC, {
            "state": "running", "run_id": run_id, "ts": time.time()})
        return cancel

    def _finish_thinking_run(self, run_id, state="complete"):
        lock, runs = self._thinking_registry()
        with lock:
            runs.pop(run_id, None)
        self.bus.publish(THINKING_STATUS_TOPIC, {
            "state": state, "run_id": run_id, "ts": time.time()})

    def _cancel_thinking_runs(self, run_id=None):
        lock, runs = self._thinking_registry()
        canceled = []
        with lock:
            selected = ([run_id] if run_id else list(runs))
            for current in selected:
                record = runs.get(current)
                if record:
                    record["cancel"].set()
                    canceled.append(current)
        self.bus.publish(THINKING_STATUS_TOPIC, {
            "state": "cancel_requested", "run_id": run_id,
            "canceled_count": len(canceled), "ts": time.time()})
        return canceled

    def on_thinking_control(self, payload):
        """Cancel or inspect thinking runs through a typed local control path."""
        payload = dict(payload or {})
        command = str(payload.get("command") or payload.get("operation") or
                      "status").lower()
        if command in {"approve_plan", "reject_plan", "cancel_plan"}:
            plan_id = str(payload.get("plan_id") or "").strip()
            if command == "approve_plan":
                if payload.get("confirmed") is not True:
                    self._publish_plan_event(
                        self._plan_manager().current(), "approval_failed",
                        "confirmed=true is required", plan_id=plan_id)
                    return
                plan = self._plan_manager().approve(plan_id)
                event = "approved" if plan and plan.get("status") == "approved" else "approval_failed"
                self._publish_plan_event(plan, event, "explicit local approval",
                                         plan_id=plan_id)
                return
            plan = self._plan_manager().reject(plan_id, "canceled" if command == "cancel_plan"
                                               else "rejected")
            self._publish_plan_event(
                plan, "canceled" if command == "cancel_plan" else "rejected",
                "explicit local plan rejection", plan_id=plan_id)
            return
        if command == "cancel":
            self._cancel_thinking_runs(payload.get("run_id"))
            return
        lock, runs = self._thinking_registry()
        with lock:
            active = [{"run_id": run_id,
                       "started_at": record.get("started_at")}
                      for run_id, record in runs.items()]
        self.bus.publish(THINKING_STATUS_TOPIC, {
            "state": "status", "active": active, "ts": time.time()})

    def _chat_with_tools(self, client, messages, tools=None):
        """One utterance, with the model allowed to call tools. Runs a
        bounded tool<->model loop: each round, execute any tool_use blocks
        (which just publish mode toggles to the daemons) and feed the
        results back so the model can produce a natural spoken reply.
        Returns the final spoken text ("" if none)."""
        convo = list(messages)
        requested_tools = THINKING_TOOLS if tools is None else list(tools)
        active_tools = [tool for tool in requested_tools
                        if tool.get("name") not in MOVEMENT_TOOL_NAMES]
        final_text = ""
        request_id = f"companion-chat-{uuid.uuid4().hex}"
        cancel_event = self._start_thinking_run(request_id)
        tool_calls = 0
        exhausted = True
        canceled = False
        for _ in range(MAX_TOOL_ROUNDS):
            if cancel_event.is_set():
                canceled = True
                exhausted = False
                break
            if isinstance(client, LLMGateway):
                try:
                    result = client.complete(
                        request_id=request_id, task="companion_chat",
                        complexity="high", system=self._compose_system_prompt(),
                        messages=convo, tools=active_tools,
                        max_tokens=REPLY_MAX_TOKENS, timeout=REPLY_TIMEOUT,
                        privacy="user_conversation", idempotent=True)
                except Exception:
                    self._finish_thinking_run(request_id, "error")
                    raise
                if not result.ok:
                    self._finish_thinking_run(request_id, "error")
                    raise RuntimeError(result.failure.get("message", "LLM unavailable"))
                text = result.text
                content = result.content
                if text:
                    final_text = text
                tool_uses = [b for b in content if b.get("type") == "tool_use"]
                if not tool_uses:
                    exhausted = False
                    break
                convo.append({"role": "assistant", "content": content})
                results = []
                for tu in tool_uses:
                    if cancel_event.is_set():
                        canceled = True
                        results.append({"type": "tool_result",
                                        "tool_use_id": tu.get("id"),
                                        "content": "Thinking task canceled; tool not run."})
                        continue
                    tool_calls += 1
                    if tool_calls > MAX_TOOL_CALLS:
                        results.append({"type": "tool_result",
                                        "tool_use_id": tu.get("id"),
                                        "content": "Tool-call budget exhausted for this turn."})
                        continue
                    out = self._run_thinking_tool(
                        tu.get("name"), tu.get("input") or {}, tu.get("id"))
                    results.append({"type": "tool_result",
                                    "tool_use_id": tu.get("id"),
                                    "content": out})
                convo.append({"role": "user", "content": results})
                continue
            request = {
                "model": COMPANION_MODEL,
                "max_tokens": REPLY_MAX_TOKENS,
                "system": self._compose_system_prompt(),
                "messages": convo,
                "timeout": REPLY_TIMEOUT,
            }
            if active_tools:
                request["tools"] = active_tools
            try:
                response = client.messages.create(**request)
            except Exception:
                self._finish_thinking_run(request_id, "error")
                raise
            text = "".join(b.text for b in response.content
                           if getattr(b, "type", None) == "text").strip()
            if text:
                final_text = text
            tool_uses = [b for b in response.content
                         if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                exhausted = False
                break
            convo.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                if cancel_event.is_set():
                    canceled = True
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": "Thinking task canceled; tool not run."})
                    continue
                tool_calls += 1
                if tool_calls > MAX_TOOL_CALLS:
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": "Tool-call budget exhausted for this turn."})
                    continue
                out = self._run_thinking_tool(
                    tu.name, getattr(tu, "input", None) or {}, tu.id)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out})
            convo.append({"role": "user", "content": results})
        try:
            if canceled:
                return "I stopped that thinking task."
            if exhausted and not final_text:
                return "I reached the safe limit for this task before I could finish."
            return final_text
        finally:
            self._finish_thinking_run(request_id, "canceled" if canceled else "complete")

    def _run_thinking_tool(self, name, tool_input, tool_use_id=None):
        """Execute and journal a non-movement tool without logging its data.

        The event logger is the durable learning boundary. It receives the
        tool name, field names, and bounded outcome only, never note text,
        source content, passwords, or arbitrary command arguments. Reflection
        can therefore learn what this individual robot actually tends to do
        and which requests succeed without turning the event log into a secret
        transcript.
        """
        fields = sorted(str(key) for key in (tool_input or {})
                        if str(key) != "confirmed")[:20]
        self.bus.publish("picarx/decision", {
            "source": "companion", "kind": "thinking_tool",
            "choice": {"tool": str(name), "phase": "requested",
                        "fields": fields},
            "tool_use_id": str(tool_use_id or "")[:100] or None,
            "reason": "the thinking conversation selected a bounded non-movement tool",
            "ts": time.time(),
        })
        outcome = self._execute_tool(name, tool_input)
        outcome_text = str(outcome).lower()
        if "request failed" in outcome_text or "didn't work" in outcome_text:
            outcome_class = "failed"
        elif "still pending" in outcome_text:
            outcome_class = "pending"
        else:
            outcome_class = "completed"
        self.bus.publish("picarx/decision", {
            "source": "companion", "kind": "thinking_tool",
            "choice": {"tool": str(name), "phase": "completed",
                        "outcome": outcome_class,
                        "ok": outcome_class != "failed"},
            "tool_use_id": str(tool_use_id or "")[:100] or None,
            "reason": "thinking tool returned a bounded result",
            "ts": time.time(),
        })
        return outcome

    def _execute_tool(self, name, tool_input):
        """Run one tool call by publishing the matching mode/request topic.
        Returns a short result string fed back to the model. Never emits a
        motion primitive - follow motion is generated by follow_daemon."""
        if name in MOVEMENT_TOOL_NAMES:
            return ("Movement is not available to the thinking conversation. "
                    "The separate local safety command path handles it.")
        if name not in THINKING_TOOL_NAMES:
            return f"Unknown tool: {name}"
        try:
            if name == "describe_tools":
                rows = []
                for tool in THINKING_TOOLS:
                    description = " ".join(str(tool.get("description") or "").split())
                    rows.append(f"{tool['name']}: {description[:180]}")
                return ("Available non-movement tools: " + "; ".join(rows) +
                        ". Movement controls are intentionally excluded.")
            if name == "get_robot_status":
                with self.lock:
                    state = dict(getattr(self, "latest_robot_state", None) or {})
                    health = dict(getattr(self, "latest_health", None) or {})
                    world = dict(getattr(self, "latest_world", None) or {})
                    gesture = dict(getattr(self, "latest_gesture_status", None) or {})
                    follow = dict(getattr(self, "latest_follow_status", None) or {})
                    remote = dict(getattr(self, "latest_remote_result", None) or {})
                    radio = dict(getattr(self, "latest_radio_state", None) or {})
                parts = [f"mode {state.get('state') or 'unknown'}"]
                claims = state.get("claims")
                if isinstance(claims, list) and claims:
                    owners = [str(c.get("owner") or "unknown") for c in claims
                              if isinstance(c, dict)]
                    if owners:
                        parts.append("claims held by " + ", ".join(owners[:6]))
                parts.append(self._format_health(health))
                if world:
                    parts.append("perception: " + self._context_blurb()[:420])
                if gesture:
                    parts.append("gesture " + str(gesture.get("state") or "unknown") +
                                 (f" ({gesture.get('phase')})" if gesture.get("phase") else ""))
                if follow:
                    parts.append("follow " + str(follow.get("state") or
                                                  follow.get("enabled") or "unknown"))
                if radio:
                    parts.append("radio " + (f"playing {radio.get('station')}"
                                              if radio.get("playing") else "off"))
                if remote:
                    parts.append("remote session " + str(
                        remote.get("command") or remote.get("state") or "active"))
                plan = self._plan_manager().current()
                if plan and plan.get("status") in {"pending", "approved"}:
                    parts.append("plan " + str(plan.get("plan_id")) +
                                 " is " + str(plan.get("status")))
                lock, runs = self._thinking_registry()
                with lock:
                    if runs:
                        parts.append(f"thinking task active ({len(runs)})")
                return "Current robot status: " + "; ".join(parts) + "."
            if name == "propose_plan":
                plan = self._plan_manager().propose(
                    tool_input.get("goal"), tool_input.get("steps"))
                self._publish_plan_event(plan, "proposed",
                                         "thinking model proposed a bounded plan")
                return (f"Plan {plan['plan_id']} proposed and waiting for explicit "
                        "local approval; do not execute its destructive steps yet.")
            if name == "cancel_current_task":
                canceled = self._cancel_thinking_runs()
                return (f"Cancellation requested for {len(canceled)} thinking task(s)."
                        if canceled else "There is no active thinking task to cancel.")
            if name == "control_radio":
                command = str(tool_input.get("command") or "").lower()
                if command not in {"play", "stop", "next", "list", "find", "status"}:
                    return "Radio command must be play, stop, next, list, find, or status."
                request = {"command": command}
                if tool_input.get("station"):
                    request["station"] = str(tool_input["station"])[:160]
                if tool_input.get("query"):
                    request["keywords"] = str(tool_input["query"])[:160]
                self.bus.publish(RADIO_TOPIC, request)
                return f"Radio {command} request sent."
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
                result = self._dispatch_thinking_request(
                    REMINDER_SET_TOPIC, req, REMINDER_RESULT_TOPIC)
                return "Reminder scheduled; " + result
            if name == "manage_reminders":
                operation = str(tool_input.get("operation") or "").lower()
                if operation not in {"list", "delete"}:
                    return "I can list or delete pending reminders."
                if operation == "delete" and not bool(tool_input.get("confirmed")):
                    return ("I need explicit approval before deleting a reminder. "
                            "Tell me which one to remove and confirm it.")
                if operation == "delete":
                    blocked = self._approved_plan_required(
                        tool_input, "manage_reminders.delete")
                    if blocked:
                        return blocked
                request = {"command": operation}
                for key in ("id", "query", "confirmed"):
                    if tool_input.get(key) not in (None, ""):
                        request[key] = tool_input[key]
                result = self._dispatch_thinking_request(
                    REMINDER_CONTROL_TOPIC, request, REMINDER_RESULT_TOPIC)
                return f"Reminder {operation}; {result}"
            if name == "create_note":
                text = str(tool_input.get("text") or "").strip()
                if not text:
                    return "No note text was provided."
                request = {"command": "create", "text": text, "source": "voice"}
                if tool_input.get("title"):
                    request["title"] = str(tool_input["title"])[:120]
                result = self._dispatch_thinking_request(
                    NOTES_TOPIC, request, NOTES_RESULT_TOPIC)
                return "Note request sent; " + result
            if name == "manage_notes":
                operation = str(tool_input.get("operation") or "").lower()
                if operation not in {"list", "get", "search", "export", "delete"}:
                    return "That notes operation is not supported."
                if operation == "delete" and not bool(tool_input.get("confirmed")):
                    return ("I need explicit approval before deleting a note or meeting log.")
                if operation == "delete":
                    blocked = self._approved_plan_required(
                        tool_input, "manage_notes.delete")
                    if blocked:
                        return blocked
                request = {"command": "list" if operation == "search" else operation,
                           "source": "voice"}
                for key in ("id", "query", "confirmed"):
                    if tool_input.get(key) not in (None, ""):
                        request[key] = tool_input[key]
                result = self._dispatch_thinking_request(
                    NOTES_TOPIC, request, NOTES_RESULT_TOPIC)
                return f"Notes {operation}; {result}"
            if name == "control_meeting_notes":
                action = str(tool_input.get("action") or "").lower()
                if action not in {"start", "pause", "resume", "stop"}:
                    return "I can start, pause, resume, or stop meeting notes."
                if action == "start" and not bool(tool_input.get("confirmed")):
                    return "I need explicit consent before starting meeting notes."
                if action == "start":
                    blocked = self._approved_plan_required(
                        tool_input, "control_meeting_notes.start")
                    if blocked:
                        return blocked
                request = {"command": action, "source": "voice"}
                for key in ("title", "id", "confirmed"):
                    if tool_input.get(key) not in (None, ""):
                        request[key] = tool_input[key]
                result = self._dispatch_thinking_request(
                    NOTES_TOPIC, request, NOTES_RESULT_TOPIC)
                return f"Meeting notes {action}; {result}"
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
            if name == "connect_remote_host":
                host = str(tool_input.get("host") or "").strip()
                if not host:
                    return "I need the host IP address or hostname first."
                request = {"command": "connect", "host": host}
                for key in ("user", "project_root", "port"):
                    value = tool_input.get(key)
                    if value not in (None, ""):
                        request[key] = value
                result = self._dispatch_thinking_request(
                    REMOTE_ASSIST_TOPIC, request, REMOTE_RESULT_TOPIC)
                return ("Connecting to the host over verified SSH; " + result)
            if name == "remote_project_operation":
                operation = str(tool_input.get("operation") or "").lower()
                allowed = {"status", "list", "read", "search", "stat", "logs",
                           "write_file", "delete_path", "preview_patch", "apply_patch",
                           "rollback", "run",
                           "authorize_write", "revoke_write", "disconnect"}
                if operation not in allowed:
                    return "That remote operation is not supported."
                if operation in {"run", "write_file", "delete_path", "authorize_write"} and \
                        not bool(tool_input.get("confirmed")):
                    return ("I need your explicit approval before I can edit the "
                            "remote project or run a remote command.")
                if operation in {"run", "write_file", "delete_path", "authorize_write",
                                 "apply_patch", "rollback"}:
                    blocked = self._approved_plan_required(tool_input,
                                                           "remote." + operation)
                    if blocked:
                        return blocked
                request = {"command": operation}
                fields = ("path", "pattern", "patch", "content", "expected_sha256",
                          "command", "cwd", "confirmed", "plan_id")
                for key in fields:
                    value = tool_input.get(key)
                    if value not in (None, ""):
                        if isinstance(value, str):
                            value = value[:20000 if key in {"patch", "content"} else 1000]
                        # `command` names the remote operation on the bus;
                        # the helper's run argument therefore travels as
                        # argv to avoid overwriting that operation name.
                        request["argv" if operation == "run" and key == "command" else key] = value
                result = self._dispatch_thinking_request(
                    REMOTE_ASSIST_TOPIC, request, REMOTE_RESULT_TOPIC)
                return f"Remote {operation}; {result}"
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
        # Reminder status is owned by reminder_daemon, but its published cache
        # makes the natural follow-up deterministic and avoids a needless or
        # hallucinated Claude answer after an asynchronous tool call.
        if self._maybe_answer_reminder(text):
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
            reply = self._chat_with_tools(client, messages,
                                          tools=tools_for_utterance(text))
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
        self.bus.subscribe(LEARNED_INTENTS_CONTROL_TOPIC,
                           self.on_learned_intent_control)
        self.bus.subscribe(DIALOG_ANSWER_TOPIC, self.on_dialog_answer)
        self.bus.subscribe(DIALOG_CLEARED_TOPIC, self.on_dialog_cleared)
        self.bus.subscribe(FEEDBACK_TOPIC, self.on_feedback)
        self.bus.subscribe(CAMERA_FRAME_TOPIC, self.on_frame)
        self.bus.subscribe("picarx/state/world", self.on_world_state)
        self.bus.subscribe(ROBOT_STATE_TOPIC, self.on_robot_state)
        self.bus.subscribe(GESTURE_STATUS_TOPIC, self.on_gesture_status)
        self.bus.subscribe(FOLLOW_STATUS_TOPIC, self.on_follow_status)
        self.bus.subscribe(REMOTE_RESULT_TOPIC, self.on_remote_result)
        self.bus.subscribe("picarx/tools/radio_state", self.on_radio_state)
        self.bus.subscribe(HEALTH_STATE_TOPIC, self.on_health)
        self.bus.subscribe(REMINDER_STATE_TOPIC, self.on_reminder_state)
        self.bus.subscribe(REMINDER_RESULT_TOPIC, self.on_reminder_result)
        self.bus.subscribe(NOTES_RESULT_TOPIC, self.on_notes_result)
        self.bus.subscribe(THINKING_CONTROL_TOPIC, self.on_thinking_control)
        self.bus.subscribe(PERCEPTION_IDENTIFY_TOPIC, self.on_identify)
        self.bus.subscribe(IMU_EVENT_TOPIC, self.on_imu_event)
        self.bus.subscribe(SELF_TRAINER_STATUS_TOPIC, self.on_self_trainer_status)

        for _ in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, daemon=True).start()

        print("Companion active, listening on picarx/audio/unhandled")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Companion().run()
