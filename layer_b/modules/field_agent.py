#!/usr/bin/env python3
# /home/picarx/layer_b/modules/field_agent.py
"""
Field Agent (Layer B) - integration test harness / onboard brain.

This is the first module that actually exercises the whole Layer B
pipeline end to end instead of bypassing it:

  - Movement requests go out as INTENTS on picarx/intent/move, picked
    up by arbiter.py, which is the only thing that talks to the
    safety daemon. This module never touches the safety socket for
    movement.
  - World knowledge comes from picarx/state/world, published by
    world_state.py (including tracked/labeled objects from
    vision_basic.py's object detector). This module does not
    re-derive it from raw sensors.
  - "History" answers come from reading event_logger.py's SQLite
    database directly (read-only queries only - this module never
    writes to that DB, event_logger.py is the sole writer).
  - Speech in and out rides the existing picarx/audio/heard and
    picarx/audio/speak topics, same as your original modules.
  - Novel objects and repeated-collision fail states are referred to
    coach.py over picarx/coach/query - see "Coach integration" below.

REQUIRES ALL OF THE FOLLOWING RUNNING FIRST:
  broker_client.py (fixed version, supports multi-topic subscribe)
  safety_daemon.py
  audio_nodes.py       (for STT input + TTS output)
  distance_sensor.py
  vision_basic.py      (required - face + labeled object tracking)
  arbiter.py           (required for exploration to actually move)
  world_state.py       (required for "what do you see" / exploration)
  event_logger.py      (required for "what have you done" / history)
  coach.py             (optional - novelty/fail-state advice; field
                        agent degrades gracefully to its own canned
                        evasion behavior if it's not running or a
                        query times out)

Voice commands understood (see handle_voice_command):
  "explore" / "start"          -> begin autonomous wandering
  "stop" / "halt"               -> stop moving, cancel any intent
  "status" / "what do you see"  -> report current world state aloud
  "objects" / "what's around"   -> list currently tracked objects
  "history" / "what have you done" / "what happened"
                                -> summarize event log aloud
  "battery" / "charge" / "level" -> report battery voltage
  "hello" / "hi"                 -> greet

Anything that doesn't match one of the above only gets forwarded for
free-form conversation (via picarx/audio/unhandled, to companion.py,
if running) when it starts with a wake phrase - see WAKE_PHRASES
below, e.g. "hey pi, what do you think of the weather". Without a
wake phrase it's just dropped (and printed to stdout), so STT
mis-hearing background noise doesn't burn an API call on every false
transcription. Hard commands above are unaffected either way.

You can also just watch stdout - every decision this module makes is
printed, not just spoken, so you can test without a working mic/speaker.

---------------------------------------------------------------------
Coach integration (novel situations + fail states)
---------------------------------------------------------------------
Two separate, independent triggers talk to coach.py over the bus:

  1. Novel object ("watch" query, non-urgent): the first time
     world_state reports an object label this module has never seen
     before, it announces it and fires a non-blocking
     picarx/coach/query. Exploration keeps running unaffected while
     waiting; if/when picarx/coach/suggestion arrives with a matching
     query_id, its action is applied for a bounded duration as a
     one-off, then picarx/coach/outcome reports whether that window
     stayed collision-free.

  2. Repeated collision ("urgent" query, blocking): if the safety
     daemon vetoes this module's own move intents VETO_FAIL_THRESHOLD
     times within VETO_WINDOW seconds (the actual "keeps running into
     an object it doesn't sense" failure mode - the vision/ultrasonic
     obstacle checks below already try to prevent this, but this is
     the backstop for whatever they miss), or the normal evasion
     state machine has had to trigger EVASION_FAIL_THRESHOLD times
     within EVASION_FAIL_WINDOW (a "stuck bouncing off the same
     thing" pattern), this module treats it as a fail state: it holds
     a safe reflex (stop, then a slow back-off) and fires a blocking
     picarx/coach/query with a bounded timeout. If a suggestion
     arrives in time, it's tried; whether or not it works (or the
     query times out with no answer at all - no network, no API key,
     coach.py not running), the canned evasion behavior is always the
     fallback, so a working coach is never a safety dependency.

Everything above involving actual motion is only ever *applied* while
explore_mode is on; novelty detection/announcement itself runs
continuously (see _perception_tick / run) since noticing things
doesn't require driving anywhere.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import sqlite3
import json
import re
import time
import random
import threading
import uuid
from collections import deque

SOURCE_NAME = "field_agent"

# Must match event_logger.py's DB_PATH - this module only ever opens
# it read-only and never writes.
DB_PATH = "/home/picarx/layer_b/data/events.db"

EXPLORE_PRIORITY = 5
EXPLORE_TICK_HZ = 5
INTENT_TTL = 0.6       # must be > 1/EXPLORE_TICK_HZ so intents don't gap out

# STT will constantly transcribe background noise/TV/conversation into
# text; without a wake phrase, every one of those would silently burn
# an Anthropic API call in companion.py. Only utterances that start
# with one of these get forwarded to the LLM chat fallback at all -
# hard commands (explore/stop/etc.) are unaffected and still work
# with or without a wake phrase, since they're matched earlier above.
WAKE_PHRASES = ("robot", "hey robot", "computer")

OBSTACLE_DISTANCE_CM = 20  # Adjusted slightly downward to prevent premature triggers
MIN_ANNOUNCEMENT_GAP = 6.0  # don't let spontaneous remarks spam the speaker

# Vision close_object cross-check: if the ultrasonic has a FRESH
# reading that says the path ahead is clearly open, a frame-filling
# SSD detection is a wall/sofa across the room (or floor texture), not
# a point-blank obstacle - field data showed vision evasions firing
# with 60-300cm of measured clear air. The vision signal exists to
# cover the ultrasonic's close-range dead zone; that dead zone only
# produces short or missing readings, never confidently-long ones, so
# trusting a long fresh reading over vision here doesn't reopen it.
VISION_OBSTACLE_ULTRASONIC_CLEAR_CM = 60

# Look-around head scan performed when exploration starts: sweep the
# camera across these pan angles (degrees, negative = left), dwelling
# at each long enough for the SSD to get a detection pass in
# (OBJECT_DETECT_INTERVAL is 1.5s), recording what's visible where.
# The result is announced, published on picarx/exploration/room_scan,
# and logged to events.db - the robot's first durable record of room
# layout. (An ackermann-steered car can't spin in place, so a head
# sweep is the practical version of "turn a circle and take in the
# room".)
SCAN_PAN_ANGLES = (-70, -35, 0, 35, 70)
SCAN_DWELL_SEC = 1.8
SCAN_TILT = 0

# Physical stuck detection: commanding forward with nothing being
# vetoed, but the camera scene isn't changing -> wheels are pushing
# against something below the ultrasonic beam (or slipping). vision's
# scene_motion (mean abs thumbnail diff, ~6.0+ while actually moving)
# stays near zero when the view is frozen.
STUCK_AFTER_SEC = 4.0          # this long of continuous forward with a static view -> stuck
STUCK_MOTION_THRESHOLD = 3.0   # scene_motion below this counts as "static"

# Evasion/coaching priorities - both outrank normal exploring (5), and
# COACH_PRIORITY outranks the canned evasion sequence (8) too, since a
# coach-directed maneuver during a fail state should win over whatever
# the plain reflex would have done.
EVADE_PRIORITY = 8
COACH_PRIORITY = 9
WATCH_PRIORITY = 6

# A depth-sensor-free obstacle signal: world_state flags a tracked
# object "approaching" if its bounding box is growing quickly while
# centered in frame. Treated exactly like a close ultrasonic reading.
#
# close_object is checked first and takes priority: it's class-agnostic
# (doesn't require the SSD to confidently recognize what the object is -
# see vision_basic.py), so it catches point-blank obstacles like a
# cabinet that "approaching" never can, since that path only fires for
# objects the SSD actually tracks by label in the first place.
def _vision_obstacle(snapshot):
    if not snapshot:
        return None
    objects = snapshot.get("objects") or {}
    if objects.get("stale", True):
        return None
    if objects.get("close_object"):
        return {"label": "something", "area_ratio": 1.0}
    best = None
    for obj in objects.get("items", []):
        if obj.get("approaching") and (best is None or obj.get("area_ratio", 0) > best.get("area_ratio", 0)):
            best = obj
    return best


def _prune_older_than(dq, window, now):
    while dq and dq[0] < now - window:
        dq.popleft()


# ---- fail-state escalation tuning ----
VETO_WINDOW = 4.0             # seconds
VETO_FAIL_THRESHOLD = 3       # this many of OUR OWN vetoed intents in the window -> fail state
EVASION_FAIL_WINDOW = 20.0    # seconds
EVASION_FAIL_THRESHOLD = 3    # this many evasion triggers in the window -> fail state (stuck bouncing)

COACH_URGENT_TIMEOUT = 3.0    # how long we'll hold a safe reflex waiting for advice
COACH_WATCH_TIMEOUT = 5.0     # how long we'll wait on a non-blocking novelty query
DEFAULT_COACH_DURATION = 1.5  # how long to follow a suggested action if it doesn't specify one

# If every direction is blocked (e.g. boxed in between an obstacle and
# a cliff/wall behind), no coach suggestion can possibly succeed - a
# minimum cooldown between failed fail-state episodes, and a hard stop
# after enough consecutive failures, keep that from turning into a
# rapid-fire query storm (each one a real API call) that never lets a
# single attempt actually run its course.
# Raised 3.0 -> 8.0 and now applied after EVERY episode (success too,
# not just failure): field data showed 68 fail-state entries in 7.5
# minutes - episodes kept "succeeding" (their action window happened to
# stay veto-free), resetting the failure counter, then instantly
# re-triggering. Each entry is an announcement plus a potential paid
# LLM call, so back-to-back re-entry has to be rate-limited regardless
# of episode outcome.
FAIL_STATE_COOLDOWN = 8.0        # min seconds between fail-state episodes
MAX_CONSECUTIVE_FAILURES = 3     # give up and wait for a human after this many straight failures


class FieldAgent:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()

        self.explore_mode = False
        self.latest_world = None
        self.face_was_detected = False
        self.known_object_labels = set()

        self.last_announcement_at = 0.0
        self.start_time = time.time()

        # State machine for non-blocking obstacle evasion / coaching.
        # These fields are only ever mutated from explore_tick's
        # thread (the run() loop) - bus callbacks only ever feed the
        # lock-protected inboxes below, never touch state directly,
        # so there's no cross-thread race on the state machine itself.
        self.state = "CRUISING"  # "CRUISING", "SCANNING", "EVADING", "COACHING"
        self.evade_stage = 0     # 0: stop, 1: reverse, 2: turn
        self.state_until = 0.0
        self.evasion_fail_events = deque()
        self.next_fail_state_allowed_at = 0.0
        self.consecutive_coach_failures = 0
        self.given_up = False    # true once MAX_CONSECUTIVE_FAILURES is hit; "explore" clears it

        # Look-around head scan state (see SCAN_PAN_ANGLES).
        self.scan_index = 0
        self.scan_dwell_until = 0.0
        self.scan_sightings = []       # [{"pan": angle, "labels": [...]}, ...]
        self.last_room_scan = None     # kept for reports; also published + logged

        # Physical stuck detection state (see STUCK_AFTER_SEC).
        self.forward_since = None      # when the current uninterrupted forward run began

        # Wander state (mirrors the old reflex explorer's behavior,
        # now expressed as intents instead of direct socket calls)
        self.last_wander = time.time()
        self.wander_interval = random.uniform(5.0, 10.0)
        self.steering_active_until = 0

        # Cross-thread inboxes (bus callbacks append, explore_tick/
        # _perception_tick drain under self.lock).
        self.veto_events = deque()
        self.pending_novel_objects = deque()
        self.pending_suggestions = deque()

        # Urgent (blocking, fail-state) coach query bookkeeping.
        self.coach_query_id = None
        self.coach_query_deadline = 0.0
        self.coach_action = None
        self.coach_action_started_at = 0.0
        self.coach_apply_until = 0.0
        self.last_coach_query_id_used = None
        self.last_coach_situation_key = None

        # Non-blocking (novelty) coach query bookkeeping.
        self.watch_query_id = None
        self.watch_query_deadline = 0.0
        self.watch_coach_action = None
        self.watch_action_started_at = 0.0
        self.watch_action_until = 0.0
        self.watch_query_id_used = None
        self.watch_situation_key = None

    # ---------- outbound: intents ----------

    def publish_intent(self, action, priority=EXPLORE_PRIORITY, ttl=INTENT_TTL):
        self.bus.publish("picarx/intent/move", {
            "source": SOURCE_NAME,
            "priority": priority,
            "action": action,
            "ttl": ttl,
        })

    def cancel_intent(self):
        self.bus.publish("picarx/intent/cancel", {"source": SOURCE_NAME})

    def publish_look(self, pan, tilt=SCAN_TILT):
        # Camera head only - rides its own topic, outside the arbiter's
        # single-winner movement channel (see arbiter.on_look).
        self.bus.publish("picarx/intent/look", {
            "source": SOURCE_NAME,
            "action": {"direction": "look", "pan": pan, "tilt": tilt},
        })

    # ---------- outbound: speech ----------

    def announce(self, text, force=False):
        now = time.time()
        if not force and (now - self.last_announcement_at) < MIN_ANNOUNCEMENT_GAP:
            print(f"(suppressed, too soon after last remark): {text}")
            return
        self.last_announcement_at = now
        print(f"Field Agent says: {text}")
        self.bus.publish("picarx/audio/speak", {"text": text})

    # ---------- inbound: world state ----------

    def on_world_state(self, payload):
        novel = []
        with self.lock:
            self.latest_world = payload
            face = payload.get("face", {})
            detected = bool(face.get("detected")) and not face.get("stale", True)
            face_became_detected = detected and not self.face_was_detected
            self.face_was_detected = detected

            objects = payload.get("objects", {})
            if not objects.get("stale", True):
                for obj in objects.get("items", []):
                    label = obj.get("label")
                    if label and label not in self.known_object_labels:
                        self.known_object_labels.add(label)
                        novel.append(obj)

        if face_became_detected:
            self.announce("I see a face.")
        if novel:
            with self.lock:
                self.pending_novel_objects.extend(novel)

    def _snapshot(self):
        with self.lock:
            return dict(self.latest_world) if self.latest_world else None

    # ---------- inbound: safety-daemon outcomes ----------

    def on_action_result(self, payload):
        if payload.get("source") != SOURCE_NAME:
            return
        result = payload.get("result") or {}
        if result.get("status") != "vetoed":
            return
        with self.lock:
            self.veto_events.append(time.time())

    # ---------- inbound: coach ----------

    def on_coach_suggestion(self, payload):
        with self.lock:
            self.pending_suggestions.append(payload)

    # ---------- inbound: voice ----------

    def on_heard(self, payload):
        text = payload.get("text", "").lower().strip()
        if not text:
            return
        print(f"Heard: '{text}'")
        self.handle_voice_command(text)

    def handle_voice_command(self, text):
        if "explore" in text or "start" in text:
            if not self.explore_mode:
                self.explore_mode = True
                self.given_up = False
                self.consecutive_coach_failures = 0
                self.next_fail_state_allowed_at = 0.0
                self.forward_since = None
                # Look around before rolling: sweep the camera across
                # the room and take stock of what's where first.
                self.state = "SCANNING"
                self.scan_index = 0
                self.scan_dwell_until = time.time() + SCAN_DWELL_SEC
                self.scan_sightings = []
                self.publish_look(SCAN_PAN_ANGLES[0])
                self.announce("Starting exploration. Let me take a look around first.", force=True)
            return

        if "stop" in text or "halt" in text:
            if self.explore_mode:
                self.explore_mode = False
                self.cancel_intent()
                self.publish_look(0, 0)  # recenter the head wherever the scan/drive left it
                self.announce("Stopping.", force=True)
            return

        if "battery" in text or "charge" in text or "level" in text:
            self.report_battery()
            return

        if "history" in text or "what have you done" in text or "what happened" in text:
            self.report_history()
            return

        if "object" in text or "what's around" in text or "whats around" in text or "what do you notice" in text:
            self.report_objects()
            return

        if "what do you see" in text or "status" in text or "report" in text:
            self.report_status()
            return

        if "hello" in text or re.search(r"\bhi\b", text):
            self.announce("Hello! I am ready to chat and explore.", force=True)
            return

        # Nothing above matched a hard command. Only forward to the LLM
        # chat fallback if a wake phrase was used (see WAKE_PHRASES) -
        # anything else is dropped, but still printed so you can see
        # what got heard and tune the wake phrases if something real
        # is being missed.
        remainder = self._strip_wake_phrase(text)
        if remainder is not None:
            self.bus.publish("picarx/audio/unhandled", {"text": remainder})
        else:
            print(f"(no wake phrase, not forwarding to chat): '{text}'")

    @staticmethod
    def _strip_wake_phrase(text):
        for phrase in WAKE_PHRASES:
            if text.startswith(phrase):
                remainder = text[len(phrase):].strip(" ,.:;-")
                return remainder if remainder else "hello"
        return None

    # ---------- spoken reports ----------

    def report_battery(self):
        snap = self._snapshot()
        if not snap or snap.get("battery", {}).get("voltage") is None:
            self.announce("I don't have a battery reading yet.", force=True)
            return
        battery = snap["battery"]
        stale_note = " though that reading is a bit old" if battery.get("stale") else ""
        self.announce(f"My battery is at {battery['voltage']:.1f} volts{stale_note}.", force=True)

    def report_status(self):
        snap = self._snapshot()
        if not snap:
            self.announce("I don't have a world state yet.", force=True)
            return

        parts = []

        face = snap.get("face", {})
        if face.get("detected") and not face.get("stale", True):
            parts.append("I see a face in front of me")
        else:
            parts.append("I don't currently see a face")

        distance = snap.get("distance_cm")
        if distance is not None and not snap.get("distance_stale", True):
            parts.append(f"the nearest obstacle is about {distance:.0f} centimeters away")
        else:
            parts.append("I don't have a fresh distance reading")

        objects = snap.get("objects", {})
        if not objects.get("stale", True) and objects.get("items"):
            labels = [o.get("label", "something") for o in objects["items"]]
            parts.append(f"I'm tracking {len(labels)} object{'s' if len(labels) != 1 else ''}: {', '.join(labels)}")

        battery = snap.get("battery", {})
        if battery.get("voltage") is not None:
            parts.append(f"my battery is at {battery['voltage']:.1f} volts")

        self.announce(". ".join(parts) + ".", force=True)

    def report_objects(self):
        snap = self._snapshot()
        objects = snap.get("objects", {}) if snap else {}
        if objects.get("stale", True) or not objects.get("items"):
            self.announce("I don't see anything I can identify right now.", force=True)
            return
        items = objects["items"]
        descriptions = []
        for obj in items:
            label = obj.get("label", "something")
            age = time.time() - obj.get("first_seen", time.time())
            if age < 3.0:
                descriptions.append(f"a {label} I just noticed")
            else:
                descriptions.append(f"a {label} I've been tracking for a bit")
        self.announce(f"I currently see {len(items)}: " + ", ".join(descriptions) + ".", force=True)

    def report_history(self):
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        except Exception as e:
            self.announce("I can't reach my event log right now.", force=True)
            print(f"History query failed to open DB: {e}")
            return

        try:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

            action_rows = conn.execute(
                "SELECT payload_json FROM events WHERE topic = ? ORDER BY id DESC LIMIT 200",
                ("picarx/action/result",),
            ).fetchall()

            vetoed = 0
            for (payload_json,) in action_rows:
                try:
                    payload = json.loads(payload_json)
                    if payload.get("result", {}).get("status") == "vetoed":
                        vetoed += 1
                except (json.JSONDecodeError, AttributeError):
                    continue

            heard_rows = conn.execute(
                "SELECT payload_json FROM events WHERE topic = ? ORDER BY id DESC LIMIT 3",
                ("picarx/audio/heard",),
            ).fetchall()
            recent_phrases = []
            for (payload_json,) in heard_rows:
                try:
                    payload = json.loads(payload_json)
                    if payload.get("text"):
                        recent_phrases.append(payload["text"])
                except (json.JSONDecodeError, AttributeError):
                    continue

            oldest_ts_row = conn.execute("SELECT MIN(ts) FROM events").fetchone()
            oldest_ts = oldest_ts_row[0] if oldest_ts_row else None
        finally:
            conn.close()

        if total == 0:
            self.announce("I don't have any history recorded yet.", force=True)
            return

        parts = [f"I have {total} recorded events"]
        if oldest_ts:
            minutes = (time.time() - oldest_ts) / 60.0
            parts.append(f"going back about {minutes:.0f} minutes")
        parts.append(f"and I've been stopped by obstacles {vetoed} times recently")
        if recent_phrases:
            parts.append(f"the last thing I heard was: {recent_phrases[0]}")

        self.announce(". ".join(parts) + ".", force=True)

    # ---------- coach integration ----------

    def _start_coach_query(self, situation, urgent, label=None, extra=None):
        query_id = str(uuid.uuid4())
        snap = self._snapshot() or {}
        payload = {
            "query_id": query_id,
            "source": SOURCE_NAME,
            "situation": situation,
            "label": label,
            "urgent": urgent,
            "requested_at": time.time(),
            "context": {
                "distance_cm": snap.get("distance_cm"),
                "distance_stale": snap.get("distance_stale", True),
                "objects": snap.get("objects", {}).get("items", []),
                "last_action": snap.get("last_action"),
                "battery": snap.get("battery"),
            },
            "extra": extra or {},
        }
        now = time.time()
        if urgent:
            self.coach_query_id = query_id
            self.coach_query_deadline = now + COACH_URGENT_TIMEOUT
        else:
            self.watch_query_id = query_id
            self.watch_query_deadline = now + COACH_WATCH_TIMEOUT
        print(f"Field Agent querying coach: situation={situation} label={label} urgent={urgent}")
        self.bus.publish("picarx/coach/query", payload)

    def _on_novel_object(self, obj):
        label = obj.get("label", "something")
        self.announce(f"I see something new - looks like a {label}.")
        if self.watch_query_id is not None:
            return  # already waiting on a novelty query, don't pile on
        self._start_coach_query(situation="novel_object", urgent=False, label=label, extra={"object": obj})

    def _apply_coach_suggestion(self, payload):
        query_id = payload.get("query_id")
        if not query_id:
            return
        action = payload.get("action")
        duration = payload.get("duration") or DEFAULT_COACH_DURATION
        rationale = payload.get("rationale")
        cached = payload.get("cached")
        now = time.time()

        if query_id == self.coach_query_id:
            self.coach_action = action
            self.coach_action_started_at = now
            self.coach_apply_until = now + duration
            self.last_coach_query_id_used = query_id
            self.last_coach_situation_key = payload.get("situation_key")
            self.coach_query_id = None
            print(f"Coach suggestion (urgent, cached={cached}): {action} - {rationale}")
            self.announce("Trying something the coach suggested.")
        elif query_id == self.watch_query_id:
            self.watch_coach_action = action
            self.watch_action_started_at = now
            self.watch_action_until = now + duration
            self.watch_query_id_used = query_id
            self.watch_situation_key = payload.get("situation_key")
            self.watch_query_id = None
            print(f"Coach suggestion (watch, cached={cached}): {action} - {rationale}")
        # else: stale/unknown query id (already timed out or superseded) - ignore

    def _finish_coach_episode(self, now):
        with self.lock:
            succeeded = not any(t > self.coach_action_started_at for t in self.veto_events)
        query_id = self.last_coach_query_id_used
        situation_key = self.last_coach_situation_key
        self.coach_action = None
        self.state = "CRUISING"
        self.last_wander = now
        self.bus.publish("picarx/coach/outcome", {
            "query_id": query_id,
            "situation_key": situation_key,
            "source": SOURCE_NAME,
            "success": succeeded,
        })
        if succeeded:
            self.consecutive_coach_failures = 0
            # Success still starts the cooldown: an instantly
            # re-triggering "success" (the suggested action's window
            # happened to stay veto-free without actually freeing the
            # robot) was re-entering a fresh episode within the same
            # second - see FAIL_STATE_COOLDOWN comment.
            self.next_fail_state_allowed_at = now + FAIL_STATE_COOLDOWN
            self.announce("That worked.")
            return

        self.consecutive_coach_failures += 1
        self.next_fail_state_allowed_at = now + FAIL_STATE_COOLDOWN
        if self.consecutive_coach_failures >= MAX_CONSECUTIVE_FAILURES:
            # Nothing suggested has worked several times in a row -
            # very likely every direction is genuinely blocked (boxed
            # in between an obstacle and a cliff/wall behind, say).
            # No further suggestion can fix that; stop actually trying
            # and say so, instead of burning more queries on repeats.
            self.given_up = True
            self.announce("I'm stuck and nothing is working. Please help me, or tell me to explore again.", force=True)
            self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
            return

        self.announce("Still stuck, backing off the normal way.")
        self.state = "EVADING"
        self.evade_stage = 0
        self.state_until = now + 0.25
        self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)

    def _finish_watch_episode(self):
        with self.lock:
            succeeded = not any(t > self.watch_action_started_at for t in self.veto_events)
        self.bus.publish("picarx/coach/outcome", {
            "query_id": self.watch_query_id_used,
            "situation_key": self.watch_situation_key,
            "source": SOURCE_NAME,
            "success": succeeded,
        })
        self.watch_coach_action = None

    def _handle_coaching_tick(self, now):
        if self.coach_action is None:
            # Still waiting on a reply - hold a safe reflex rather than
            # doing nothing (or worse, drifting forward) while we wait.
            if now > self.coach_query_deadline:
                self.announce("No answer from the coach, handling it myself.")
                self.coach_query_id = None
                self.state = "EVADING"
                self.evade_stage = 0
                self.state_until = now + 0.25
                self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
                return
            if now < self.coach_action_started_at + 0.3:
                self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
            else:
                self.publish_intent({"direction": "backward", "speed": 25}, priority=COACH_PRIORITY)
            return

        if now < self.coach_apply_until:
            self.publish_intent(self.coach_action, priority=COACH_PRIORITY, ttl=0.6)
            return

        self._finish_coach_episode(now)

    def _enter_collision_fail_state(self, reason):
        now = time.time()

        if self.state == "COACHING":
            # Already mid-episode - a fresh trigger here (e.g. the
            # coach's own suggested action itself getting vetoed) used
            # to abandon the in-progress episode and fire a brand new
            # query immediately, which is how this turned into a
            # same-second query storm. Let the current episode run its
            # course and reach its own pass/fail conclusion instead.
            return

        if self.given_up:
            # Already told the user we're stuck; hold still instead of
            # silently resuming the retry loop. "explore" clears this.
            self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
            return

        if now < self.next_fail_state_allowed_at:
            # Cooling down after a recent failed attempt - a fresh
            # trigger during this window (still stuck) just holds a
            # plain stop rather than hammering the coach again.
            self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
            return

        self.evasion_fail_events.append(now)
        _prune_older_than(self.evasion_fail_events, EVASION_FAIL_WINDOW, now)
        stuck_pattern = len(self.evasion_fail_events) >= EVASION_FAIL_THRESHOLD
        with self.lock:
            self.veto_events.clear()
        self.forward_since = None
        self.state = "COACHING"
        self.coach_action = None
        self.coach_apply_until = 0.0
        self.coach_action_started_at = now
        self.announce("I keep running into something. Let me get some advice.", force=True)
        self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
        self._start_coach_query(
            situation="collision_loop", urgent=True,
            extra={"reason": reason, "stuck_pattern": stuck_pattern},
        )

    def _begin_evasion(self, reason):
        now = time.time()
        self.forward_since = None
        self.evasion_fail_events.append(now)
        _prune_older_than(self.evasion_fail_events, EVASION_FAIL_WINDOW, now)
        if len(self.evasion_fail_events) >= EVASION_FAIL_THRESHOLD:
            self._enter_collision_fail_state(f"evasion_loop:{reason}")
            return
        self.state = "EVADING"
        self.evade_stage = 0
        self.state_until = now + 0.25
        self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)

    # ---------- perception (always runs, independent of explore_mode) ----------

    def _perception_tick(self):
        with self.lock:
            novel = list(self.pending_novel_objects)
            self.pending_novel_objects.clear()
            suggestions = list(self.pending_suggestions)
            self.pending_suggestions.clear()

        for obj in novel:
            self._on_novel_object(obj)
        for payload in suggestions:
            self._apply_coach_suggestion(payload)

    # ---------- look-around head scan ----------

    def _handle_scanning_tick(self, now):
        # Stationary the whole time - hold an explicit stop so the
        # arbiter doesn't fall through to some stale lower-priority
        # intent while our head is turned.
        self.publish_intent({"direction": "stop"})
        if now < self.scan_dwell_until:
            return

        # Dwell at this angle is over - record what's visible here.
        snap = self._snapshot() or {}
        objects = snap.get("objects", {})
        labels = sorted({
            o.get("label") for o in objects.get("items", [])
            if o.get("label") and not objects.get("stale", True)
        })
        self.scan_sightings.append({"pan": SCAN_PAN_ANGLES[self.scan_index], "labels": labels})

        self.scan_index += 1
        if self.scan_index < len(SCAN_PAN_ANGLES):
            self.publish_look(SCAN_PAN_ANGLES[self.scan_index])
            self.scan_dwell_until = now + SCAN_DWELL_SEC
            return

        # Sweep complete - recenter, remember, publish, announce, roll.
        self.publish_look(0, 0)
        self.last_room_scan = {"scanned_at": now, "sightings": self.scan_sightings}
        self.bus.publish("picarx/exploration/room_scan", self.last_room_scan)

        seen = sorted({label for s in self.scan_sightings for label in s["labels"]})
        if seen:
            self.announce(f"I looked around and I can see: {', '.join(seen)}. Off I go.", force=True)
        else:
            self.announce("I looked around but didn't recognize anything. Exploring anyway.", force=True)
        self.state = "CRUISING"
        self.last_wander = now

    # ---------- exploration behavior ----------

    def explore_tick(self):
        now = time.time()

        with self.lock:
            _prune_older_than(self.veto_events, VETO_WINDOW, now)
            veto_count = len(self.veto_events)

        if self.state == "SCANNING":
            self._handle_scanning_tick(now)
            return

        if self.state == "COACHING":
            self._handle_coaching_tick(now)
            return

        if self.watch_coach_action is not None:
            if now < self.watch_action_until:
                self.publish_intent(self.watch_coach_action, priority=WATCH_PRIORITY, ttl=0.6)
                return
            self._finish_watch_episode()

        if veto_count >= VETO_FAIL_THRESHOLD:
            self._enter_collision_fail_state("repeated_veto")
            return

        snap = self._snapshot()
        distance = snap.get("distance_cm") if snap else None
        distance_stale = snap.get("distance_stale", True) if snap else True
        vision_obstacle = _vision_obstacle(snap)

        # --- Handle Evasion State Machine ---
        if self.state == "EVADING":
            if now < self.state_until:
                # Continue executing current stage behavior
                if self.evade_stage == 0:
                    self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 1:
                    self.publish_intent({"direction": "backward", "speed": 30}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 2:
                    # Maintain the turn angle chosen when entering this stage
                    pass
                return
            else:
                # Progress to next step of evasion
                self.evade_stage += 1
                if self.evade_stage == 1:
                    # Move backward for 1.2 seconds
                    self.state_until = now + 1.2
                    self.publish_intent({"direction": "backward", "speed": 30}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 2:
                    # Choose random direction to pivot away for 0.6 seconds
                    angle = random.choice([-30, 30])
                    self.state_until = now + 0.6
                    self.publish_intent({"direction": "turn", "angle": angle}, priority=EVADE_PRIORITY)
                else:
                    # Evasion complete, clean slate
                    self.publish_intent({"direction": "turn", "angle": 0}, priority=EVADE_PRIORITY)
                    self.state = "CRUISING"
                    self.last_wander = now
                return

        # --- Handle a vision-flagged approaching object (covers the
        # ultrasonic's blind spots - this is the fix for driving
        # straight into things the distance sensor never saw) ---
        # Cross-checked against the ultrasonic: a fresh, clearly-long
        # distance reading means the frame-filling detection is the
        # room itself (wall/sofa/floor), not a point-blank obstacle -
        # see VISION_OBSTACLE_ULTRASONIC_CLEAR_CM.
        ultrasonic_says_clear = (
            distance is not None and not distance_stale
            and distance > VISION_OBSTACLE_ULTRASONIC_CLEAR_CM
        )
        if vision_obstacle is not None and not ultrasonic_says_clear:
            label = vision_obstacle.get("label", "something")
            if label == "something":
                self.announce("Something's right in front of me, backing away.")
            else:
                self.announce(f"A {label} is closing in, backing away.")
            self._begin_evasion("vision")
            return

        # --- Handle Trustworthiness of Sensor Data ---
        if distance is None or distance_stale or distance < 0:
            # Fallback cautious crawl if we are blind
            if self._note_forward_and_check_stuck(now, snap):
                return
            self.publish_intent({"direction": "forward", "speed": 15})
            return

        # --- Handle New Obstacle Detection ---
        if distance < OBSTACLE_DISTANCE_CM:
            self.announce("Obstacle ahead, backing away.")
            self._begin_evasion("ultrasonic")
            return

        # --- Handle Timed Steering Reset during standard wander ---
        if self.steering_active_until != 0 and now >= self.steering_active_until:
            self.publish_intent({"direction": "turn", "angle": 0})
            self.steering_active_until = 0
            return

        # --- Handle Periodic Spontaneous Wandering ---
        if now - self.last_wander > self.wander_interval:
            angle = random.randint(-25, 25)
            print(f"Wandering with angle: {angle}")
            self.publish_intent({"direction": "turn", "angle": angle})
            self.steering_active_until = now + 1.5
            self.wander_interval = random.uniform(5.0, 15.0)
            self.last_wander = now
            return

        # --- Standard Base Case ---
        if self._note_forward_and_check_stuck(now, snap):
            return
        self.publish_intent({"direction": "forward", "speed": 25})

    def _note_forward_and_check_stuck(self, now, snap):
        """Physical stuck detection: continuously commanding forward
        while the camera view stays static means the wheels are pushing
        against something below the ultrasonic beam (or spinning in
        place). Returns True if it triggered an evasion (tick consumed).
        Any visibly-moving scene sample restarts the window, so firing
        requires STUCK_AFTER_SEC of forward with a frozen view."""
        if self.forward_since is None:
            self.forward_since = now
            return False
        if now - self.forward_since < STUCK_AFTER_SEC:
            return False
        objects = (snap or {}).get("objects", {})
        motion = objects.get("scene_motion")
        if objects.get("stale", True) or motion is None:
            return False  # no usable vision signal - can't judge, keep driving
        if motion >= STUCK_MOTION_THRESHOLD:
            self.forward_since = now  # scene is changing - genuinely moving
            return False
        self.forward_since = None
        self.announce("I've been pushing forward but the view isn't changing. I think I'm stuck, backing off.")
        self._begin_evasion("no_visual_motion")
        return True

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.bus.subscribe("picarx/state/world", self.on_world_state)
        self.bus.subscribe("picarx/action/result", self.on_action_result)
        self.bus.subscribe("picarx/coach/suggestion", self.on_coach_suggestion)

        print("Field Agent active. Say 'explore', 'stop', 'status', 'objects', 'history', or 'battery'.")
        self.announce("Field agent online.", force=True)

        period = 1.0 / EXPLORE_TICK_HZ
        while True:
            self._perception_tick()
            if self.explore_mode:
                self.explore_tick()
            time.sleep(period)


if __name__ == "__main__":
    FieldAgent().run()
