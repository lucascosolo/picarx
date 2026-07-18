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
from spatial_store import SpatialStore
import person_memory
import speech_match

# Smooth Ackermann steering controller (fail-soft: if it can't import,
# the discrete _steer_away_angle law below keeps working unchanged).
try:
    from steering_controller import SteeringController
except Exception as _sc_err:  # pragma: no cover - exercised on-robot only
    print(f"Field agent: steering controller unavailable ({_sc_err}) - "
          f"falling back to the discrete steer-around law")
    SteeringController = None

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

# Utterances containing these are TOOL commands (tools_registry.py
# routes them to their own modules). They must be ignored here so that
# e.g. "stop radio" reaches the radio instead of tripping the
# robot-wide "stop". Movement/safety words never appear in this list.
# "music"/"song" are here because tools_registry now treats them as
# radio synonyms AND escalates its own unparsed radio-ish utterances -
# without them here both modules would escalate the same text twice.
TOOL_KEYWORDS = ("radio", "station", "tools", "tune", "frequency", "dial", "fm",
                 "music", "song")

# After anyone has interacted with the robot (a command, or a wake-
# phrase chat), keep treating unmatched speech as conversation for this
# long - so a follow-up question doesn't need "robot ..." again. Kept
# short: it's a reply window, not an always-open mic to the LLM.
CONVERSATION_WINDOW_SEC = 45.0

# Person identity (person_memory.py, optional): greet a recognized person
# by name, but not every time their face is re-confirmed - once per
# arrival is friendly, once per second is unbearable.
PERSON_GREET_COOLDOWN = 300.0
PERSON_FRESH_SEC = 10.0     # an identity older than this isn't "who I see NOW"

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

# Lighter periodic "glance" done WHILE cruising (as opposed to the full
# startup sweep above): fewer angles, shorter dwell, no spoken report.
# The point is to catch objects off to the side that the forward-facing
# camera+ultrasonic miss when approaching at an angle (the "drives
# diagonally and clips a table leg" failure), and to keep the
# escape-direction estimate below fresh. Silent so it doesn't narrate a
# look-around every half minute.
SCAN_PAN_ANGLES_QUICK = (-55, 0, 55)
SCAN_DWELL_QUICK_SEC = 1.0
CRUISE_SCAN_INTERVAL = 25.0    # re-glance around at least this often while cruising

# Physical stuck detection: commanding forward with nothing being
# vetoed, but the camera scene isn't changing -> wheels are pushing
# against something below the ultrasonic beam (or slipping). vision's
# scene_motion (mean abs thumbnail diff, ~6.0+ while actually moving)
# stays near zero when the view is frozen.
STUCK_AFTER_SEC = 4.0          # this long of continuous forward with a static view -> stuck
STUCK_MOTION_THRESHOLD = 3.0   # scene_motion below this counts as "static"

# Active hypothesis testing: when the ultrasonic reports an obstacle
# but fresh vision sees NOTHING (no tracked objects, nothing filling
# the frame), the sensors disagree - it might be a phantom reading
# (glancing echo, sensor noise) or something vision can't recognize.
# Instead of always evading, occasionally run a bounded micro-probe:
# creep forward very slowly and watch whether the reading tracks like
# a real surface. Hard-bounded per the roadmap's safety mitigations
# (speed <= 15, movement < 1s) and the safety daemon still vetoes at
# its own SAFE_DISTANCE_CM underneath us, so the worst case of a wrong
# guess is a vetoed creep - which itself resolves the hypothesis.
PROBE_SPEED = 12               # <= the roadmap's low-speed probing cap of 15
PROBE_CREEP_SEC = 0.7          # < 1s of actual movement
PROBE_SETTLE_SEC = 0.4         # hold still before/after to get clean readings
PROBE_COOLDOWN = 60.0          # at most one probe a minute
PROBE_MIN_DISTANCE_CM = 16     # below this, don't probe - just evade
PROBE_TIMEOUT = 5.0            # hard upper bound on the whole probe (safety net)

# Second hypothesis: a location the map says keeps vetoing us. Before
# cruising into it normally, creep in slowly (speed <= 10, under the probe
# cap) and STOP a safe buffer early, then just watch: if the safety daemon
# doesn't flag anything for VETO_PROBE_CLEAR_SEC, the area might be clear
# now. The daemon stays the ONLY thing that physically stops the robot -
# this test never bypasses it, it only asks it a question.
VETO_PRONE_THRESHOLD = 3          # veto_count that marks a place "veto-prone"
VETO_PRONE_PROBE_COOLDOWN = 90.0  # min seconds between veto-prone probes
VETO_PROBE_SPEED = 10             # hard speed cap for the careful approach
VETO_PROBE_APPROACH_SEC = 1.0     # brief slow creep in, then stop early
VETO_PROBE_CLEAR_SEC = 3.0        # no veto for this long -> "might be clear"

# Curiosity bias (explorer.py's uncertainty scores, all fail-soft):
# when the CURRENT location scores below this - i.e. it's already well
# understood - wander steering leans toward the side the last scan
# found clearer, to drift somewhere newer instead of re-pacing a known
# patch. At or above it (still learning here, or no explorer running)
# wander stays uniform random, which is exactly the old behavior.
CURIOSITY_SETTLED_SCORE = 0.45
CURIOSITY_BIAS_PROB = 0.7      # biased wanders still keep 30% pure randomness

# ---------------------------------------------------------------------
# RC mode: passive demonstration learning
# ---------------------------------------------------------------------
# While the human drives (picarx/rc/mode active), the AI doesn't act -
# but it WATCHES. When an obstacle-like situation appears (the same
# triggers exploration reacts to), a bounded "demonstration" episode
# opens: the situation context is snapshotted and the human's RC
# commands are collected (consecutive duplicates collapsed) until the
# path clears or the window closes. One picarx/rc/demonstration event
# per episode goes to events.db, where reflection later distills
# repeated demonstrations into durable tactics - a second coach whose
# suggestions come from the human's own driving. Rate-limited so a
# session of joyriding never floods episodic memory with turn-by-turn
# noise.
RC_DEMO_TRIGGER_CM = 30        # fresh ultrasonic below this opens an episode
RC_DEMO_CLEAR_CM = 45          # ...and above this (with clear vision) resolves it
RC_DEMO_MAX_SEC = 10.0         # hard bound on one episode
RC_DEMO_COOLDOWN = 30.0        # min seconds between episodes

# Evasion/coaching priorities - both outrank normal exploring (5), and
# COACH_PRIORITY outranks the canned evasion sequence (8) too, since a
# coach-directed maneuver during a fail state should win over whatever
# the plain reflex would have done.
EVADE_PRIORITY = 8
COACH_PRIORITY = 9
WATCH_PRIORITY = 6

# ---------------------------------------------------------------------
# Reactive steer-around: a deterministic perception->heading law
# ---------------------------------------------------------------------
# Cruising used to drive dead straight until some threshold tripped, at
# which point the only steering that ever happened was the emergency
# reverse-arc - so the robot's whole repertoire looked like "drive at
# things, then back-and-turn away from them." The steering servo has
# always accepted any angle while moving (follow_daemon steers
# proportionally and continuously the same way); the reflexes just never
# used that freedom. This law closes the gap: every cruise tick, fresh
# tracked objects that are in the path ahead each contribute a
# counter-steer away from their side, proportional to how big they loom
# (weight saturates at AVOID_SATURATION_AREA) and how central they sit
# (a dead-ahead object steers hardest, one at the cone's edge barely).
# Contributions SUM, so two objects flanking a gap cancel and the robot
# threads between them instead of ping-ponging - and everything stays a
# vetoable picarx/intent/move like all other motion. No LLM anywhere in
# this path; it's the same class of deterministic control loop as
# follow_daemon. The emergency evade/hypothesis reflexes above it in the
# tick keep priority: this only shapes the heading while the path is
# still considered clear.
AVOID_MIN_AREA = 0.06         # ignore specks; approaching objects count regardless
AVOID_SATURATION_AREA = 0.30  # an object this big steers at full weight
AVOID_CONE_FRAC = 0.75        # only objects within this fraction of half-frame are "in the path"
AVOID_MAX_ANGLE = 22          # cap - stays short of the +/-30 emergency reflexes
AVOID_MIN_ANGLE = 4           # smaller than this isn't worth moving the servo
AVOID_RESEND_DELTA = 4        # re-aim only when the target angle really moved (follow_daemon pattern)
AVOID_SEND_DEADBAND = 1.0     # smooth controller: min angle change (deg) worth a steer tick
AVOID_HOLD_SEC = 0.8          # steering-reset window kept refreshed while actively avoiding
AVOID_SPEED = 20              # ease off (from cruise 25) while maneuvering around something


def _steer_away_angle(snapshot):
    """Signed steering angle to flow around what's visible ahead, or None
    when there's nothing worth reacting to. Pure function of the world
    snapshot (unit-testable off-robot). Returns {"angle", "labels"}."""
    if not snapshot:
        return None
    objects = snapshot.get("objects") or {}
    if objects.get("stale", True):
        return None
    total = 0.0
    labels = []
    for obj in objects.get("items", []):
        area = obj.get("area_ratio") or 0.0
        if not obj.get("approaching") and area < AVOID_MIN_AREA:
            continue
        frame_w = obj.get("frame_width") or 0
        if frame_w <= 0:
            continue
        offset_frac = obj.get("center_offset", 0) / (frame_w / 2.0)
        # Dead-center has no side to prefer (that's the evasion reflex's
        # call), and far off-path isn't in our way - both contribute nothing.
        if offset_frac == 0 or abs(offset_frac) > AVOID_CONE_FRAC:
            continue
        weight = min(1.0, area / AVOID_SATURATION_AREA)
        # "Approaching" means the box is growing FAST - it's closing on us
        # whatever its current size, so it never weighs less than half.
        if obj.get("approaching"):
            weight = max(weight, 0.5)
        side = 1.0 if offset_frac > 0 else -1.0
        total += -side * (1.0 - abs(offset_frac)) * weight * AVOID_MAX_ANGLE
        labels.append(obj.get("label", "something"))
    if not labels:
        return None
    angle = int(round(max(-AVOID_MAX_ANGLE, min(AVOID_MAX_ANGLE, total))))
    if abs(angle) < AVOID_MIN_ANGLE:
        return None
    return {"angle": angle, "labels": labels}


# A depth-sensor-free obstacle signal: world_state flags a tracked
# object "approaching" if its bounding box is growing quickly while
# centered in frame. Treated exactly like a close ultrasonic reading.
#
# Priority order, most-blind-spot-covering first:
#   1. overhead - a head-height OVERHANG (counter lip, table edge) the low
#      bumper ultrasonic can't see over its beam. This is the one the
#      ultrasonic's "clear" reading must NOT be allowed to dismiss (the beam
#      passes under it into open air); flagged with overhead=True so the
#      caller's cross-check knows to trust vision for the head's path.
#   2. close_object - class-agnostic frame-filler; catches point-blank
#      obstacles like a cabinet that "approaching" never can, since that path
#      only fires for objects the SSD actually tracks by label.
#   3. approaching - a labeled tracked object growing fast and centered.
# Returned dict is normalized: {label, area_ratio, overhead, approaching}.
def _vision_obstacle(snapshot):
    if not snapshot:
        return None
    objects = snapshot.get("objects") or {}
    if objects.get("stale", True):
        return None
    overhead = objects.get("overhead")
    if overhead:
        return {"label": "something", "area_ratio": overhead.get("area_ratio", 1.0),
                "overhead": True, "approaching": bool(overhead.get("approaching"))}
    if objects.get("close_object"):
        return {"label": "something", "area_ratio": 1.0,
                "overhead": False, "approaching": False}
    best = None
    for obj in objects.get("items", []):
        if obj.get("approaching") and (best is None or obj.get("area_ratio", 0) > best.get("area_ratio", 0)):
            best = obj
    if best is None:
        return None
    return {"label": best.get("label", "something"), "area_ratio": best.get("area_ratio", 0),
            "overhead": False, "approaching": True,
            # Which side the object sits on, so the escape can swing AWAY
            # from it instead of picking a side blind (see evade_away_hint).
            "center_offset": best.get("center_offset", 0),
            "frame_width": best.get("frame_width", 0)}


def _prune_older_than(dq, window, now, key=None):
    """Drop leading entries older than the window. `key` extracts the
    timestamp when entries aren't bare floats (veto_events holds
    (ts, reason_code) tuples so episode records can say WHICH veto)."""
    ts = key if key is not None else (lambda e: e)
    while dq and ts(dq[0]) < now - window:
        dq.popleft()


# ---------------------------------------------------------------------
# Spoken memory/navigation query parsing (pure functions, unit-testable)
# ---------------------------------------------------------------------
# These parse the raw utterance, not the canonicalized form, because the
# captured group is a NAME the user chose ("kitchen", "watering can") and
# canonicalization is lossy on names.

def parse_place_name_command(text):
    """'call this place the kitchen' / 'name this room lucas office'
    -> 'kitchen' / 'lucas office', else None."""
    m = re.search(r"\b(?:call|name) this (?:place|room|spot|area)\s+"
                  r"(?:the\s+)?([a-z][a-z' ]{1,40})", text)
    if not m:
        m = re.search(r"\bthis (?:place|room) is (?:called\s+)?"
                      r"(?:the\s+)?([a-z][a-z' ]{1,40})", text)
    return m.group(1).strip() if m else None


def parse_go_to_command(text):
    """'go to the kitchen' / 'go back to the living room' -> 'kitchen' /
    'living room', else None. Deliberately narrow: motion only ever
    starts from this literal spoken shape, matching the strict local
    'explore' rule - never from a fuzzy repair or an LLM guess."""
    m = re.search(r"\bgo (?:back )?to (?:the\s+)?([a-z][a-z' ]{1,40})", text)
    return m.group(1).strip() if m else None


def parse_where_is_query(text):
    """'where is the bottle' / \"where's my cup\" / 'where did you see
    the chair' -> 'bottle' / 'cup' / 'chair', else None."""
    if "where are you" in text:
        return None  # that's the map report, not an object query
    m = re.search(r"\bwhere(?:'s| is| was| did you (?:last )?see)\s+"
                  r"(?:the |my |a |an )?([a-z][a-z' ]{1,40})", text)
    return m.group(1).strip() if m else None


def parse_whats_in_query(text):
    """\"what's in the kitchen\" / 'what is in the living room' / 'what
    have you seen in the kitchen' -> 'kitchen' etc., else None."""
    m = re.search(r"\bwhat(?:'s| is| have you seen| did you see)\s+"
                  r"(?:is )?in (?:the\s+)?([a-z][a-z' ]{1,40})", text)
    return m.group(1).strip() if m else None


def spoken_age(seconds):
    """'just now' / '5 minutes ago' / 'about 3 hours ago' - for spoken
    answers about when something was last seen."""
    if seconds < 90:
        return "just now"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} minutes ago"
    hours = minutes / 60.0
    if hours < 36:
        return f"about {hours:.0f} hour{'s' if round(hours) != 1 else ''} ago"
    return f"about {hours / 24.0:.0f} days ago"


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


# ---------------------------------------------------------------------
# Active physical hypothesis testing (generic framework)
# ---------------------------------------------------------------------
# The robot occasionally has a QUESTION about the physical world it can
# only answer by carefully doing something and watching the result -
# "is that ultrasonic blip real or a phantom?", "is this spot that keeps
# stopping me still blocked?". Each such test is a HypothesisTask: a
# small, bounded, single-at-a-time state machine.
#
# INVARIANTS every task must uphold (the point of the shared base):
#   - Standard lifecycle fields: type, state (init -> testing ->
#     resolving), started_at, timeout. Nothing runs unbounded.
#   - It NEVER issues a motion command the safety daemon can't veto, and
#     never touches the safety socket. The daemon is the only thing that
#     physically stops the robot; a task only asks it questions by
#     moving slowly and watching for its vetoes.
#   - Every resolution goes out the SAME way (resolve()): the
#     picarx/exploration/hypothesis topic plus a decision-journal entry,
#     so the existing contract holds no matter how many task types exist.
class HypothesisTask:
    TYPE = "hypothesis"
    QUESTION = "hypothesis"
    START_KIND = "hypothesis_probe"

    def __init__(self, agent, now, timeout):
        self.agent = agent
        self.type = self.TYPE
        self.state = "init"          # init -> testing -> resolving
        self.started_at = now
        self.timeout = timeout
        self.resolution = None

    # --- lifecycle template: the agent calls run() once per tick ---
    def run(self, now):
        """Advance one tick. Returns True while still running, False once
        resolved. Enforces the shared timeout before delegating to the
        task-specific tick(), so no subclass can run past its bound."""
        if self.state == "resolving":
            return False
        if now - self.started_at > self.timeout:
            self.on_timeout(now)
            return False
        self.state = "testing"
        return self.tick(now)

    # --- to be provided by subclasses ---
    def start_choice(self):
        """Small dict describing what set this probe off (decision journal)."""
        return {}

    def tick(self, now):
        raise NotImplementedError

    def on_timeout(self, now):
        """Default: an unfinished probe is treated as unresolved-and-safe."""
        self.resolve(now, "inconclusive",
                     "hypothesis timed out before resolving - treating as unresolved")

    def follow_up(self, now):
        """What the agent should do after this task resolves (evade, cruise,
        ...). Default: nothing - just go back to cruising."""
        self.agent.state = "CRUISING"
        self.agent.last_wander = now

    # --- shared helpers ---
    def _vetoed_since_start(self):
        """Did the safety daemon veto any of OUR intents since the probe
        began? That is the physical answer we are listening for."""
        with self.agent.lock:
            return any(t > self.started_at for t, _code in self.agent.veto_events)

    def resolve(self, now, resolution, why, **details):
        """Single exit point for EVERY hypothesis type: publish the
        outcome on picarx/exploration/hypothesis and mirror it into the
        decision journal, exactly as the original probe did."""
        self.state = "resolving"
        self.resolution = resolution
        self.agent.bus.publish("picarx/exploration/hypothesis", {
            "question": self.QUESTION,
            "resolution": resolution,
            "location": self.agent._location_context(),
            "ts": now,
            **details,
        })
        self.agent.publish_decision("hypothesis_resolved",
                                    {"resolution": resolution}, why, **details)
        print(f"Hypothesis [{self.type}]: {resolution} - {why}")


class SensorDisagreementProbe(HypothesisTask):
    """Original probe, migrated verbatim: the ultrasonic reports an
    obstacle but fresh vision sees nothing. Creep forward very slowly and
    see whether the reading tracks like a real surface, closes in (real),
    or opens up (phantom). A safety-daemon veto during the creep resolves
    it immediately as real."""
    TYPE = "sensor_disagreement"
    QUESTION = "ultrasonic_obstacle_vs_empty_vision"

    def __init__(self, agent, now, d0):
        super().__init__(agent, now, timeout=PROBE_TIMEOUT)
        self.d0 = d0                  # ultrasonic reading that started the probe
        self.d1 = None
        self.stage = 0                # 0 settle, 1 creep, 2 settle+judge
        self.stage_until = now + PROBE_SETTLE_SEC

    def start_choice(self):
        return {"d0": self.d0}

    def tick(self, now):
        agent = self.agent
        if self._vetoed_since_start():
            self.resolve(now, "real_obstacle",
                         "the safety layer vetoed the creep - it's real",
                         d0=self.d0, d1=None)
            return False
        if now < self.stage_until:
            if self.stage == 1:
                agent.publish_intent({"direction": "forward", "speed": PROBE_SPEED},
                                     priority=EVADE_PRIORITY)
            else:
                agent.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
            return True
        self.stage += 1
        if self.stage == 1:
            self.stage_until = now + PROBE_CREEP_SEC
            return True
        if self.stage == 2:
            self.stage_until = now + PROBE_SETTLE_SEC
            agent.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
            return True
        # stage 3: judge the second reading against the first.
        snap = agent._snapshot() or {}
        d1 = None
        if snap.get("distance_cm") is not None and not snap.get("distance_stale", True):
            d1 = snap["distance_cm"]
        self.d1 = d1
        if d1 is not None and d1 <= self.d0 + 2:
            self.resolve(now, "real_obstacle",
                         "the reading tracked like a real surface while creeping",
                         d0=self.d0, d1=d1)
        elif d1 is not None and d1 > self.d0 + 15:
            self.resolve(now, "phantom_reading",
                         "the path opened right up - the reading was a phantom",
                         d0=self.d0, d1=d1)
        else:
            self.resolve(now, "inconclusive",
                         "couldn't get a clean second reading - treating it as real to be safe",
                         d0=self.d0, d1=d1)
        return False

    def follow_up(self, now):
        agent = self.agent
        if self.resolution == "phantom_reading":
            agent.announce("False alarm - the way is actually clear.")
            agent.state = "CRUISING"
            agent.last_wander = now
            return
        # real_obstacle / inconclusive both get the normal escape.
        agent.announce("It's really there. Backing away.")
        agent._begin_evasion("ultrasonic")


class VetoProneLocationProbe(HypothesisTask):
    """Second hypothesis: this place has vetoed us repeatedly before, so
    the map calls it veto-prone. Question: "is this area still blocked?"
    Approach VERY slowly (speed <= 10) for a brief moment, then STOP a
    safe distance early and simply watch. If the safety daemon flags an
    obstacle, it is still blocked; if nothing is flagged for the whole
    watch window, the area MIGHT be clear now. We never decide "clear" by
    driving through it - only by the daemon staying silent."""
    TYPE = "veto_prone_location"
    QUESTION = "is_veto_prone_area_still_blocked"

    def __init__(self, agent, now, location_id, veto_count):
        super().__init__(agent, now, timeout=VETO_PROBE_CLEAR_SEC)
        self.location_id = location_id
        self.veto_count = veto_count

    def start_choice(self):
        return {"location_id": self.location_id, "veto_count": self.veto_count}

    def tick(self, now):
        agent = self.agent
        if self._vetoed_since_start():
            self.resolve(now, "still_blocked",
                         "the safety daemon flagged an obstacle - the area is still blocked",
                         location_id=self.location_id, veto_count=self.veto_count)
            return False
        if now - self.started_at < VETO_PROBE_APPROACH_SEC:
            # Creep in slowly. Speed is hard-capped and the safety daemon
            # still owns the actual stop - this only nudges toward the spot.
            agent.publish_intent({"direction": "forward", "speed": VETO_PROBE_SPEED},
                                 priority=EVADE_PRIORITY)
        else:
            # Stop a safe distance early and just listen for a veto until
            # the watch window (the task timeout) elapses.
            agent.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
        return True

    def on_timeout(self, now):
        # The window closing with no veto IS the resolution here (re-check
        # once more so a veto landing right at the edge still counts).
        if self._vetoed_since_start():
            self.resolve(now, "still_blocked",
                         "the safety daemon flagged an obstacle - the area is still blocked",
                         location_id=self.location_id, veto_count=self.veto_count)
        else:
            self.resolve(now, "maybe_clear",
                         "no veto for the full watch window - the area might be clear now",
                         location_id=self.location_id, veto_count=self.veto_count)

    def follow_up(self, now):
        agent = self.agent
        if self.resolution == "maybe_clear":
            agent.announce("No block this time. This spot might be clear now.")
            agent.state = "CRUISING"
            agent.last_wander = now
            return
        agent.announce("Still blocked here. Backing away.")
        agent._begin_evasion("veto_prone_location")


class FieldAgent:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()

        self.last_interaction_at = 0.0  # last time speech was clearly for us
        self.explore_mode = False
        self.latest_world = None
        self.face_was_detected = False
        self.known_object_labels = set()

        # Person identity (from person_memory.py, optional/fail-soft).
        self.latest_person = None       # last identity payload + received_at
        self.last_greeted_person = None
        self.last_greeted_at = 0.0

        # The most recent human utterance handled (never a repair or a
        # console correction) - what spoken feedback like "that's not
        # what I meant" refers to, so the intent teacher knows WHICH
        # phrasing to unlearn or re-map.
        self.last_utterance = None

        # RC mode (picarx/rc/mode): the human has the wheel; we only
        # observe. rc_demo holds the in-flight demonstration episode (or
        # None); rc_pending_actions is the cross-thread inbox of the
        # human's executed/vetoed RC commands.
        self.rc_active = False
        self.rc_demo = None
        self.rc_pending_actions = deque()
        self.last_rc_demo_at = 0.0

        self.last_announcement_at = 0.0
        self.start_time = time.time()

        # State machine for non-blocking obstacle evasion / coaching.
        # These fields are only ever mutated from explore_tick's
        # thread (the run() loop) - bus callbacks only ever feed the
        # lock-protected inboxes below, never touch state directly,
        # so there's no cross-thread race on the state machine itself.
        self.state = "CRUISING"  # CRUISING, SCANNING, EVADING, COACHING, HYPOTHESIS
        self.evade_stage = 0     # 0:stop 1:pre-turn 2:reverse-arc 3:straighten+go
        self.evade_angle = 0     # steering angle held through the reverse arc
        self.evade_reason = None # what triggered the current evasion (journaled)
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
        self.scan_angles = SCAN_PAN_ANGLES   # which sweep this scan is running
        self.scan_dwell_sec = SCAN_DWELL_SEC
        self.scan_is_startup = True    # startup scan announces; periodic glance stays quiet
        self.last_scan_at = 0.0        # when the last sweep finished (paces periodic glances)
        # Escape direction learned from the most recent scan: turn toward
        # whichever side had fewer objects, instead of a coin flip. None
        # until a scan has produced an asymmetry.
        self.preferred_escape_angle = None

        # Physical stuck detection state (see STUCK_AFTER_SEC).
        self.forward_since = None      # when the current uninterrupted forward run began

        # Active-hypothesis framework state (see HypothesisTask). At most
        # one bounded, safety-daemon-gated probe runs at a time (None when
        # not testing); the last_*_at fields rate-limit each trigger.
        self.hypothesis = None               # current HypothesisTask, or None
        self.last_probe_at = 0.0             # sensor-disagreement probe cooldown
        self.last_veto_prone_probe_at = 0.0  # veto-prone location probe cooldown

        # Wander state (mirrors the old reflex explorer's behavior,
        # now expressed as intents instead of direct socket calls)
        self.last_wander = time.time()
        self.wander_interval = random.uniform(5.0, 10.0)
        self.steering_active_until = 0

        # Reactive steer-around state (see _steer_away_angle): the angle
        # currently commanded to flow around a visible object (None while
        # not avoiding), and a one-shot escape-side hint - which way the
        # NEXT evasion should swing, taken from the side the triggering
        # obstacle was actually seen on.
        self.avoid_active_angle = None
        self.evade_away_hint = None

        # Smooth steering controller (optional - None falls back to the
        # discrete law). The two _avoid_* fields drive the one-primitive-
        # per-tick alternation: the arbiter holds ONE intent per source,
        # so publishing turn and forward back to back would let the
        # forward overwrite the turn before the arbiter ever samples it.
        # Instead we steer on ticks where the angle materially moved
        # (never two in a row, so forward - and the safety daemon's
        # forward checks - keep flowing) and drive on the rest; the
        # daemon's MotionSmoother holds both targets between commands.
        # Same pattern follow_daemon and the evade stages already use.
        self.steering = SteeringController() if SteeringController else None
        self._avoid_sent_angle = 0.0
        self._avoid_turn_last_tick = False

        # Spatial context (location_graph.py + explorer.py, both
        # optional): where the robot currently believes it is, and how
        # uncertain each known place still is. None/empty when those
        # modules aren't running - every use below falls back to the
        # old spatially-blind behavior.
        self.current_location = None       # last location_change payload
        self.uncertainty_scores = {}       # location_id -> score
        self.spatial = SpatialStore(readonly=True)  # map queries for spoken reports

        # Active exploration subgoal (goal_manager.py, optional): if a
        # scan spots any of the goal place's landmark labels, wander
        # leans toward that side until the goal changes.
        self.active_goal = None            # last active_goal payload (or None)
        self.goal_bias_angle = None        # signed angle toward last goal sighting

        # Cross-thread inboxes (bus callbacks append, explore_tick/
        # _perception_tick drain under self.lock).
        self.veto_events = deque()
        self.last_veto_code = None   # reason_code of the most recent veto
        self.last_veto_at = 0.0
        self.pending_novel_objects = deque()
        self.pending_suggestions = deque()

        # Urgent (blocking, fail-state) coach query bookkeeping.
        self.coach_query_id = None
        self.coach_query_deadline = 0.0
        # A coach suggestion is now an ordered list of steps
        # ([{"action","duration"}, ...]) run back to back, not a single
        # action. coach_steps None = still waiting on a reply.
        self.coach_steps = None
        self.coach_step_index = 0
        self.coach_step_until = 0.0
        self.coach_motion_max = None   # peak scene_motion seen during the maneuver
        self.coach_action_started_at = 0.0
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
        # ts lets audio_nodes drop announcements that sat in the playback
        # queue too long - narrating a decision from 20 seconds ago while
        # already doing something else is worse than staying quiet.
        self.bus.publish("picarx/audio/speak", {"text": text, "ts": now})

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
        if payload.get("source") == "rc":
            # Human RC commands: collected only while a demonstration
            # episode is open (see _rc_observer_tick), never acted on.
            with self.lock:
                if self.rc_demo is not None:
                    self.rc_pending_actions.append(
                        (time.time(), payload.get("action") or {},
                         (payload.get("result") or {}).get("status")))
            return
        if payload.get("source") != SOURCE_NAME:
            return
        result = payload.get("result") or {}
        if result.get("status") != "vetoed":
            return
        with self.lock:
            # (ts, reason_code): the code rides along so episode/outcome
            # records can say WHICH veto ended a maneuver, not just THAT
            # one did - a cliff veto and an unseen-obstacle veto call for
            # different corrections.
            code = result.get("reason_code", "unknown")
            self.veto_events.append((time.time(), code))
            self.last_veto_code = code
            self.last_veto_at = time.time()

    # ---------- inbound: follow-mode coordination ----------

    def on_follow_state(self, payload):
        """Follow mode owns the wheel while it's on (priority 7 beats
        exploring's 5/6 in the arbiter), but leaving explore_mode running
        underneath means this module keeps publishing headings - and its
        EVADE/COACH reflexes (priority 8/9) can outrank and fight the
        follow behaviour, e.g. treating the approaching person as an
        obstacle to reverse away from. One driver at a time: following
        pauses exploration outright."""
        if payload.get("enabled") and self.explore_mode:
            self.explore_mode = False
            self.cancel_intent()
            print("Field agent: pausing exploration - follow mode took the wheel")
            self.publish_decision(
                "explore_paused", {"by": "follow"},
                "following was enabled, so I stopped exploring instead of "
                "fighting it for the wheel")

    def _stop_following(self):
        """The inverse handoff: starting to explore (or navigate to a
        place) takes the wheel back from follow mode. A no-op in
        follow_daemon when following isn't active."""
        self.bus.publish("picarx/tools/follow/set", {"enabled": False})

    # ---------- inbound: RC mode (human takes the wheel) ----------

    def on_rc_mode(self, payload):
        active = bool(payload.get("active"))
        if active == self.rc_active:
            return
        self.rc_active = active
        with self.lock:
            self.rc_demo = None
            self.rc_pending_actions.clear()
        if active:
            if self.explore_mode:
                self.explore_mode = False
                self.cancel_intent()
            self._stop_following()
            print("Field agent: RC mode ON - standing down, watching the human drive")
            self.publish_decision(
                "rc_mode", {"active": True},
                "the user took manual control, so I stopped driving and am "
                "watching how they handle things")
        else:
            print("Field agent: RC mode off - autonomy available again (say explore)")
            self.publish_decision("rc_mode", {"active": False},
                                  "manual control ended")

    # ---------- RC observation (passive demonstration learning) ----------

    def _rc_situation(self, snap):
        """Obstacle-like trigger, mirroring what exploration reacts to.
        Returns a situation string or None."""
        if snap is None:
            return None
        distance = snap.get("distance_cm")
        if (distance is not None and not snap.get("distance_stale", True)
                and 0 < distance < RC_DEMO_TRIGGER_CM):
            return "obstacle_ahead"
        if _vision_obstacle(snap) is not None:
            return "vision_obstacle"
        return None

    def _rc_situation_cleared(self, snap):
        if snap is None:
            return False
        distance = snap.get("distance_cm")
        ultra_clear = (distance is not None and not snap.get("distance_stale", True)
                       and distance > RC_DEMO_CLEAR_CM)
        return ultra_clear and _vision_obstacle(snap) is None

    @staticmethod
    def _compress_rc_actions(raw):
        """[(ts, action, status), ...] -> deduped [{"action","status",
        "count","duration"}] - the SHAPE of the maneuver, not every
        repeated tick, but WITH real timing: "backed up for 1.2 seconds"
        is the correction-relevant fact, and duration also makes these
        steps schema-compatible with the coach's own {"action","duration"}
        arms. Duration spans first to last confirmation of the identical
        command plus one arbiter tick (the command is held ~0.1s past its
        final confirmation). Held stops are under-counted (the arbiter
        dedups repeated stops) - fine, stop length rarely matters."""
        steps = []
        for ts, action, status in raw:
            if steps and steps[-1]["action"] == action and steps[-1]["status"] == status:
                steps[-1]["count"] += 1
                steps[-1]["_last_ts"] = ts
                continue
            steps.append({"action": action, "status": status, "count": 1,
                          "_first_ts": ts, "_last_ts": ts})
        for s in steps:
            s["duration"] = round(s.pop("_last_ts") - s.pop("_first_ts") + 0.1, 2)
        return steps

    @staticmethod
    def _demo_object(obj):
        """Compact, geometry-aware object record for demonstration
        context: which side it sat on (same l/c/r convention as place
        fingerprints), how big it loomed, and whether it was closing."""
        frame_w = obj.get("frame_width") or 0
        offset_frac = (obj.get("center_offset", 0) / (frame_w / 2.0)) if frame_w else 0
        side = "l" if offset_frac < -0.15 else ("r" if offset_frac > 0.15 else "c")
        return {"label": obj.get("label", "something"), "side": side,
                "area_ratio": round(obj.get("area_ratio") or 0.0, 2),
                "approaching": bool(obj.get("approaching"))}

    def _rc_observer_tick(self, now):
        snap = self._snapshot()
        with self.lock:
            demo = self.rc_demo
            pending = list(self.rc_pending_actions)
            self.rc_pending_actions.clear()

        if demo is None:
            if now - self.last_rc_demo_at < RC_DEMO_COOLDOWN:
                return
            situation = self._rc_situation(snap)
            if situation is None:
                return
            demo = {
                "situation": situation,
                "started_at": now,
                "raw_actions": [],
                "context": {
                    "distance_cm": (snap or {}).get("distance_cm"),
                    # Geometry matters for corrections: an obstacle on the
                    # LEFT is a different lesson than one on the right.
                    "objects": [self._demo_object(o) for o in
                                ((snap or {}).get("objects") or {}).get("items", [])],
                    "location": self._location_context(),
                },
            }
            with self.lock:
                self.rc_demo = demo
            print(f"Field agent: watching how the user handles: {situation}")
            return

        demo["raw_actions"].extend(pending)
        cleared = self._rc_situation_cleared(snap)
        if not cleared and now - demo["started_at"] < RC_DEMO_MAX_SEC:
            with self.lock:
                self.rc_demo = demo
            return

        # Episode over: publish one compact demonstration record - but
        # only if the human actually DID something (an empty window is
        # noise, not a lesson).
        with self.lock:
            self.rc_demo = None
        self.last_rc_demo_at = now
        steps = self._compress_rc_actions(demo["raw_actions"])
        if not steps:
            return
        record = {
            "situation": demo["situation"],
            "context": demo["context"],
            "actions": steps,
            "resolved": cleared,
            "duration": round(now - demo["started_at"], 1),
            "ts": now,
        }
        moves = ",".join(s["action"].get("direction", "?") for s in steps)
        print(f"Field agent: recorded RC demonstration ({demo['situation']} -> "
              f"{moves} -> {'cleared' if cleared else 'unresolved'})")
        self.bus.publish("picarx/rc/demonstration", record)
        self.publish_decision(
            "rc_demonstration",
            {"situation": demo["situation"], "moves": moves, "resolved": cleared},
            "I watched how the user drove out of a situation I usually "
            "struggle with, and saved it to learn from")

    # ---------- inbound: person identity (optional module) ----------

    def on_person(self, payload):
        name = payload.get("name")
        now = time.time()
        with self.lock:
            self.latest_person = {**payload, "received_at": now}
        if not name:
            return
        if (name == self.last_greeted_person
                and now - self.last_greeted_at < PERSON_GREET_COOLDOWN):
            return
        self.last_greeted_person = name
        self.last_greeted_at = now
        self.announce(f"Hello, {name}! Good to see you.")

    # ---------- inbound: coach ----------

    def on_coach_suggestion(self, payload):
        with self.lock:
            self.pending_suggestions.append(payload)

    # ---------- inbound: spatial context (optional modules) ----------

    def on_location_change(self, payload):
        with self.lock:
            self.current_location = payload
        if payload.get("is_new"):
            self.announce(f"I think this is somewhere new. I'll call it {payload.get('label')}.")

    def on_uncertainty_map(self, payload):
        scores = {e["id"]: e["score"] for e in payload.get("locations", [])}
        with self.lock:
            self.uncertainty_scores = scores

    def on_active_goal(self, payload):
        with self.lock:
            self.active_goal = payload if payload.get("location_id") is not None else None
            self.goal_bias_angle = None  # re-derived from the next scan

    def _location_context(self):
        """Compact {id,label,score} of where we are, or None."""
        with self.lock:
            loc = dict(self.current_location) if self.current_location else None
            scores = self.uncertainty_scores
        if not loc:
            return None
        return {"id": loc.get("location_id"), "label": loc.get("label"),
                "uncertainty": scores.get(loc.get("location_id"))}

    # ---------- decision journal ----------

    def publish_decision(self, kind, choice, reason, **extra):
        """Introspection hook: every non-trivial choice goes onto the
        bus with WHY it was made (event_logger persists them), so
        'why did you do that?' has a real answer instead of a shrug."""
        self.bus.publish("picarx/decision", {
            "source": SOURCE_NAME, "kind": kind, "choice": choice,
            "reason": reason, "location": self._location_context(),
            "ts": time.time(), **extra,
        })

    # ---------- inbound: voice ----------

    def on_heard(self, payload):
        text = payload.get("text", "").lower().strip()
        if not text:
            return
        print(f"Heard: '{text}'")
        self.handle_voice_command(text,
                                  confidence=payload.get("confidence"),
                                  source=payload.get("source"))

    def handle_voice_command(self, text, confidence=None, source=None):
        # Checks below run against the raw text PLUS its canonicalized
        # form, so filler never blocks a match and an STT near-miss
        # ("batery", "radial") still lands on the right word. Single
        # keywords are matched as whole TOKENS, not substrings - "we
        # stopped by earlier" must not halt exploration, and "who's in
        # charge here" must not read out the battery. Phrases stay
        # substring checks.
        canon = speech_match.canonicalize(text)
        match_text = f"{text} {canon}"
        toks = set(speech_match.tokens(match_text))
        # Repaired text comes from the LLM intent arbiter. Its allowlist
        # already excludes motion, but enforce the invariant here too:
        # motion (explore / go to) only ever starts from literally heard
        # words, never from any model's rewrite of a garbled transcript.
        from_repair = source == "intent_repair"
        from_human = source not in ("intent_repair", "user_correction")

        # Spoken feedback on the last interpretation ("that's not what I
        # meant" / "good robot") goes to the intent teacher (companion),
        # tagged with the utterance being judged - never treated as a
        # command or chat itself.
        if from_human:
            verdict = speech_match.parse_feedback(text)
            if verdict:
                self._mark_interaction()
                print(f"(intent feedback '{verdict}' on: '{self.last_utterance}')")
                self.bus.publish("picarx/intent/feedback", {
                    "verdict": verdict, "utterance": self.last_utterance,
                    "origin": "voice", "ts": time.time()})
                return
            self.last_utterance = text

        # Tool commands belong to tools_registry.py / their tool module.
        if any(k in match_text for k in TOOL_KEYWORDS):
            print(f"(tool command, leaving it to the tools registry): '{text}'")
            self._mark_interaction()
            return

        # Only an explicit "explore" starts driving. "start" was too loose -
        # STT mishearing background noise/TV as "start" could auto-launch
        # exploration on its own, which is exactly the unwanted
        # "moves around on boot" behavior. Default state is stationary.
        # NOTE this checks raw text only, deliberately: motion must never
        # start off a fuzzy repair, only off the literal word.
        if "explore" in text and not from_repair:
            self._mark_interaction()
            if self.rc_active:
                # The human literally has the wheel - autonomy resumes
                # only after RC mode is switched off at the console.
                self.announce("You're driving me right now. Turn off R C mode "
                              "first and I'll explore.", force=True)
                return
            if not self.explore_mode:
                self._stop_following()
                self.explore_mode = True
                self.given_up = False
                self.consecutive_coach_failures = 0
                self.next_fail_state_allowed_at = 0.0
                # Look around before rolling: sweep the camera across
                # the room and take stock of what's where first.
                self._enter_scanning(time.time(), startup=True)
                self.announce("Starting exploration. Let me take a look around first.", force=True)
            return

        if "stop" in toks or "halt" in toks:
            self._mark_interaction()
            if self.explore_mode:
                self.explore_mode = False
                self.cancel_intent()
                self.publish_look(0, 0)  # recenter the head wherever the scan/drive left it
                self.announce("Stopping.", force=True)
            return

        # Deliberately narrow: just "battery"/"power" as whole tokens.
        # Looser phrasings ("how's your charge?") escalate to the intent
        # arbiter below, get mapped to "battery" once, and are cached -
        # so precision here costs nothing but the first-ever API call.
        if "battery" in toks or "power" in toks:
            self._mark_interaction()
            self.report_battery()
            return

        if "history" in toks or "what have you done" in text or "what happened" in text:
            self._mark_interaction()
            self.report_history()
            return

        if "object" in toks or "objects" in toks or "what's around" in text or "whats around" in text or "what do you notice" in text:
            self._mark_interaction()
            self.report_objects()
            return

        # --- spatial/person memory commands (all local, no LLM) ---
        place_name = parse_place_name_command(text)
        if place_name:
            self._mark_interaction()
            self.name_current_place(place_name)
            return

        # NOTE raw text only, same rule as "explore" above: motion must
        # never start off a fuzzy repair, only off the literal words.
        destination = parse_go_to_command(text) if not from_repair else None
        if destination:
            self._mark_interaction()
            self.go_to_place(destination)
            return

        where_query = parse_where_is_query(text)
        if where_query:
            self._mark_interaction()
            if not self.report_object_location(where_query):
                # Nothing in spatial memory matches - that makes this a
                # QUESTION, not a failed command. Hand it to the chat
                # layer, which has both the sighting-store tool and
                # general knowledge, instead of a canned brush-off.
                print(f"(no sighting match for '{where_query}', forwarding to chat): '{text}'")
                self.bus.publish("picarx/audio/unhandled",
                                 {"text": text, "confidence": confidence})
            return

        whats_in = parse_whats_in_query(text)
        if whats_in:
            self._mark_interaction()
            self.report_place_contents(whats_in)
            return

        if ("who am i" in text or "who do you see" in text
                or "do you know me" in text or "who is this" in text):
            self._mark_interaction()
            self.report_person()
            return

        if "why" in text and ("why did you" in text or text.strip() == "why"):
            self._mark_interaction()
            self.report_why()
            return

        if "where are you" in text or "map" in match_text or "places" in match_text:
            self._mark_interaction()
            self.report_map()
            return

        if "what do you see" in text or "status" in match_text or "report" in match_text:
            self._mark_interaction()
            self.report_status()
            return

        if "hello" in match_text or re.search(r"\bhi\b", text):
            self._mark_interaction()
            self.announce("Hello! I am ready to chat and explore.", force=True)
            return

        # Nothing above matched a hard command. Three ways it can still
        # mean something, tried in order:
        #   1. wake phrase -> LLM chat (as before);
        #   2. we're mid-conversation (someone addressed the robot within
        #      CONVERSATION_WINDOW_SEC) -> LLM chat without the wake word,
        #      so follow-ups don't need "robot ..." every single time;
        #   3. it LOOKS like a garbled command (contains robot vocabulary
        #      yet matched nothing) -> the LLM intent arbiter, which maps
        #      it onto a known command if it can, and teaches the local
        #      phrase cache so next time this stays fully on-board.
        # Everything else is dropped, printed so the misses stay visible.
        remainder = self._strip_wake_phrase(text)
        if remainder is not None:
            self._mark_interaction()
            self.bus.publish("picarx/audio/unhandled",
                             {"text": remainder, "confidence": confidence})
            return

        if time.time() - self.last_interaction_at < CONVERSATION_WINDOW_SEC:
            # Deliberately does NOT re-mark the interaction: only wake
            # phrases and matched commands extend the window. If chatting
            # itself extended it, a talkative TV within 45s of one real
            # command would hold the window open (and burn API calls)
            # indefinitely.
            print(f"(in-conversation, forwarding without wake phrase): '{text}'")
            self.bus.publish("picarx/audio/unhandled",
                             {"text": text, "confidence": confidence})
            return

        # Loop guard: text the arbiter already repaired never re-escalates -
        # if its best repair still matched nothing, it dies here.
        # Two independent "this was probably meant for the robot" signals:
        # robot vocabulary anywhere in it (looks_command_like), or an
        # imperative sentence shape ("take me to the kitchen", "come with
        # me") that contains no domain word at all (looks_directed_command).
        # Either way the LLM arbiter makes the real intent call, and its
        # phrase cache means each phrasing is only ever paid for once.
        if not from_repair and (speech_match.looks_command_like(canon)
                                or speech_match.looks_directed_command(text)):
            print(f"(command-shaped but unmatched, escalating to arbiter): '{text}'")
            self.bus.publish("picarx/audio/uncertain", {
                "text": text, "confidence": confidence, "from": SOURCE_NAME})
            return

        print(f"(no wake phrase, not forwarding to chat): '{text}'")

    def _mark_interaction(self):
        """Speech was clearly directed at the robot - keeps the
        no-wake-word conversation window (CONVERSATION_WINDOW_SEC) open."""
        self.last_interaction_at = time.time()

    @staticmethod
    def _strip_wake_phrase(text):
        for phrase in WAKE_PHRASES:
            if not text.startswith(phrase):
                continue
            # Whole-word match only: "robotics class was fun" starts with
            # "robot" but was never addressed to the robot.
            rest = text[len(phrase):]
            if rest and rest[0].isalnum():
                continue
            remainder = rest.strip(" ,.:;-!?")
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
        known_places = {}   # label -> place name, from the sighting store
        for obj in items:
            label = obj.get("label", "something")
            age = time.time() - obj.get("first_seen", time.time())
            if age < 3.0:
                descriptions.append(f"a {label} I just noticed")
            else:
                descriptions.append(f"a {label} I've been tracking for a bit")
            if label not in known_places:
                places = self.spatial.object_locations(label, limit=1)
                if places and places[0]["times_seen"] > 1:
                    known_places[label] = places[0]["place"]
        speech = f"I currently see {len(items)}: " + ", ".join(descriptions)
        # Spatial recall makes this more than a live list: say where the
        # robot usually finds one of these, when it actually knows.
        for label, place in list(known_places.items())[:2]:
            speech += f". I usually see a {label} at {place}"
        self.announce(speech + ".", force=True)

    def report_map(self):
        """Spoken summary of the spatial map: where am I, how much of
        the world do I know, what's still mysterious. All read-only
        and fail-soft (no spatial modules -> honest 'no map yet')."""
        count = self.spatial.location_count()
        if count == 0:
            self.announce("I haven't built a map yet. Tell me to explore and I'll start one.",
                          force=True)
            return
        parts = [f"I know {count} place{'s' if count != 1 else ''}"]
        loc = self._location_context()
        if loc and loc.get("label"):
            parts.append(f"right now I believe I'm at {loc['label']}")
        with self.lock:
            scores = dict(self.uncertainty_scores)
            goal = dict(self.active_goal) if self.active_goal else None
        if scores:
            mystery_id = max(scores, key=scores.get)
            mystery = self.spatial.get_location(mystery_id)
            if mystery and scores[mystery_id] >= 0.3:
                parts.append(f"the place I understand least is {mystery['label']}")
        if goal:
            parts.append(f"my current mission is to reach {goal.get('label')}")
        self.announce(". ".join(parts) + ".", force=True)

    def name_current_place(self, name):
        """Route 'call this place the kitchen' to location_graph (the
        sole spatial.db writer); it renames, confirms out loud, and
        journals the change. We only update our own cached label."""
        loc = self._location_context()
        if not loc or loc.get("id") is None:
            self.announce("I'm not sure where I am yet. Tell me to explore first, "
                          "then name the place.", force=True)
            return
        self.bus.publish("picarx/exploration/name_place",
                         {"location_id": loc["id"], "name": name})
        with self.lock:
            if self.current_location:
                self.current_location["label"] = name

    def go_to_place(self, place_query):
        """'go to the kitchen': adopt a user goal for a known place and
        start exploring toward it. All motion stays the normal wander/
        goal-bias pipeline - vetoable intents, nothing new mechanically."""
        loc = self.spatial.find_location_by_name(place_query)
        if loc is None:
            known = [l["label"] for l in self.spatial.all_locations()]
            hint = f" I know: {', '.join(known[:5])}." if known else \
                " I haven't mapped anywhere yet - tell me to explore."
            self.announce(f"I don't know a place called {place_query} yet.{hint}",
                          force=True)
            return
        self.bus.publish("picarx/exploration/goal_request",
                         {"location_id": loc["id"], "label": loc["label"]})
        self.publish_decision("go_to_place",
                              {"location_id": loc["id"], "label": loc["label"]},
                              f"the user asked me to go to {loc['label']}")
        if not self.explore_mode:
            self._stop_following()
            self.explore_mode = True
            self.given_up = False
            self.consecutive_coach_failures = 0
            self.next_fail_state_allowed_at = 0.0
            self._enter_scanning(time.time(), startup=True)
        self.announce(f"Heading toward {loc['label']}. I'll keep an eye out for it "
                      f"as I go.", force=True)

    def report_object_location(self, query):
        """'where is the bottle' - answered from the sighting store, no
        LLM: which place, how long ago, how reliably. Returns True if it
        answered; False when the store has no match, so the caller can
        route the utterance to chat instead of a canned miss."""
        label = speech_match.best_label_match(query, self.spatial.sighting_labels())
        places = self.spatial.object_locations(label) if label else []
        if not places:
            return False
        top = places[0]
        when = spoken_age(time.time() - top["last_seen"])
        speech = f"I last saw a {label} at {top['place']}, {when}"
        if top["times_seen"] > 1:
            speech += f". I've spotted it there {top['times_seen']} times"
        if len(places) > 1:
            speech += f". I've also seen one at {places[1]['place']}"
        self.announce(speech + ".", force=True)
        return True

    def report_place_contents(self, place_query):
        """'what's in the kitchen' - the sighting store's inventory of a
        known place."""
        loc = self.spatial.find_location_by_name(place_query)
        if loc is None:
            self.announce(f"I don't know a place called {place_query} yet.", force=True)
            return
        objects = self.spatial.location_objects(loc["id"])
        if not objects:
            self.announce(f"I haven't recorded any objects at {loc['label']} yet.",
                          force=True)
            return
        names = ", ".join(o["label"] for o in objects[:6])
        self.announce(f"At {loc['label']} I've seen: {names}.", force=True)

    def report_person(self):
        """'who am I' / 'who do you see' - answered from person_memory's
        published identity, falling back to honest 'a face I don't know'."""
        with self.lock:
            person = dict(self.latest_person) if self.latest_person else None
        now = time.time()
        if (person and person.get("name")
                and now - person.get("received_at", 0) < PERSON_FRESH_SEC):
            self.announce(f"You're {person['name']}, of course.", force=True)
            return
        snap = self._snapshot() or {}
        face = snap.get("face", {})
        if face.get("detected") and not face.get("stale", True):
            self.announce("I can see someone, but I don't recognize the face. "
                          "Say remember me, I am, and then your name, and I'll "
                          "learn it.", force=True)
        else:
            self.announce("I don't see anyone right now.", force=True)

    def report_why(self):
        """Self-explanation from the decision journal: read back the
        most recent logged decisions WITH the reasons recorded at the
        moment they were made - evidence, not reconstruction."""
        rows = []
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT payload_json FROM events WHERE topic = ? ORDER BY id DESC LIMIT 3",
                    ("picarx/decision",)).fetchall()
            finally:
                conn.close()
        except Exception as e:
            print(f"Why query failed: {e}")
        decisions = []
        for (payload_json,) in rows:
            try:
                d = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            if d.get("reason"):
                decisions.append(d)
        if not decisions:
            self.announce("I haven't recorded any decisions yet.", force=True)
            return
        latest = decisions[0]
        speech = f"My last decision was {latest.get('kind', 'a choice')}: {latest['reason']}"
        if len(decisions) > 1:
            speech += f". Before that, {decisions[1].get('kind', 'a choice')}: {decisions[1]['reason']}"
        self.announce(speech + ".", force=True)

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
            if minutes < 120:
                parts.append(f"going back about {minutes:.0f} minutes")
            else:
                parts.append(f"going back about {minutes / 60.0:.0f} hours")
        parts.append(f"and I've been stopped by obstacles {vetoed} times recently")
        # Fold in what the robot has actually LEARNED, not just what it
        # logged: map coverage and the people it can recognize.
        place_count = self.spatial.location_count()
        if place_count:
            parts.append(f"I've mapped {place_count} place{'s' if place_count != 1 else ''}")
        people = person_memory.known_people()
        if people:
            parts.append(f"I can recognize {', '.join(people)}")
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
                "location": self._location_context(),
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
        # Only consult the coach about how to react to a new object while
        # we're actually exploring - a novelty maneuver is pointless (and
        # a wasted LLM/bandit call) when parked and awaiting commands.
        # Noticing/announcing still happens either way.
        if not self.explore_mode:
            return
        if self.watch_query_id is not None:
            return  # already waiting on a novelty query, don't pile on
        self._start_coach_query(situation="novel_object", urgent=False, label=label, extra={"object": obj})

    def _apply_coach_suggestion(self, payload):
        query_id = payload.get("query_id")
        if not query_id:
            return
        steps = self._normalize_steps(payload)
        rationale = payload.get("rationale")
        cached = payload.get("cached")
        confidence = payload.get("confidence")
        experimental = payload.get("experimental", False)
        now = time.time()

        if query_id == self.coach_query_id:
            self.coach_steps = steps
            self.coach_step_index = 0
            self.coach_step_until = now + steps[0]["duration"]
            self.coach_motion_max = None
            self.coach_action_started_at = now
            self.last_coach_query_id_used = query_id
            self.last_coach_situation_key = payload.get("situation_key")
            self.coach_query_id = None
            print(f"Coach suggestion (urgent, cached={cached}): {steps} - {rationale}")
            self.announce(self._coach_speech(steps, rationale, cached, confidence, experimental),
                          force=True)
        elif query_id == self.watch_query_id:
            # Novelty reactions stay simple: run the first step only.
            self.watch_coach_action = steps[0]["action"]
            self.watch_action_started_at = now
            self.watch_action_until = now + steps[0]["duration"]
            self.watch_query_id_used = query_id
            self.watch_situation_key = payload.get("situation_key")
            self.watch_query_id = None
            print(f"Coach suggestion (watch, cached={cached}): {steps[0]} - {rationale}")
            self.announce(self._coach_speech(steps, rationale, cached, confidence, experimental),
                          force=True)

    @staticmethod
    def _normalize_steps(payload):
        """Coach now sends a "steps" list; tolerate a legacy single
        "action"/"duration" payload too. Always returns a non-empty list
        of {"action","duration"} with sane durations."""
        raw = payload.get("steps")
        if not raw:
            action = payload.get("action") or {"direction": "stop"}
            raw = [{"action": action, "duration": payload.get("duration") or DEFAULT_COACH_DURATION}]
        steps = []
        for s in raw:
            action = s.get("action") or {"direction": "stop"}
            try:
                duration = float(s.get("duration") or DEFAULT_COACH_DURATION)
            except (TypeError, ValueError):
                duration = DEFAULT_COACH_DURATION
            steps.append({"action": action, "duration": duration})
        return steps or [{"action": {"direction": "stop"}, "duration": DEFAULT_COACH_DURATION}]

    @staticmethod
    def _coach_speech(steps, rationale, cached, confidence=None, experimental=False):
        # Spoken so the user can hear the LLM coach is actually engaged.
        # Names the source (fresh model call vs. remembered past advice),
        # is honest about how sure it is (confidence = the arm's actual
        # observed success rate; None = untested guess), and flags
        # deliberate experiments as such.
        if experimental:
            source = "Let me try an experiment - something I've never done here. I'll"
        elif cached and confidence is not None and confidence >= 0.75:
            source = "I know what works here. I'll"
        elif cached:
            source = "From memory, though I'm not certain, I'll"
        else:
            source = "I'm guessing, but the coach suggests I"
        verbs = {"backward": "back away", "forward": "go forward",
                 "stop": "hold still", "turn": "turn"}
        moves = [verbs.get((s.get("action") or {}).get("direction"), "move") for s in steps]
        # De-dupe consecutive identical verbs so "turn, turn" reads cleanly.
        seq = []
        for m in moves:
            if not seq or seq[-1] != m:
                seq.append(m)
        maneuver = ", then ".join(seq)
        if rationale:
            return f"{source} {maneuver}. {rationale}"
        return f"{source} {maneuver}."

    def _episode_moved(self, steps, motion_max):
        """Did the robot actually GO anywhere during a maneuver? "No veto"
        alone is not success: pushing against something the sensors can't
        see (table edge above the ultrasonic beam) never vetoes, so every
        useless maneuver got recorded as a win, the bandit reinforced it,
        and the robot proudly re-announced learned no-ops forever. Require
        visual evidence of motion whenever the maneuver contains any
        moving step. No vision signal at all -> benefit of the doubt."""
        if steps and all(s["action"].get("direction") == "stop" for s in steps):
            return True  # a pure hold-still maneuver is supposed to not move
        if motion_max is None:
            return True
        return motion_max >= STUCK_MOTION_THRESHOLD

    def _finish_coach_episode(self, now):
        with self.lock:
            episode_vetoes = [(t, code) for t, code in self.veto_events
                              if t > self.coach_action_started_at]
        veto_free = not episode_vetoes
        succeeded = veto_free and self._episode_moved(self.coach_steps, self.coach_motion_max)
        query_id = self.last_coach_query_id_used
        situation_key = self.last_coach_situation_key
        self.coach_steps = None
        # Straighten the wheels before resuming: a coach maneuver (often a
        # turn, or a reverse while the wheels were already turned) leaves
        # the steering angled, so plain forward cruise would arc right
        # back into the same object. Reset to straight first.
        self.publish_intent({"direction": "turn", "angle": 0}, priority=COACH_PRIORITY)
        self.state = "CRUISING"
        self.last_wander = now
        # The bare boolean isn't enough to correct from: say WHY it
        # failed (a veto, and of which kind, vs. grinding in place with
        # no visual motion) and how long the whole maneuver ran.
        self.bus.publish("picarx/coach/outcome", {
            "query_id": query_id,
            "situation_key": situation_key,
            "source": SOURCE_NAME,
            "success": succeeded,
            "vetoed": not veto_free,
            "veto_code": episode_vetoes[-1][1] if episode_vetoes else None,
            "motion_max": self.coach_motion_max,
            "duration": round(now - self.coach_action_started_at, 1),
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
            succeeded = not any(t > self.watch_action_started_at
                                for t, _code in self.veto_events)
        self.bus.publish("picarx/coach/outcome", {
            "query_id": self.watch_query_id_used,
            "situation_key": self.watch_situation_key,
            "source": SOURCE_NAME,
            "success": succeeded,
        })
        self.watch_coach_action = None

    def _handle_coaching_tick(self, now):
        if self.coach_steps is None:
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
            # Safe holding reflex while waiting: brief stop, ease back,
            # then hold stop. The back-off is bounded WELL under the
            # safety daemon's 2s continuous-reverse cap so the reflex
            # itself never generates reverse-limit vetoes.
            if now < self.coach_action_started_at + 0.3:
                self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
            elif now < self.coach_action_started_at + 1.8:
                self.publish_intent({"direction": "backward", "speed": 25}, priority=COACH_PRIORITY)
            else:
                self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
            return

        # Sample movement evidence while the maneuver runs (see
        # _episode_moved - this is what tells a real escape apart from
        # uselessly grinding against an unseen obstacle).
        snap = self._snapshot() or {}
        motion = (snap.get("objects") or {}).get("scene_motion")
        if motion is not None:
            self.coach_motion_max = motion if self.coach_motion_max is None \
                else max(self.coach_motion_max, motion)

        # Run the suggested step sequence back to back: hold each step
        # until its duration elapses, then advance to the next; finish
        # after the last. The arbiter/safety layer only ever sees the one
        # primitive we publish this tick, so multi-step maneuvers never
        # leak into the hardware-gate layers.
        while self.coach_step_index < len(self.coach_steps) and now >= self.coach_step_until:
            self.coach_step_index += 1
            if self.coach_step_index < len(self.coach_steps):
                self.coach_step_until = now + self.coach_steps[self.coach_step_index]["duration"]

        if self.coach_step_index >= len(self.coach_steps):
            self._finish_coach_episode(now)
            return

        self.publish_intent(self.coach_steps[self.coach_step_index]["action"],
                            priority=COACH_PRIORITY, ttl=0.6)

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
        self.coach_steps = None
        self.coach_step_index = 0
        self.coach_action_started_at = now
        self.announce("I keep running into something. Let me get some advice.", force=True)
        self.publish_intent({"direction": "stop"}, priority=COACH_PRIORITY)
        # Which sensor-level failure actually caused this? Only trust a
        # veto code from the recent past - a stale one from minutes ago
        # describes a different problem.
        with self.lock:
            failure_mode = self.last_veto_code if (now - self.last_veto_at) < 10.0 else None
        self._start_coach_query(
            situation="collision_loop", urgent=True,
            extra={"reason": reason, "stuck_pattern": stuck_pattern,
                   "failure_mode": failure_mode},
        )

    def _begin_evasion(self, reason, away_hint=None):
        now = time.time()
        self.forward_since = None
        self.evade_reason = reason   # journaled with the chosen angle at stage 1
        # Escape-side hint from the caller (signed angle, or None): set
        # fresh on EVERY entry so a stale hint from a previous vision
        # evasion can never steer an unrelated later escape.
        self.evade_away_hint = away_hint
        self.evasion_fail_events.append(now)
        _prune_older_than(self.evasion_fail_events, EVASION_FAIL_WINDOW, now)
        if len(self.evasion_fail_events) >= EVASION_FAIL_THRESHOLD:
            self._enter_collision_fail_state(f"evasion_loop:{reason}")
            return
        self.state = "EVADING"
        self.evade_stage = 0
        self.state_until = now + 0.25
        self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)

    # ---------- active hypothesis testing (generic framework) ----------

    def _start_hypothesis(self, task, announce, decision_reason):
        """Enter the single shared HYPOTHESIS state with `task` as the
        active probe: announce it, log why on the decision journal (same
        'hypothesis_probe' kind as before), and hold a stop so nothing
        stale drives us while the probe's own state machine takes over."""
        self.hypothesis = task
        self.state = "HYPOTHESIS"
        self.forward_since = None
        self.announce(announce)
        self.publish_decision(task.START_KIND, task.start_choice(), decision_reason)
        self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)

    def _maybe_start_sensor_probe(self, now, snap, distance):
        """Sensor-disagreement hypothesis: ultrasonic says obstacle but
        fresh vision sees an empty scene. Returns True if the probe
        started (tick consumed); False means fall through to evasion."""
        if now - self.last_probe_at < PROBE_COOLDOWN:
            return False
        if distance < PROBE_MIN_DISTANCE_CM:
            return False
        objects = (snap or {}).get("objects", {})
        vision_disagrees = (not objects.get("stale", True)
                            and not objects.get("close_object")
                            and not objects.get("items"))
        if not vision_disagrees:
            return False
        self.last_probe_at = now
        self._start_hypothesis(
            SensorDisagreementProbe(self, now, distance),
            announce="My distance sensor says something's there but I can't see it. Testing carefully.",
            decision_reason=("ultrasonic reports an obstacle but fresh vision sees nothing - "
                             "creeping forward slowly to find out which sensor is right"))
        return True

    def _maybe_start_veto_prone_probe(self, now):
        """Veto-prone-location hypothesis: the near path is clear, but the
        map says this place keeps vetoing us. Before cruising in normally,
        test whether it's still blocked. Returns True if the probe started.
        Fail-soft: no spatial map / unknown place -> never triggers."""
        if now - self.last_veto_prone_probe_at < VETO_PRONE_PROBE_COOLDOWN:
            return False
        loc = self._location_context()
        if not loc or loc.get("id") is None:
            return False
        location = self.spatial.get_location(loc["id"])
        if not location or location.get("veto_count", 0) < VETO_PRONE_THRESHOLD:
            return False
        veto_count = location["veto_count"]
        self.last_veto_prone_probe_at = now
        self._start_hypothesis(
            VetoProneLocationProbe(self, now, loc["id"], veto_count),
            announce="I've been blocked here before. Let me test carefully if it's still blocked.",
            decision_reason=(f"approaching {loc.get('label')}, which has vetoed me "
                             f"{veto_count} times - testing whether it is still blocked"))
        return True

    def _handle_hypothesis_tick(self, now):
        """Drive the active HypothesisTask one tick. When it resolves, run
        its follow-up (evade / resume cruising) and leave HYPOTHESIS."""
        task = self.hypothesis
        if task is None:
            self.state = "CRUISING"
            return
        if not task.run(now):
            self.hypothesis = None
            task.follow_up(now)

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

    def _enter_scanning(self, now, startup):
        """Begin a camera head sweep. startup=True is the full sweep on
        'explore' (announces what it sees); startup=False is the lighter,
        silent periodic glance done while cruising."""
        self.state = "SCANNING"
        self.scan_is_startup = startup
        self.scan_angles = SCAN_PAN_ANGLES if startup else SCAN_PAN_ANGLES_QUICK
        self.scan_dwell_sec = SCAN_DWELL_SEC if startup else SCAN_DWELL_QUICK_SEC
        self.scan_index = 0
        self.scan_sightings = []
        self.scan_dwell_until = now + self.scan_dwell_sec
        self.forward_since = None
        # Arm the timed steering reset so the FIRST cruise tick after this
        # scan straightens the wheels on a dedicated tick. The servo holds
        # whatever angle the previous behaviour (smooth avoidance, follow,
        # a wander arc) left behind, and driving off on a stale angle is
        # how "resume cruising" turns into an unintended circle.
        self.steering_active_until = now
        self.publish_look(self.scan_angles[0])

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
        self.scan_sightings.append({"pan": self.scan_angles[self.scan_index], "labels": labels})

        self.scan_index += 1
        if self.scan_index < len(self.scan_angles):
            self.publish_look(self.scan_angles[self.scan_index])
            self.scan_dwell_until = now + self.scan_dwell_sec
            return

        # Sweep complete - recenter, remember, publish, set escape bias, roll.
        # The forward ultrasonic reading rides along (the sensor is
        # body-fixed, so one reading covers the whole stationary sweep):
        # location_graph folds it into the place fingerprint so two
        # featureless scans can still tell a tight corner from open floor.
        self.publish_look(0, 0)
        scan_distance = None
        if snap.get("distance_cm") is not None and not snap.get("distance_stale", True):
            scan_distance = snap["distance_cm"]
        self.last_room_scan = {"scanned_at": now, "sightings": self.scan_sightings,
                               "distance_cm": scan_distance}
        self.bus.publish("picarx/exploration/room_scan", self.last_room_scan)
        self.preferred_escape_angle = self._escape_angle_from_scan()
        self.goal_bias_angle = self._goal_angle_from_scan()
        self.last_scan_at = now
        self.state = "CRUISING"
        self.last_wander = now

        if self.scan_is_startup:
            seen = sorted({label for s in self.scan_sightings for label in s["labels"]})
            if seen:
                self.announce(f"I looked around and I can see: {', '.join(seen)}. Off I go.", force=True)
            else:
                self.announce("I looked around but didn't recognize anything. Exploring anyway.", force=True)

    def _escape_angle_from_scan(self):
        """Turn toward whichever side had fewer objects in the last sweep.
        Camera-based (the head pans, so left/right sightings are real),
        which sidesteps the forward-fixed ultrasonic entirely. Returns a
        signed angle, or None if the scan was symmetric / empty (caller
        falls back to a random pick)."""
        left = sum(len(s["labels"]) for s in self.scan_sightings if s["pan"] < 0)
        right = sum(len(s["labels"]) for s in self.scan_sightings if s["pan"] > 0)
        if left == right:
            return None
        return -30 if left < right else 30

    def _goal_angle_from_scan(self):
        """If the last sweep saw any of the active goal's landmark
        labels, return a signed angle toward the side they appeared on
        (None when no goal, no match, or it was dead ahead). This is
        the only steering the goal system gets - purely a lean."""
        with self.lock:
            goal = dict(self.active_goal) if self.active_goal else None
        targets = set((goal or {}).get("target_labels") or [])
        if not targets:
            return None
        weighted = 0
        for s in self.scan_sightings:
            hits = targets.intersection(s["labels"])
            if hits and s["pan"]:
                weighted += (1 if s["pan"] > 0 else -1) * len(hits)
        if weighted == 0:
            return None
        return 25 if weighted > 0 else -25

    # ---------- exploration behavior ----------

    def explore_tick(self):
        now = time.time()

        with self.lock:
            _prune_older_than(self.veto_events, VETO_WINDOW, now, key=lambda e: e[0])
            veto_count = len(self.veto_events)

        if self.state == "SCANNING":
            self._handle_scanning_tick(now)
            return

        if self.state == "COACHING":
            self._handle_coaching_tick(now)
            return

        if self.state == "HYPOTHESIS":
            self._handle_hypothesis_tick(now)
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
        # Ackermann-correct: a car can only change heading while it is
        # MOVING with the wheels turned. The old sequence turned the
        # wheels while stationary (which does nothing on this chassis)
        # then drove off straight - so it returned to the same spot and
        # re-hit the same object. Now the wheels are pre-turned toward
        # the escape side and the robot REVERSES along that arc, so its
        # heading actually swings away before it drives on.
        if self.state == "EVADING":
            if now < self.state_until:
                # Hold the current stage's command (re-published so the
                # arbiter's per-source intent doesn't expire mid-stage).
                if self.evade_stage == 0:
                    self.publish_intent({"direction": "stop"}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 1:
                    # Wheels turning to the escape angle; hold still.
                    self.publish_intent({"direction": "turn", "angle": self.evade_angle}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 2:
                    # Reverse with wheels still turned -> arcs away.
                    self.publish_intent({"direction": "backward", "speed": 30}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 3:
                    # Straighten and ease forward onto the new heading.
                    self.publish_intent({"direction": "forward", "speed": 20}, priority=EVADE_PRIORITY)
                return
            else:
                self.evade_stage += 1
                if self.evade_stage == 1:
                    # Choose + command the escape steering angle, best
                    # information first: away from the side the triggering
                    # obstacle was actually SEEN on (evade_away_hint), else
                    # toward whichever side the last scan found clearer;
                    # coin flip only with no asymmetry to go on at all.
                    self.evade_angle = self.evade_away_hint
                    self.evade_away_hint = None
                    if self.evade_angle is None:
                        self.evade_angle = self.preferred_escape_angle
                    if self.evade_angle is None:
                        self.evade_angle = random.choice([-30, 30])
                    self.state_until = now + 0.3
                    # Journal the maneuver the moment its shape is decided:
                    # evasions are the robot's most common maneuver, and
                    # without this "why did you back up?" had no answer and
                    # reflection couldn't see evasion choices at all.
                    self.publish_decision(
                        "evade",
                        {"angle": self.evade_angle,
                         "trigger": getattr(self, "evade_reason", "unknown")},
                        f"escaping {getattr(self, 'evade_reason', 'an obstacle')} by "
                        f"reversing along a {'left' if self.evade_angle < 0 else 'right'} arc")
                    self.publish_intent({"direction": "turn", "angle": self.evade_angle}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 2:
                    # Reverse along the arc (wheels stay turned - backward
                    # only changes speed, not the held steering angle).
                    self.state_until = now + 1.3
                    self.publish_intent({"direction": "backward", "speed": 30}, priority=EVADE_PRIORITY)
                elif self.evade_stage == 3:
                    # Straighten wheels, then commit forward briefly so we
                    # actually leave on the new heading instead of arcing
                    # right back.
                    self.state_until = now + 0.6
                    self.publish_intent({"direction": "turn", "angle": 0}, priority=EVADE_PRIORITY)
                    self.publish_intent({"direction": "forward", "speed": 20}, priority=EVADE_PRIORITY)
                else:
                    self.publish_intent({"direction": "turn", "angle": 0}, priority=EVADE_PRIORITY)
                    self.state = "CRUISING"
                    self.last_wander = now
                return

        # --- Handle a vision-flagged obstacle (covers the ultrasonic's
        # blind spots - this is the fix for driving straight into things
        # the distance sensor never saw) ---
        # Cross-checked against the ultrasonic: a fresh, clearly-long
        # distance reading normally means a frame-filling detection is the
        # room itself (wall/sofa/floor), not a point-blank obstacle - see
        # VISION_OBSTACLE_ULTRASONIC_CLEAR_CM.
        #
        # BUT that cross-check only holds for obstacles the ultrasonic could
        # actually have seen. The sensor rides low on the front bumper; an
        # OVERHANG at head height (a counter lip, a table edge) sits ABOVE its
        # beam, so the beam passes underneath into clear air and reads long
        # even as the camera head is about to smack the object - the exact
        # "base drives under the counter, head hits the side" failure. So a
        # clear reading may only dismiss a NON-overhead obstacle; for an
        # overhead one we trust vision. To keep a genuinely distant high wall
        # from tripping it (both fill the upper frame), an overhead mass that
        # coincides with a clear-long reading must also be actively LOOMING
        # (growing) before we evade.
        if vision_obstacle is not None:
            ultrasonic_says_clear = (
                distance is not None and not distance_stale
                and distance > VISION_OBSTACLE_ULTRASONIC_CLEAR_CM
            )
            if vision_obstacle.get("overhead"):
                if not ultrasonic_says_clear or vision_obstacle.get("approaching"):
                    self.announce("Something's right at head height, backing away before I hit it.")
                    self._begin_evasion("overhead")
                    return
            elif not ultrasonic_says_clear:
                label = vision_obstacle.get("label", "something")
                if label == "something":
                    self.announce("Something's right in front of me, backing away.")
                else:
                    self.announce(f"A {label} is closing in, backing away.")
                # If we saw which side it's on, escape AWAY from that side
                # (same sign convention as _escape_angle_from_scan) instead
                # of leaving it to scan memory or a coin flip. Only trust a
                # clearly off-center sighting; near-center says nothing.
                away_hint = None
                frame_w = vision_obstacle.get("frame_width") or 0
                if frame_w > 0:
                    offset_frac = vision_obstacle.get("center_offset", 0) / (frame_w / 2.0)
                    if abs(offset_frac) > 0.1:
                        away_hint = -30 if offset_frac > 0 else 30
                self._begin_evasion("vision", away_hint=away_hint)
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
            if self._maybe_start_sensor_probe(now, snap, distance):
                return
            self.announce("Obstacle ahead, backing away.")
            self._begin_evasion("ultrasonic")
            return

        # --- Veto-prone location hypothesis ---
        # The near path is clear (obstacle checks above already returned),
        # but the map says this spot keeps stopping us. Before cruising in
        # normally, run the careful, safety-daemon-gated "is it still
        # blocked?" test instead of just driving forward.
        if self._maybe_start_veto_prone_probe(now):
            return

        # --- Periodic look-around while cruising ---
        # Path is clear here (obstacle/evasion checks above already
        # returned). Every so often, stop and glance side to side so
        # objects approached at an angle get noticed before a corner
        # clips them, and so the escape-direction bias stays current.
        if now - self.last_scan_at > CRUISE_SCAN_INTERVAL:
            self._enter_scanning(now, startup=False)
            return

        # --- Reactive steer-around (deterministic, vision-based) ---
        # The path isn't an emergency (the checks above already returned),
        # but something visible is looming off-center - bend the heading
        # away from it NOW and keep rolling, instead of driving straight
        # at it until the evasion reflex has to back out. The smooth
        # SteeringController turns the snapshot into a continuous
        # float-degree arc with curvature/proximity-scaled speed; the
        # discrete _steer_away_angle law below is the fail-soft fallback.
        # Either way this outranks/overrides wander steering while
        # active, and steering_active_until is kept refreshed so the
        # existing timed-reset block straightens the wheels automatically
        # once the object is cleared and this stops firing.
        if self.steering is not None:
            cmd = self.steering.compute_command(snap, now=now)
            if cmd["active"]:
                angle = cmd["steering_angle_deg"]
                if self.avoid_active_angle is None:
                    self.publish_decision(
                        "steer_around",
                        {"angle": round(angle, 1), "speed": round(cmd["speed"], 1)},
                        cmd["reason"])
                self.steering_active_until = now + AVOID_HOLD_SEC
                # One primitive per tick (see the __init__ note): steer
                # when the angle materially moved, never twice in a row.
                steer_tick = (not self._avoid_turn_last_tick
                              and (self.avoid_active_angle is None
                                   or abs(angle - self._avoid_sent_angle)
                                   >= AVOID_SEND_DEADBAND))
                self.avoid_active_angle = angle
                if steer_tick:
                    self._avoid_turn_last_tick = True
                    self._avoid_sent_angle = angle
                    self.publish_intent({"direction": "turn", "angle": angle})
                    return
                self._avoid_turn_last_tick = False
                if self._note_forward_and_check_stuck(now, snap):
                    return
                self.publish_intent({"direction": "forward", "speed": cmd["speed"]})
                return
            self._avoid_turn_last_tick = False
            self.avoid_active_angle = None
        else:
            avoid = _steer_away_angle(snap)
            if avoid is not None:
                angle = avoid["angle"]
                if self.avoid_active_angle is None:
                    self.publish_decision(
                        "steer_around", {"angle": angle},
                        f"steering around {', '.join(sorted(set(avoid['labels'])))} "
                        f"seen off-center ahead")
                if (self.avoid_active_angle is None
                        or abs(angle - self.avoid_active_angle) >= AVOID_RESEND_DELTA):
                    self.publish_intent({"direction": "turn", "angle": angle})
                    self.avoid_active_angle = angle
                self.steering_active_until = now + AVOID_HOLD_SEC
                if self._note_forward_and_check_stuck(now, snap):
                    return
                self.publish_intent({"direction": "forward", "speed": AVOID_SPEED})
                return
            self.avoid_active_angle = None

        # --- Handle Timed Steering Reset during standard wander ---
        if self.steering_active_until != 0 and now >= self.steering_active_until:
            self.publish_intent({"direction": "turn", "angle": 0})
            self.steering_active_until = 0
            if self.steering is not None:
                # Keep the smooth controller's internal angle mirror in
                # sync with the servo we just zeroed, so its next active
                # command slews from reality instead of a stale model.
                self.steering._angle = 0.0
            return

        # --- Handle Periodic Spontaneous Wandering ---
        if now - self.last_wander > self.wander_interval:
            angle, reason = self._choose_wander_angle()
            print(f"Wandering with angle: {angle} ({reason})")
            self.publish_decision("wander", {"angle": angle}, reason)
            self.publish_intent({"direction": "turn", "angle": angle})
            self.steering_active_until = now + 1.5
            self.wander_interval = random.uniform(5.0, 15.0)
            self.last_wander = now
            return

        # --- Standard Base Case ---
        if self._note_forward_and_check_stuck(now, snap):
            return
        self.publish_intent({"direction": "forward", "speed": 25})

    def _choose_wander_angle(self):
        """Pick the next wander steering angle, curiosity-aware when the
        spatial modules are up. Returns (angle, reason) - the reason is
        the honest explanation that goes into the decision journal."""
        # An active subgoal whose landmarks were sighted outranks
        # curiosity drift: we know which way progress is.
        with self.lock:
            goal = dict(self.active_goal) if self.active_goal else None
            goal_angle = self.goal_bias_angle
        if goal is not None and goal_angle is not None and random.random() < CURIOSITY_BIAS_PROB:
            side = 1 if goal_angle > 0 else -1
            angle = side * random.randint(12, 25)
            return angle, (f"leaning toward {goal.get('label')} - "
                           f"its landmarks were sighted on that side")

        loc = self._location_context()
        settled = (loc is not None and loc.get("uncertainty") is not None
                   and loc["uncertainty"] < CURIOSITY_SETTLED_SCORE)
        if settled and self.preferred_escape_angle is not None \
                and random.random() < CURIOSITY_BIAS_PROB:
            # Well-understood spot: lean toward the side the last sweep
            # found clearer, to drift somewhere with more left to learn.
            side = 1 if self.preferred_escape_angle > 0 else -1
            angle = side * random.randint(8, 25)
            return angle, (f"{loc['label']} is already well understood "
                           f"(uncertainty {loc['uncertainty']:.2f}), drifting toward the clearer side")
        angle = random.randint(-25, 25)
        if loc is None or loc.get("uncertainty") is None:
            return angle, "no spatial map available, wandering at random"
        if settled:
            return angle, "well-known area but keeping some randomness"
        return angle, (f"still learning {loc['label']} "
                       f"(uncertainty {loc['uncertainty']:.2f}), poking around it at random")

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
        self.bus.subscribe("picarx/exploration/location_change", self.on_location_change)
        self.bus.subscribe("picarx/exploration/uncertainty_map", self.on_uncertainty_map)
        self.bus.subscribe("picarx/exploration/active_goal", self.on_active_goal)
        self.bus.subscribe("picarx/vision/person", self.on_person)
        self.bus.subscribe("picarx/tools/follow/state", self.on_follow_state)
        self.bus.subscribe("picarx/rc/mode", self.on_rc_mode)

        print("Field Agent active. Say 'explore', 'stop', 'status', 'objects', 'history', or 'battery'.")
        self.announce("Field agent online and standing by. Say explore when you want me to drive.", force=True)

        period = 1.0 / EXPLORE_TICK_HZ
        while True:
            self._perception_tick()
            if self.explore_mode:
                self.explore_tick()
            elif self.rc_active:
                self._rc_observer_tick(time.time())
            time.sleep(period)


if __name__ == "__main__":
    FieldAgent().run()
