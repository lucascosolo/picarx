#!/usr/bin/env python3
# layer_b/modules/expressions.py
"""
Expressions (Layer B) - the robot's ambient personality.

Everything else in Layer B is task-driven: field_agent explores, curiosity
asks about ambiguous sightings, companion converses when spoken to, coach
picks maneuvers. Between those, a real creature is never fully inert - it
glances around, cocks its head at something new, mutters to itself, and now
and then commits something worth keeping to memory. This module supplies that
connective tissue and nothing more.

It NEVER drives the wheels and never competes for the movement channel. It
dispatches only a few gentle "tools", all fail-soft and all heavily throttled
because they share ONE speaker and ONE camera head with the whole system:

  - speak         -> picarx/audio/speak   (plain, untagged speech)
  - look around   -> picarx/intent/look   (a short pan sweep, recentres)
  - curious tilt  -> picarx/intent/look   (a head cock toward something new;
                                           for an IDENTIFIED subject it then
                                           holds and follows it briefly instead
                                           of snapping straight back to centre)
  - emote gesture -> picarx/intent/look   (a mood made legible WITHOUT words:
                                           a curious cock, a satisfied nod, a
                                           frustrated shake - see AffectState)
  - remember      -> picarx/memory/note   (reflection persists it as a fact)

Three ways an expression fires:

  * CONTEXT-BASED - a reaction to picarx/state/world: a familiar person
    arrives (greet + turn the head toward them + remember the meeting), a
    confidently-new object appears (a curious head-tilt + a remark + a note so
    the moment is remembered), something looms close while the robot sits idle
    (a small startled remark). It stays out of curiosity.py's lane: genuinely
    AMBIGUOUS sightings belong to the question-asker, so expressions reacts
    only to confident, novel labels.

  * INTERNAL-STATE / AFFECT - the robot has moods (curiosity, frustration,
    satisfaction) that build from its OWN experience on the bus: it gets
    curious at novelty (a new place/object), frustrated when it keeps getting
    vetoed/stuck in a loop, satisfied when something finally works (a spot it
    probed turns out clear, a goal is reached, it learns a label, a practice
    run pays off). Those levels decay over time; the strongest past a threshold
    is expressed as a head gesture (occasionally with a short spoken tag), so
    the robot's internal state is legible even when it says nothing. See
    AffectState and _maybe_emote.

  * RANDOM / AMBIENT - when nothing has happened for a while, an occasional
    self-directed musing or an idle look-around drawn from a small repertoire.
    This is the "alive when idle" behaviour.

Deference is the whole reason it can share the hardware safely. Expressions
stands completely down while the robot is busy - moving, being spoken to (or
having just spoken), driven by a human in RC mode, in low power, or while
another module is actively using the camera head. It holds one global cooldown
between expressions and does one thing at a time. Disable it in
module_registry.json and the robot loses only its idle charm; every task
behaviour is untouched.

The decision helpers (is_busy / choose_context_acts / pick_idle_acts) are pure
- no hardware, no bus - and unit-tested off-robot; the class is just their
throttled dispatcher.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus

import random
import threading
import time

WORLD_TOPIC = "picarx/state/world"
HEARD_TOPIC = "picarx/audio/heard"
SPEAK_TOPIC = "picarx/audio/speak"
LOOK_TOPIC = "picarx/intent/look"
NOTE_TOPIC = "picarx/memory/note"
RC_MODE_TOPIC = "picarx/rc/mode"
SOURCE_NAME = "expressions"

# --- Throttling / deference (this shares one speaker + one head) ---
EXPRESSION_COOLDOWN = 45.0     # min seconds between any two expressions
IDLE_BEFORE_AMBIENT = 40.0     # seconds of nothing before a random musing fires
IDLE_TICK_SEC = 5.0            # how often the idle loop considers an ambient act
IDLE_EXPRESS_CHANCE = 0.25     # probability per idle tick once eligible
SPOKE_QUIET_SEC = 8.0          # our own recent speech still counts as "talking"
HEARD_QUIET_SEC = 12.0         # a human spoke recently -> a conversation is live
FOREIGN_LOOK_QUIET_SEC = 6.0   # don't grab the head if someone else just moved it
MOVING_FRESH_SEC = 2.0         # a drive action this recent means we're moving

# --- Context reaction gates ---
NOVEL_OBJECT_CONF = 0.75       # only react to CONFIDENT sightings...
NOVEL_OBJECT_TTL = 600.0       # ...and to a given label at most this often
PERSON_GREET_TTL = 900.0       # re-greet the same person at most this often

# --- Head gesture geometry (clamped again by the safety daemon) ---
SCAN_TILT = 0
LOOK_SWEEP_PANS = (-45, 0, 45, 0)   # a gentle look-around, ending centred
LOOK_STEP_SEC = 0.6                 # dwell between sweep positions
CURIOUS_PAN = 25                    # how far to cock the head toward a subject
CURIOUS_TILT = 20                   # ...and up, the attentive "huh?" angle
CURIOUS_HOLD_SEC = 2.0              # hold the expressive cock before settling into a gaze

# --- Gaze hold (keep watching an identified subject) ---
# After the cock, an IDENTIFIED subject (a greeted person, a confidently
# labelled object) is FOLLOWED for a short window rather than dropped by an
# instant recentre - a hard snap to centre routinely threw the thing back out
# of the narrow frame and lost track of it. The hold re-aims the head from each
# fresh world snapshot so it tracks the subject as either of them moves, but it
# is strictly bounded and deferential: it releases (recentres) the instant the
# subject is lost, after GAZE_HOLD_STEPS re-aims, or if another module takes the
# head / the robot moves / a human takes the wheel. A startled glance at a close
# blob, or an idle look at nothing, still just recentres as before.
GAZE_HOLD_STEPS = 10               # max re-aims before releasing the head
GAZE_STEP_SEC = 0.5                # dwell between gaze re-aims
GAZE_TILT = 0                      # look level at the subject while following (pan-only tracking)
GAZE_DEADBAND_FRAC = 0.12          # don't twitch the head for a near-centred subject
GAZE_PAN_GAIN_DEG = 28.0           # deg of pan per unit of normalised offset (~camera half-FOV)
GAZE_MAX_STEP_DEG = 18             # cap one re-aim so the head eases rather than snaps
GAZE_PAN_LIMIT = 65                # soft pan bound (the safety daemon clamps again at +-80)

# Ambient musings: generic, non-factual, never claim anything about the world
# (so they can't mislead or pollute memory). One is picked at random.
IDLE_MUSINGS = (
    "Hm. All quiet.",
    "I wonder what's around the corner.",
    "Just taking it all in.",
    "Nothing much going on. That's alright.",
    "I could go exploring.",
    "It's peaceful right now.",
    "I like it here.",
)
# Said while cocking the head at nothing in particular.
IDLE_CURIOUS_MUSINGS = (
    "Did I hear something?",
    "Hm, what was that?",
    "Something caught my eye.",
)
GREETINGS = (
    "Oh, hello {name}.",
    "Hi {name}, good to see you.",
    "There you are, {name}.",
    "Hello again, {name}.",
)
OBJECT_REMARKS = (
    "Oh, a {label}.",
    "Huh, a {label}. Interesting.",
    "I see a {label} over there.",
    "A {label}. Noted.",
)
CLOSE_REMARKS = (
    "Whoa, that's close.",
    "Oh, hello there. That's right in my face.",
    "Something's right in front of me.",
)

# --- Affect: expressive behaviour tied to INTERNAL state ---
# The robot has moods that build from its own recent experience and make its
# internal state legible WITHOUT words: a curious head-cock at novelty, a
# frustrated head-shake when it keeps getting stuck, a satisfied nod when
# something finally works. Moods are scalar levels in [0,1] that DECAY over
# time; the strongest one past a threshold is expressed, then consumed so it
# doesn't loop. Same deference/cooldown/head-free gates as everything else.
AFFECT_HALF_LIFE = 90.0            # seconds for a mood to fade to half
AFFECT_EXPRESS_THRESHOLD = 0.4     # a mood must reach this to be worth expressing
AFFECT_CONSUME_FACTOR = 0.25       # fraction of a mood left after expressing it
EMOTE_TAG_CHANCE = 0.5             # chance an emote also gets a short spoken tag

# What each internal-state event does to which mood, and by how much. A single
# veto barely registers; a burst of them (the actual "stuck in a loop" signal)
# accumulates into real frustration before it decays away.
AFFECT_NUDGES = {
    "veto":           ("frustration", 0.22),   # a safety veto: something's in the way
    "still_blocked":  ("frustration", 0.40),   # a probed spot is STILL blocked
    "real_obstacle":  ("frustration", 0.25),
    "goal_abandoned": ("frustration", 0.35),
    "path_cleared":   ("satisfaction", 0.50),  # phantom reading / spot turned out clear
    "goal_reached":   ("satisfaction", 0.80),
    "learned_label":  ("satisfaction", 0.40),
    "trained":        ("satisfaction", 0.70),
    "greeted":        ("satisfaction", 0.30),
    "new_place":      ("curiosity", 0.55),
    "novel_object":   ("curiosity", 0.35),
}

# Legible-without-words head gestures per mood: (pan, tilt) step sequences that
# always end centred - a cock for curiosity, a nod for satisfaction, a shake for
# frustration. The safety daemon clamps these angles again.
EMOTE_GESTURES = {
    "curiosity":    [(CURIOUS_PAN, CURIOUS_TILT), (0, 0)],
    "satisfaction": [(0, -12), (0, 10), (0, -6), (0, 0)],
    "frustration":  [(-22, 0), (22, 0), (-14, 0), (0, 0)],
}
EMOTE_DWELL = {"curiosity": 1.2, "satisfaction": 0.3, "frustration": 0.3}
EMOTE_TAGS = {
    "curiosity":    ("Hm?", "Ooh.", "What's this?"),
    "satisfaction": ("There we go.", "Nice.", "Got it."),
    "frustration":  ("Hmph.", "Ugh.", "Not again."),
}


# ---------- pure decision helpers (no bus, no hardware) ----------

def _is_moving(world, now):
    """True if the robot is actively driving (not merely a stale/stopped
    last_action). The arbiter re-sends any non-stop action every tick, so a
    fresh forward/backward/turn means real motion; a stale one does not."""
    la = (world or {}).get("last_action") or {}
    action = la.get("action") or {}
    updated = la.get("updated_at")
    if updated is None or (now - updated) > MOVING_FRESH_SEC:
        return False
    direction = action.get("direction")
    if direction in ("forward", "backward"):
        return True
    return direction == "turn" and bool(action.get("angle"))


def is_busy(world, now, rc_active, last_speak_at):
    """Whether expressions should stay completely quiet right now. Any one of:
    a human has the wheel, low/critical battery (conserve), the robot spoke or
    a human spoke very recently (a conversation is live - don't talk over it),
    or the robot is moving."""
    if rc_active:
        return True
    battery = (world or {}).get("battery") or {}
    if battery.get("low") or battery.get("critical"):
        return True
    if now - last_speak_at < SPOKE_QUIET_SEC:
        return True
    heard = (world or {}).get("last_heard") or {}
    heard_at = heard.get("updated_at")
    if heard_at is not None and not heard.get("stale") \
            and (now - heard_at) < HEARD_QUIET_SEC:
        return True
    return _is_moving(world, now)


def _pan_dir(offset):
    """A subject's frame_center_offset -> which way to cock the head: +1 right,
    -1 left, 0 roughly centred (don't bother turning for a tiny offset)."""
    if offset is None:
        return 0
    if offset > 40:
        return 1
    if offset < -40:
        return -1
    return 0


def _subject_offset(world, track):
    """Where the tracked subject sits in the current frame, as
    (center_offset_px, frame_width), or None if it can't be located right now
    (gone, stale, or no position reported yet - the caller treats that as
    "lost" and releases the head).

    `track` is ("person", name) or ("object", label). A person is located by
    the face box when the camera has a fresh one, falling back to a detected
    "person"-labelled object; an object is located by its label."""
    world = world or {}
    kind, _key = track
    if kind == "person":
        face = world.get("face") or {}
        if face.get("detected") and not face.get("stale") \
                and face.get("frame_center_offset") is not None:
            return face["frame_center_offset"], face.get("frame_width")
        kind, _key = "object", "person"   # fall back to a person-shaped detection
    if kind == "object":
        objects = world.get("objects") or {}
        if objects.get("stale"):
            return None
        for obj in objects.get("items") or []:
            if obj.get("label") == _key and obj.get("center_offset") is not None:
                return obj["center_offset"], obj.get("frame_width")
    return None


def _aim_pan(current_pan, offset_px, frame_width):
    """Head pan (deg) that re-centres a subject currently `offset_px` right(+)/
    left(-) of frame centre. Proportional with a dead-band (so a centred subject
    doesn't jitter the head) and a per-step cap (so the head eases toward it
    rather than snapping); clamped to a soft range - the safety daemon clamps
    again to the servo's physical limits."""
    if not frame_width:
        return current_pan                       # no scale -> hold rather than guess
    frac = offset_px / (frame_width / 2.0)        # -1 (left edge) .. +1 (right edge)
    if abs(frac) < GAZE_DEADBAND_FRAC:
        return current_pan
    delta = max(-GAZE_MAX_STEP_DEG, min(GAZE_MAX_STEP_DEG, frac * GAZE_PAN_GAIN_DEG))
    return int(max(-GAZE_PAN_LIMIT, min(GAZE_PAN_LIMIT, current_pan + delta)))


def _first_novel_object(items, reacted_objects, now):
    """The first confident, UNAMBIGUOUS object whose label we haven't reacted
    to recently. Ambiguous sightings (alt_label set) are deliberately skipped -
    those are curiosity.py's job to ask about, not ours to editorialize."""
    for obj in items or []:
        label = obj.get("label")
        if not label or obj.get("alt_label"):
            continue
        conf = obj.get("confidence")
        if conf is None or conf < NOVEL_OBJECT_CONF:
            continue
        last = reacted_objects.get(label)
        if last is not None and (now - last) < NOVEL_OBJECT_TTL:
            continue
        return obj
    return None


def choose_context_acts(world, now, reacted_objects, greeted_people, rng):
    """Pick at most one context-driven reaction to the current world snapshot.

    Returns (acts, updates): `acts` is a list of tool dispatches (see the
    module docstring), `updates` records what the caller should remember it
    reacted to (so the same person/object isn't reacted to again within its
    TTL). Pure: it reads the passed-in state and never mutates it.

    Priority: greet a returning person > remark on a confidently-new object >
    a small startled note when something looms close while idle.
    """
    world = world or {}
    acts, updates = [], {}

    # 1) A familiar person is in view -> greet them, turn toward them, and
    #    remember the meeting (once per session; the store dedups across runs).
    person = world.get("person") or {}
    name = person.get("name")
    if name and not person.get("stale"):
        last = greeted_people.get(name)
        if last is None or (now - last) >= PERSON_GREET_TTL:
            face = world.get("face") or {}
            pan_dir = _pan_dir(face.get("frame_center_offset")) if not face.get("stale") else 0
            acts.append({"tool": "curious_tilt", "pan_dir": pan_dir,
                         "track": ("person", name)})
            acts.append({"tool": "speak", "text": rng.choice(GREETINGS).format(name=name)})
            if last is None:   # first greeting this session -> worth keeping
                acts.append({"tool": "remember", "subject": name,
                             "fact": f"I greeted {name} when they came into view",
                             "confidence": 0.6})
            updates["greeted"] = name
            return acts, updates

    objects = world.get("objects") or {}
    items = objects.get("items") or []
    stale_objects = objects.get("stale")

    # 2) A confident, novel object -> a curious head-cock toward it, a remark,
    #    and a note (building a durable little inventory of what it has seen).
    if not stale_objects:
        obj = _first_novel_object(items, reacted_objects, now)
        if obj is not None:
            label = obj["label"]
            acts.append({"tool": "curious_tilt",
                         "pan_dir": _pan_dir(obj.get("center_offset")),
                         "track": ("object", label)})
            acts.append({"tool": "speak",
                         "text": rng.choice(OBJECT_REMARKS).format(label=label)})
            acts.append({"tool": "remember", "subject": label,
                         "fact": f"I have seen a {label}", "confidence": 0.55})
            updates["reacted_object"] = label
            return acts, updates

        # 3) Something is filling the frame while we sit idle -> a small
        #    startled remark and a look at it. (Evasion is field_agent's job;
        #    is_busy already ruled out that we're moving.)
        if objects.get("close_object"):
            acts.append({"tool": "curious_tilt", "pan_dir": 0})
            acts.append({"tool": "speak", "text": rng.choice(CLOSE_REMARKS)})
            updates["close_reacted"] = True
            return acts, updates

    return acts, updates


def pick_idle_acts(rng):
    """A random ambient expression for a quiet moment: a musing, a look-around,
    or a curious look paired with a quiet remark. Pure - the caller gates it on
    idleness/cooldown and supplies the rng."""
    roll = rng.random()
    if roll < 0.4:
        return [{"tool": "speak", "text": rng.choice(IDLE_MUSINGS)}]
    if roll < 0.7:
        return [{"tool": "look_around"}]
    return [{"tool": "curious_tilt", "pan_dir": rng.choice((-1, 1))},
            {"tool": "speak", "text": rng.choice(IDLE_CURIOUS_MUSINGS)}]


class AffectState:
    """The robot's current mood as decaying scalar levels per feeling. Pure and
    hardware-free: the Expressions dispatcher owns the clock and the lock and
    calls in. Levels rise on internal-state events (nudge) and fade with time
    (decay), so a mood reflects RECENT experience, not all of history."""

    MOODS = ("curiosity", "frustration", "satisfaction")

    def __init__(self, half_life=AFFECT_HALF_LIFE):
        self.half_life = half_life
        self.levels = {m: 0.0 for m in self.MOODS}
        self.updated_at = None

    def _decay_to(self, now):
        if self.updated_at is None:
            self.updated_at = now
            return
        dt = now - self.updated_at
        if dt > 0:
            factor = 0.5 ** (dt / self.half_life)
            for m in self.levels:
                self.levels[m] *= factor
            self.updated_at = now

    def nudge(self, mood, amount, now):
        """Raise a mood by `amount` (clamped to 1.0), after decaying to `now`."""
        if mood not in self.levels:
            return
        self._decay_to(now)
        self.levels[mood] = min(1.0, self.levels[mood] + amount)

    def dominant(self, now, threshold=AFFECT_EXPRESS_THRESHOLD):
        """The strongest current mood past `threshold` as (mood, level), else
        (None, 0.0). Ties break by MOODS order."""
        self._decay_to(now)
        mood = max(self.MOODS, key=lambda m: self.levels[m])
        level = self.levels[mood]
        return (mood, level) if level >= threshold else (None, 0.0)

    def consume(self, mood, now, factor=AFFECT_CONSUME_FACTOR):
        """Damp a mood after it's been expressed, so it doesn't fire again on
        the very next tick but a still-elevated feeling can re-surface."""
        self._decay_to(now)
        if mood in self.levels:
            self.levels[mood] *= factor


def pick_emote_acts(mood, rng):
    """A legible expression of `mood`: the head gesture, and - sometimes - a
    short spoken tag. Pure; the caller gates it on deference and supplies rng."""
    acts = [{"tool": "emote_gesture", "mood": mood}]
    if rng.random() < EMOTE_TAG_CHANCE:
        tags = EMOTE_TAGS.get(mood)
        if tags:
            acts.append({"tool": "speak", "text": rng.choice(tags)})
    return acts


class Expressions:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.rng = random.Random()
        self.latest_world = None
        self.rc_active = False
        self.last_expression_at = 0.0
        self.last_activity_at = time.time()   # last heard/moved/expressed
        self.last_speak_at = 0.0              # last speech on the bus (any source)
        self.last_foreign_look_at = 0.0       # last head move by another module
        self.reacted_objects = {}             # label -> when we last remarked
        self.greeted_people = {}              # name  -> when we last greeted
        self.affect = AffectState()           # curiosity/frustration/satisfaction
        # Seams so head gestures (which sleep between servo steps) run in a
        # background thread on-robot but synchronously and instantly in tests.
        self._spawn = lambda fn: threading.Thread(target=fn, daemon=True).start()
        self._sleep = time.sleep

    # ---------- inbound bus state ----------

    def on_world(self, payload):
        now = time.time()
        with self.lock:
            self.latest_world = payload
            rc_active = self.rc_active
            last_speak_at = self.last_speak_at
            if _is_moving(payload, now):
                self.last_activity_at = now
        if is_busy(payload, now, rc_active, last_speak_at):
            return
        with self.lock:
            if now - self.last_expression_at < EXPRESSION_COOLDOWN:
                return
            reacted = dict(self.reacted_objects)
            greeted = dict(self.greeted_people)
        acts, updates = choose_context_acts(payload, now, reacted, greeted, self.rng)
        # Perceiving something novel or a friend nudges mood too, so the affect
        # layer feels curious/pleased even when the context reaction itself is
        # throttled off.
        if updates.get("reacted_object"):
            self._nudge("novel_object", now)
        if updates.get("greeted"):
            self._nudge("greeted", now)
        if acts:
            self._dispatch(acts, now, updates)

    # ---------- inbound: internal-state signals feed the affect layer ----------

    def _nudge(self, kind, now=None):
        """Map an internal-state event to a mood bump (see AFFECT_NUDGES)."""
        spec = AFFECT_NUDGES.get(kind)
        if not spec:
            return
        mood, amount = spec
        with self.lock:
            self.affect.nudge(mood, amount, now or time.time())

    def on_hypothesis(self, payload):
        """A physical hypothesis resolved: relief when a spot turns out clear,
        a touch of frustration when it's confirmed blocked."""
        kind = {"phantom_reading": "path_cleared", "maybe_clear": "path_cleared",
                "still_blocked": "still_blocked",
                "real_obstacle": "real_obstacle"}.get(payload.get("resolution"))
        if kind:
            self._nudge(kind)

    def on_action_result(self, payload):
        """A safety-daemon veto: something got in the way. A burst of these
        (the real 'stuck in a loop' signal) accumulates into frustration."""
        if (payload.get("result") or {}).get("status") == "vetoed":
            self._nudge("veto")

    def on_label(self, payload):
        """Learned what something is - a small satisfaction, and it keeps the
        curiosity warm."""
        now = time.time()
        with self.lock:
            self.affect.nudge("satisfaction", 0.4, now)
            self.affect.nudge("curiosity", 0.2, now)

    def on_location_change(self, payload):
        if payload.get("is_new"):
            self._nudge("new_place")

    def on_goal_progress(self, payload):
        kind = {"reached": "goal_reached",
                "abandoned": "goal_abandoned"}.get(payload.get("status"))
        if kind:
            self._nudge(kind)

    def on_self_trainer_status(self, payload):
        if payload.get("state") == "published":
            self._nudge("trained")

    def on_heard(self, payload):
        # A human spoke: a conversation is live. Mark activity so the ambient
        # timer resets; is_busy() keeps us quiet for HEARD_QUIET_SEC after.
        if not (payload.get("text") or "").strip():
            return
        with self.lock:
            self.last_activity_at = time.time()

    def on_speak(self, payload):
        with self.lock:
            self.last_speak_at = time.time()

    def on_look(self, payload):
        # Another module is driving the camera head; back off head gestures so
        # we don't yank it out from under an exploration sweep.
        if payload.get("source") == SOURCE_NAME:
            return
        with self.lock:
            self.last_foreign_look_at = time.time()

    def on_rc_mode(self, payload):
        with self.lock:
            self.rc_active = bool(payload.get("active"))

    # ---------- ambient (random) path ----------

    def _maybe_idle_express(self):
        now = time.time()
        with self.lock:
            world = self.latest_world
            rc_active = self.rc_active
            last_speak_at = self.last_speak_at
            idle_for = now - self.last_activity_at
            cooled = (now - self.last_expression_at) >= EXPRESSION_COOLDOWN
        if idle_for < IDLE_BEFORE_AMBIENT or not cooled:
            return
        if is_busy(world, now, rc_active, last_speak_at):
            return
        if self.rng.random() >= IDLE_EXPRESS_CHANCE:
            return
        self._dispatch(pick_idle_acts(self.rng), now, {})

    # ---------- affect (internal-state) path ----------

    def _maybe_emote(self):
        """Express the strongest current mood as a legible gesture, if one is
        past threshold and deference allows. Checked before the ambient path so
        an actual feeling wins over a generic musing; consumes the mood so it
        doesn't repeat every tick."""
        now = time.time()
        with self.lock:
            world = self.latest_world
            rc_active = self.rc_active
            last_speak_at = self.last_speak_at
            cooled = (now - self.last_expression_at) >= EXPRESSION_COOLDOWN
            head_free = (now - self.last_foreign_look_at) >= FOREIGN_LOOK_QUIET_SEC
            mood, _level = self.affect.dominant(now)
        # A mood expressed WITHOUT the head gesture isn't legible, so wait for
        # the head to be free rather than blurt a bare tag over a busy sweep.
        if mood is None or not cooled or not head_free:
            return
        if is_busy(world, now, rc_active, last_speak_at):
            return
        with self.lock:
            self.affect.consume(mood, now)
        self._dispatch(pick_emote_acts(mood, self.rng), now, {})

    # ---------- dispatch ----------

    def _dispatch(self, acts, now, updates):
        with self.lock:
            self.last_expression_at = now
            self.last_activity_at = now
            if "greeted" in updates:
                self.greeted_people[updates["greeted"]] = now
            if "reacted_object" in updates:
                self.reacted_objects[updates["reacted_object"]] = now
            head_free = (now - self.last_foreign_look_at) >= FOREIGN_LOOK_QUIET_SEC
        for act in acts:
            tool = act.get("tool")
            if tool == "speak":
                self._speak(act["text"])
            elif tool == "remember":
                self._remember(act["subject"], act["fact"], act.get("confidence", 0.5))
            elif tool in ("look_around", "curious_tilt", "emote_gesture") and head_free:
                if tool == "look_around":
                    self._look_around()
                elif tool == "curious_tilt":
                    self._curious_tilt(act.get("pan_dir", 0), act.get("track"))
                else:
                    self._emote_gesture(act["mood"])

    def _speak(self, text):
        self.bus.publish(SPEAK_TOPIC, {"text": text, "ts": time.time()})
        print(f"Expressions: say '{text}'")

    def _remember(self, subject, fact, confidence):
        self.bus.publish(NOTE_TOPIC, {
            "subject": subject, "fact": fact, "confidence": confidence,
            "source": SOURCE_NAME, "ts": time.time()})
        print(f"Expressions: remember [{subject}] {fact}")

    def _publish_look(self, pan, tilt):
        self.bus.publish(LOOK_TOPIC, {
            "source": SOURCE_NAME,
            "action": {"direction": "look", "pan": pan, "tilt": tilt},
        })

    def _look_around(self):
        self._spawn(lambda: self._sweep_worker(LOOK_SWEEP_PANS, SCAN_TILT))

    def _curious_tilt(self, pan_dir, track=None):
        pan = CURIOUS_PAN * pan_dir
        self._spawn(lambda: self._curious_worker(pan, CURIOUS_TILT, track))

    def _emote_gesture(self, mood):
        self._spawn(lambda: self._emote_worker(mood))

    def _emote_worker(self, mood):
        """Play a mood's head-gesture sequence, ending centred. Legible without
        words: a cock, a nod, a shake."""
        seq = EMOTE_GESTURES.get(mood) or [(0, 0)]
        dwell = EMOTE_DWELL.get(mood, LOOK_STEP_SEC)
        for pan, tilt in seq:
            self._publish_look(pan, tilt)
            self._sleep(dwell)

    def _sweep_worker(self, pans, tilt):
        for pan in pans:
            self._publish_look(pan, tilt)
            self._sleep(LOOK_STEP_SEC)

    def _curious_worker(self, pan, tilt, track=None):
        self._publish_look(pan, tilt)       # the expressive "huh?" cock
        self._sleep(CURIOUS_HOLD_SEC)
        if track is not None:
            self._hold_gaze(pan, track)     # then keep watching an identified subject
        self._publish_look(0, 0)            # release: always recentre when done

    def _hold_gaze(self, pan, track):
        """Follow an identified subject for a bounded window after the cock, so a
        hard recentre doesn't throw it out of the narrow frame. Re-aims from each
        fresh world snapshot; stops the moment the subject is lost, another module
        takes the head, or the robot starts moving / a human takes over. The
        caller always recentres afterwards."""
        for _ in range(GAZE_HOLD_STEPS):
            if self._gaze_interrupted():
                return
            loc = _subject_offset(self._world_snapshot(), track)
            if loc is None:
                return                       # lost it -> stop holding
            pan = _aim_pan(pan, loc[0], loc[1])
            self._publish_look(pan, GAZE_TILT)
            self._sleep(GAZE_STEP_SEC)

    def _world_snapshot(self):
        with self.lock:
            return self.latest_world

    def _gaze_interrupted(self):
        """True if the gaze must yield the head right now: a human took the
        wheel, another module moved the head, or the robot began driving."""
        now = time.time()
        with self.lock:
            rc = self.rc_active
            foreign = (now - self.last_foreign_look_at) < FOREIGN_LOOK_QUIET_SEC
            world = self.latest_world
        return rc or foreign or _is_moving(world, now)

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(WORLD_TOPIC, self.on_world)
        self.bus.subscribe(HEARD_TOPIC, self.on_heard)
        self.bus.subscribe(SPEAK_TOPIC, self.on_speak)
        self.bus.subscribe(LOOK_TOPIC, self.on_look)
        self.bus.subscribe(RC_MODE_TOPIC, self.on_rc_mode)
        # Internal-state signals that drive the affect layer (all optional).
        self.bus.subscribe("picarx/exploration/hypothesis", self.on_hypothesis)
        self.bus.subscribe("picarx/action/result", self.on_action_result)
        self.bus.subscribe("picarx/perception/label", self.on_label)
        self.bus.subscribe("picarx/exploration/location_change", self.on_location_change)
        self.bus.subscribe("picarx/exploration/goal_progress", self.on_goal_progress)
        self.bus.subscribe("picarx/self_trainer/status", self.on_self_trainer_status)
        print(f"Expressions active - ambient personality + affect "
              f"(cooldown {EXPRESSION_COOLDOWN:.0f}s, ambient after "
              f"{IDLE_BEFORE_AMBIENT:.0f}s idle)")
        while True:
            self._maybe_emote()        # express internal state (mood) first
            self._maybe_idle_express()
            time.sleep(IDLE_TICK_SEC)


if __name__ == "__main__":
    Expressions().run()
