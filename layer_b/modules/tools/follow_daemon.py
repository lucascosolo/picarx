#!/usr/bin/env python3
# /home/picarx/layer_b/modules/tools/follow_daemon.py
"""
Follow daemon (Layer B tool) - "follow me" person tracking.

Enabled/disabled by a MODE toggle on picarx/tools/follow/set
{"enabled": bool}. companion.py's start_following / stop_following LLM
tools publish that toggle - the model only flips a switch, it never
computes a motion. All actual movement is generated HERE, deterministically
from vision, and published as ordinary intents on picarx/intent/move, so
arbiter.py + the safety daemon gate every command exactly like any other
motion: if the person walks toward a cliff or an obstacle, the safety
layer vetoes and the robot does not follow into danger. This daemon never
touches the safety socket and never bypasses that veto.

Vision reuse (CPU budget): it does NOT open its own camera - only
vision_basic.py owns picamera2. It consumes vision_basic's published
detections: the SSD/YOLO "person" track on picarx/vision/objects (works
whichever way the person faces, and its area_ratio is a depth-free
distance proxy) and, as a faster-updating fallback for centering, the
Haar face on picarx/vision/faces.

Control: steer proportionally to the target's horizontal offset to keep
it centered, drive forward at a bounded low speed to close distance, and
STOP once the bounding box gets large enough (person close). If the
target is lost, hold still and reacquire; after a longer gap, give up and
disable. A literal spoken "stop"/"halt" on picarx/audio/heard also
disables follow immediately - the kill switch never depends on the LLM.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import threading
import time

CONTROL_TOPIC = "picarx/tools/follow/set"
STATE_TOPIC = "picarx/tools/follow/state"
OBJECTS_TOPIC = "picarx/vision/objects"
FACES_TOPIC = "picarx/vision/faces"
HEARD_TOPIC = "picarx/audio/heard"
INTENT_TOPIC = "picarx/intent/move"
SPEAK_TOPIC = "picarx/audio/speak"

SOURCE_NAME = "follow"
# Above plain exploring (5) and novelty-watch (6) - a user "follow me" wins
# those - but below the safety-reflex evasion (8) and coach (9), so a
# genuine escape maneuver always outranks following. The safety DAEMON is
# the real backstop regardless of this number.
FOLLOW_PRIORITY = 7
INTENT_TTL = 0.5           # short: if this daemon dies, the intent lapses fast

CONTROL_HZ = 8.0
MAX_STEER_ANGLE = 30       # PiCar-X steering limit used elsewhere
STEER_DEADBAND = 6         # ignore tiny offsets so we don't wobble on center
ANGLE_RESEND_DELTA = 4     # only re-aim the servo when the target angle really moved
FOLLOW_SPEED = 18          # bounded, gentle approach speed
STOP_AREA_RATIO = 0.35     # person's box fills this much of the frame -> close enough, stop
FRESH_TARGET_SEC = 1.2     # a detection older than this is stale (SSD updates ~every 1.5s)
LOST_HOLD_SEC = 2.0        # target stale this long -> hold still (stop)
LOST_GIVEUP_SEC = 15.0     # ...this long -> give up and switch follow off


def steer_angle(offset, frame_width):
    """Signed steering angle to re-center a target whose box center is
    `offset` px right(+)/left(-) of frame center. Normalized so an offset
    at the frame edge maps to full lock; deadbanded near center."""
    if not frame_width or abs(offset) <= STEER_DEADBAND:
        return 0
    norm = max(-1.0, min(1.0, (offset / (frame_width / 2.0))))
    return int(round(norm * MAX_STEER_ANGLE))


def drive_decision(area_ratio):
    """('stop'|'forward', speed) from the target's box area. Big box = the
    person is close, so hold position; otherwise ease forward."""
    if area_ratio is not None and area_ratio >= STOP_AREA_RATIO:
        return "stop", 0
    return "forward", FOLLOW_SPEED


def pick_person(objects_payload):
    """Largest 'person' box in a vision objects payload, or None."""
    best = None
    for item in (objects_payload or {}).get("objects", []):
        if item.get("label") != "person":
            continue
        if best is None or item.get("area_ratio", 0) > best.get("area_ratio", 0):
            best = item
    return best


class FollowDaemon:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.enabled = False
        self.enabled_at = 0.0
        # Latest target info: (offset_px, frame_width, area_ratio_or_None, ts)
        self.person = None
        self.face = None
        self.last_sent_angle = 0
        self.lost_announced = False

    # ---------- inbound ----------

    def on_control(self, payload):
        want = bool(payload.get("enabled"))
        with self.lock:
            was = self.enabled
            self.enabled = want
            self.last_sent_angle = 0
            self.lost_announced = False
            if want and not was:
                # Fresh start: forget sightings from before this session (a
                # person box from an hour ago must not count as "just lost")
                # and remember when following began, so the lost-target
                # timers measure from NOW, not from epoch 0 - without this,
                # enabling follow before the detector has produced a person
                # made gone_for huge and the daemon gave up ("I lost you")
                # in the very first tick after saying it would follow.
                self.enabled_at = time.time()
                self.person = None
                self.face = None
        if want and not was:
            self.bus.publish(SPEAK_TOPIC, {"text": "Okay, I'll follow you.", "ts": time.time()})
        elif was and not want:
            self._release()
            self.bus.publish(SPEAK_TOPIC, {"text": "Okay, I'll stop following.", "ts": time.time()})
        self.bus.publish(STATE_TOPIC, {"enabled": want, "ts": time.time()})

    def on_objects(self, payload):
        person = pick_person(payload)
        if person is None:
            return
        with self.lock:
            self.person = (person.get("center_offset", 0),
                           person.get("frame_width"),
                           person.get("area_ratio"),
                           time.time())

    def on_faces(self, payload):
        if not payload.get("detected"):
            return
        with self.lock:
            self.face = (payload.get("frame_center_offset", 0),
                         payload.get("frame_width"),
                         None,                      # a face gives no distance proxy
                         time.time())

    def on_heard(self, payload):
        # Literal spoken kill switch - never depends on the LLM. Matches the
        # same words field_agent treats as a hard stop.
        text = (payload.get("text") or "").lower()
        if not self.enabled:
            return
        if "stop" in text or "halt" in text:
            with self.lock:
                self.enabled = False
            self._release()
            self.bus.publish(STATE_TOPIC, {"enabled": False, "ts": time.time()})
            print("Follow daemon: stopped by spoken command")

    # ---------- motion ----------

    def _release(self):
        """Give up our motion intent so the arbiter falls back to whatever
        else wants the robot (or a safe stop). One explicit stop first."""
        self._publish_intent({"direction": "stop"})
        self.bus.publish("picarx/intent/cancel", {"source": SOURCE_NAME})

    def _publish_intent(self, action):
        self.bus.publish(INTENT_TOPIC, {
            "source": SOURCE_NAME, "priority": FOLLOW_PRIORITY,
            "action": action, "ttl": INTENT_TTL})

    def _fresh_target(self, now):
        """Prefer a fresh person track (has distance); else a fresh face
        (centering only). Returns (offset, frame_width, area_ratio) or None."""
        with self.lock:
            person, face = self.person, self.face
        if person and now - person[3] < FRESH_TARGET_SEC:
            return person[:3]
        if face and now - face[3] < FRESH_TARGET_SEC:
            return face[:3]
        return None

    def _tick(self, now):
        target = self._fresh_target(now)
        if target is None:
            self._handle_lost(now)
            return
        self.lost_announced = False
        offset, frame_width, area_ratio = target
        drive, speed = drive_decision(area_ratio)
        if drive == "stop":
            # Close enough - hold position (still gated by the safety daemon).
            self._publish_intent({"direction": "stop"})
            return
        target_angle = steer_angle(offset, frame_width)
        # Ackermann: aim the servo on ticks where the heading needs to change,
        # then drive forward on the others (the servo holds its angle). This
        # matches how field_agent arcs - one primitive per tick to the arbiter.
        if abs(target_angle - self.last_sent_angle) >= ANGLE_RESEND_DELTA:
            self._publish_intent({"direction": "turn", "angle": target_angle})
            self.last_sent_angle = target_angle
        else:
            self._publish_intent({"direction": "forward", "speed": speed})

    def _handle_lost(self, now):
        # Find the freshest sighting timestamp to measure how long it's been.
        # Never seen anyone this session -> measure from when follow was
        # enabled, so the robot waits the normal windows to acquire a target
        # instead of instantly concluding it lost one it never had.
        with self.lock:
            stamps = [t[3] for t in (self.person, self.face) if t]
            enabled_at = self.enabled_at
        last_seen = max(stamps) if stamps else enabled_at
        gone_for = now - last_seen
        if gone_for >= LOST_GIVEUP_SEC:
            with self.lock:
                self.enabled = False
            self._release()
            self.bus.publish(STATE_TOPIC, {"enabled": False, "reason": "target_lost",
                                           "ts": now})
            self.bus.publish(SPEAK_TOPIC, {"text": "I lost you, so I'll stop following.",
                                           "ts": now})
            print("Follow daemon: target lost, disabling")
            return
        # Briefly lost: hold still and wait to reacquire.
        self._publish_intent({"direction": "stop"})
        if gone_for >= LOST_HOLD_SEC and not self.lost_announced:
            self.lost_announced = True
            self.bus.publish(SPEAK_TOPIC, {"text": "Where did you go?", "ts": now})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(CONTROL_TOPIC, self.on_control)
        self.bus.subscribe(OBJECTS_TOPIC, self.on_objects)
        self.bus.subscribe(FACES_TOPIC, self.on_faces)
        self.bus.subscribe(HEARD_TOPIC, self.on_heard)
        print(f"Follow daemon active, waiting for {CONTROL_TOPIC} (motion via {INTENT_TOPIC})")
        period = 1.0 / CONTROL_HZ
        while True:
            time.sleep(period)
            if self.enabled:
                try:
                    self._tick(time.time())
                except Exception as e:
                    print(f"Follow daemon: tick error: {e}")


if __name__ == "__main__":
    FollowDaemon().run()
