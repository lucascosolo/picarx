#!/usr/bin/env python3
# /home/picarx/layer_b/modules/field_agent.py
"""
Field Agent (Layer B) - integration test harness.

This is the first module that actually exercises the whole Layer B
pipeline end to end instead of bypassing it:

  - Movement requests go out as INTENTS on picarx/intent/move, picked
    up by arbiter.py, which is the only thing that talks to the
    safety daemon. This module never touches the safety socket for
    movement.
  - World knowledge comes from picarx/state/world, published by
    world_state.py. This module does not re-derive it from raw
    sensors.
  - "History" answers come from reading event_logger.py's SQLite
    database directly (read-only queries only - this module never
    writes to that DB, event_logger.py is the sole writer).
  - Speech in and out rides the existing picarx/audio/heard and
    picarx/audio/speak topics, same as your original modules.

REQUIRES ALL OF THE FOLLOWING RUNNING FIRST:
  broker_client.py (fixed version, supports multi-topic subscribe)
  safety_daemon.py
  audio_nodes.py       (for STT input + TTS output)
  distance_sensor.py
  vision_basic.py      (optional but recommended - enables face reports)
  arbiter.py           (required for exploration to actually move)
  world_state.py       (required for "what do you see" / exploration)
  event_logger.py      (required for "what have you done" / history)

Voice commands understood (see handle_voice_command):
  "explore" / "start"          -> begin autonomous wandering
  "stop" / "halt"               -> stop moving, cancel any intent
  "status" / "what do you see"  -> report current world state aloud
  "history" / "what have you done" / "what happened"
                                -> summarize event log aloud
  "battery" / "charge" / "level" -> report battery voltage
  "hello" / "hi"                 -> greet

You can also just watch stdout - every decision this module makes is
printed, not just spoken, so you can test without a working mic/speaker.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import sqlite3
import json
import time
import random
import threading

SOURCE_NAME = "field_agent"

# Must match event_logger.py's DB_PATH - this module only ever opens
# it read-only and never writes.
DB_PATH = "/home/picarx/layer_b/data/events.db"

EXPLORE_PRIORITY = 5
EXPLORE_TICK_HZ = 5
INTENT_TTL = 0.6       # must be > 1/EXPLORE_TICK_HZ so intents don't gap out

OBSTACLE_DISTANCE_CM = 20  # Adjusted slightly downward to prevent premature triggers
MIN_ANNOUNCEMENT_GAP = 6.0  # don't let spontaneous remarks spam the speaker


class FieldAgent:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()

        self.explore_mode = False
        self.latest_world = None
        self.face_was_detected = False

        self.last_announcement_at = 0.0
        self.start_time = time.time()

        # State machine for non-blocking obstacle evasion
        self.state = "CRUISING"  # "CRUISING", "EVADING"
        self.evade_stage = 0     # 0: stop, 1: reverse, 2: turn
        self.state_until = 0.0

        # Wander state (mirrors the old reflex explorer's behavior,
        # now expressed as intents instead of direct socket calls)
        self.last_wander = time.time()
        self.wander_interval = random.uniform(5.0, 10.0)
        self.steering_active_until = 0

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
        with self.lock:
            self.latest_world = payload
            face = payload.get("face", {})
            detected = bool(face.get("detected")) and not face.get("stale", True)

            if detected and not self.face_was_detected:
                self.announce("I see a face.")
            self.face_was_detected = detected

    def _snapshot(self):
        with self.lock:
            return dict(self.latest_world) if self.latest_world else None

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
                self.state = "CRUISING"
                self.announce("Starting exploration.", force=True)
            return

        if "stop" in text or "halt" in text:
            if self.explore_mode:
                self.explore_mode = False
                self.cancel_intent()
                self.announce("Stopping.", force=True)
            return

        if "battery" in text or "charge" in text or "level" in text:
            self.report_battery()
            return

        if "history" in text or "what have you done" in text or "what happened" in text:
            self.report_history()
            return

        if "what do you see" in text or "status" in text or "report" in text:
            self.report_status()
            return

        if "hello" in text or "hi" in text:
            self.announce("Hello! I am ready to chat and explore.", force=True)
            return

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

        battery = snap.get("battery", {})
        if battery.get("voltage") is not None:
            parts.append(f"my battery is at {battery['voltage']:.1f} volts")

        self.announce(". ".join(parts) + ".", force=True)

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

    # ---------- exploration behavior ----------

    def explore_tick(self):
        now = time.time()
        snap = self._snapshot()
        distance = snap.get("distance_cm") if snap else None
        distance_stale = snap.get("distance_stale", True) if snap else True

        # --- Handle Evasion State Machine ---
        if self.state == "EVADING":
            if now < self.state_until:
                # Continue executing current stage behavior
                if self.evade_stage == 0:
                    self.publish_intent({"direction": "stop"}, priority=8)
                elif self.evade_stage == 1:
                    self.publish_intent({"direction": "backward", "speed": 30}, priority=8)
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
                    self.publish_intent({"direction": "backward", "speed": 30}, priority=8)
                elif self.evade_stage == 2:
                    # Choose random direction to pivot away for 0.6 seconds
                    angle = random.choice([-30, 30])
                    self.state_until = now + 0.6
                    self.publish_intent({"direction": "turn", "angle": angle}, priority=8)
                else:
                    # Evasion complete, clean slate
                    self.publish_intent({"direction": "turn", "angle": 0}, priority=8)
                    self.state = "CRUISING"
                    self.last_wander = now
                return

        # --- Handle Trustworthiness of Sensor Data ---
        if distance is None or distance_stale or distance < 0:
            # Fallback cautious crawl if we are blind
            self.publish_intent({"direction": "forward", "speed": 15})
            return

        # --- Handle New Obstacle Detection ---
        if distance < OBSTACLE_DISTANCE_CM:
            self.announce("Obstacle ahead, backing away.")
            self.state = "EVADING"
            self.evade_stage = 0
            self.state_until = now + 0.25  # Quick stop window
            self.publish_intent({"direction": "stop"}, priority=8)
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
        self.publish_intent({"direction": "forward", "speed": 25})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.bus.subscribe("picarx/state/world", self.on_world_state)

        print("Field Agent active. Say 'explore', 'stop', 'status', 'history', or 'battery'.")
        self.announce("Field agent online.", force=True)

        period = 1.0 / EXPLORE_TICK_HZ
        while True:
            if self.explore_mode:
                self.explore_tick()
            time.sleep(period)


if __name__ == "__main__":
    FieldAgent().run()