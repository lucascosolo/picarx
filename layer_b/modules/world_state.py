#!/usr/bin/env python3
# /home/picarx/layer_b/modules/world_state.py
"""
World State Aggregator (Layer B).

Subscribes to the various raw sensor/event topics and maintains one
coherent, timestamped snapshot of "what the robot currently knows,"
republished on picarx/state/world at a fixed rate. This exists so
that:

  1. Every future consumer (person-following, dialogue manager, and
     eventually the Layer C / LLM coach) reads ONE topic instead of
     each reassembling context from five raw feeds.
  2. Each field carries an age/staleness flag, so a consumer can tell
     "no face detected" apart from "haven't heard from vision in 30
     seconds because it crashed" - those are very different facts and
     collapsing them is how you get a robot that acts confidently on
     dead data.

Inputs (subscribed):
  picarx/vision/faces     - from vision_basic.py
  picarx/vision/objects   - from vision_basic.py (tracked/labeled objects)
  picarx/sensors/distance - from distance_sensor.py
  picarx/audio/heard      - from audio_nodes.py
  picarx/action/result    - from arbiter.py

Inputs (polled directly, read-only query - safe to decentralize per
the same reasoning as distance_sensor.py's direct socket query):
  safety daemon "battery_status" query, since nothing currently
  publishes battery state onto the bus.

Output (published):
  picarx/state/world, at PUBLISH_HZ, shaped like:
  {
    "timestamp": <float, unix time this snapshot was built>,
    "face": {
        "detected": bool,
        "x", "y", "w", "h", "frame_width", "frame_center_offset": ...
            (only present if detected),
        "updated_at": <float or None>,
        "stale": bool
    },
    "distance_cm": <float or None>,
    "distance_stale": bool,
    "objects": {
        "items": [ {"id", "label", "confidence", "x","y","w","h",
                    "frame_width", "frame_height", "area_ratio",
                    "center_offset", "first_seen", "last_seen",
                    "approach_rate", "approaching"}, ... ],
        "close_object": bool,  # class-agnostic - see vision_basic.py;
                                # true if something is filling most of
                                # the frame regardless of what it is
        "stale": bool
    },
    "battery": {
        "voltage": <float or None>,
        "low": bool,
        "critical": bool,
        "updated_at": <float or None>,
        "stale": bool
    },
    "last_heard": {
        "text": <str or None>,
        "updated_at": <float or None>,
        "stale": bool
    },
    "last_action": {
        "source": <str or None>,
        "action": <dict or None>,
        "result": <dict or None>,
        "updated_at": <float or None>
    }
  }

NOTE: "stale" means "older than the threshold in STALE_AFTER for that
field," NOT "no data ever received" - in that case updated_at is None
and stale is always True.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import socket
import json
import time
import threading

SOCKET_PATH = "/tmp/picarx_safety.sock"

PUBLISH_HZ = 2
BATTERY_POLL_INTERVAL = 5.0

# How old (seconds) a field's data can be before we flag it stale.
STALE_AFTER = {
    "face": 2.0,
    "distance": 2.0,
    "objects": 2.0,
    "heard": 15.0,
    "battery": 20.0,
}

# An object counts as "approaching" once its bounding-box area grows
# this fraction of the frame per second while sitting within the
# center portion of the frame - a depth-sensor-free stand-in for
# "something is closing in on us," used to catch obstacles the
# ultrasonic sensor's narrow cone misses entirely.
APPROACH_RATE_THRESHOLD = 0.12  # area_ratio growth per second
APPROACH_CENTER_FRACTION = 0.5  # |center_offset| must be within this
                                 # fraction of half the frame width


class WorldState:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()

        self.state = {
            "face": {"detected": False, "updated_at": None},
            "distance_cm": None,
            "distance_updated_at": None,
            "objects": {},  # id -> tracked object record (see on_objects)
            "objects_updated_at": None,
            "close_object": False,
            "battery": {"voltage": None, "low": False, "critical": False, "updated_at": None},
            "last_heard": {"text": None, "updated_at": None},
            "last_action": {"source": None, "action": None, "result": None, "updated_at": None},
        }

    # ---------- bus callbacks ----------

    def on_face(self, payload):
        with self.lock:
            self.state["face"] = {**payload, "updated_at": time.time()}

    def on_distance(self, payload):
        with self.lock:
            self.state["distance_cm"] = payload.get("distance_cm")
            self.state["distance_updated_at"] = time.time()

    def on_objects(self, payload):
        now = time.time()
        with self.lock:
            existing = self.state["objects"]
            updated = {}
            for obj in payload.get("objects", []):
                tid = obj["id"]
                prev = existing.get(tid)
                approach_rate = 0.0
                if prev is not None:
                    dt = now - prev["_ts"]
                    if dt > 0:
                        approach_rate = (obj["area_ratio"] - prev["area_ratio"]) / dt
                record = dict(obj)
                record["approach_rate"] = approach_rate
                record["approaching"] = (
                    approach_rate > APPROACH_RATE_THRESHOLD
                    and abs(obj.get("center_offset", 0)) <
                        (obj.get("frame_width", 1) / 2.0) * APPROACH_CENTER_FRACTION
                )
                record["_ts"] = now
                updated[tid] = record
            self.state["objects"] = updated
            self.state["objects_updated_at"] = now
            self.state["close_object"] = bool(payload.get("close_object", False))

    def on_heard(self, payload):
        with self.lock:
            self.state["last_heard"] = {
                "text": payload.get("text"),
                "updated_at": time.time(),
            }

    def on_action_result(self, payload):
        with self.lock:
            self.state["last_action"] = {
                "source": payload.get("source"),
                "action": payload.get("action"),
                "result": payload.get("result"),
                "updated_at": time.time(),
            }

    # ---------- battery polling (direct read-only socket query) ----------

    def query_battery(self):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.3)
            s.connect(SOCKET_PATH)
            s.sendall(json.dumps({"query": "battery_status"}).encode())
            data = s.recv(1024)
            s.close()
            return json.loads(data.decode())
        except Exception as e:
            print(f"World state: battery query failed: {e}")
            return None

    def battery_poll_loop(self):
        while True:
            result = self.query_battery()
            if result and "voltage" in result:
                with self.lock:
                    self.state["battery"] = {
                        "voltage": result.get("voltage"),
                        "low": result.get("low", False),
                        "critical": result.get("critical", False),
                        "updated_at": time.time(),
                    }
            time.sleep(BATTERY_POLL_INTERVAL)

    # ---------- snapshot assembly ----------

    def _is_stale(self, updated_at, key):
        if updated_at is None:
            return True
        return (time.time() - updated_at) > STALE_AFTER[key]

    def build_snapshot(self):
        with self.lock:
            face = dict(self.state["face"])
            distance_cm = self.state["distance_cm"]
            distance_updated_at = self.state["distance_updated_at"]
            objects = {tid: dict(obj) for tid, obj in self.state["objects"].items()}
            objects_updated_at = self.state["objects_updated_at"]
            close_object = self.state["close_object"]
            battery = dict(self.state["battery"])
            heard = dict(self.state["last_heard"])
            last_action = dict(self.state["last_action"])

        return {
            "timestamp": time.time(),
            "face": {
                **face,
                "stale": self._is_stale(face.get("updated_at"), "face"),
            },
            "distance_cm": distance_cm,
            "distance_stale": self._is_stale(distance_updated_at, "distance"),
            "objects": {
                "items": [
                    {k: v for k, v in obj.items() if k != "_ts"}
                    for obj in objects.values()
                ],
                "close_object": close_object,
                "stale": self._is_stale(objects_updated_at, "objects"),
            },
            "battery": {
                **battery,
                "stale": self._is_stale(battery.get("updated_at"), "battery"),
            },
            "last_heard": {
                **heard,
                "stale": self._is_stale(heard.get("updated_at"), "heard"),
            },
            "last_action": last_action,
        }

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/vision/faces", self.on_face)
        self.bus.subscribe("picarx/vision/objects", self.on_objects)
        self.bus.subscribe("picarx/sensors/distance", self.on_distance)
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.bus.subscribe("picarx/action/result", self.on_action_result)

        threading.Thread(target=self.battery_poll_loop, daemon=True).start()

        print("World State Aggregator active, publishing to picarx/state/world")
        period = 1.0 / PUBLISH_HZ
        while True:
            snapshot = self.build_snapshot()
            self.bus.publish("picarx/state/world", snapshot)
            time.sleep(period)


if __name__ == "__main__":
    WorldState().run()