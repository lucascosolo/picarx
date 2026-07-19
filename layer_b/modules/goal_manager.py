#!/usr/bin/env python3
# layer_b/modules/goal_manager.py
"""
Goal Manager (Layer B) - gives exploration a direction without ever
touching the wheels.

Long-horizon loop: pick the known place the robot understands least
(explorer.py's uncertainty scores over spatial.db), declare it the
active subgoal, and let field_agent lean its wander decisions toward
any scan sighting that matches the goal's fingerprint. Reaching the
goal (location_graph resolves a scan to it) ends the episode as a
success; a deadline expiry ends it as abandoned. Places that keep
being abandoned are marked unreachable (persisted across restarts)
and stop being chosen - the roadmap's "don't chase unexplorable
zones" mitigation.

Everything is advisory: no goal, a crashed goal manager, or an
unreachable goal leaves field_agent wandering exactly as it does
today. Goals only START once exploration is actually producing scans
(the first location_change after boot), so a parked robot never
announces missions.

Publishes:
  picarx/exploration/active_goal    {goal_id, location_id, label,
                                     target_labels, reason, deadline}
  picarx/exploration/goal_progress  {goal_id, status: reached|abandoned,
                                     location_id, label, elapsed}
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config
from spatial_store import SpatialStore

import json
import time
import threading
import uuid

STATE_PATH = robot_config.data_path("goal_state.json")
GOAL_DEADLINE_SEC = 300.0     # give up on a subgoal after this long
MAX_GOAL_FAILURES = 3         # abandoned this often -> marked unreachable
CHECK_INTERVAL = 15.0
MIN_GOAL_SCORE = 0.30         # nothing is uncertain enough -> no goal at all


class GoalManager:
    def __init__(self):
        self.bus = Bus()
        self.store = SpatialStore(readonly=True)
        self.lock = threading.Lock()
        self.scores = {}          # location_id -> uncertainty score
        self.current_id = None    # where the robot is (last location_change)
        self.active = None        # {"goal_id","location_id","label","started_at","deadline"}
        self.failures = self._load_failures()

    # ---------- persistence (which places keep defeating us) ----------

    def _load_failures(self):
        try:
            with open(STATE_PATH) as f:
                return {int(k): v for k, v in (json.load(f).get("failures") or {}).items()}
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    def _save_failures(self):
        try:
            tmp = f"{STATE_PATH}.tmp"
            with open(tmp, "w") as f:
                json.dump({"failures": {str(k): v for k, v in self.failures.items()}}, f)
            os.replace(tmp, STATE_PATH)
        except OSError as e:
            print(f"Goal manager: couldn't persist state: {e}")

    # ---------- inbound ----------

    def on_uncertainty_map(self, payload):
        with self.lock:
            self.scores = {e["id"]: e["score"] for e in payload.get("locations", [])}

    def on_location_change(self, payload):
        now = time.time()
        with self.lock:
            self.current_id = payload.get("location_id")
            active = dict(self.active) if self.active else None
        if active and self.current_id == active["location_id"]:
            self._finish_goal(active, "reached", now)
        elif active is None:
            self._maybe_adopt_goal(now)

    def on_goal_request(self, payload):
        """A user asked for a destination out loud ("go to the kitchen",
        routed by field_agent). A user goal replaces whatever curiosity
        goal is active and skips the uncertainty-score bar - the person
        outranks the robot's own wanderlust. It still expires on the
        normal deadline, and the unreachable-place blacklist is bypassed
        too: an explicit ask deserves a fresh attempt."""
        location_id = payload.get("location_id")
        if location_id is None:
            return
        loc = self.store.get_location(location_id)
        if loc is None:
            return
        now = time.time()
        with self.lock:
            previous = dict(self.active) if self.active else None
        if previous is not None and previous["location_id"] != location_id:
            # Superseded, not abandoned: the robot didn't fail to reach the
            # old goal, the user changed the plan - so no failure is counted
            # against the old goal's location.
            self.bus.publish("picarx/exploration/goal_progress", {
                "goal_id": previous["goal_id"], "status": "superseded",
                "location_id": previous["location_id"], "label": previous["label"],
                "elapsed": round(now - previous["started_at"], 1), "ts": now,
            })
        target_labels = sorted({l.split(":", 1)[1]
                                for l in loc["fingerprint"].get("labels") or []})
        goal = {
            "goal_id": str(uuid.uuid4()),
            "location_id": location_id,
            "label": loc["label"],
            "started_at": now,
            "deadline": now + GOAL_DEADLINE_SEC,
            "user_requested": True,
        }
        with self.lock:
            self.active = goal
        reason = "the user asked me to go there"
        print(f"Goal manager: user goal -> {loc['label']}")
        self.bus.publish("picarx/exploration/active_goal", {
            "goal_id": goal["goal_id"], "location_id": location_id,
            "label": loc["label"], "target_labels": target_labels,
            "reason": reason, "deadline": goal["deadline"], "ts": now,
        })
        self.bus.publish("picarx/decision", {
            "source": "goal_manager", "kind": "goal_adopted",
            "choice": {"location_id": location_id, "label": loc["label"],
                       "user_requested": True},
            "reason": reason, "ts": now,
        })

    # ---------- goal lifecycle ----------

    def _maybe_adopt_goal(self, now):
        with self.lock:
            scores, current = dict(self.scores), self.current_id
        if current is None or not scores:
            return
        # Prefer a directly-connected neighbor (we might actually know
        # how to get there); fall back to anywhere sufficiently unknown.
        neighbors = set(self.store.neighbors(current))
        candidates = [
            (score + (0.1 if lid in neighbors else 0.0), lid)
            for lid, score in scores.items()
            if lid != current and self.failures.get(lid, 0) < MAX_GOAL_FAILURES
        ]
        if not candidates:
            return
        best_score, best_id = max(candidates)
        if best_score < MIN_GOAL_SCORE:
            return  # everywhere is well understood - free wandering is fine
        loc = self.store.get_location(best_id)
        if loc is None:
            return
        target_labels = sorted({l.split(":", 1)[1]
                                for l in loc["fingerprint"].get("labels") or []})
        goal = {
            "goal_id": str(uuid.uuid4()),
            "location_id": best_id,
            "label": loc["label"],
            "started_at": now,
            "deadline": now + GOAL_DEADLINE_SEC,
        }
        with self.lock:
            self.active = goal
        reason = (f"least-understood reachable place "
                  f"(uncertainty {scores[best_id]:.2f}"
                  f"{', a known neighbor' if best_id in neighbors else ''})")
        print(f"Goal manager: new subgoal -> {loc['label']} ({reason})")
        self.bus.publish("picarx/exploration/active_goal", {
            "goal_id": goal["goal_id"], "location_id": best_id,
            "label": loc["label"], "target_labels": target_labels,
            "reason": reason, "deadline": goal["deadline"], "ts": now,
        })
        self.bus.publish("picarx/decision", {
            "source": "goal_manager", "kind": "goal_adopted",
            "choice": {"location_id": best_id, "label": loc["label"]},
            "reason": reason, "ts": now,
        })
        self.bus.publish("picarx/audio/speak", {
            "text": f"New mission: find my way back to {loc['label']}.", "ts": now})

    def _finish_goal(self, goal, status, now):
        with self.lock:
            self.active = None
        if status == "abandoned":
            self.failures[goal["location_id"]] = self.failures.get(goal["location_id"], 0) + 1
            self._save_failures()
            if self.failures[goal["location_id"]] >= MAX_GOAL_FAILURES:
                print(f"Goal manager: {goal['label']} marked unreachable "
                      f"after {MAX_GOAL_FAILURES} failed attempts")
        else:
            self.failures.pop(goal["location_id"], None)
            self._save_failures()
        print(f"Goal manager: subgoal {goal['label']} {status}")
        self.bus.publish("picarx/exploration/goal_progress", {
            "goal_id": goal["goal_id"], "status": status,
            "location_id": goal["location_id"], "label": goal["label"],
            "elapsed": round(now - goal["started_at"], 1), "ts": now,
        })
        # Clearing the goal on the bus so field_agent drops its bias.
        self.bus.publish("picarx/exploration/active_goal", {
            "goal_id": None, "location_id": None, "ts": now})
        if status == "reached":
            self.bus.publish("picarx/audio/speak", {
                "text": f"Made it back to {goal['label']}. Mission complete.", "ts": now})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/exploration/uncertainty_map", self.on_uncertainty_map)
        self.bus.subscribe("picarx/exploration/location_change", self.on_location_change)
        self.bus.subscribe("picarx/exploration/goal_request", self.on_goal_request)
        print(f"Goal manager active ({len(self.failures)} places with failed attempts on record)")
        while True:
            time.sleep(CHECK_INTERVAL)
            now = time.time()
            with self.lock:
                active = dict(self.active) if self.active else None
            if active and now > active["deadline"]:
                self._finish_goal(active, "abandoned", now)


if __name__ == "__main__":
    GoalManager().run()
