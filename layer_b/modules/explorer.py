#!/usr/bin/env python3
# layer_b/modules/explorer.py
"""
Curiosity Explorer (Layer B) - scores how well the robot understands
each place it knows about, and tells the rest of the system where its
knowledge is thin.

Every UPDATE_INTERVAL it reads spatial.db (read-only - location_graph
owns it) and computes an uncertainty score per location from:
  novelty        - barely visited places score high
  staleness      - places not seen in hours score high (world changes)
  trouble        - places with many safety vetoes per visit score high
                   (something there the sensors don't handle well)
  unpredictability - coach outcomes there keep NOT going as expected

Outputs, both fail-soft optional for every consumer:
  - picarx/exploration/uncertainty_map (published only when scores
    materially change, so the bus and events.db aren't spammed with
    identical heatmaps)
  - data/uncertainty_map.json - the human-review artifact: open it to
    see which corners of the house are still mysterious.

field_agent uses the current location's score to decide between
"this area is well understood, drift toward open space" and "still
learning here, keep poking around" when it picks wander angles.
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

UPDATE_INTERVAL = 60.0
MAP_JSON_PATH = robot_config.data_path("uncertainty_map.json")
PUBLISH_DELTA = 0.05      # min score movement (any location) worth republishing

STALE_FULL_HOURS = 6.0    # not visited for this long -> full staleness

WEIGHTS = {"novelty": 0.40, "staleness": 0.25, "trouble": 0.20, "unpredictability": 0.15}


def score_location(loc, now):
    """Pure scoring - returns (score, components). All components 0..1,
    higher = more uncertain / more worth revisiting."""
    novelty = 1.0 / (1.0 + loc["visit_count"])
    age_h = max(0.0, (now - loc["last_visited_at"]) / 3600.0)
    staleness = min(1.0, age_h / STALE_FULL_HOURS)
    trouble = min(1.0, loc["veto_count"] / (3.0 * max(1, loc["visit_count"])))
    pulls = loc["coach_wins"] + loc["coach_losses"]
    unpredictability = (loc["coach_losses"] / pulls) if pulls else 0.5
    components = {"novelty": novelty, "staleness": staleness,
                  "trouble": trouble, "unpredictability": unpredictability}
    score = sum(WEIGHTS[k] * v for k, v in components.items())
    return round(score, 3), {k: round(v, 3) for k, v in components.items()}


def build_map(locations, now):
    entries = []
    for loc in locations:
        score, components = score_location(loc, now)
        entries.append({
            "id": loc["id"], "label": loc["label"], "score": score,
            "visits": loc["visit_count"], "vetoes": loc["veto_count"],
            "components": components,
        })
    entries.sort(key=lambda e: e["score"], reverse=True)
    return {"generated_at": now, "locations": entries}


class Explorer:
    def __init__(self):
        self.bus = Bus()
        self.store = SpatialStore(readonly=True)
        self.last_scores = {}
        self.recompute_asap = threading.Event()

    def on_location_change(self, payload):
        # A scan just resolved - refresh soon so field_agent's next
        # wander decision sees scores that include this visit.
        if payload.get("changed") or payload.get("is_new"):
            self.recompute_asap.set()

    def tick(self, now=None):
        now = now if now is not None else time.time()
        umap = build_map(self.store.all_locations(), now)
        if not umap["locations"]:
            return
        scores = {e["id"]: e["score"] for e in umap["locations"]}
        moved = (set(scores) != set(self.last_scores)) or any(
            abs(scores[i] - self.last_scores.get(i, -1)) > PUBLISH_DELTA for i in scores)
        if not moved:
            return
        self.last_scores = scores
        self.bus.publish("picarx/exploration/uncertainty_map", umap)
        try:
            tmp = f"{MAP_JSON_PATH}.tmp"
            with open(tmp, "w") as f:
                json.dump(umap, f, indent=1)
            os.replace(tmp, MAP_JSON_PATH)
        except OSError as e:
            print(f"Explorer: couldn't write map json: {e}")
        top = umap["locations"][0]
        print(f"Explorer: {len(scores)} places scored; most uncertain: "
              f"{top['label']} ({top['score']})")

    def run(self):
        self.bus.subscribe("picarx/exploration/location_change", self.on_location_change)
        print("Curiosity explorer active, scoring spatial.db every "
              f"{UPDATE_INTERVAL:.0f}s")
        while True:
            # Event doubles as an interruptible sleep: a location change
            # wakes the scorer early, otherwise it's the normal cadence.
            self.recompute_asap.wait(timeout=UPDATE_INTERVAL)
            self.recompute_asap.clear()
            try:
                self.tick()
            except Exception as e:
                print(f"Explorer: tick failed: {e}")


if __name__ == "__main__":
    Explorer().run()
