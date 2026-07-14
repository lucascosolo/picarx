#!/usr/bin/env python3
# /home/picarx/layer_b/modules/location_graph.py
"""
Location Graph (Layer B) - maintains the robot's topological map.

Turns each completed look-around head sweep (picarx/exploration/
room_scan, from field_agent) into a node in spatial.db via
spatial_store.SpatialStore: match the sweep's perceptual fingerprint
against known places, or mint a new one. Consecutive distinct places
become graph edges ("you can get from here to there"), and safety
vetoes / coach outcomes are counted against the place they happened
in, so downstream consumers can ask "where does the robot struggle?".

Publishes picarx/exploration/location_change after every resolved
scan (changed=false when it's the same place re-confirmed), so
field_agent can tag its coach queries and decisions with WHERE they
happened without doing any spatial reasoning itself.

Deliberately conservative (per the rollout-risk notes): location
inference happens ONLY on a completed scan - never inferred from
wander progress or single detections - so a bad frame can't teleport
the map. Everything here is enrichment; if this module is down the
robot explores exactly as before.

This module is the SOLE writer to spatial.db.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus
from spatial_store import SpatialStore, fingerprint_from_scan

import time
import threading


class LocationGraph:
    def __init__(self):
        self.bus = Bus()
        self.store = SpatialStore(readonly=False)
        self.lock = threading.Lock()
        self.current_id = None      # location of the most recent scan
        self.current_label = None

    # ---------- inbound: scans resolve to locations ----------

    def on_room_scan(self, payload):
        fingerprint = fingerprint_from_scan(
            payload.get("sightings"), payload.get("distance_cm"))
        now = time.time()
        loc = self.store.match_or_create(fingerprint, now)

        with self.lock:
            prev_id = self.current_id
            changed = loc["id"] != prev_id
            if changed and prev_id is not None:
                self.store.note_edge(prev_id, loc["id"], now)
            self.current_id = loc["id"]
            self.current_label = loc["label"]

        if loc["is_new"]:
            print(f"Location graph: discovered new {loc['label']} "
                  f"({self.store.location_count()} places known)")
        self.bus.publish("picarx/exploration/location_change", {
            "location_id": loc["id"],
            "label": loc["label"],
            "is_new": loc["is_new"],
            "changed": changed,
            "new_visit": loc["new_visit"],
            "visit_count": loc["visit_count"],
            "veto_count": loc["veto_count"],
            "ts": now,
        })

    # ---------- inbound: outcomes get pinned to the current place ----------

    def on_action_result(self, payload):
        if (payload.get("result") or {}).get("status") != "vetoed":
            return
        with self.lock:
            loc_id = self.current_id
        if loc_id is not None:
            self.store.note_veto(loc_id)

    def on_coach_episode(self, payload):
        with self.lock:
            loc_id = self.current_id
        if loc_id is not None:
            self.store.note_coach_outcome(loc_id, bool(payload.get("success")))

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/exploration/room_scan", self.on_room_scan)
        self.bus.subscribe("picarx/action/result", self.on_action_result)
        self.bus.subscribe("picarx/coach/episode", self.on_coach_episode)
        print(f"Location graph active ({self.store.location_count()} places known)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    LocationGraph().run()
