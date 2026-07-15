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

Belief revision for places (the map half of the hypothesis loop): it
also listens on picarx/exploration/hypothesis. When field_agent's
VetoProneLocationProbe physically re-tests a veto-prone spot and the
safety daemon stays silent ("maybe_clear"), this module eases that
location's veto_count back down - so a place the robot learned to fear
can be un-feared once the obstacle is actually gone. The write stays
here because location_graph is the SOLE writer to spatial.db;
field_agent only reports the physical finding, it never writes the map.

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

# The hypothesis-outcome contract from field_agent's VetoProneLocationProbe
# (see modules/field_agent.py). Matching on the question string keeps this
# decoupled from the other hypothesis types that share the topic.
VETO_PRONE_QUESTION = "is_veto_prone_area_still_blocked"
MAYBE_CLEAR = "maybe_clear"
# How much to ease veto_count per confirmed clear re-test. 1 = gradual:
# with the veto-prone threshold at 3, a place needs a few clean passes to
# stop being treated as veto-prone, so one lucky window can't erase a real
# recurring hazard.
VETO_RELAX_STEP = 1


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

    # ---------- decision journal ----------

    def publish_decision(self, kind, choice, reason, location=None):
        """Mirror field_agent's journal convention: a non-trivial map change
        lands on picarx/decision WITH the reason it happened, so the robot
        can answer 'why did you do that?' from event_logger's record instead
        of confabulating."""
        self.bus.publish("picarx/decision", {
            "source": "location_graph", "kind": kind, "choice": choice,
            "reason": reason, "location": location, "ts": time.time(),
        })

    # ---------- inbound: physical hypothesis outcomes (map decay) ----------

    def on_hypothesis(self, payload):
        """A VetoProneLocationProbe resolving 'maybe_clear' means the spot
        that kept vetoing us re-tested clean - ease its veto_count so the
        robot can eventually stop treating it as blocked, and journal WHY so
        'why did you go back in there?' has a real answer. Other hypothesis
        types (and 'still_blocked') are ignored. Fail-soft on a payload
        missing / naming an unknown location."""
        if payload.get("question") != VETO_PRONE_QUESTION:
            return
        if payload.get("resolution") != MAYBE_CLEAR:
            return
        # Prefer the explicit location_id in the outcome detail; fall back
        # to the location-context block field_agent stamps on every probe.
        loc_id = payload.get("location_id")
        if loc_id is None:
            loc_id = (payload.get("location") or {}).get("id")
        if loc_id is None:
            return
        # One read up front yields both the label (for the reason) and the
        # count. Skip entirely if there's nothing left to relax, so a floored
        # location can't spam the journal with non-events - and the new count
        # is computed locally, no second read.
        loc = self.store.get_location(loc_id)
        if loc is None or loc["veto_count"] <= 0:
            return
        label = loc["label"]
        self.store.relax_veto(loc_id, VETO_RELAX_STEP)
        remaining = max(0, loc["veto_count"] - VETO_RELAX_STEP)
        print(f"Location graph: {label} re-tested clear - eased veto_count to {remaining}")
        self.publish_decision(
            "map_update",
            {"location_id": loc_id, "change": "veto_relaxed", "veto_count": remaining},
            f"I relaxed my caution about {label} because a physical test showed "
            f"it might be clear now",
            location={"id": loc_id, "label": label})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/exploration/room_scan", self.on_room_scan)
        self.bus.subscribe("picarx/action/result", self.on_action_result)
        self.bus.subscribe("picarx/coach/episode", self.on_coach_episode)
        self.bus.subscribe("picarx/exploration/hypothesis", self.on_hypothesis)
        print(f"Location graph active ({self.store.location_count()} places known)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    LocationGraph().run()
