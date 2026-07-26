#!/usr/bin/env python3
# layer_b/modules/location_graph.py
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
from spatial_store import SpatialStore, fingerprint_from_scan

import time
import threading
import uuid

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

# An ambiguous visual match is cheap to probe with the camera head.  Keep the
# probe deliberately small: two offset looks are enough to change which
# landmark is in-frame without driving the robot into an unknown space.
DISAMBIGUATION_PANS = (-50, 50)
DISAMBIGUATION_MAX_ATTEMPTS = 1


class LocationGraph:
    def __init__(self):
        self.bus = Bus()
        self.store = SpatialStore(readonly=False)
        self.lock = threading.Lock()
        self.current_id = None      # location of the most recent scan
        self.current_label = None
        self._disambiguation_pending = None

    @staticmethod
    def _id(value=None):
        """Return a caller-supplied correlation id or a fresh UUID."""
        return str(value) if value else uuid.uuid4().hex

    @staticmethod
    def _evidence_ids(payload, scan_id):
        """Normalize bounded, transport-safe evidence references.

        Event logger row ids are deliberately not used here: room scans and
        location changes cross an asynchronous broker boundary.  The scan id
        is therefore the stable evidence reference for this resolution.
        """
        values = payload.get("evidence_ids") or []
        if isinstance(values, str):
            values = [values]
        out = [str(value) for value in values[:8] if value]
        if scan_id not in out:
            out.append(scan_id)
        return out[:8]

    # ---------- inbound: scans resolve to locations ----------

    def on_room_scan(self, payload):
        # ``on_room_scan`` is also the completion callback for an active
        # disambiguation probe. Consume the one-shot marker before matching so
        # an ambiguous probe result cannot recursively schedule probes.
        with self.lock:
            probe = getattr(self, "_disambiguation_pending", None)
            self._disambiguation_pending = None

        scan_id = self._id(payload.get("scan_id"))
        resolution_id = self._id(
            (probe or {}).get("resolution_id") or payload.get("resolution_id"))
        evidence_ids = self._evidence_ids(payload, scan_id)
        if probe:
            for evidence_id in probe.get("evidence_ids") or []:
                if evidence_id and evidence_id not in evidence_ids:
                    evidence_ids.append(str(evidence_id))
            evidence_ids = evidence_ids[:8]

        # Keep positional compatibility with small test/adaptor
        # implementations that still expose the historical two-argument
        # helper, while forwarding pose data when a producer supplies it.
        if "imu_delta" in payload or "pose_delta" in payload:
            fingerprint = fingerprint_from_scan(
                payload.get("sightings"), payload.get("distance_cm"),
                payload.get("imu_delta", payload.get("pose_delta")))
        else:
            fingerprint = fingerprint_from_scan(
                payload.get("sightings"), payload.get("distance_cm"))
        now = time.time()
        loc = self.store.match_or_create(fingerprint, now)
        candidate_scores = list(loc.get("candidate_scores") or [])[:3]

        # A probe is only allowed to confirm one of the candidates that made
        # the original scan ambiguous. This prevents an unrelated follow-up
        # frame from silently teleporting the graph to a different place.
        candidate_ids = set((probe or {}).get("candidate_location_ids") or [])
        if probe is not None and loc.get("resolved") and candidate_ids and \
                loc.get("id") not in candidate_ids:
            loc = {**loc, "id": None, "label": None, "resolved": False,
                   "is_new": False, "new_visit": False,
                   "reason": "disambiguation_disagreed",
                   "ambiguity": {"candidate_location_ids": sorted(candidate_ids)}}

        # A sparse scan or one that fits several remembered places is not a
        # location fix. Publishing that uncertainty is important: consumers
        # must not quietly retain an old place and mistake it for an arrival.
        if not loc.get("resolved"):
            request = None
            if loc.get("reason") == "ambiguous_match" and probe is None:
                request = self._schedule_disambiguation(
                    loc, now, scan_id=scan_id, resolution_id=resolution_id,
                    evidence_ids=evidence_ids)
            with self.lock:
                self.current_id = None
                self.current_label = None
            self.bus.publish("picarx/exploration/location_change", {
                "location_id": None, "label": None, "is_new": False,
                "changed": False, "new_visit": False, "visit_count": None,
                "veto_count": None, "localized": False,
                "confidence": loc.get("similarity"),
                "reason": loc.get("reason"), "ambiguity": loc.get("ambiguity"),
                "scan_id": scan_id, "resolution_id": resolution_id,
                "probe_id": (probe or {}).get("probe_id"),
                "candidate_scores": candidate_scores,
                "evidence_ids": evidence_ids,
                "disambiguation": (
                    {"attempt": 0, "outcome": "pending",
                     "probe_id": request.get("probe_id")} if request else
                    ({"attempt": int((probe or {}).get("attempt", 1)),
                      "outcome": "unresolved",
                      "probe_id": (probe or {}).get("probe_id")}
                     if probe else None)),
                "ts": now,
            })
            return

        # Object-place memory: every label this sweep saw is recorded
        # against the resolved place, so "where is the bottle?" has a
        # durable answer ("the kitchen, 20 minutes ago") instead of only
        # the last few seconds of tracked objects.
        labels = set()
        for sighting in payload.get("sightings") or []:
            if not isinstance(sighting, dict):
                continue
            raw_labels = sighting.get("labels") or sighting.get("objects") or []
            if isinstance(raw_labels, (str, dict)):
                raw_labels = [raw_labels]
            for raw in raw_labels:
                name = (raw.get("name", raw.get("label"))
                        if isinstance(raw, dict) else raw)
                if name:
                    labels.add(str(name))
        if labels:
            self.store.note_sightings(loc["id"], labels, now)

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
            "localized": True,
            "confidence": loc.get("similarity"),
            "reason": loc.get("reason"),
            "scan_id": scan_id, "resolution_id": resolution_id,
            "probe_id": (probe or {}).get("probe_id"),
            "candidate_scores": candidate_scores,
            "evidence_ids": evidence_ids,
            "disambiguation": (
                {"attempt": int((probe or {}).get("attempt", 1)),
                 "outcome": "resolved", "probe_id": (probe or {}).get("probe_id")}
                if probe else None),
            "ts": now,
        })

    def _schedule_disambiguation(self, loc, now, scan_id=None,
                                 resolution_id=None, evidence_ids=None):
        """Request one richer scan for an ambiguous location match.

        The location graph cannot manufacture a camera sweep, so it publishes
        a durable request for field_agent (or another actuator owner) and also
        nudges the head immediately.  The next ``room_scan`` consumes the
        one-shot marker and is matched normally; an unresolved result stays
        unresolved rather than creating a third duplicate place.
        """
        ambiguity = loc.get("ambiguity") or {}
        candidate_ids = [value for value in (
            ambiguity.get("best_location_id"),
            ambiguity.get("second_location_id")) if value is not None]
        candidate_scores = list(loc.get("candidate_scores") or [])[:3]
        if not candidate_scores:
            # Keep compatibility with small test stores and old adapters that
            # only returned the historical ambiguity block.
            best_id = ambiguity.get("best_location_id")
            if best_id is not None:
                candidate_scores.append({"location_id": best_id,
                                         "similarity": loc.get("similarity")})
            second_id = ambiguity.get("second_location_id")
            second_similarity = ambiguity.get("second_similarity")
            if second_id is not None and second_similarity is not None:
                candidate_scores.append({"location_id": second_id,
                                         "similarity": second_similarity})
        scan_id = self._id(scan_id)
        resolution_id = self._id(resolution_id)
        evidence_ids = list(evidence_ids or [])[:8]
        if scan_id not in evidence_ids:
            evidence_ids.append(scan_id)
        probe_id = self._id()
        pan_offsets = list(DISAMBIGUATION_PANS)
        request = {
            "reason": "ambiguous_match",
            "ambiguity": ambiguity,
            "candidate_location_ids": candidate_ids,
            "scan_attempt": 1,
            "attempt": 1,
            "max_scan_attempts": DISAMBIGUATION_MAX_ATTEMPTS,
            "action": "head_sweep",
            "pans": pan_offsets,
            "pan_offsets": pan_offsets,
            "probe": {"type": "head_sweep", "pan_offsets": pan_offsets,
                      "saccades": 2, "quick": True},
            "scan_id": scan_id, "resolution_id": resolution_id,
            "probe_id": probe_id, "candidate_scores": candidate_scores,
            "evidence_ids": evidence_ids[:8],
            "quick": True,
            "ts": now,
        }
        with self.lock:
            # The guard makes a single ambiguous scan idempotent even if a
            # broker redelivers it before the probe result arrives.
            if getattr(self, "_disambiguation_pending", None) is not None:
                return
            self._disambiguation_pending = request

        self.bus.publish("picarx/exploration/disambiguation_needed", request)
        return request

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

    # ---------- inbound: user names a place ----------

    def on_name_place(self, payload):
        """Voice command routed by field_agent ("call this place the
        kitchen"): rename a location. The write stays here because
        location_graph is the SOLE writer to spatial.db. Defaults to the
        current location when no explicit id is given."""
        name = (payload.get("name") or "").strip()
        if not name:
            return
        loc_id = payload.get("location_id")
        if loc_id is None:
            with self.lock:
                loc_id = self.current_id
        if loc_id is None:
            self.bus.publish("picarx/audio/speak", {
                "text": "I'm not sure where I am yet - let me scan around first.",
                "ts": time.time()})
            return
        old = self.store.get_location(loc_id)
        new_label = self.store.rename_location(loc_id, name)
        if new_label is None:
            return
        with self.lock:
            if self.current_id == loc_id:
                self.current_label = new_label
        print(f"Location graph: renamed '{old['label'] if old else loc_id}' -> '{new_label}'")
        self.publish_decision(
            "place_named",
            {"location_id": loc_id, "label": new_label},
            f"the user told me this place is called {new_label}",
            location={"id": loc_id, "label": new_label})
        self.bus.publish("picarx/audio/speak", {
            "text": f"Got it, this is {new_label}.", "ts": time.time()})

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
        self.bus.subscribe("picarx/exploration/name_place", self.on_name_place)
        print(f"Location graph active ({self.store.location_count()} places known)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    LocationGraph().run()
