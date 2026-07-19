#!/usr/bin/env python3
# layer_b/modules/event_logger.py
"""
Episodic Event Logger (Layer B).

Writes a persistent, timestamped history of what the robot sensed and
did to a local SQLite database. This module makes zero decisions and
controls no hardware - its only job is to make sure that by the time
you're ready to build the Layer C / LLM coach, there is actually a
history for it to learn from or reflect on. Skipping this now means
having no data later; there's no way to backfill history you didn't
record.

What gets logged, and why each is handled differently:

  picarx/audio/heard    - logged immediately, every message. Low
                           volume (only fires on recognized speech),
                           and every utterance is a distinct event
                           worth keeping in full.

  picarx/action/result  - logged immediately, every message. This is
                           the record of what the robot actually did
                           and whether the safety layer allowed or
                           vetoed it - the core "what happened" trail.

  picarx/coach/episode  - logged immediately, every message. One row
                           per completed coach query: the situation,
                           the action tried (cached arm or fresh LLM
                           suggestion), and whether it succeeded. This
                           is the actual inspectable training history
                           behind coach.py's policy cache - the cache
                           itself only keeps aggregate counts, this is
                           the full record of every episode that fed it.

  picarx/state/world    - NOT logged on every publish (it's emitted
                           at PUBLISH_HZ from world_state.py, which
                           would flood the database with near-duplicate
                           rows). Instead:
                             (a) a periodic snapshot is written every
                                 SNAPSHOT_INTERVAL seconds, and
                             (b) a snapshot is written immediately,
                                 out of cycle, whenever battery
                                 "critical" flips from False to True,
                                 since that's a significant edge worth
                                 capturing exactly when it happens
                                 rather than up to SNAPSHOT_INTERVAL
                                 seconds late.

All rows share one schema: (id, ts, topic, payload_json). This is
intentionally generic/append-only rather than one bespoke table per
topic - it keeps this module simple, and a normalized schema can
always be built later by reading back through payload_json once you
know what Layer C actually needs to query for.

Requires: world_state.py must be running and publishing to
picarx/state/world for the periodic/edge-triggered snapshot logging
to do anything.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config

import sqlite3
import json
import time
import threading

DB_DIR = robot_config.data_path()
DB_PATH = f"{DB_DIR}/events.db"

SNAPSHOT_INTERVAL = 15.0  # seconds between routine world-state log rows


class EventLogger:
    def __init__(self):
        self.bus = Bus()
        self.db_lock = threading.Lock()

        os.makedirs(DB_DIR, exist_ok=True)
        # check_same_thread=False: mqtt callbacks and the snapshot
        # timer run on different threads than __init__. All access is
        # serialized through self.db_lock, so this is safe.
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # WAL + synchronous=NORMAL collapses the per-commit fsync cost on the
        # movement hot path (action/result republishes up to ~10x/sec), cutting
        # SD wear with no durability loss that matters for a telemetry log.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

        self.latest_world_state = None
        self.last_battery_critical = False
        self._last_action_sig = None

    def _init_schema(self):
        with self.db_lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    topic TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic)
            """)
            self.conn.commit()

    def log_event(self, topic, payload, ts=None):
        if ts is None:
            ts = time.time()
        with self.db_lock:
            self.conn.execute(
                "INSERT INTO events (ts, topic, payload_json) VALUES (?, ?, ?)",
                (ts, topic, json.dumps(payload)),
            )
            self.conn.commit()

    # ---------- bus callbacks ----------

    def on_heard(self, payload):
        self.log_event("picarx/audio/heard", payload)

    def on_action_result(self, payload):
        # The arbiter re-sends the winning action every tick, so during steady
        # cruising this fires ~10x/sec with an identical (action, status). Skip
        # the duplicates: log only transitions and every veto. Downstream
        # consumers (pattern_miner, reflection) want the transitions, not a
        # sampled full history.
        result = payload.get("result") or {}
        status = result.get("status")
        sig = (payload.get("action"), status)
        if status != "vetoed" and sig == self._last_action_sig:
            return
        self._last_action_sig = sig
        self.log_event("picarx/action/result", payload)

    def on_coach_episode(self, payload):
        self.log_event("picarx/coach/episode", payload)

    def on_coach_surprise(self, payload):
        # Rare by construction (a proven arm failing / a written-off
        # arm succeeding): the raw material for reflection's "what
        # should have worked but didn't" ideation.
        self.log_event("picarx/coach/surprise", payload)

    def on_room_scan(self, payload):
        # One row per completed look-around head sweep (field_agent):
        # what was visible at each camera pan angle. Low volume (one
        # per "explore" command), and it's the robot's only durable
        # record of room layout - the starting point for any future
        # spatial memory/mapping work.
        self.log_event("picarx/exploration/room_scan", payload)

    def on_location_change(self, payload):
        # One row per resolved scan (location_graph): which known place
        # the robot decided it was in, or that it minted a new one.
        self.log_event("picarx/exploration/location_change", payload)

    def on_uncertainty_map(self, payload):
        # explorer.py only publishes when scores materially move, so
        # logging every publish is already change-triggered, not periodic.
        self.log_event("picarx/exploration/uncertainty_map", payload)

    def on_active_goal(self, payload):
        # Goal adoptions and clears (goal_manager). Low volume - at
        # most one every few minutes while exploring.
        self.log_event("picarx/exploration/active_goal", payload)

    def on_goal_progress(self, payload):
        # One row per finished subgoal episode: reached or abandoned.
        self.log_event("picarx/exploration/goal_progress", payload)

    def on_hypothesis(self, payload):
        # One row per resolved sensor-disagreement probe (field_agent):
        # what was ambiguous, what the test found. Reflection turns
        # repeated resolutions into durable facts ("that corner gives
        # phantom ultrasonic readings").
        self.log_event("picarx/exploration/hypothesis", payload)

    def on_rc_demonstration(self, payload):
        # One row per RC-mode demonstration episode (field_agent): the
        # obstacle situation the human drove into and the maneuver they
        # used to get out - training material from a human coach.
        self.log_event("picarx/rc/demonstration", payload)

    def on_intent_feedback(self, payload):
        # Explicit user judgments on how an utterance was interpreted
        # (web console check/X, or spoken "that's not what I meant").
        # The highest-signal training data the robot ever gets - a human
        # labeled an example - so it must be durable for reflection.
        self.log_event("picarx/intent/feedback", payload)

    def on_decision(self, payload):
        # The decision journal: every non-trivial choice any module
        # makes, with its stated reason. This is what lets the robot
        # answer "why did you do that?" from evidence instead of
        # confabulating - and lets reflection notice its own habits.
        self.log_event("picarx/decision", payload)

    def on_world_state(self, payload):
        # Always keep the freshest snapshot around for the timer loop
        # to write out on its own schedule.
        self.latest_world_state = payload

        # Edge-trigger: log immediately the moment battery goes
        # critical, rather than waiting for the next periodic tick.
        battery = payload.get("battery", {})
        is_critical = bool(battery.get("critical"))
        if is_critical and not self.last_battery_critical:
            self.log_event("picarx/state/world:battery_critical", payload)
        self.last_battery_critical = is_critical

    # ---------- periodic snapshot ----------

    def snapshot_loop(self):
        while True:
            time.sleep(SNAPSHOT_INTERVAL)
            if self.latest_world_state is not None:
                self.log_event("picarx/state/world:periodic", self.latest_world_state)

    # ---------- optional inspection helper ----------

    def print_recent(self, limit=20):
        with self.db_lock:
            rows = self.conn.execute(
                "SELECT ts, topic, payload_json FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        for ts, topic, payload_json in reversed(rows):
            print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] {topic}: {payload_json}")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.bus.subscribe("picarx/action/result", self.on_action_result)
        self.bus.subscribe("picarx/coach/episode", self.on_coach_episode)
        self.bus.subscribe("picarx/coach/surprise", self.on_coach_surprise)
        self.bus.subscribe("picarx/exploration/room_scan", self.on_room_scan)
        self.bus.subscribe("picarx/exploration/location_change", self.on_location_change)
        self.bus.subscribe("picarx/exploration/uncertainty_map", self.on_uncertainty_map)
        self.bus.subscribe("picarx/exploration/hypothesis", self.on_hypothesis)
        self.bus.subscribe("picarx/exploration/active_goal", self.on_active_goal)
        self.bus.subscribe("picarx/exploration/goal_progress", self.on_goal_progress)
        self.bus.subscribe("picarx/rc/demonstration", self.on_rc_demonstration)
        self.bus.subscribe("picarx/intent/feedback", self.on_intent_feedback)
        self.bus.subscribe("picarx/decision", self.on_decision)
        self.bus.subscribe("picarx/state/world", self.on_world_state)

        threading.Thread(target=self.snapshot_loop, daemon=True).start()

        print(f"Event Logger active, writing to {DB_PATH}")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    import sys as _sys
    logger = EventLogger()
    if "--recent" in _sys.argv:
        logger.print_recent()
    else:
        logger.run()