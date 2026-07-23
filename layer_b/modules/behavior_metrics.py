#!/usr/bin/env python3
# layer_b/modules/behavior_metrics.py
"""
Behavior metrics (Layer B) - instrument the robot's real-world collision/veto
rates so the self_trainer -> coach adopt learning loop can be PROVEN to help
(or not) before it's trusted.

The round-trip's whole premise is that sim-trained coach arms reduce how often
the robot gets stuck or vetoed on the actual carpet. This module measures that
directly and cheaply, and tags every measurement with the A/B condition
(experiment.py, chosen by coach) so an offline report (ab_report.py) can compare
adopt vs control sessions across many runs.

Signals consumed (all fail-soft; a missing producer just leaves a counter at 0):
  - picarx/action/result   -> motion ATTEMPTS and safety-daemon VETOES (by
                              reason_code). Human RC driving is excluded - it's
                              not the robot's own behaviour. veto_rate =
                              vetoes / attempts is the headline metric.
  - picarx/sensors/imu/event (impact) -> collisions actually FELT (0 while the
                              IMU is unavailable - fine, vetoes carry the signal).
  - picarx/coach/query (collision_loop) -> fail-state loops (the "keeps running
                              into something" pattern).
  - picarx/experiment/condition -> this session's A/B condition + id.

Output: a rolling checkpoint summary appended to data/behavior_metrics.jsonl
every CHECKPOINT_SEC (and once at startup), so an abrupt power-off loses at most
one interval. The report keeps the LATEST checkpoint per session_id. The counts
are cumulative for the session; the writer self-truncates the file so it can't
fill the SD card. Issues no motion and writes no database.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config

import json
import threading
import time

DATA_DIR = robot_config.data_path()
METRICS_PATH = f"{DATA_DIR}/behavior_metrics.jsonl"
MAX_LOG_BYTES = 2 * 1024 * 1024   # self-truncate past this, keep the newer half

CHECKPOINT_SEC = float(robot_config.get(
    "experiment", "checkpoint_sec", 30.0, env="EXPERIMENT_CHECKPOINT_SEC"))

MOTION_DIRECTIONS = ("forward", "backward", "turn")


class SessionMetrics:
    """Pure per-session counters. No bus, no IO - unit-testable off-robot."""

    def __init__(self, session_id=None, condition="unknown", started_at=0.0):
        self.session_id = session_id
        self.condition = condition
        self.started_at = started_at
        self.move_attempts = 0
        self.vetoes = 0
        self.veto_reasons = {}
        self.impacts = 0
        self.fail_loops = 0

    def record_action(self, source, action, status, reason_code=None):
        """One picarx/action/result. Counts a motion attempt (and a veto, if it
        was vetoed). Human RC driving isn't the robot's own behaviour, so it's
        excluded from the rate."""
        if source == "rc":
            return
        direction = (action or {}).get("direction")
        if direction not in MOTION_DIRECTIONS:
            return
        self.move_attempts += 1
        if status == "vetoed":
            self.vetoes += 1
            code = reason_code or "unknown"
            self.veto_reasons[code] = self.veto_reasons.get(code, 0) + 1

    def record_impact(self):
        self.impacts += 1

    def record_fail_loop(self):
        self.fail_loops += 1

    def summary(self, now):
        attempts = self.move_attempts
        return {
            "type": "behavior_metrics",
            "ts": now,
            "session_id": self.session_id,
            "condition": self.condition,
            "uptime_sec": round(now - self.started_at, 1) if self.started_at else None,
            "move_attempts": attempts,
            "vetoes": self.vetoes,
            "veto_rate": round(self.vetoes / attempts, 4) if attempts else 0.0,
            "impacts": self.impacts,
            "fail_loops": self.fail_loops,
            "veto_reasons": dict(self.veto_reasons),
        }


class BehaviorMetrics:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        # Until coach announces the A/B condition, tag with our own boot time so
        # the data is never orphaned; the experiment message aligns the id.
        boot = time.time()
        self.metrics = SessionMetrics(session_id=boot, condition="unknown",
                                      started_at=boot)

    # ---------- inbound ----------

    def on_action_result(self, payload):
        result = payload.get("result") or {}
        with self.lock:
            self.metrics.record_action(
                payload.get("source"), payload.get("action"),
                result.get("status"), result.get("reason_code"))

    def on_imu_event(self, payload):
        if payload.get("kind") == "impact":
            with self.lock:
                self.metrics.record_impact()

    def on_coach_query(self, payload):
        if payload.get("situation") == "collision_loop":
            with self.lock:
                self.metrics.record_fail_loop()

    def on_experiment(self, payload):
        condition = payload.get("condition")
        session_id = payload.get("session_id")
        with self.lock:
            if condition:
                self.metrics.condition = condition
            if session_id is not None:
                self.metrics.session_id = session_id

    # ---------- checkpoint writer (self-truncating) ----------

    def _write_checkpoint(self, now=None):
        now = now if now is not None else time.time()
        with self.lock:
            entry = self.metrics.summary(now)
        os.makedirs(DATA_DIR, exist_ok=True)
        line = json.dumps(entry) + "\n"
        try:
            if os.path.exists(METRICS_PATH) and os.path.getsize(METRICS_PATH) > MAX_LOG_BYTES:
                with open(METRICS_PATH, "rb") as f:
                    f.seek(-MAX_LOG_BYTES // 2, os.SEEK_END)
                    tail = f.read()
                _, _, tail = tail.partition(b"\n")   # drop the partial leading line
                with open(METRICS_PATH, "wb") as f:
                    f.write(tail)
            with open(METRICS_PATH, "a") as f:
                f.write(line)
        except OSError as e:
            print(f"Behavior metrics: could not write checkpoint ({e})")
        return entry

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/action/result", self.on_action_result)
        self.bus.subscribe("picarx/sensors/imu/event", self.on_imu_event)
        self.bus.subscribe("picarx/coach/query", self.on_coach_query)
        self.bus.subscribe("picarx/experiment/condition", self.on_experiment)
        self._write_checkpoint()   # a baseline row so a session always appears
        print(f"Behavior metrics active - collision/veto instrumentation "
              f"(checkpoint every {CHECKPOINT_SEC:.0f}s -> {METRICS_PATH})")
        while True:
            time.sleep(CHECKPOINT_SEC)
            self._write_checkpoint()


if __name__ == "__main__":
    BehaviorMetrics().run()
