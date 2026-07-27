#!/usr/bin/env python3
"""Exclusive resource/state manager for camera, head, audio and tools.

Modules claim a state over the bus instead of independently deciding that they
own the camera or head.  Claims are leases: a crashed module cannot hold a
resource forever.  The manager never sends a wheel command and never replaces
the safety daemon; ``SAFETY_STOP`` is an observable high-priority state, while
the daemon remains the final veto.

Claim topic::

    picarx/state/claim {"owner":"gesture", "state":"GESTURE_TRACKING",
                         "ttl":1.0, "reason":"user enabled gesture mode"}

Release topic::

    picarx/state/release {"owner":"gesture"}

The winning state is published on ``picarx/state/current``.  Every state is
exclusive; a SPEAKING claim can temporarily preempt gesture processing, and
the gesture claim becomes active again when speech releases its lease.
"""
import enum
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus


class RobotState(str, enum.Enum):
    IDLE = "IDLE"
    GESTURE_TRACKING = "GESTURE_TRACKING"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    SPEAKING = "SPEAKING"
    REMOTE_ASSIST = "REMOTE_ASSIST"
    RC = "RC"
    SAFETY_STOP = "SAFETY_STOP"


# Higher priority wins, but claims at the same priority are deterministic:
# the most recently renewed claim wins.  Safety is still enforced below this
# layer by safety_daemon.py.
STATE_PRIORITY = {
    RobotState.IDLE: 0,
    RobotState.OBJECT_DETECTION: 20,
    RobotState.GESTURE_TRACKING: 30,
    # A persistent remote session must not prevent TTS from taking the
    # camera/CPU budget; its file operations are paused while speech owns the
    # active state and resume when the speech lease is released.
    RobotState.REMOTE_ASSIST: 40,
    RobotState.SPEAKING: 50,
    RobotState.RC: 80,
    RobotState.SAFETY_STOP: 100,
}


def parse_state(value):
    """Return a RobotState or None for malformed external input."""
    if isinstance(value, RobotState):
        return value
    try:
        return RobotState(str(value).upper())
    except (TypeError, ValueError):
        return None


class StateManager:
    """Thread-safe pure lease manager; the MQTT wrapper is below."""

    def __init__(self):
        self.lock = threading.RLock()
        self.claims = {}  # owner -> {state, priority, expires_at, reason, renewed_at}
        self._sequence = 0

    def _winner_locked(self, now):
        expired = [owner for owner, claim in self.claims.items()
                   if claim["expires_at"] <= now]
        for owner in expired:
            del self.claims[owner]
        if not self.claims:
            return None
        return max(self.claims.values(),
                   key=lambda c: (c["priority"], c["renewed_at"], c["owner"]))

    def winner(self, now=None):
        now = time.time() if now is None else float(now)
        with self.lock:
            claim = self._winner_locked(now)
            return dict(claim) if claim else {
                "owner": "robot_state", "state": RobotState.IDLE.value,
                "priority": 0, "expires_at": now, "reason": "no active claim",
                "renewed_at": now,
            }

    def claim(self, owner, state, ttl=1.0, priority=None, reason="", now=None):
        """Renew/replace an owner's lease.

        Invalid claims are rejected without changing the current winner.  A
        zero/negative TTL is not accepted because a lease that expires as it is
        installed creates unsafe, nondeterministic transitions.
        """
        now = time.time() if now is None else float(now)
        owner = str(owner or "").strip()[:80]
        parsed = parse_state(state)
        try:
            ttl = float(ttl)
        except (TypeError, ValueError):
            ttl = 0.0
        if not owner or parsed is None or ttl <= 0:
            return {"accepted": False, "reason": "invalid claim", "state": self.winner(now)}
        with self.lock:
            self._sequence += 1
            p = STATE_PRIORITY[parsed] if priority is None else int(priority)
            self.claims[owner] = {
                "owner": owner,
                "state": parsed.value,
                "priority": p,
                "expires_at": now + min(ttl, 3600.0),
                "reason": str(reason or "")[:240],
                "renewed_at": self._sequence,
            }
            winner = self._winner_locked(now)
            return {"accepted": True, "state": dict(winner), "claim": dict(self.claims[owner])}

    def release(self, owner, now=None):
        now = time.time() if now is None else float(now)
        owner = str(owner or "").strip()[:80]
        with self.lock:
            removed = self.claims.pop(owner, None) is not None
            winner = self._winner_locked(now)
            return {"released": removed, "state": dict(winner) if winner else {
                "owner": "robot_state", "state": RobotState.IDLE.value,
                "priority": 0, "expires_at": now, "reason": "no active claim",
                "renewed_at": now,
            }}

    def snapshot(self, now=None):
        now = time.time() if now is None else float(now)
        with self.lock:
            self._winner_locked(now)
            return [dict(c) for c in self.claims.values()]


class RobotStateModule:
    def __init__(self):
        self.bus = Bus()
        self.manager = StateManager()
        self._last_signature = None

    def _publish(self, reason="state update", force=False, now=None):
        now = time.time() if now is None else now
        state = self.manager.winner(now)
        payload = {
            "state": state["state"], "owner": state["owner"],
            "priority": state["priority"], "reason": state.get("reason", reason),
            "expires_at": state["expires_at"], "claims": self.manager.snapshot(now),
            "ts": now,
        }
        signature = (payload["state"], payload["owner"],
                     tuple((c["owner"], c["state"], c["priority"])
                           for c in payload["claims"]))
        if force or signature != self._last_signature:
            self._last_signature = signature
            self.bus.publish("picarx/state/current", payload)
            self.bus.publish("picarx/decision", {
                "source": "robot_state", "kind": "state_transition",
                "choice": {"state": payload["state"], "owner": payload["owner"]},
                "reason": payload["reason"], "ts": now,
            })

    def on_claim(self, payload):
        result = self.manager.claim(
            payload.get("owner"), payload.get("state"), payload.get("ttl", 1.0),
            payload.get("priority"), payload.get("reason"), time.time())
        if not result.get("accepted"):
            self.bus.publish("picarx/state/rejected", result)
        self._publish(force=True)

    def on_release(self, payload):
        self.manager.release(payload.get("owner"), time.time())
        self._publish(force=True)

    def on_query(self, payload):
        self._publish(force=True)

    def run(self):
        self.bus.subscribe("picarx/state/claim", self.on_claim)
        self.bus.subscribe("picarx/state/release", self.on_release)
        self.bus.subscribe("picarx/state/query", self.on_query)
        self._publish(force=True)
        print("Robot state manager active")
        while True:
            time.sleep(0.2)
            self._publish()


if __name__ == "__main__":
    RobotStateModule().run()
