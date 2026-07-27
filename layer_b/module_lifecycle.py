#!/usr/bin/env python3
"""Pure need-driven module lifecycle policy.

The registry describes what a module is capable of; this planner decides
which entries are needed now.  Keeping the policy free of subprocess and MQTT
code makes lifecycle decisions deterministic and preserves the existing
module tests.

Activation modes:

``always``
    Long-lived infrastructure or observers.  These are the small set of
    modules needed to hear commands, maintain safety-adjacent state, or make
    the robot autonomous.

``state``
    Active only while RobotState is one of the configured states.  Vision is
    the first consumer: it should not be alive to compete with gesture mode.

``demand``
    Started by a typed bus request.  Persistent controls (follow and gesture)
    remain active until their explicit false/stop request; one-shot tools get
    a bounded lease and are stopped after their quiet period.

An entry without an activation field retains the old ``enabled`` behavior and
is treated as ``always``.  That compatibility rule lets local overlays and
third-party modules migrate incrementally.
"""
import time


STATE_TOPIC = "picarx/state/current"


def _activation(entry):
    value = entry.get("activation", "always")
    if isinstance(value, str):
        return {"mode": value}
    if isinstance(value, dict):
        return dict(value)
    return {"mode": "always"}


def _enabled(entry):
    # enabled remains an availability/kill switch for emergency rollback and
    # old local overlays; it is no longer the normal way to select behavior.
    return bool(entry.get("enabled", True))


class NeedPlanner:
    """Track current robot state and externally expressed module needs."""

    def __init__(self, registry, clock=None):
        self.registry = {entry["name"]: dict(entry) for entry in registry}
        self.clock = clock or time.time
        self.state = "IDLE"
        self._needs = {}       # name -> {expires_at, payload}
        self._last_payload = {}  # name -> most recent trigger for replay

    def replace_registry(self, registry):
        self.registry = {entry["name"]: dict(entry) for entry in registry}
        known = set(self.registry)
        self._needs = {name: value for name, value in self._needs.items()
                       if name in known}
        self._last_payload = {name: value for name, value in self._last_payload.items()
                              if name in known}

    def set_state(self, payload):
        if isinstance(payload, dict):
            self.state = str(payload.get("state") or "IDLE").upper()
        else:
            self.state = "IDLE"

    def _entry_for_topic(self, topic):
        for name, entry in self.registry.items():
            activation = _activation(entry)
            if activation.get("mode") != "demand":
                continue
            if activation.get("topic") == topic:
                return name, entry, activation
        return None, None, None

    @staticmethod
    def _is_off(payload, activation):
        payload = payload if isinstance(payload, dict) else {}
        field = activation.get("enabled_field")
        if field and field in payload and not bool(payload.get(field)):
            return True
        command = str(payload.get("command") or payload.get("op") or "").lower()
        return command in set(activation.get("stop_commands", []))

    def observe_demand(self, topic, payload, now=None):
        """Apply a demand-topic event and return its module name, if any.

        The return value lets the process supervisor start a module and replay
        the triggering request when the module was not already alive.
        """
        now = self.clock() if now is None else float(now)
        name, entry, activation = self._entry_for_topic(topic)
        if name is None:
            return None
        if self._is_off(payload, activation):
            self._needs.pop(name, None)
            self._last_payload.pop(name, None)
            return name

        persistent = bool(activation.get("persistent", False))
        ttl = float(activation.get("ttl_sec", 120.0))
        self._needs[name] = {
            "expires_at": None if persistent else now + max(1.0, ttl),
            "payload": dict(payload or {}),
        }
        self._last_payload[name] = dict(payload or {})
        return name

    def observe_state_signal(self, topic, payload, now=None):
        """Refresh a demand lease from a module's published state topic."""
        now = self.clock() if now is None else float(now)
        payload = payload if isinstance(payload, dict) else {}
        for name, entry in self.registry.items():
            activation = _activation(entry)
            if activation.get("mode") != "demand" or activation.get("state_topic") != topic:
                continue
            field = activation.get("state_field", "enabled")
            if field not in payload:
                continue
            active = bool(payload.get(field))
            if active:
                ttl = float(activation.get("ttl_sec", 120.0))
                self._needs[name] = {"expires_at": now + max(1.0, ttl),
                                      "payload": self._last_payload.get(name, {})}
            elif activation.get("state_clears", True):
                self._needs.pop(name, None)

    def _need_active(self, name, now):
        need = self._needs.get(name)
        if need is None:
            return False
        expires_at = need.get("expires_at")
        if expires_at is not None and expires_at <= now:
            del self._needs[name]
            return False
        return True

    def desired_names(self, now=None):
        now = self.clock() if now is None else float(now)
        desired = set()
        for name, entry in self.registry.items():
            if not _enabled(entry):
                continue
            activation = _activation(entry)
            mode = activation.get("mode", "always")
            if mode == "always":
                desired.add(name)
            elif mode == "state":
                states = {str(s).upper() for s in activation.get("states", [])}
                if self.state in states:
                    desired.add(name)
            elif mode == "demand" and self._need_active(name, now):
                desired.add(name)
        return desired

    def replay_payload(self, name):
        payload = self._last_payload.get(name)
        return dict(payload) if payload is not None else None

    def status(self, now=None):
        now = self.clock() if now is None else float(now)
        return {
            "state": self.state,
            "desired": sorted(self.desired_names(now)),
            "needs": sorted(name for name in self._needs if self._need_active(name, now)),
            "ts": now,
        }
