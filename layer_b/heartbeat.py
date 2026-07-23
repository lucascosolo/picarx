#!/usr/bin/env python3
# layer_b/heartbeat.py
"""
Unified per-module liveness heartbeat.

The per-module `.../status` beacons (imu.py's picarx/sensors/imu/status,
self_trainer.py's picarx/self_trainer/status) proved their worth for field
debugging - a bus-visible "is this thing alive and what's it doing" you can
watch with `mosquitto_sub`. This generalizes that to EVERY module, for free and
with no per-module code: every Layer B module runs as its own process and builds
a Bus, so the Bus itself heartbeats on a single shared topic.

  topic:   picarx/module/heartbeat
  payload: {name, pid, ts, uptime_sec, seq, status?}

`name` is derived from the process entrypoint (field_agent.py -> "field_agent").
A module that wants to fold in a little self-reported status can register a
status_fn returning a small dict (see Bus.set_heartbeat_status); it rides under
"status".

Liveness semantics AND their limit: the heartbeat runs on a daemon thread, so it
proves the PROCESS is alive and the bus is publishing - if a module crashes or
exits, its heartbeats stop and a consumer (debug_monitor) notices it went
silent. It does NOT prove the module's main loop isn't wedged while the process
lingers; debug_monitor's per-process CPU view is the complement for that case.

The emitter is pure given an injected publish function and clock, so it's
unit-testable off-robot with no threads or broker.
"""
import os
import sys
import threading
import time

HEARTBEAT_TOPIC = "picarx/module/heartbeat"
DEFAULT_INTERVAL_SEC = 10.0


def module_name(argv0=None):
    """A stable module name from the process entrypoint path
    ('.../field_agent.py' -> 'field_agent'). Falls back to 'unknown'."""
    argv0 = argv0 if argv0 is not None else (sys.argv[0] if sys.argv else "")
    base = os.path.basename(argv0 or "")
    if base.endswith(".py"):
        base = base[:-3]
    return base or "unknown"


class HeartbeatEmitter:
    """Builds and paces heartbeat payloads. Pure given an injected publish_fn
    (called publish_fn(topic, dict)) and clock, so tests drive it directly with
    no threads or broker."""

    def __init__(self, publish_fn, name, pid, interval=DEFAULT_INTERVAL_SEC,
                 clock=time.time, status_fn=None, started_at=None):
        self.publish_fn = publish_fn
        self.name = name
        self.pid = pid
        self.interval = interval
        self.clock = clock
        self.status_fn = status_fn
        self.started_at = started_at if started_at is not None else clock()
        self.seq = 0
        self._last_emit = None

    def payload(self, now):
        p = {"name": self.name, "pid": self.pid, "ts": now,
             "uptime_sec": round(now - self.started_at, 1), "seq": self.seq}
        if self.status_fn is not None:
            try:
                status = self.status_fn()
                if status:
                    p["status"] = status
            except Exception:
                pass   # a broken status_fn must never silence the heartbeat
        return p

    def due(self, now):
        return self._last_emit is None or (now - self._last_emit) >= self.interval

    def tick(self, now=None):
        """Emit a heartbeat if due; return the payload emitted, else None."""
        now = self.clock() if now is None else now
        if not self.due(now):
            return None
        payload = self.payload(now)
        self.seq += 1
        self._last_emit = now
        try:
            self.publish_fn(HEARTBEAT_TOPIC, payload)
        except Exception:
            pass       # a publish hiccup must never kill the heartbeat loop
        return payload


def start(publish_fn, name=None, interval=DEFAULT_INTERVAL_SEC, status_fn=None,
          spawn=None):
    """Spawn a daemon thread that heartbeats forever. `publish_fn(topic, dict)`
    is the Bus's publish. Returns the emitter (mainly for tests / status_fn
    wiring). Fail-soft: any error building it returns None rather than raising,
    so a heartbeat problem can never stop a module from coming up."""
    try:
        emitter = HeartbeatEmitter(publish_fn, name or module_name(), os.getpid(),
                                   interval=interval, status_fn=status_fn)
    except Exception:
        return None

    def _loop():
        while True:
            emitter.tick()
            time.sleep(max(0.5, emitter.interval))

    spawn = spawn or (lambda fn: threading.Thread(
        target=fn, name="heartbeat", daemon=True).start())
    spawn(_loop)
    return emitter
