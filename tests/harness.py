"""
Shared off-robot test harness.

The Layer B modules are written to run on the Pi: they `import broker_client`
(which needs paho-mqtt) and pull in hardware/vision stacks (vosk, picamera2,
cv2, numpy). None of that exists on a CI box, so importing this module FIRST
stubs those dependencies and puts layer_b/ + layer_b/modules/ on sys.path, so
the real application code can be imported and exercised unchanged.

Every test file starts with:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import harness  # noqa: E402  - installs stubs + sys.path, exposes FakeBus

Nothing here touches real hardware, the network, or the robot's DB paths;
tests always pass explicit tmp db_path=... values into the stores.
"""
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER_B = os.path.join(REPO_ROOT, "layer_b")
MODULES = os.path.join(LAYER_B, "modules")

# Stub the hardware / C-extension dependencies that aren't installed off-robot.
# Import-time only: tests never call into these, they drive pure logic.
for _name in ("numpy", "cv2", "vosk", "picamera2",
              "paho", "paho.mqtt", "paho.mqtt.client"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

# A couple of modules do `from vosk import Model, KaldiRecognizer` /
# `from picamera2 import Picamera2` at import time, so the bare stub module
# isn't enough - give it the names (trivial no-op classes; tests never call
# into real speech/camera code).
for _attr in ("Model", "KaldiRecognizer"):
    if not hasattr(sys.modules["vosk"], _attr):
        setattr(sys.modules["vosk"], _attr,
                type(_attr, (), {"__init__": lambda self, *a, **k: None}))
if not hasattr(sys.modules["picamera2"], "Picamera2"):
    sys.modules["picamera2"].Picamera2 = type("Picamera2", (), {})

# vision_basic calls cv2.setNumThreads() at import time (module level, before
# any test runs); give the bare cv2 stub a no-op so the module imports off-robot
# and its pure helpers (e.g. pick_overhead) can be exercised. Tests never touch
# the real DNN/camera paths.
if not hasattr(sys.modules["cv2"], "setNumThreads"):
    sys.modules["cv2"].setNumThreads = lambda *a, **k: None

for _p in (MODULES, LAYER_B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import broker_client  # noqa: E402  - real module, but we swap its Bus below


class FakeBus:
    """In-memory stand-in for broker_client.Bus.

    Records everything published (assert on it) and every subscription (so a
    test can hand a module a payload exactly as the broker would, via
    deliver()). No threads, no network - callbacks run synchronously on the
    calling thread, which makes ordering in tests deterministic.
    """

    def __init__(self, *args, **kwargs):
        self.published = []            # [(topic, payload_dict), ...]
        self.subscriptions = {}        # topic -> [callback, ...]
        self.heartbeat_status_fn = None  # last fn passed to set_heartbeat_status

    def subscribe(self, topic, callback):
        self.subscriptions.setdefault(topic, []).append(callback)

    def set_heartbeat_status(self, status_fn):
        """Mirror Bus.set_heartbeat_status: record the module's status_fn so a
        test can assert it was registered and exercise it directly."""
        self.heartbeat_status_fn = status_fn

    def publish(self, topic, payload):
        self.published.append((topic, dict(payload)))

    # --- test helpers ---
    def deliver(self, topic, payload):
        """Fire a payload to whatever subscribed to `topic`, like the broker."""
        for callback in list(self.subscriptions.get(topic, [])):
            callback(payload)

    def of(self, topic):
        """Every payload published on `topic`, in order."""
        return [p for (t, p) in self.published if t == topic]

    def last(self, topic):
        msgs = self.of(topic)
        return msgs[-1] if msgs else None

    def clear(self):
        self.published.clear()


# Any module doing `from broker_client import Bus` now gets FakeBus.
broker_client.Bus = FakeBus
