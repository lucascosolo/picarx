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

    def subscribe(self, topic, callback):
        self.subscriptions.setdefault(topic, []).append(callback)

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
