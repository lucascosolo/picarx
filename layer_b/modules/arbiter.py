#!/usr/bin/env python3
# layer_b/modules/arbiter.py
"""
Central motion arbiter for Layer B.

This is the ONLY module that should ever open a socket to the safety
daemon for movement commands (queries like battery/distance status are
harmless to leave decentralized, but 'direction'/'turn' actions must
all flow through here). Every other Layer B module publishes an
*intent* on the bus instead of touching the socket directly.

Intent message format (published to picarx/intent/move):
{
    "source": "reflex_explorer",   # who wants this
    "priority": 10,                 # higher wins
    "action": {"direction": "forward", "speed": 25},
    "ttl": 1.0                       # seconds this intent stays valid
                                     # if no newer one arrives from
                                     # the same source
}

Sources are expected to keep re-publishing their intent while they
still want it (e.g. every loop tick). If a source goes silent for
longer than its ttl, its intent is dropped automatically. This means
a crashed or hung module can never leave the robot stuck mid-action.

Result of each executed/vetoed action is published back to
picarx/action/result so sources (and later, a logger) can see what
actually happened to their request.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus

import socket
import json
import time
import threading

SOCKET_PATH = "/tmp/picarx_safety.sock"
TICK_HZ = 10  # how often we resolve intents and send to the daemon
DEFAULT_TTL = 1.0
STOP_ACTION = {"direction": "stop"}


class Arbiter:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        # source_name -> {"priority", "action", "expires_at"}
        self.intents = {}
        self.last_sent_action = None
        self.last_look_sent = None

    # ---------- intent bookkeeping ----------

    def on_intent(self, payload):
        source = payload.get("source")
        action = payload.get("action")
        if not source or not action:
            print(f"Arbiter: dropping malformed intent {payload}")
            return

        priority = payload.get("priority", 0)
        ttl = payload.get("ttl", DEFAULT_TTL)

        with self.lock:
            self.intents[source] = {
                "priority": priority,
                "action": action,
                "expires_at": time.time() + ttl,
            }

    def on_cancel(self, payload):
        """A source can explicitly give up its intent early."""
        source = payload.get("source")
        with self.lock:
            self.intents.pop(source, None)

    def on_look(self, payload):
        """
        Camera head (pan/tilt) channel - deliberately OUTSIDE the
        priority/winner system above. Look actions don't drive the
        wheels, so they must not compete with (or starve) movement
        intents for the single winner slot; they're forwarded to the
        safety daemon directly, deduplicated so a re-published
        identical head position doesn't hit the socket again.
        """
        action = payload.get("action") or {}
        if action.get("direction") != "look":
            return
        key = (action.get("pan", 0), action.get("tilt", 0))
        if key == self.last_look_sent:
            return
        self.last_look_sent = key
        self.send_to_safety(action)

    def _prune_expired(self):
        now = time.time()
        with self.lock:
            expired = [s for s, i in self.intents.items() if i["expires_at"] < now]
            for s in expired:
                del self.intents[s]

    def _pick_winner(self):
        with self.lock:
            if not self.intents:
                return None, None
            source = max(self.intents, key=lambda s: self.intents[s]["priority"])
            return source, self.intents[source]["action"]

    # ---------- talking to the safety daemon ----------

    def send_to_safety(self, action):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                s.connect(SOCKET_PATH)
                s.sendall(json.dumps(action).encode())
                data = s.recv(1024)
            return json.loads(data.decode())
        except Exception as e:
            print(f"Arbiter: safety link error: {e}")
            return None

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/intent/move", self.on_intent)
        self.bus.subscribe("picarx/intent/cancel", self.on_cancel)
        self.bus.subscribe("picarx/intent/look", self.on_look)
        print("Arbiter active. Waiting for motion intents.")

        period = 1.0 / TICK_HZ
        while True:
            self._prune_expired()
            source, action = self._pick_winner()

            if action is None:
                action = STOP_ACTION
                source = None

            # Avoid spamming identical repeated actions to the daemon
            # every tick if nothing has changed AND it's a no-op stop.
            if action != self.last_sent_action or action != STOP_ACTION:
                result = self.send_to_safety(action)
                self.last_sent_action = action

                if result is not None:
                    self.bus.publish("picarx/action/result", {
                        "source": source,
                        "action": action,
                        "result": result,
                    })
                    if result.get("status") == "vetoed":
                        # Safety layer overruled us; drop the losing
                        # intent so we don't just re-request the same
                        # vetoed action next tick.
                        with self.lock:
                            if source in self.intents:
                                del self.intents[source]

            time.sleep(period)


if __name__ == "__main__":
    Arbiter().run()