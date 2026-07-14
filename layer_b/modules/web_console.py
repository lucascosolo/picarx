#!/usr/bin/env python3
# /home/picarx/layer_b/modules/web_console.py
"""
Web Console (Layer B) - phone/laptop control panel for testing without
shouting across a loud room.

One design rule keeps this honest: the console does NOT get its own
command paths. Every button and text box submits a phrase to POST /say,
which is published on picarx/audio/heard - byte-for-byte what the
microphone would have produced. field_agent, tools_registry and
companion react exactly as they would to voice, so anything you test
here is tested for voice too. The only non-voice control is the mic
kill-switch (picarx/audio/mic_control), which by definition can't be a
voice command (a disabled mic can't hear you re-enable it).

Endpoints (all JSON):
  GET  /        the single-page console (web_ui/console.html)
  GET  /state   latest mic/radio/world/location/goal + speak/heard log
  POST /say     {"text": "..."}    -> picarx/audio/heard
  POST /mic     {"enabled": bool}  -> picarx/audio/mic_control

Serves plain HTTP on the LAN with no authentication - anyone on your
network can drive the robot. That's the right trade-off for a hobby
robot on a home network; do not port-forward it to the internet.

Stdlib only (ThreadingHTTPServer); ~zero idle cost. Fail-soft: if the
port is taken the module logs and idles rather than crash-looping.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("WEB_CONSOLE_PORT", "8088"))
HTML_PATH = "/home/picarx/layer_b/web_ui/console.html"
LOG_LINES = 40


class ConsoleState:
    """Latest-value cache + rolling event log, fed by MQTT callbacks."""
    def __init__(self):
        self.lock = threading.Lock()
        self.mic_enabled = True
        self.radio = {}
        self.location = {}
        self.goal = {}
        self.world = {}
        self.log = deque(maxlen=LOG_LINES)

    def add_log(self, kind, text):
        with self.lock:
            self.log.appendleft({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})

    def snapshot(self):
        with self.lock:
            battery = (self.world.get("battery") or {})
            return {
                "mic_enabled": self.mic_enabled,
                "radio": self.radio,
                "location": self.location,
                "goal": self.goal,
                "battery_v": battery.get("voltage"),
                "battery_low": battery.get("low"),
                "distance_cm": self.world.get("distance_cm"),
                "log": list(self.log),
            }


STATE = ConsoleState()
BUS = None  # set in main


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request stderr spam

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(200, b"<h1>PicarX console</h1><p>web_ui/console.html missing.</p>",
                           "text/html; charset=utf-8")
        elif self.path == "/state":
            self._send(200, STATE.snapshot())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._read_json()
        if body is None:
            self._send(400, {"error": "bad json"})
            return
        if self.path == "/say":
            text = (body.get("text") or "").strip().lower()
            if not text:
                self._send(400, {"error": "empty text"})
                return
            # Exactly what the microphone would publish - see module doc.
            BUS.publish("picarx/audio/heard", {"text": text})
            STATE.add_log("you", text)
            self._send(200, {"ok": True})
        elif self.path == "/mic":
            BUS.publish("picarx/audio/mic_control", {"enabled": bool(body.get("enabled", True))})
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})


# ---------- MQTT feeds ----------

def on_speak(p):
    if p.get("text"):
        STATE.add_log("robot", p["text"])

def on_heard(p):
    text = p.get("text")
    if not text:
        return
    # Our own /say publishes onto this same topic, and we already
    # logged it as "you" - don't double-log the echo.
    with STATE.lock:
        if STATE.log and STATE.log[0]["kind"] == "you" and STATE.log[0]["text"] == text:
            return
    STATE.add_log("heard", text)

def on_mic_state(p):
    with STATE.lock:
        STATE.mic_enabled = bool(p.get("enabled", True))

def on_radio_state(p):
    with STATE.lock:
        STATE.radio = p

def on_world(p):
    with STATE.lock:
        STATE.world = p

def on_location(p):
    with STATE.lock:
        STATE.location = p

def on_goal(p):
    with STATE.lock:
        STATE.goal = p if p.get("location_id") is not None else {}


def main():
    global BUS
    BUS = Bus()
    BUS.subscribe("picarx/audio/speak", on_speak)
    BUS.subscribe("picarx/audio/heard", on_heard)
    BUS.subscribe("picarx/audio/mic_state", on_mic_state)
    BUS.subscribe("picarx/tools/radio_state", on_radio_state)
    BUS.subscribe("picarx/state/world", on_world)
    BUS.subscribe("picarx/exploration/location_change", on_location)
    BUS.subscribe("picarx/exploration/active_goal", on_goal)

    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    except OSError as e:
        print(f"Web console: cannot bind port {PORT} ({e}) - idling (fail-soft)")
        while True:
            time.sleep(60)
    print(f"Web console active: http://<robot-ip>:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
