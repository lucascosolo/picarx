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
import robot_config

import base64
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(robot_config.get("web_console", "port", 8088, env="WEB_CONSOLE_PORT"))
HTML_PATH = "/home/picarx/layer_b/web_ui/console.html"
LOG_LINES = 40

# Live camera view. vision_basic.py owns the camera and only encodes
# frames while we ask it to (picarx/vision/stream_control), so the view
# costs nothing until someone opens it. We tell vision to start on the
# first frame request and, via a watchdog, to stop once the browser has
# gone quiet for CAMERA_IDLE_SEC (tab closed / live toggle off) - so a
# forgotten tab can't pin vision's CPU encoding frames nobody sees.
VISION_STREAM_CONTROL = "picarx/vision/stream_control"
VISION_FRAME_TOPIC = "picarx/vision/frame"
CAMERA_IDLE_SEC = 5.0


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
        # Live camera view
        self.camera_jpeg = None       # latest JPEG bytes, or None until first frame
        self.camera_ts = 0.0          # when that frame was captured (robot clock)
        self.last_view_request = 0.0  # last time a browser fetched /camera.jpg
        self.stream_on = False        # whether we've told vision to encode frames

    def add_log(self, kind, text):
        with self.lock:
            self.log.appendleft({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})

    def set_camera_frame(self, jpeg_bytes, ts):
        with self.lock:
            self.camera_jpeg = jpeg_bytes
            self.camera_ts = ts

    def note_view_demand(self):
        """A browser asked for a frame. Records the demand and reports
        whether this is a fresh off->on transition (caller then tells
        vision to start streaming)."""
        with self.lock:
            self.last_view_request = time.time()
            was_on = self.stream_on
            self.stream_on = True
            return not was_on

    def set_stream(self, on):
        """Force stream state (explicit toggle / watchdog). Returns True
        if it actually changed, so the caller only publishes on edges."""
        with self.lock:
            if self.stream_on == on:
                return False
            self.stream_on = on
            if on:
                self.last_view_request = time.time()
            return True

    def camera_idle_expired(self, now):
        """True (once, on the transition) if streaming is on but no frame
        has been requested for CAMERA_IDLE_SEC - time to stop vision."""
        with self.lock:
            if self.stream_on and now - self.last_view_request > CAMERA_IDLE_SEC:
                self.stream_on = False
                return True
            return False

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
        elif self.path == "/camera.jpg" or self.path.startswith("/camera.jpg?"):
            # Fetching a frame is itself the "someone is watching" signal:
            # start vision's stream on the first request, keep the
            # watchdog fed on every one.
            if STATE.note_view_demand():
                BUS.publish(VISION_STREAM_CONTROL, {"enabled": True})
            with STATE.lock:
                jpeg = STATE.camera_jpeg
            if jpeg is None:
                # Streaming just started; no frame has arrived yet. 503
                # tells the client to keep polling without logging an error.
                self._send(503, {"error": "no frame yet"})
            else:
                self._send(200, jpeg, "image/jpeg")
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
        elif self.path == "/camera":
            # Explicit toggle from the live-view switch. Turning it off
            # stops vision encoding immediately rather than waiting out
            # the idle watchdog; turning it on pre-warms the stream.
            enabled = bool(body.get("enabled", False))
            if STATE.set_stream(enabled):
                BUS.publish(VISION_STREAM_CONTROL, {"enabled": enabled})
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

def on_vision_frame(p):
    b64 = p.get("jpeg")
    if not b64:
        return
    try:
        STATE.set_camera_frame(base64.b64decode(b64), float(p.get("ts") or 0.0))
    except (ValueError, TypeError):
        pass  # malformed frame - just skip it, next one will be fine


def camera_watchdog():
    """Stop vision's encoder when the browser stops asking for frames
    (live view toggled off, or the tab was simply closed). Runs off the
    request path so it fires even when nothing is hitting the server."""
    while True:
        time.sleep(1.0)
        if STATE.camera_idle_expired(time.time()):
            BUS.publish(VISION_STREAM_CONTROL, {"enabled": False})
            print("Web console: live view idle - stopping camera stream")


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
    BUS.subscribe(VISION_FRAME_TOPIC, on_vision_frame)

    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    except OSError as e:
        print(f"Web console: cannot bind port {PORT} ({e}) - idling (fail-soft)")
        while True:
            time.sleep(60)
    threading.Thread(target=camera_watchdog, daemon=True).start()
    print(f"Web console active: http://<robot-ip>:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
