#!/usr/bin/env python3
# layer_b/modules/web_console.py
"""
Web Console (Layer B) - phone/laptop control panel for testing without
shouting across a loud room.

The features outgrew one screen, so the UI is split into pages that share a
top nav, a common stylesheet (web_ui/app.css) and helper script (web_ui/app.js):

  /          Dashboard   - status overview, quick commands, say-anything, log
  /drive     Drive & Cam - live camera + bounding boxes + manual RC driving
  /training  Training    - label the object in view, ask about objects/history
  /people    People      - face enrolment, following, places & navigation
  /audio     Audio+Radio - mic/speaker kill-switches, internet radio
  /config    Config      - every config.json knob, editable in the browser

One design rule keeps most of this honest: the console does NOT get its own
command paths. Almost every button and text box submits a phrase to POST /say,
which is published on picarx/audio/heard - byte-for-byte what the microphone
would have produced. field_agent, tools_registry and companion react exactly
as they would to voice, so anything tested here is tested for voice too. The
few non-voice controls exist because they can't BE voice commands: the mic
kill-switch (a disabled mic can't hear "mic on"), RC driving, camera streaming,
perception relabels, and editing the config file.

The live camera is demand-gated: vision_basic only encodes frames while a
browser is actually fetching /camera.jpg (with an idle watchdog), so the camera
costs nothing unless someone is on the Drive page watching it.

HTTP endpoints (JSON unless noted):
  GET  /,/drive,/training,/people,/audio,/config   the pages (HTML)
  GET  /app.css /app.js                            shared static assets
  GET  /state         status cache + speak/heard log + places/people
  GET  /boxes         camera-overlay boxes for the current frame
  GET  /objects       objects currently tracked in view (id/label/confidence)
  GET  /camera.jpg    latest JPEG frame (also arms the stream)
  GET  /facts[?q=]    recent (or searched) semantic-memory facts
  GET  /config/data   the full config tree + per-knob help + env note
  POST /say /mic /speaker /feedback /label /rc /rc/drive /camera   (as before)
  POST /config/save   {"config": {section: {key: value}}} -> config.json

Serves plain HTTP on the LAN with no authentication - anyone on your network
can drive the robot (and now edit its config). That's the right trade-off for a
hobby robot on a home network; do not port-forward it to the internet.

Stdlib only (ThreadingHTTPServer); ~zero idle cost. Fail-soft: if the port is
taken the module logs and idles rather than crash-looping.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from broker_client import Bus
import robot_config
from spatial_store import SpatialStore
from semantic_store import SemanticStore
import person_memory

import base64
import json
import os.path
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(robot_config.get("web_console", "port", 8088, env="WEB_CONSOLE_PORT"))
WEB_UI_DIR = robot_config.base_path("web_ui")
LOG_LINES = 40

# Page routes -> HTML file. Method disambiguates from the POST API endpoints
# (GET /config is the page; POST /config is the camera toggle), so page paths
# and action paths can safely overlap.
PAGES = {
    "/": "dashboard.html",
    "/drive": "drive.html",
    "/training": "training.html",
    "/people": "people.html",
    "/audio": "audio.html",
    "/config": "config.html",
}
# Shared static assets, whitelisted (never serve an arbitrary path from disk).
ASSETS = {
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

# Live camera view. vision_basic.py owns the camera and only encodes
# frames while we ask it to (picarx/vision/stream_control), so the view
# costs nothing until someone opens it. We tell vision to start on the
# first frame request and, via a watchdog, to stop once the browser has
# gone quiet for CAMERA_IDLE_SEC (tab closed / live toggle off) - so a
# forgotten tab can't pin vision's CPU encoding frames nobody sees.
VISION_STREAM_CONTROL = "picarx/vision/stream_control"
VISION_FRAME_TOPIC = "picarx/vision/frame"
CAMERA_IDLE_SEC = 5.0

# ---- RC mode ----
# Manual driving from the console. Commands are ORDINARY vetoable
# intents on picarx/intent/move - the arbiter and safety daemon chain is
# untouched - but at RC_PRIORITY they outrank every AI source (explore 5,
# watch 6, follow 7, evade 8, coach 9), so the human preempts any queued
# AI motion. picarx/rc/mode tells the AI side to stand down entirely
# (field_agent pauses exploring/following and just observes).
# Dead-man layering: intents carry a short TTL, the browser re-posts its
# control state ~4x/s, a 10Hz publisher thread re-asserts + alternates
# primitives, and RC_DEADMAN_SEC of client silence force-stops. A client
# gone for RC_MODE_TIMEOUT_SEC exits RC mode altogether.
# Perception label feedback: the check/X on an identification line publishes
# here, the SAME topic curiosity.py uses for spoken relabels, and reflection.py
# turns it into a durable semantic fact (it is the sole semantic.db writer).
LABEL_TOPIC = "picarx/perception/label"

RC_MODE_TOPIC = "picarx/rc/mode"
RC_SOURCE = "rc"
RC_PRIORITY = 10
RC_SPEED = 25
RC_TURN_ANGLE = 25
RC_INTENT_TTL = 0.5
RC_DEADMAN_SEC = 0.8
RC_MODE_TIMEOUT_SEC = 60.0
RC_TICK_SEC = 0.1


class ConsoleState:
    """Latest-value cache + rolling event log, fed by MQTT callbacks."""
    def __init__(self):
        self.lock = threading.Lock()
        self.mic_enabled = True
        self.speaker_enabled = True
        self.radio = {}
        self.location = {}
        self.goal = {}
        self.world = {}
        self.follow = {}
        self.log = deque(maxlen=LOG_LINES)
        # Most recent user text ("you" or "heard") - each robot log line
        # records it as "re", so the check/X feedback buttons know which
        # utterance a response was interpreting.
        self.last_user_text = None
        # Live camera view
        self.camera_jpeg = None       # latest JPEG bytes, or None until first frame
        self.camera_ts = 0.0          # when that frame was captured (robot clock)
        self.last_view_request = 0.0  # last time a browser fetched /camera.jpg
        self.stream_on = False        # whether we've told vision to encode frames

    def add_log(self, kind, text, obs=None):
        with self.lock:
            entry = {"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text}
            if kind in ("you", "heard"):
                self.last_user_text = text
            elif kind == "robot":
                entry["re"] = self.last_user_text
                # An identification/question the robot volunteered ("looks
                # like a chair"): tag it so the console's check/X grade the
                # ID, not the command interpretation (see /label).
                if obs:
                    entry["obs"] = obs
            self.log.appendleft(entry)

    def mark_feedback(self, response_text, verdict):
        """Stamp the newest un-judged robot log line matching this
        response with the user's verdict, so the UI shows it persistently
        across re-renders. Returns True if a line was marked."""
        with self.lock:
            for entry in self.log:
                if (entry["kind"] == "robot" and entry["text"] == response_text
                        and "fb" not in entry):
                    entry["fb"] = verdict
                    return True
        return False

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
            objects = (self.world.get("objects") or {})
            person = (self.world.get("person") or {})
            seen = sorted({o.get("label") for o in objects.get("items", [])
                           if o.get("label")}) if not objects.get("stale", True) else []
            return {
                "mic_enabled": self.mic_enabled,
                "speaker_enabled": self.speaker_enabled,
                "radio": self.radio,
                "location": self.location,
                "goal": self.goal,
                "battery_v": battery.get("voltage"),
                "battery_low": battery.get("low"),
                "distance_cm": self.world.get("distance_cm"),
                "sees": seen,
                "person": person.get("name") if not person.get("stale", True) else None,
                "follow": bool(self.follow.get("enabled")),
                "log": list(self.log),
            }


def build_boxes(world):
    """Camera-overlay payload from a world snapshot: the freshest tracked
    objects (and confirmed face) in frame-pixel coordinates for the
    client to scale onto the displayed JPEG. A recognized person's name
    replaces the generic 'person'/'face' label."""
    objects = (world.get("objects") or {})
    person = (world.get("person") or {})
    face = (world.get("face") or {})
    person_name = person.get("name") if not person.get("stale", True) else None
    boxes, frame_w, frame_h, named = [], None, None, False
    if not objects.get("stale", True):
        for o in objects.get("items", []):
            if not all(k in o for k in ("x", "y", "w", "h")):
                continue
            frame_w = o.get("frame_width") or frame_w
            frame_h = o.get("frame_height") or frame_h
            label = o.get("label", "?")
            if label == "person" and person_name and not named:
                label, named = person_name, True
            boxes.append({"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"],
                          "label": label, "kind": "object",
                          "confidence": round(float(o.get("confidence") or 0.0), 2)})
    if (face.get("detected") and not face.get("stale", True)
            and all(k in face for k in ("x", "y", "w", "h"))):
        frame_w = frame_w or face.get("frame_width")
        boxes.append({"x": face["x"], "y": face["y"], "w": face["w"], "h": face["h"],
                      "label": person_name or "face", "kind": "face"})
    return {"frame_w": frame_w or 320, "frame_h": frame_h or 240, "boxes": boxes}


class RcController:
    """RC drive state -> vetoable intents. update() is called from HTTP
    handlers on every client post; step() runs on a 10Hz thread doing the
    steer/drive primitive alternation (the arbiter holds ONE intent per
    source), TTL keep-alive, dead-man stop, and mode timeout."""

    def __init__(self, bus):
        self.bus = bus
        self.lock = threading.Lock()
        self.enabled = False
        self.f = 0                  # -1 back / 0 stop / +1 forward
        self.t = 0                  # -1 left / 0 straight / +1 right
        self.last_update = 0.0      # last client post (drive OR keepalive)
        self._last_turn_sent = None
        self._steered_last = False

    def _publish(self, action):
        self.bus.publish("picarx/intent/move", {
            "source": RC_SOURCE, "priority": RC_PRIORITY,
            "action": action, "ttl": RC_INTENT_TTL})

    def _drive_action(self, f):
        if f > 0:
            return {"direction": "forward", "speed": RC_SPEED}
        if f < 0:
            return {"direction": "backward", "speed": RC_SPEED}
        return {"direction": "stop"}

    def set_mode(self, enabled, now=None):
        now = now if now is not None else time.time()
        with self.lock:
            was, self.enabled = self.enabled, enabled
            self.f = self.t = 0
            self.last_update = now
            self._last_turn_sent = None
            self._steered_last = False
        if was == enabled:
            return
        if not enabled:
            # Leave the wheel clean: stop, straighten, drop our intent so
            # the arbiter falls back to its default safe stop.
            self._publish({"direction": "turn", "angle": 0})
            self._publish({"direction": "stop"})
            self.bus.publish("picarx/intent/cancel", {"source": RC_SOURCE})
        self.bus.publish(RC_MODE_TOPIC, {"active": enabled, "ts": now})
        print(f"Web console: RC mode {'ON - manual driving' if enabled else 'off'}")

    def update(self, f, t, now=None):
        """A client control post. Stop latency matters most, so a full
        release publishes stop synchronously instead of waiting a tick."""
        now = now if now is not None else time.time()
        f = max(-1, min(1, int(f)))
        t = max(-1, min(1, int(t)))
        with self.lock:
            if not self.enabled:
                return
            changed = (f, t) != (self.f, self.t)
            self.f, self.t = f, t
            self.last_update = now
        if changed and f == 0:
            self._publish({"direction": "stop"})
            self._steered_last = False

    def step(self, now=None):
        """One 10Hz publisher pass. Returns the action published (for
        tests), or None when idle."""
        now = now if now is not None else time.time()
        with self.lock:
            if not self.enabled:
                return None
            if now - self.last_update > RC_MODE_TIMEOUT_SEC:
                pass  # fall through to set_mode below, outside the lock
            elif now - self.last_update > RC_DEADMAN_SEC and (self.f or self.t):
                self.f = self.t = 0
                self._publish({"direction": "stop"})
                print("Web console: RC dead-man - client went quiet, stopping")
                return {"direction": "stop"}
            f, t, last_update = self.f, self.t, self.last_update
        if now - last_update > RC_MODE_TIMEOUT_SEC:
            print("Web console: RC client gone - leaving RC mode")
            self.set_mode(False, now)
            return None
        desired_turn = t * RC_TURN_ANGLE
        if desired_turn != self._last_turn_sent and not self._steered_last:
            self._last_turn_sent = desired_turn
            self._steered_last = True
            action = {"direction": "turn", "angle": desired_turn}
            self._publish(action)
            return action
        self._steered_last = False
        if f == 0 and t == 0 and self._last_turn_sent in (0, None):
            return None   # idle: stay quiet, the arbiter's default stop holds
        action = self._drive_action(f)
        self._publish(action)
        return action

    def loop(self):
        while True:
            time.sleep(RC_TICK_SEC)
            try:
                self.step()
            except Exception as e:
                print(f"Web console: RC step failed: {e}")


STATE = ConsoleState()
BUS = None  # set in main
RC = None   # RcController, set in main
# Read-only store access. location_graph owns spatial.db and reflection owns
# semantic.db; the console only ever reads them, fail-soft to empty like every
# other reader.
SPATIAL = SpatialStore(readonly=True)
SEMANTIC = SemanticStore(readonly=True)


def _memory_snapshot():
    """Known places + enrolled people for the UI (place buttons, roster).
    Cheap reads at poll cadence; both fail soft to empty lists."""
    try:
        places = [l["label"] for l in SPATIAL.all_locations()]
    except Exception:
        places = []
    return {"places": places[:20], "people": person_memory.known_people()}


def objects_snapshot(world):
    """The objects currently tracked IN VIEW, for the Training page to relabel.
    Only fresh detections (stale frames give nothing to point at), one entry
    per confirmed track with its id so a correction can train that exact
    object via /label."""
    objects = (world or {}).get("objects") or {}
    if objects.get("stale", True):
        return []
    out = []
    for o in objects.get("items", []):
        if not o.get("id") or not o.get("label"):
            continue
        out.append({
            "id": o["id"],
            "label": o["label"],
            "confidence": round(float(o.get("confidence") or 0.0), 2),
            "alt_label": o.get("alt_label"),
            "area_ratio": round(float(o.get("area_ratio") or 0.0), 3),
            # Only part of the object is in frame (a border cuts it off).
            "truncated": bool(o.get("truncated")),
        })
    return out


def facts_snapshot(query=None, limit=25):
    """Recent (or searched) semantic-memory facts for the Training page's
    'ask about its objects / history' view. Read-only, fail-soft to []."""
    try:
        q = (query or "").strip()
        rows = SEMANTIC.search_facts(q, limit=limit) if q \
            else SEMANTIC.recent_facts(limit=limit)
        facts = [{"subject": r["subject"], "fact": r["fact"],
                  "confidence": round(float(r["confidence"]), 2),
                  "seen_count": r["seen_count"]} for r in rows]
        count = SEMANTIC.fact_count()
    except Exception:
        facts, count = [], 0
    return {"facts": facts, "count": count}


def config_data():
    """Every tunable for the Config page, built from the knob REGISTRY (the
    single source of truth) so the page is exhaustive by construction. Each
    knob carries its type, help, default, the file value being edited, and -
    when set - the environment variable currently shadowing it, so a stale
    export can't silently defeat an edit here."""
    cfg = robot_config.all_config()
    knobs = []
    for k in robot_config.knobs():
        section = cfg.get(k["section"])
        section = section if isinstance(section, dict) else {}
        file_val = section.get(k["key"])
        knobs.append({
            "section": k["section"], "key": k["key"], "type": k["type"],
            "desc": k["desc"], "default": k["default"],
            # The value the input shows/edits: the file's value, or the default
            # when the file doesn't pin this knob.
            "value": file_val if file_val is not None else k["default"],
            "env": k["env"],
            "env_override": robot_config.env_override(k["env"]),
        })
    return {
        "knobs": knobs,
        "note": ("Saved to config.json. Environment variables still override "
                 "these at runtime, and most modules read config at startup - "
                 "so a change takes effect the next time that module restarts."),
    }


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

    def _serve_file(self, filename, ctype):
        try:
            with open(os.path.join(WEB_UI_DIR, filename), "rb") as f:
                self._send(200, f.read(), ctype)
        except OSError:
            self._send(200, f"<h1>PicarX console</h1><p>{filename} missing.</p>".encode(),
                       "text/html; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/index.html":
            path = "/"
        if path in PAGES:
            self._serve_file(PAGES[path], "text/html; charset=utf-8")
        elif path in ASSETS:
            self._serve_file(*ASSETS[path])
        elif self.path == "/state":
            self._send(200, {**STATE.snapshot(), **_memory_snapshot(),
                             "rc_enabled": RC.enabled if RC else False})
        elif self.path == "/boxes":
            with STATE.lock:
                world = dict(STATE.world)
            self._send(200, build_boxes(world))
        elif self.path == "/objects":
            with STATE.lock:
                world = dict(STATE.world)
            self._send(200, {"objects": objects_snapshot(world)})
        elif path == "/facts":
            from urllib.parse import parse_qs, urlparse
            q = (parse_qs(urlparse(self.path).query).get("q") or [""])[0]
            self._send(200, facts_snapshot(q))
        elif self.path == "/config/data":
            self._send(200, config_data())
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
        elif self.path == "/speaker":
            # audio_nodes gates TTS on this and re-runs the amp-enable
            # command (robot_hat enable_speaker) on the off->on press.
            BUS.publish("picarx/audio/speaker_control",
                        {"enabled": bool(body.get("enabled", True))})
            self._send(200, {"ok": True})
        elif self.path == "/feedback":
            # Check/X on a robot response: grade how the last utterance
            # was interpreted. Rides the same MQTT bus as everything else
            # (picarx/intent/feedback -> companion's intent teacher). A
            # typed correction additionally executes through the normal
            # heard pipeline, exactly like the /say box - same trust.
            verdict = body.get("verdict")
            if verdict not in ("correct", "incorrect"):
                self._send(400, {"error": "verdict must be correct|incorrect"})
                return
            utterance = (body.get("utterance") or "").strip().lower()
            response = (body.get("response") or "").strip()
            correction = (body.get("correction") or "").strip().lower()
            payload = {"verdict": verdict, "utterance": utterance,
                       "response": response, "origin": "web", "ts": time.time()}
            if correction:
                payload["correction"] = correction
            BUS.publish("picarx/intent/feedback", payload)
            STATE.mark_feedback(response, verdict)
            if correction:
                BUS.publish("picarx/audio/heard",
                            {"text": correction, "source": "user_correction"})
                STATE.add_log("you", correction)
            self._send(200, {"ok": True})
        elif self.path == "/label":
            # Check/X on an identification line: grade what the robot saw,
            # not how it read a command. A correcting (or confirming) label
            # rides picarx/perception/label -> reflection.py writes it to the
            # semantic store, exactly like a spoken relabel via curiosity.py.
            correct = (body.get("label") or "").strip().lower()
            if not correct:
                self._send(400, {"error": "empty label"})
                return
            guess = (body.get("guess") or "").strip().lower()
            verdict = "correct" if correct == guess else "incorrect"
            BUS.publish(LABEL_TOPIC, {
                "correct_label": correct, "guess": guess,
                "object_id": body.get("object_id"), "origin": "web",
                "ts": time.time()})
            response = (body.get("response") or "").strip()
            if response:
                STATE.mark_feedback(response, verdict)
            if verdict == "incorrect":
                STATE.add_log("you", f"that's a {correct}")
            self._send(200, {"ok": True})
        elif self.path == "/rc":
            RC.set_mode(bool(body.get("enabled", False)))
            self._send(200, {"ok": True})
        elif self.path == "/rc/drive":
            try:
                RC.update(body.get("f", 0), body.get("t", 0))
            except (TypeError, ValueError):
                self._send(400, {"error": "f and t must be -1, 0 or 1"})
                return
            self._send(200, {"ok": True})
        elif self.path == "/camera":
            # Explicit toggle from the live-view switch. Turning it off
            # stops vision encoding immediately rather than waiting out
            # the idle watchdog; turning it on pre-warms the stream.
            enabled = bool(body.get("enabled", False))
            if STATE.set_stream(enabled):
                BUS.publish(VISION_STREAM_CONTROL, {"enabled": enabled})
            self._send(200, {"ok": True})
        elif self.path == "/config/save":
            # Persist edited knobs to config.json (merged, so untouched keys and
            # the _readme survive). Announce the change for any future live
            # re-readers; today most modules pick it up on their next restart.
            edits = body.get("config")
            if not isinstance(edits, dict):
                self._send(400, {"error": "config must be an object"})
                return
            try:
                robot_config.merge_and_save(edits)
            except ValueError as e:
                self._send(400, {"error": str(e)})
                return
            except OSError as e:
                self._send(500, {"error": f"could not write config: {e}"})
                return
            BUS.publish("picarx/config/reload", {"ts": time.time()})
            print("Web console: config.json updated from the browser")
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})


# ---------- MQTT feeds ----------

def on_speak(p):
    if not p.get("text"):
        return
    # Modules tag identification/uncertainty remarks with kind=observation
    # /question (+ the label they went with); carry that onto the log line
    # so the console can offer an ID-correction affordance instead of the
    # command-interpretation feedback (see the console's renderLog / /label).
    obs = None
    if p.get("kind") in ("observation", "question"):
        items = p.get("objects")
        if not items and p.get("label"):
            items = [{"label": p["label"], "id": None}]
        obs = {"kind": p["kind"], "items": items or [],
               "subject": p.get("subject")}
    STATE.add_log("robot", p["text"], obs=obs)

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

def on_speaker_state(p):
    with STATE.lock:
        STATE.speaker_enabled = bool(p.get("enabled", True))

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

def on_follow_state(p):
    with STATE.lock:
        STATE.follow = p

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
    global BUS, RC
    BUS = Bus()
    RC = RcController(BUS)
    threading.Thread(target=RC.loop, name="rc-publisher", daemon=True).start()
    BUS.subscribe("picarx/audio/speak", on_speak)
    BUS.subscribe("picarx/audio/heard", on_heard)
    BUS.subscribe("picarx/audio/mic_state", on_mic_state)
    BUS.subscribe("picarx/audio/speaker_state", on_speaker_state)
    BUS.subscribe("picarx/tools/radio_state", on_radio_state)
    BUS.subscribe("picarx/state/world", on_world)
    BUS.subscribe("picarx/exploration/location_change", on_location)
    BUS.subscribe("picarx/exploration/active_goal", on_goal)
    BUS.subscribe("picarx/tools/follow/state", on_follow_state)
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
