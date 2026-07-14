#!/usr/bin/env python3
# /home/picarx/layer_b/modules/radio.py
"""
Radio (Layer B tool) - internet radio streaming through the existing
speaker.

Hardware honesty: the PiCar-X has no radio tuner and no RTL-SDR is
assumed installed, so "radio" means streaming station URLs over the
network through the same ALSA output audio_nodes.py uses for TTS.
Entirely fail-soft on all three dependencies:
  - no mpv/ffplay/mplayer installed -> announces "no radio capability"
  - no network -> the player exits, we announce the failure
  - no stations file -> built-in default list (SomaFM, stable free
    streams) is written on first run so it's easy to edit

Listens on picarx/tools/radio (routed by tools_registry.py):
  {"command": "play"}                  - resume/last station
  {"command": "play", "station": "x"}  - fuzzy name match
  {"command": "stop"} / {"command": "next"}

Publishes picarx/tools/radio_state on every change. The player runs
in its own process group and is killed on module stop/restart, so an
orchestrator restart never leaks a background stream. Note: while the
radio plays, STT will hear music - expect voice commands to need more
repetition (the mic and speaker fight; that's physics, not a bug).
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import json
import shutil
import signal
import subprocess
import time

STATIONS_PATH = "/home/picarx/layer_b/data/radio_stations.json"

# Hardware honesty: there is no FM tuner on this robot, so a "dial"
# (e.g. "98.7") is NOT a radio frequency being received - it's a label
# you attach to an internet stream so you can ask for a station the way
# you'd say it out loud. Most real FM stations also stream online; put
# the stream URL of YOUR local 98.7 here and "tune to 98.7" plays it.
# Edit this file (data/radio_stations.json) to map your own dials.
# The two dial entries below are EXAMPLES using public streams - change
# their urls to your actual local stations.
DEFAULT_STATIONS = [
    {"name": "groove salad", "url": "http://ice1.somafm.com/groovesalad-128-mp3"},
    {"name": "drone zone", "url": "http://ice1.somafm.com/dronezone-128-mp3"},
    {"name": "secret agent", "url": "http://ice1.somafm.com/secretagent-128-mp3"},
    {"name": "lush", "url": "http://ice1.somafm.com/lush-128-mp3"},
    {"dial": "98.7", "name": "example dial ninety eight seven (edit me)",
     "url": "http://ice1.somafm.com/indiepop-128-mp3"},
    {"dial": "101.5", "name": "example dial one oh one five (edit me)",
     "url": "http://ice1.somafm.com/bootliquor-128-mp3"},
]


def _norm_dial(value):
    """Compare dials loosely: '98.7', '98 7', ' 98.70 ' all match."""
    if value is None:
        return None
    return "".join(ch for ch in str(value) if ch.isdigit() or ch == ".").strip(".")

# Preference order; all take a bare URL and play to the default ALSA out.
PLAYERS = (
    ("mpv", ["mpv", "--no-video", "--really-quiet"]),
    ("ffplay", ["ffplay", "-nodisp", "-loglevel", "quiet", "-autoexit"]),
    ("mplayer", ["mplayer", "-really-quiet"]),
)


class Radio:
    def __init__(self):
        self.bus = Bus()
        self.player_cmd = next((cmd for name, cmd in PLAYERS if shutil.which(name)), None)
        self.stations = self._load_stations()
        self.index = 0
        self.proc = None

    def _load_stations(self):
        try:
            with open(STATIONS_PATH) as f:
                stations = json.load(f)
            if stations:
                return stations
        except (OSError, json.JSONDecodeError):
            pass
        try:
            os.makedirs(os.path.dirname(STATIONS_PATH), exist_ok=True)
            with open(STATIONS_PATH, "w") as f:
                json.dump(DEFAULT_STATIONS, f, indent=1)
        except OSError:
            pass
        return list(DEFAULT_STATIONS)

    def _say(self, text):
        self.bus.publish("picarx/audio/speak", {"text": text, "ts": time.time()})

    def _publish_state(self, playing):
        station = self.stations[self.index]
        self.bus.publish("picarx/tools/radio_state", {
            "playing": playing,
            "station": station["name"] if playing else None,
            "dial": station.get("dial") if playing else None,
            "ts": time.time(),
        })

    # ---------- player process ----------

    def _stop_player(self):
        if self.proc is None:
            return
        try:
            # Whole process group: mpv may have forked helpers.
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc = None

    def _start_player(self):
        station = self.stations[self.index]
        try:
            self.proc = subprocess.Popen(
                self.player_cmd + [station["url"]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError as e:
            print(f"Radio: failed to start player: {e}")
            self.proc = None
            return False

    # ---------- station lookup ----------

    def _phrase(self, station):
        dial = station.get("dial")
        return f"{station['name']} on {dial}" if dial else station["name"]

    def _find_by_dial(self, dial):
        want = _norm_dial(dial)
        for i, s in enumerate(self.stations):
            if _norm_dial(s.get("dial")) == want:
                return i
        return None

    def _find_by_name(self, name):
        wanted = name.lower()
        for i, s in enumerate(self.stations):
            if wanted in s.get("name", "").lower():
                return i
        return None

    # ---------- commands ----------

    def on_command(self, payload):
        command = payload.get("command")
        if self.player_cmd is None:
            self._say("Sorry, I don't have radio capability on this hardware.")
            return

        if command == "stop":
            was_playing = self.proc is not None
            self._stop_player()
            self._publish_state(False)
            if was_playing:
                self._say("Radio off.")
            return

        if command == "list":
            names = ", ".join(self._phrase(s) for s in self.stations)
            self._say(f"I have {len(self.stations)} stations: {names}.")
            return

        if command == "status":
            if self.proc is not None:
                self._say(f"Now playing {self._phrase(self.stations[self.index])}.")
            else:
                self._say("The radio is off.")
            return

        if command == "next" and self.proc is not None:
            self.index = (self.index + 1) % len(self.stations)
        elif command == "play" and payload.get("dial"):
            idx = self._find_by_dial(payload["dial"])
            if idx is None:
                self._say(f"I don't have a station saved for {payload['dial']}. "
                          f"Add its stream to my stations file and I'll tune there.")
                return
            self.index = idx
        elif command == "play" and payload.get("station"):
            idx = self._find_by_name(payload["station"])
            if idx is None:
                self._say(f"I don't know a station called {payload['station']}. "
                          f"Playing {self._phrase(self.stations[self.index])} instead.")
            else:
                self.index = idx
        elif command not in ("play", "next"):
            return

        self._stop_player()
        if self._start_player():
            self._say(f"Tuning to {self._phrase(self.stations[self.index])}.")
            self._publish_state(True)
        else:
            self._say("I couldn't start the radio stream.")
            self._publish_state(False)

    # ---------- main loop ----------

    def _shutdown(self, signum, frame):
        self._stop_player()
        sys.exit(0)

    def run(self):
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
        self.bus.subscribe("picarx/tools/radio", self.on_command)
        print(f"Radio active (player: {self.player_cmd[0] if self.player_cmd else 'NONE - degraded'}, "
              f"{len(self.stations)} stations)")
        while True:
            time.sleep(5)
            # A dead stream (network drop, bad URL) shouldn't pretend to
            # be playing forever.
            if self.proc is not None and self.proc.poll() is not None:
                print("Radio: stream ended/died")
                self.proc = None
                self._publish_state(False)


if __name__ == "__main__":
    Radio().run()
