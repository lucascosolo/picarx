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

DEFAULT_STATIONS = [
    {"name": "groove salad", "url": "http://ice1.somafm.com/groovesalad-128-mp3"},
    {"name": "drone zone", "url": "http://ice1.somafm.com/dronezone-128-mp3"},
    {"name": "secret agent", "url": "http://ice1.somafm.com/secretagent-128-mp3"},
    {"name": "lush", "url": "http://ice1.somafm.com/lush-128-mp3"},
]

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
        self.bus.publish("picarx/tools/radio_state", {
            "playing": playing,
            "station": self.stations[self.index]["name"] if playing else None,
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

        if command == "next" and self.proc is not None:
            self.index = (self.index + 1) % len(self.stations)
        elif command == "play" and payload.get("station"):
            wanted = payload["station"].lower()
            for i, s in enumerate(self.stations):
                if wanted in s["name"].lower():
                    self.index = i
                    break
            else:
                self._say(f"I don't know a station called {payload['station']}. "
                          f"Playing {self.stations[self.index]['name']} instead.")
        elif command not in ("play", "next"):
            return

        self._stop_player()
        if self._start_player():
            self._say(f"Tuning to {self.stations[self.index]['name']}.")
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
