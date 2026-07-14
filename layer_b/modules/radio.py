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
from radio_browser import RadioBrowser

import json
import shutil
import signal
import subprocess
import tempfile
import threading
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

# The stream must go to the SAME speaker the TTS uses, not ALSA's
# "default" sink (often HDMI / the wrong card -> the classic "says
# tuning, plays nothing"). This is the ALSA PCM *name* (as in
# asound.conf), matching audio_nodes.py's plug:robot_speaker output.
RADIO_ALSA_DEVICE = os.environ.get("RADIO_ALSA_DEVICE", "robot_speaker")
# How long to watch a freshly-started player before trusting it - a bad
# stream/URL/device makes the player exit within a second, and we'd
# rather report that than announce a station and sit silent.
PLAYER_HEALTHCHECK_SEC = 1.3
# The hifiberry is exclusive (only one app at a time), so the spoken
# "Tuning to ..." and the stream can't share it. Give the announcement
# this long to finish before the player seizes the device. Set to 0 if
# you switch robot_speaker to a mixing (dmix) PCM so both can overlap.
TTS_SETTLE_SEC = float(os.environ.get("RADIO_TTS_SETTLE", "2.0"))

# Per-player argv builders. Each returns the full command list,
# pointed at RADIO_ALSA_DEVICE so audio lands on the real speaker.
def _mpv_cmd(url, dev):
    args = ["mpv", "--no-video", "--really-quiet"]
    if dev:
        args.append(f"--audio-device=alsa/{dev}")
    return args + [url]

def _ffplay_cmd(url, dev):
    # ffplay (SDL) has no clean device flag; AUDIODEV is honored by the
    # SDL/ALSA backend. Set it in the child env in _start_player.
    return ["ffplay", "-nodisp", "-loglevel", "error", "-autoexit", url]

def _mplayer_cmd(url, dev):
    args = ["mplayer", "-really-quiet"]
    if dev:
        args += ["-ao", f"alsa:device={dev}"]
    return args + [url]

# Preference order.
PLAYERS = (
    ("mpv", _mpv_cmd),
    ("ffplay", _ffplay_cmd),
    ("mplayer", _mplayer_cmd),
)


class Radio:
    def __init__(self):
        self.bus = Bus()
        self.player_name, self.player_build = next(
            ((name, build) for name, build in PLAYERS if shutil.which(name)), (None, None))
        self.stations = self._load_stations()
        self.index = 0
        self.proc = None
        self._errfile = None
        # Live directory search (radio-browser.info): "radio find soft
        # rock" fills search_results and switches mode; "next station"
        # then walks the RESULTS until you hear one you like. Saved
        # stations/dials still work exactly as before and playing one
        # switches mode back.
        self.browser = RadioBrowser()
        self.search_results = []
        self.search_pos = 0
        self.mode = "saved"          # "saved" | "search"

    def _current(self):
        if self.mode == "search" and self.search_results:
            return self.search_results[self.search_pos]
        return self.stations[self.index]

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
        station = self._current()
        self.bus.publish("picarx/tools/radio_state", {
            "playing": playing,
            "station": station["name"] if playing else None,
            "dial": station.get("dial") if playing else None,
            "mode": self.mode,
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
        # Wait for it to actually exit before returning. The hifiberry
        # is an exclusive (non-mixing) ALSA device, so if we start the
        # next player while this one is still tearing down, the new one
        # can't open the speaker and dies silently - that's the
        # intermittent "next station does nothing". Block until the
        # device is genuinely free, escalating to SIGKILL if needed.
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=1.0)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                pass
        self.proc = None
        if self._errfile is not None:
            try:
                self._errfile.close()
            except OSError:
                pass
            self._errfile = None

    def _start_player(self):
        """Start the player and confirm it's actually alive. Returns
        True only if it's still streaming after the healthcheck window;
        False (with a logged reason) if it died on startup."""
        station = self._current()
        argv = self.player_build(station["url"], RADIO_ALSA_DEVICE)
        env = dict(os.environ)
        if self.player_name == "ffplay" and RADIO_ALSA_DEVICE:
            env["AUDIODEV"] = RADIO_ALSA_DEVICE
        # stderr to a temp file (not a PIPE): a live player we never
        # read from could fill a pipe buffer and stall; a file can't.
        self._errfile = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=self._errfile,
                start_new_session=True, env=env,
            )
        except OSError as e:
            print(f"Radio: failed to start player: {e}")
            self.proc = None
            return False

        # A healthy stream keeps running; a bad URL/device/network makes
        # the player exit within ~1s. Watch briefly before we trust it.
        deadline = time.time() + PLAYER_HEALTHCHECK_SEC
        while time.time() < deadline:
            if self.proc.poll() is not None:
                print(f"Radio: player exited on startup (code {self.proc.returncode}): "
                      f"{self._read_err()}")
                self.proc = None
                return False
            time.sleep(0.1)
        return True

    def _read_err(self):
        if self._errfile is None:
            return ""
        try:
            self._errfile.seek(0)
            return self._errfile.read().decode("utf-8", "replace").strip()[-300:]
        except OSError:
            return ""

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
        if self.player_build is None:
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
                self._say(f"Now playing {self._phrase(self._current())}.")
            else:
                self._say("The radio is off.")
            return

        if command == "find":
            keywords = (payload.get("keywords") or "").strip()
            if not keywords:
                return
            results = self.browser.search(keywords)
            if not results:
                self._say(f"I couldn't find any stations for {keywords}.")
                return
            self.search_results = results
            self.search_pos = 0
            self.mode = "search"
            self._say(f"Found {len(results)} stations for {keywords}. "
                      f"Say next station to try another.")
            self._tune_and_play()
            return

        if command == "next" and self.proc is not None:
            if self.mode == "search" and self.search_results:
                self.search_pos = (self.search_pos + 1) % len(self.search_results)
            else:
                self.index = (self.index + 1) % len(self.stations)
        elif command == "play" and payload.get("dial"):
            idx = self._find_by_dial(payload["dial"])
            if idx is None:
                self._say(f"I don't have a station saved for {payload['dial']}. "
                          f"Add its stream to my stations file and I'll tune there.")
                return
            self.index = idx
            self.mode = "saved"
        elif command == "play" and payload.get("station"):
            idx = self._find_by_name(payload["station"])
            if idx is None:
                self._say(f"I don't know a station called {payload['station']}. "
                          f"Playing {self._phrase(self._current())} instead.")
            else:
                self.index = idx
                self.mode = "saved"
        elif command == "play":
            pass  # resume whatever is current (saved or search result)
        elif command != "next":
            return

        self._tune_and_play()

    def _tune_and_play(self):
        station = self._current()
        phrase = self._phrase(station)
        # Free the speaker (stop any current stream) BEFORE announcing,
        # then speak while it's free, then let that short announcement
        # finish before the player grabs the exclusive device again.
        self._stop_player()
        self._say(f"Tuning to {phrase}.")
        if TTS_SETTLE_SEC > 0:
            time.sleep(TTS_SETTLE_SEC)

        # One retry: the first attempt can lose the device race with the
        # announcement we just made, or hit a transient network hiccup;
        # a second try a moment later clears both.
        ok = self._start_player()
        if not ok:
            time.sleep(1.0)
            ok = self._start_player()

        if ok:
            self._publish_state(True)
            # Directory etiquette: report the play so radio-browser
            # learns the station is alive/popular. Off-thread so a slow
            # directory can never delay playback handling.
            if station.get("uuid"):
                threading.Thread(target=self.browser.click,
                                 args=(station["uuid"],), daemon=True).start()
        else:
            self._say("I couldn't reach that station.")
            self._publish_state(False)

    # ---------- main loop ----------

    def _shutdown(self, signum, frame):
        self._stop_player()
        sys.exit(0)

    def run(self):
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
        self.bus.subscribe("picarx/tools/radio", self.on_command)
        print(f"Radio active (player: {self.player_name or 'NONE - degraded'} -> "
              f"{RADIO_ALSA_DEVICE or 'default'}, {len(self.stations)} stations)")
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
