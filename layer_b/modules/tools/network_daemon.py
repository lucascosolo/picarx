#!/usr/bin/env python3
# /home/picarx/layer_b/modules/tools/network_daemon.py
"""
Network daemon (Layer B tool) - share a phone's connection with the robot.

When the robot rolls somewhere its home wifi can't reach, this lets it
fall back to YOUR phone so radio.py and the LLM modules keep working.
Two zero-fuss paths:

  1. USB tethering (no config): plug the phone into the Pi's USB and turn
     on USB tethering. NetworkManager brings up the usb0 interface with
     DHCP automatically; this daemon just notices the internet came back
     and says so.

  2. Phone wifi hotspot: store the hotspot's SSID + password once in
     data/networks.json (below). Then a spoken "share your connection"
     (companion.py's share_connection LLM tool -> picarx/tools/network/
     connect) makes the Pi join it via nmcli. If auto_failover is on, the
     daemon also joins it on its own after the internet has been
     unreachable for a grace period, and NetworkManager prefers the
     higher-priority home network again once it's back in range.

data/networks.json (created with an editable template on first run):
  {"hotspots": [{"ssid": "MyPhone", "password": "secret"}],
   "auto_failover": true, "check_interval": 20}

Everything is fail-soft: no nmcli installed, no config, or a failed join
just gets announced on TTS - never raised. This daemon issues NO motion.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import json
import shutil
import socket
import subprocess
import threading
import time

CONNECT_TOPIC = "picarx/tools/network/connect"
STATE_TOPIC = "picarx/tools/network/state"
SPEAK_TOPIC = "picarx/audio/speak"

DATA_DIR = "/home/picarx/layer_b/data"
NETWORKS_PATH = f"{DATA_DIR}/networks.json"

DEFAULT_CONFIG = {
    "hotspots": [{"ssid": "MyPhoneHotspot", "password": "change-me"}],
    "auto_failover": True,
    "check_interval": 20,        # seconds between connectivity checks
    "offline_grace": 60,         # unreachable this long before auto-failover
}
CONNECT_TIMEOUT = 30             # nmcli join timeout (seconds)


def load_config(path):
    """Config dict, fail-soft to DEFAULT_CONFIG (and write the template on
    first run so it's easy to edit)."""
    try:
        with open(path) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return dict(DEFAULT_CONFIG)
        merged = dict(DEFAULT_CONFIG)
        merged.update(cfg)
        return merged
    except FileNotFoundError:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
        except OSError:
            pass
        return dict(DEFAULT_CONFIG)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)


def pick_hotspot(config, name=None):
    """Choose a configured hotspot: by (case-insensitive) name/ssid match
    if given, else the first one. None if none are configured."""
    hotspots = config.get("hotspots") or []
    if not hotspots:
        return None
    if name:
        want = str(name).strip().lower()
        for h in hotspots:
            if want in (str(h.get("ssid", "")).lower(), str(h.get("name", "")).lower()):
                return h
    return hotspots[0]


def build_wifi_connect_cmd(ssid, password):
    """nmcli argv (list form - no shell) to join a wifi network."""
    cmd = ["nmcli", "device", "wifi", "connect", ssid]
    if password:
        cmd += ["password", password]
    return cmd


def internet_reachable(host="1.1.1.1", port=53, timeout=2.0):
    """True if a TCP connection to a public DNS resolver succeeds - a
    dependency-free 'are we actually online?' check (works over wifi,
    USB tether, anything)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class NetworkDaemon:
    def __init__(self, reachable=internet_reachable):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.config = load_config(NETWORKS_PATH)
        self._reachable = reachable
        self.online = None            # None = unknown until first check
        self.offline_since = None
        self.have_nmcli = shutil.which("nmcli") is not None
        if not self.have_nmcli:
            print("Network daemon: nmcli not found - can report connectivity but "
                  "can't switch networks on this host")

    # ---------- joining a network ----------

    def _join_hotspot(self, name=None):
        hotspot = pick_hotspot(self.config, name)
        if hotspot is None:
            self._speak("I don't have any phone hotspots saved yet. Add one to my "
                        "networks file first.")
            return False
        ssid = hotspot.get("ssid")
        if not ssid:
            return False
        if not self.have_nmcli:
            self._speak("I can't switch wifi on my own here. Plug your phone into my "
                        "USB and turn on tethering instead.")
            return False
        self._speak(f"Trying to join {ssid}.")
        try:
            proc = subprocess.run(
                build_wifi_connect_cmd(ssid, hotspot.get("password")),
                capture_output=True, text=True, timeout=CONNECT_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"Network daemon: nmcli failed: {e}")
            self._speak(f"I couldn't join {ssid}.")
            return False
        ok = proc.returncode == 0
        print(f"Network daemon: join {ssid} -> "
              f"{'ok' if ok else proc.stderr.strip() or 'failed'}")
        self._speak(f"Connected to {ssid}." if ok else f"I couldn't join {ssid}.")
        self.bus.publish(STATE_TOPIC, {"event": "join", "ssid": ssid,
                                       "ok": ok, "ts": time.time()})
        return ok

    # ---------- inbound ----------

    def on_connect(self, payload):
        """Spoken 'share your connection' (or an explicit ssid/password)."""
        name = payload.get("name") or payload.get("ssid")
        # An ad-hoc hotspot passed inline gets tried without needing the file.
        if payload.get("ssid") and payload.get("password") is not None:
            with self.lock:
                self.config.setdefault("hotspots", []).insert(
                    0, {"ssid": payload["ssid"], "password": payload["password"]})
        threading.Thread(target=self._join_hotspot, args=(name,), daemon=True).start()

    # ---------- connectivity monitor ----------

    def _speak(self, text):
        self.bus.publish(SPEAK_TOPIC, {"text": text, "ts": time.time()})

    def _check_once(self, now):
        online = self._reachable()
        was = self.online
        self.online = online
        if online:
            self.offline_since = None
            if was is False:                 # transition offline -> online
                self._speak("I'm back online.")
                self.bus.publish(STATE_TOPIC, {"online": True, "ts": now})
            return
        # offline
        if self.offline_since is None:
            self.offline_since = now
        if was in (True, None):              # transition (or first check) -> offline
            self.bus.publish(STATE_TOPIC, {"online": False, "ts": now})
        # Auto-failover once we've been offline past the grace period.
        if (self.config.get("auto_failover")
                and now - self.offline_since >= self.config.get("offline_grace", 60)
                and pick_hotspot(self.config) is not None
                and self.have_nmcli):
            print("Network daemon: offline past grace, trying phone hotspot")
            if self._join_hotspot():
                self.offline_since = now      # give the new link time before retrying

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(CONNECT_TOPIC, self.on_connect)
        interval = float(self.config.get("check_interval", 20))
        print(f"Network daemon active (nmcli={'yes' if self.have_nmcli else 'no'}), "
              f"listening on {CONNECT_TOPIC}")
        while True:
            try:
                self._check_once(time.time())
            except Exception as e:
                print(f"Network daemon: monitor error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    NetworkDaemon().run()
