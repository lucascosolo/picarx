#!/usr/bin/env python3
# /home/picarx/layer_b/modules/tools/bluetooth_daemon.py
"""
Bluetooth daemon (Layer B tool) - share a phone's connection over BLUETOOTH.

Replaces the earlier wifi-hotspot fallback. Wifi on this Pi is managed by
the OS (and the user's setup auto-spins-up its OWN wifi hotspot when it
can't reach the internet, which fought any attempt to join a phone's wifi
hotspot). Bluetooth Personal Area Networking (PAN) sidesteps that entirely:
it never touches the wifi radio, so the robot can tether to a phone's data
over Bluetooth while the wifi failsafe does whatever it likes.

One-time setup (system tools, done once by a human):
  1. On the phone, enable "Bluetooth tethering".
  2. On the Pi, pair + trust the phone once:
       bluetoothctl
         scan on            # find the phone's MAC
         pair AA:BB:CC:DD:EE:FF
         trust AA:BB:CC:DD:EE:FF
         quit
  3. Put the MAC in data/bluetooth.json (a template is written on first run):
       {"devices": [{"mac": "AA:BB:CC:DD:EE:FF", "name": "MyPhone"}],
        "auto_failover": true, "check_interval": 20, "offline_grace": 60}

Then a spoken "share your connection" (companion.py's share_connection LLM
tool -> picarx/tools/bluetooth/connect) brings up the PAN link via nmcli;
with auto_failover on, it also connects on its own after the internet has
been unreachable for a grace period. The connect command is a template
(BT_CONNECT_CMD) so you can swap nmcli for bluez-tools' bt-network if you
prefer. Everything is fail-soft (no tool / no config / failed link is just
announced). This daemon issues NO motion and never touches the wifi radio.
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

CONNECT_TOPIC = "picarx/tools/bluetooth/connect"
STATE_TOPIC = "picarx/tools/bluetooth/state"
SPEAK_TOPIC = "picarx/audio/speak"

DATA_DIR = "/home/picarx/layer_b/data"
BLUETOOTH_PATH = f"{DATA_DIR}/bluetooth.json"

# How to bring up the PAN link to a PAIRED phone's {mac}. NetworkManager
# handles a trusted Bluetooth NAP device with a plain `device connect`.
# Override for a different stack, e.g. "bt-network -c {mac} nap".
BT_CONNECT_CMD = os.environ.get("BT_CONNECT_CMD", "nmcli device connect {mac}")
CONNECT_TIMEOUT = 30

DEFAULT_CONFIG = {
    "devices": [{"mac": "AA:BB:CC:DD:EE:FF", "name": "MyPhone"}],
    "auto_failover": True,
    "check_interval": 20,        # seconds between connectivity checks
    "offline_grace": 60,         # unreachable this long before auto-tethering
}


def load_config(path):
    """Config dict, fail-soft to DEFAULT_CONFIG (writing an editable template
    on first run)."""
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


def pick_device(config, name=None):
    """Choose a paired phone: by (case-insensitive) name/mac match if given,
    else the first one. None if none are configured."""
    devices = config.get("devices") or []
    if not devices:
        return None
    if name:
        want = str(name).strip().lower()
        for d in devices:
            if want in (str(d.get("name", "")).lower(), str(d.get("mac", "")).lower()):
                return d
    return devices[0]


def build_pan_connect_cmd(mac, template=None):
    """argv (list form - no shell) to bring up the Bluetooth PAN link."""
    return (template or BT_CONNECT_CMD).format(mac=mac).split()


def internet_reachable(host="1.1.1.1", port=53, timeout=2.0):
    """True if a TCP connection to a public DNS resolver succeeds - a
    transport-agnostic 'are we actually online?' check."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class BluetoothDaemon:
    def __init__(self, reachable=internet_reachable):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.config = load_config(BLUETOOTH_PATH)
        self._reachable = reachable
        self.online = None            # None = unknown until first check
        self.offline_since = None
        self.connect_tool = BT_CONNECT_CMD.split()[0]
        self.have_tool = shutil.which(self.connect_tool) is not None
        if not self.have_tool:
            print(f"Bluetooth daemon: '{self.connect_tool}' not found - can report "
                  f"connectivity but can't bring up a PAN link on this host")

    # ---------- bringing up the link ----------

    def _tether(self, name=None):
        device = pick_device(self.config, name)
        if device is None:
            self._speak("I don't have a paired phone saved yet. Pair one and add it "
                        "to my bluetooth file first.")
            return False
        mac = device.get("mac")
        if not mac:
            return False
        if not self.have_tool:
            self._speak("I can't bring up Bluetooth tethering on my own here.")
            return False
        label = device.get("name") or mac
        self._speak(f"Tethering to {label} over Bluetooth.")
        try:
            proc = subprocess.run(build_pan_connect_cmd(mac),
                                  capture_output=True, text=True, timeout=CONNECT_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"Bluetooth daemon: connect failed: {e}")
            self._speak(f"I couldn't tether to {label}. Is it paired and sharing?")
            return False
        ok = proc.returncode == 0
        print(f"Bluetooth daemon: tether {mac} -> "
              f"{'ok' if ok else (proc.stderr.strip() or 'failed')}")
        self._speak(f"Connected to {label}." if ok
                    else f"I couldn't tether to {label}. Is it paired and sharing?")
        self.bus.publish(STATE_TOPIC, {"event": "tether", "mac": mac,
                                       "ok": ok, "ts": time.time()})
        return ok

    # ---------- inbound ----------

    def on_connect(self, payload):
        name = payload.get("name") or payload.get("mac")
        # An inline mac gets tried without needing the file (still must be paired).
        if payload.get("mac"):
            with self.lock:
                self.config.setdefault("devices", []).insert(
                    0, {"mac": payload["mac"], "name": payload.get("name")})
        threading.Thread(target=self._tether, args=(name,), daemon=True).start()

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
        if (self.config.get("auto_failover")
                and now - self.offline_since >= self.config.get("offline_grace", 60)
                and pick_device(self.config) is not None
                and self.have_tool):
            print("Bluetooth daemon: offline past grace, trying phone tether")
            if self._tether():
                self.offline_since = now      # give the new link time before retrying

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(CONNECT_TOPIC, self.on_connect)
        interval = float(self.config.get("check_interval", 20))
        print(f"Bluetooth daemon active ({self.connect_tool}="
              f"{'yes' if self.have_tool else 'no'}), listening on {CONNECT_TOPIC}")
        while True:
            try:
                self._check_once(time.time())
            except Exception as e:
                print(f"Bluetooth daemon: monitor error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    BluetoothDaemon().run()
