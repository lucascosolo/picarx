#!/usr/bin/env python3
# layer_b/modules/tools/reminder_daemon.py
"""
Reminder daemon (Layer B tool) - time-aware reminders.

companion.py exposes a `schedule_reminder` LLM tool; when the model
decides to use it, companion publishes the request here on
picarx/tools/reminder/set. This daemon holds the timers asynchronously
(one threading.Timer per pending reminder) and, when a timer expires,
speaks the reminder text on the existing TTS topic picarx/audio/speak.

Set-request payload (all times are LOCAL):
  {"message": "take the cake out",           # required
   "delay_minutes": 20}                       # relative, OR
  {"message": "...", "at": "18:30"}            # exact "HH:MM" today/tomorrow, OR
  {"message": "...", "at": "2026-07-15 18:30"} # exact date+time

Pending reminders are persisted to data/reminders.json and re-armed on
start, so a module restart never silently drops one (a reminder already
due at load time fires immediately). Everything is fail-soft: an
unparseable time is announced and dropped, never raised. This daemon
only ever SPEAKS - it issues no motion and touches no other DB.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from broker_client import Bus
import robot_config

import json
import threading
import time
import uuid
from datetime import datetime, timedelta

SET_TOPIC = "picarx/tools/reminder/set"
STATE_TOPIC = "picarx/tools/reminder/state"
SPEAK_TOPIC = "picarx/audio/speak"

DATA_DIR = robot_config.data_path()
REMINDERS_PATH = f"{DATA_DIR}/reminders.json"

MAX_PENDING = 50            # ignore new sets past this, so a runaway can't pile up timers
MAX_HORIZON_SEC = 7 * 86400  # refuse absurdly far-future reminders (likely a parse error)


def parse_at(at_str, now):
    """Turn an exact-time string into an epoch, in LOCAL time. Accepts
    'HH:MM' (today, or tomorrow if that clock time already passed) and
    'YYYY-MM-DD HH:MM' / 'YYYY-MM-DDTHH:MM'. Returns None if unparseable."""
    if not at_str:
        return None
    s = str(at_str).strip().replace("T", " ")
    base = datetime.fromtimestamp(now)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(s, fmt).time()
        except ValueError:
            continue
        cand = base.replace(hour=t.hour, minute=t.minute,
                            second=getattr(t, "second", 0), microsecond=0)
        if cand.timestamp() <= now:          # that clock time already passed today
            cand += timedelta(days=1)
        return cand.timestamp()
    return None


def resolve_fire_at(payload, now):
    """Epoch at which a set-request should fire, or None if it can't be
    resolved to a sane future moment."""
    delay = payload.get("delay_minutes")
    if delay is not None:
        try:
            secs = float(delay) * 60.0
        except (TypeError, ValueError):
            return None
        if secs <= 0 or secs > MAX_HORIZON_SEC:
            return None
        return now + secs
    fire_at = parse_at(payload.get("at"), now)
    if fire_at is None or fire_at - now > MAX_HORIZON_SEC:
        return None
    return fire_at


def humanize(seconds):
    """A short spoken duration: '20 minutes', 'about 2 hours'."""
    minutes = max(1, round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = seconds / 3600.0
    return f"about {hours:.0f} hour{'s' if round(hours) != 1 else ''}"


class ReminderDaemon:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.reminders = {}   # id -> {"fire_at": epoch, "message": str, "timer": Timer}
        self._load()

    # ---------- persistence ----------

    def _persist(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with self.lock:
            snapshot = {rid: {"fire_at": r["fire_at"], "message": r["message"]}
                        for rid, r in self.reminders.items()}
        tmp = f"{REMINDERS_PATH}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(snapshot, f, indent=2)
            os.replace(tmp, REMINDERS_PATH)
        except OSError as e:
            print(f"Reminder daemon: could not persist reminders: {e}")

    def _load(self):
        try:
            with open(REMINDERS_PATH) as f:
                saved = json.load(f)
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError) as e:
            print(f"Reminder daemon: could not load reminders, starting fresh: {e}")
            return
        now = time.time()
        for rid, r in saved.items():
            try:
                self._arm(rid, float(r["fire_at"]), str(r["message"]), now=now)
            except (KeyError, TypeError, ValueError):
                continue
        print(f"Reminder daemon: reloaded {len(self.reminders)} pending reminder(s)")

    # ---------- arming / firing ----------

    def _arm(self, rid, fire_at, message, now=None):
        now = now if now is not None else time.time()
        delay = max(0.0, fire_at - now)
        timer = threading.Timer(delay, self._fire, args=(rid,))
        timer.daemon = True
        with self.lock:
            self.reminders[rid] = {"fire_at": fire_at, "message": message, "timer": timer}
        timer.start()

    def _fire(self, rid):
        with self.lock:
            r = self.reminders.pop(rid, None)
        if not r:
            return
        text = f"Reminder: {r['message']}"
        print(f"Reminder daemon: firing '{r['message']}'")
        self.bus.publish(SPEAK_TOPIC, {"text": text, "ts": time.time()})
        self.bus.publish(STATE_TOPIC, {"event": "fired", "id": rid,
                                       "message": r["message"], "ts": time.time()})
        self._persist()

    # ---------- inbound ----------

    def on_set(self, payload):
        message = str(payload.get("message") or "").strip()
        if not message:
            return
        now = time.time()
        fire_at = resolve_fire_at(payload, now)
        if fire_at is None:
            # Fail-soft: tell the user rather than silently doing nothing.
            self.bus.publish(SPEAK_TOPIC, {
                "text": "I couldn't tell when to remind you, so I didn't set that one.",
                "ts": now})
            return
        with self.lock:
            full = len(self.reminders) >= MAX_PENDING
        if full:
            print("Reminder daemon: too many pending reminders, dropping new one")
            return
        rid = str(payload.get("id") or uuid.uuid4())
        self._arm(rid, fire_at, message, now=now)
        self._persist()
        print(f"Reminder daemon: set '{message}' for +{humanize(fire_at - now)}")
        self.bus.publish(STATE_TOPIC, {"event": "set", "id": rid, "message": message,
                                       "fire_at": fire_at, "ts": now})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(SET_TOPIC, self.on_set)
        print(f"Reminder daemon active ({len(self.reminders)} pending), listening on {SET_TOPIC}")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    ReminderDaemon().run()
