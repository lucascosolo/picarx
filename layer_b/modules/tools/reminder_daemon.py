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
CONTROL_TOPIC = "picarx/tools/reminder/control"
RESULT_TOPIC = "picarx/tools/reminder/result"
STATE_TOPIC = "picarx/tools/reminder/state"
SPEAK_TOPIC = "picarx/audio/speak"
ROBOT_STATE_TOPIC = "picarx/state/current"

DATA_DIR = robot_config.data_path()
REMINDERS_PATH = f"{DATA_DIR}/reminders.json"

MAX_PENDING = 50            # ignore new sets past this, so a runaway can't pile up timers
MAX_HORIZON_SEC = 7 * 86400  # refuse absurdly far-future reminders (likely a parse error)
BOOT_ANNOUNCE_DELAY = 3.0
BOOT_MAX_REMINDERS = 5


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
    def __init__(self, bus=None):
        self.bus = bus or Bus()
        self.lock = threading.Lock()
        self.reminders = {}   # id -> {"fire_at": epoch, "message": str, "timer": Timer}
        self.robot_state = "IDLE"
        self.boot_announced = False
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

    def snapshot(self, now=None, limit=MAX_PENDING):
        now = time.time() if now is None else float(now)
        try:
            limit = max(1, min(MAX_PENDING, int(limit)))
        except (TypeError, ValueError):
            limit = MAX_PENDING
        with self.lock:
            rows = [{"id": rid, "message": r["message"],
                     "fire_at": r["fire_at"],
                     "remaining_sec": max(0.0, r["fire_at"] - now)}
                    for rid, r in self.reminders.items()]
        rows.sort(key=lambda r: r["fire_at"])
        return rows[:limit]

    def _result(self, payload, command, ok=True, result=None, error=None,
                speak=None):
        reply = {"ok": bool(ok), "command": command,
                 "request_id": payload.get("request_id"), "ts": time.time()}
        if result is not None:
            reply["result"] = result
        if error:
            reply["error"] = str(error)[:500]
        self.bus.publish(RESULT_TOPIC, reply)
        if speak:
            self.bus.publish(SPEAK_TOPIC, {"text": str(speak)[:400],
                                           "ts": time.time()})
        return reply

    def _cancel_id(self, rid):
        with self.lock:
            record = self.reminders.pop(str(rid), None)
        if record:
            record["timer"].cancel()
            self._persist()
        return record

    def _resolve_id(self, payload):
        rid = str(payload.get("id") or "").strip()
        if rid:
            return rid
        query = str(payload.get("query") or payload.get("message") or "").strip().lower()
        if not query:
            raise ValueError("reminder id or search text is required")
        matches = [row for row in self.snapshot()
                   if query in row["message"].lower()]
        if len(matches) != 1:
            raise ValueError("search must identify exactly one pending reminder")
        return matches[0]["id"]

    def on_control(self, payload):
        payload = dict(payload or {})
        command = str(payload.get("command") or payload.get("op") or "").lower()
        source = str(payload.get("source") or "voice")
        try:
            if command in {"list", "status"}:
                rows = self.snapshot()
                spoken = self._speak_list(rows) if source != "web" else None
                return self._result(payload, command,
                                    result={"reminders": rows}, speak=spoken)
            if command in {"delete", "cancel", "remove"}:
                if not payload.get("confirmed"):
                    raise ValueError("explicit confirmation is required to delete a reminder")
                rid = self._resolve_id(payload)
                record = self._cancel_id(rid)
                if record is None:
                    raise ValueError("pending reminder not found")
                self.bus.publish(STATE_TOPIC, {"event": "deleted", "id": rid,
                                               "message": record["message"],
                                               "ts": time.time()})
                return self._result(payload, "delete", result={"id": rid,
                                    "message": record["message"]},
                                    speak="I cancelled that reminder."
                                    if source != "web" else None)
            raise ValueError(f"unsupported reminder command: {command}")
        except (TypeError, ValueError) as exc:
            return self._result(payload, command, ok=False, error=exc,
                                speak=f"I couldn't update reminders: {exc}"
                                if source != "web" else None)

    @staticmethod
    def _speak_list(rows):
        if not rows:
            return "You have no pending reminders."
        messages = "; ".join(row["message"] for row in rows[:5])
        suffix = " and more" if len(rows) > 5 else ""
        return "Pending reminders: " + messages + suffix + "."

    def on_robot_state(self, payload):
        self.robot_state = str(payload.get("state") or "IDLE").upper()

    def announce_pending(self):
        """Speak a bounded boot briefing once, unless a safety owner is active."""
        if self.boot_announced:
            return False
        if self.robot_state in {"RC", "SAFETY_STOP", "SPEAKING"}:
            return False
        rows = self.snapshot(limit=BOOT_MAX_REMINDERS)
        self.boot_announced = True
        if not rows:
            return False
        if len(rows) == 1:
            text = f"You have one reminder pending: {rows[0]['message']}."
        else:
            messages = "; ".join(row["message"] for row in rows)
            text = f"You have {len(rows)} reminders pending: {messages}."
        self.bus.publish(SPEAK_TOPIC, {"text": text[:400], "ts": time.time()})
        self.bus.publish(STATE_TOPIC, {"event": "boot_briefing",
                                       "count": len(rows), "reminders": rows,
                                       "ts": time.time()})
        return True

    def _delayed_boot_announce(self):
        if self.announce_pending():
            return
        if not self.boot_announced:
            timer = threading.Timer(BOOT_ANNOUNCE_DELAY, self._delayed_boot_announce)
            timer.daemon = True
            timer.start()

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
        with self.lock:
            if rid in self.reminders:
                self.bus.publish(SPEAK_TOPIC, {
                    "text": "That reminder already exists, so I did not duplicate it.",
                    "ts": now})
                return
        self._arm(rid, fire_at, message, now=now)
        self._persist()
        print(f"Reminder daemon: set '{message}' for +{humanize(fire_at - now)}")
        self.bus.publish(STATE_TOPIC, {"event": "set", "id": rid, "message": message,
                                       "fire_at": fire_at, "ts": now})
        if str(payload.get("source") or "voice") != "web":
            self.bus.publish(SPEAK_TOPIC, {
                "text": f"Reminder set for {humanize(fire_at - now)}: {message}.",
                "ts": time.time()})
        self._result(payload, "set", result={"id": rid, "message": message,
                                              "fire_at": fire_at})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(SET_TOPIC, self.on_set)
        self.bus.subscribe(CONTROL_TOPIC, self.on_control)
        self.bus.subscribe(ROBOT_STATE_TOPIC, self.on_robot_state)
        timer = threading.Timer(BOOT_ANNOUNCE_DELAY, self._delayed_boot_announce)
        timer.daemon = True
        timer.start()
        print(f"Reminder daemon active ({len(self.reminders)} pending), listening on {SET_TOPIC}")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    ReminderDaemon().run()
