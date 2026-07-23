#!/usr/bin/env python3
# layer_b/modules/debug_monitor.py
"""
Debug Monitor (Layer B) - always-on resource/telemetry logger.

Exists to answer two concrete questions from the field: "which process
is actually spiking CPU, and when" and "why does the robot keep saying
'I keep running into something, let me get some advice' over and over
instead of eventually giving up or actually getting unstuck." Neither
question is answerable from event_logger.py's events.db alone - that
records what the robot decided and sensed, not what the OS/CPU was
doing at the time, and it has no per-process resource visibility at
all.

Two independent things get logged here, both to LOG_PATH as one JSON
object per line (plain text, greppable/tailable without SQL - deliberately
NOT routed through event_logger.py, which is the sole writer to events.db
and has its own topic whitelist; this is a separate concern with a
separate file):

  1. A resource sample every SAMPLE_INTERVAL seconds: overall CPU%
     (from /proc/stat), per-tracked-module CPU% (from /proc/<pid>/stat,
     matched to a module name via /proc/<pid>/cmdline against
     TRACKED_MODULES - this is what actually answers "which process"),
     load average, memory, and (less often, since it shells out)
     vcgencmd temperature/throttle status. All read straight from
     /proc rather than a dependency like psutil, both to avoid another
     "pip package didn't land in the systemd service's exact python3"
     footgun and because a handful of small file reads every few
     seconds is already about as cheap as this can be.

  2. A "collision_loop_triggered" entry every time field_agent.py fires
     an urgent coach query for a collision fail state (the trigger
     behind the "I keep running into something" line), with the
     world-state context at that moment (distance/objects/battery) and
     a rolling count of how many times that's happened recently. If
     that count crosses FAIL_LOOP_THRESHOLD within FAIL_LOOP_WINDOW,
     the entry is flagged - the actual "over and over" signal, since a
     single fail-state trip is normal/expected but a tight repeating
     cluster is not.

Both are cheap and bounded: resource sampling never spawns a
subprocess more than once per TEMP_CHECK_INTERVAL, and the log file
self-truncates at MAX_LOG_BYTES so it can't slowly fill the SD card
over a long uptime.
"""
import os
import sys
import time
import json
import subprocess
import threading
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config
import heartbeat

DATA_DIR = robot_config.data_path()
LOG_PATH = f"{DATA_DIR}/resource_log.jsonl"
MAX_LOG_BYTES = 5 * 1024 * 1024   # self-truncate past this size, keep the newer half

SAMPLE_INTERVAL = 5.0             # seconds between resource samples
TEMP_CHECK_INTERVAL = 30.0        # vcgencmd shells out - check it less often than everything else
SPIKE_REPORT_PCT = 90.0           # print (not just log) when a module/overall CPU hits this

FAIL_LOOP_WINDOW = 120.0          # seconds - how far back to look for a repeating fail-state pattern
FAIL_LOOP_THRESHOLD = 3           # this many collision_loop queries within the window -> flag it

# A module is "silent" (presumed dead/wedged) if its last unified heartbeat
# (see heartbeat.py) is older than this. Default: three heartbeat intervals, so
# one dropped beat is fine but a crashed module is caught quickly.
_HB_INTERVAL = float(robot_config.get(
    "observability", "heartbeat_interval_sec", heartbeat.DEFAULT_INTERVAL_SEC,
    env="PICARX_HEARTBEAT_INTERVAL"))
HEARTBEAT_STALE_SEC = max(15.0, _HB_INTERVAL * 3)


def evaluate_liveness(heartbeats, now, stale_after=HEARTBEAT_STALE_SEC):
    """Split the modules we've ever heard a heartbeat from into alive vs silent
    by how long ago each was last seen. Pure - the caller owns the clock and the
    seen-map. `heartbeats` is {name: {"last_seen": ts, ...}}."""
    alive, silent = [], []
    for name, hb in heartbeats.items():
        (alive if (now - hb.get("last_seen", 0)) <= stale_after else silent).append(name)
    return {"alive": sorted(alive), "silent": sorted(silent)}

# Matched against /proc/<pid>/cmdline substrings - these are the actual
# entrypoint filenames from module_registry.json (plus this module itself,
# so its own overhead is visible too, not hidden from the log it writes).
TRACKED_MODULES = [
    "vision_basic.py", "audio_nodes.py", "coach.py", "companion.py",
    "field_agent.py", "arbiter.py", "world_state.py", "event_logger.py",
    "distance_sensor.py", "debug_monitor.py", "safety_daemon.py",
]


# ---------- pure /proc readers (no MQTT/state - kept standalone and testable) ----------

def _read_cpu_totals():
    """(idle_jiffies, total_jiffies) summed across all cores from /proc/stat."""
    with open("/proc/stat") as f:
        line = f.readline()
    parts = [int(x) for x in line.split()[1:]]
    idle = parts[3] + parts[4]   # idle + iowait
    total = sum(parts)
    return idle, total


def _read_loadavg():
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return {"load1": float(parts[0]), "load5": float(parts[1]), "load15": float(parts[2])}


def _read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                info[key] = int(rest.strip().split()[0])   # kB
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    used_pct = round(100.0 * (total - available) / total, 1) if total else None
    return {"mem_used_pct": used_pct, "mem_available_kb": available}


def _read_pid_jiffies(pid):
    """utime+stime (jiffies) for one pid, or None if it's already gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # comm field can itself contain spaces/parens - split after the LAST
    # closing paren so the rest of the fields are safe to just .split().
    after = raw.rsplit(")", 1)[-1].split()
    utime, stime = int(after[11]), int(after[12])
    return utime + stime


def _list_tracked_pids():
    """pid -> module name, by scanning /proc/*/cmdline for our known scripts."""
    found = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read().decode(errors="ignore")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for name in TRACKED_MODULES:
            if name in cmdline:
                found[int(entry)] = name
                break
    return found


def _read_temp_and_throttle():
    """Best-effort - silently returns {} on anything that isn't a Pi with vcgencmd."""
    result = {}
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2)
        if out.returncode == 0:
            result["temp_c"] = float(out.stdout.strip().split("=")[1].split("'")[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        if out.returncode == 0:
            flags = int(out.stdout.strip().split("=")[1], 16)
            result["under_voltage_now"] = bool(flags & 0x1)
            result["freq_capped_now"] = bool(flags & 0x2)
            result["throttled_now"] = bool(flags & 0x4)
            result["throttled_ever"] = bool(flags & 0xF0000)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return result


class DebugMonitor:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.fail_events = deque()   # timestamps of collision_loop coach queries
        self.latest_context = {}     # last known world-state snapshot, for correlating fail events
        self.heartbeats = {}         # module name -> last heartbeat {last_seen, pid, seq, status}
        self._silent = set()         # modules currently flagged silent (to log transitions once)

    # ---------- bus callbacks ----------

    def on_heartbeat(self, payload):
        name = payload.get("name")
        if not name:
            return
        with self.lock:
            self.heartbeats[name] = {
                "last_seen": payload.get("ts") or time.time(),
                "pid": payload.get("pid"), "seq": payload.get("seq"),
                "status": payload.get("status")}

    def _check_liveness(self, now):
        """Log current module liveness and print transitions (a module going
        silent, or coming back). Returns the liveness dict."""
        with self.lock:
            snapshot = {n: dict(hb) for n, hb in self.heartbeats.items()}
            prev_silent = set(self._silent)
        live = evaluate_liveness(snapshot, now)
        silent = set(live["silent"])
        newly_silent = silent - prev_silent
        recovered = prev_silent - silent
        with self.lock:
            self._silent = silent
        for name in sorted(newly_silent):
            last = snapshot[name]["last_seen"]
            print(f"Debug Monitor: module '{name}' went SILENT - no heartbeat for "
                  f"{now - last:.0f}s (crashed, exited, or wedged?)")
        for name in sorted(recovered):
            print(f"Debug Monitor: module '{name}' is heartbeating again")
        if newly_silent or recovered:
            self._write({"ts": now, "type": "module_liveness",
                         "alive": live["alive"], "silent": live["silent"]})
        return live

    def on_world_state(self, payload):
        with self.lock:
            self.latest_context = {
                "distance_cm": payload.get("distance_cm"),
                "objects": [o.get("label") for o in payload.get("objects", {}).get("items", [])],
                "battery_v": (payload.get("battery") or {}).get("voltage"),
            }

    def on_coach_query(self, payload):
        if payload.get("situation") != "collision_loop":
            return
        now = time.time()
        with self.lock:
            self.fail_events.append(now)
            while self.fail_events and now - self.fail_events[0] > FAIL_LOOP_WINDOW:
                self.fail_events.popleft()
            recent_count = len(self.fail_events)
            context = dict(self.latest_context)

        looping = recent_count >= FAIL_LOOP_THRESHOLD
        self._write({
            "ts": now,
            "type": "collision_loop_triggered",
            "reason": (payload.get("extra") or {}).get("reason"),
            "recent_count": recent_count,
            "looping": looping,
            "context": context,
        })
        if looping:
            print(f"Debug Monitor: collision_loop has fired {recent_count} times in the last "
                  f"{FAIL_LOOP_WINDOW:.0f}s - looks like a real repeating stuck pattern, not one-off noise")

    # ---------- log writer (shared by both producers, self-truncating) ----------

    def _write(self, entry):
        os.makedirs(DATA_DIR, exist_ok=True)
        line = json.dumps(entry) + "\n"
        with self.lock:
            try:
                if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > MAX_LOG_BYTES:
                    with open(LOG_PATH, "rb") as f:
                        f.seek(-MAX_LOG_BYTES // 2, os.SEEK_END)
                        tail = f.read()
                    # the seek almost certainly landed mid-line - drop
                    # everything up to (and including) the next newline
                    # so every remaining line is a complete JSON object.
                    _, _, tail = tail.partition(b"\n")
                    with open(LOG_PATH, "wb") as f:
                        f.write(tail)
            except OSError:
                pass
            with open(LOG_PATH, "a") as f:
                f.write(line)

    # ---------- resource sampling loop ----------

    def _resource_loop(self):
        clock_ticks = os.sysconf("SC_CLK_TCK")
        last_idle, last_total = _read_cpu_totals()
        last_pid_jiffies = {}
        last_sample_time = time.time()
        next_temp_check = 0.0

        while True:
            time.sleep(SAMPLE_INTERVAL)
            now = time.time()
            elapsed = now - last_sample_time
            last_sample_time = now

            idle, total = _read_cpu_totals()
            d_idle, d_total = idle - last_idle, total - last_total
            last_idle, last_total = idle, total
            overall_cpu_pct = round(100.0 * (1 - d_idle / d_total), 1) if d_total > 0 else None

            per_module = {}
            new_pid_jiffies = {}
            for pid, name in _list_tracked_pids().items():
                jiffies = _read_pid_jiffies(pid)
                if jiffies is None:
                    continue
                new_pid_jiffies[pid] = jiffies
                prev = last_pid_jiffies.get(pid)
                if prev is not None and elapsed > 0:
                    pct = round(100.0 * (jiffies - prev) / (clock_ticks * elapsed), 1)
                    per_module[name] = max(per_module.get(name, 0.0), pct)
            last_pid_jiffies = new_pid_jiffies

            entry = {
                "ts": now,
                "type": "resource_sample",
                "cpu_overall_pct": overall_cpu_pct,
                "per_module_cpu_pct": per_module,
            }
            entry.update(_read_loadavg())
            entry.update(_read_meminfo())
            if now >= next_temp_check:
                entry.update(_read_temp_and_throttle())
                next_temp_check = now + TEMP_CHECK_INTERVAL

            # Module self-reported liveness (unified heartbeat) rides along on
            # every sample, and going-silent/recovering transitions are logged
            # and printed separately by _check_liveness.
            entry["module_liveness"] = self._check_liveness(now)

            self._write(entry)

            spiking = [f"{name}={pct}%" for name, pct in per_module.items() if pct >= SPIKE_REPORT_PCT]
            if spiking or (overall_cpu_pct is not None and overall_cpu_pct >= SPIKE_REPORT_PCT):
                extra = f" modules={','.join(spiking)}" if spiking else ""
                print(f"Debug Monitor: CPU spike - overall={overall_cpu_pct}%{extra}")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/state/world", self.on_world_state)
        self.bus.subscribe("picarx/coach/query", self.on_coach_query)
        self.bus.subscribe(heartbeat.HEARTBEAT_TOPIC, self.on_heartbeat)

        threading.Thread(target=self._resource_loop, daemon=True).start()

        print(f"Debug Monitor active, logging to {LOG_PATH}")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    DebugMonitor().run()
