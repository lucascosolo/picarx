#!/usr/bin/env python3
# layer_b/modules/self_trainer.py
"""
Idle self-trainer (Layer B) - practises in the sibling picarx-training
simulator while the robot is idle, refining its OWN learning, and folds the
result back in through the owning modules. Disabled by default
(module_registry.json); enable it only on a robot with picarx-training checked
out alongside.

The round-trip, once per eligible idle window:

  1. copy the live data dir (coach_policy.json + events.db/semantic.db) into a
     throwaway scratch dir, so the training subprocess never opens the live DBs.
  2. run  picarx-training/run_training.py <scenario>
          --knowledge-dir <scratch> --seed-from <scratch> --speedf <low> --quiet
     nice-d, as a subprocess. It seeds from the robot's own policy, so the pack
     it produces is a same-lineage refinement (Steps A/B1).
  3. on clean success, load the produced pack and publish it to the online
     intakes the owning modules expose:
        picarx/coach/adopt    -> coach folds arms into coach_policy.json (mode
                                 "adopt": this robot's own round-trip, so shared
                                 arms take the refined counts, never summed)
        picarx/memory/note    -> reflection persists transferable facts
        picarx/memory/pattern -> reflection persists mined patterns
     This module NEVER writes coach_policy.json or semantic.db itself -
     single-writer ownership is preserved.

Non-negotiables (mirrors reflection.py's idle discipline, and then some):
  * Live behaviour ALWAYS wins. Any movement intent, heard speech, or coach
    query resets the idle clock AND, if a session is running, SIGTERMs the
    subprocess and aborts immediately. run_training tears down cleanly on
    SIGTERM (its own handler), so the kill is instant and safe.
  * Fail-soft: a missing sibling repo, a crashed/killed/timed-out session, or a
    bad pack degrades to "no self-training this window" - never stuck, never a
    direct DB write, never a second writer.
  * Safety isolation is inherited: the training subprocess talks only to the
    sim's private /tmp/picarx_train_<port>.sock and an ephemeral bus port -
    never /tmp/picarx_safety.sock or localhost:1883, so it can't drive hardware.

All the decisions (idle/eligibility, scenario choice, pack->messages) are pure
module-level functions, unit-tested under tests/harness.py; the class keeps only
thin subprocess/file glue.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config

import glob
import json
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PICARX_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))   # dir holding layer_b/
DATA_DIR = robot_config.data_path()

# Files copied from live data/ to seed a session. Mirrors picarx-training's
# launcher.SEED_FILES: coach_policy.json is what matters (the sim refines it);
# events.db seeds the pattern-mining corpus; semantic.db rides along. spatial.db
# is deliberately excluded - place memories don't transfer.
SEED_FILES = ("coach_policy.json", "events.db", "semantic.db")

CHECK_INTERVAL = 30.0     # how often the idle/eligibility check runs
POLL_INTERVAL = 1.0       # how often a running session is checked for abort/timeout
NICE = 10                 # subprocess niceness - training must never starve live work
STATUS_TOPIC = "picarx/self_trainer/status"   # bus-visible lifecycle heartbeat

# Battery is only a proxy for "on the dock / topped up": there's no explicit
# charging line on picarx/state/world, so a healthy, high pack stands in.
BATT_HEALTHY_V = 7.0      # matches health_daemon.RECOVER_BATTERY_V

# ---- config (registered in robot_config.KNOBS, shown on the Config page) ----
IDLE_AFTER_SEC = float(robot_config.get(
    "self_trainer", "idle_after_sec", 600.0, env="SELF_TRAIN_IDLE_AFTER"))
COOLDOWN_SEC = float(robot_config.get(
    "self_trainer", "cooldown_sec", 10800.0, env="SELF_TRAIN_COOLDOWN"))
SPEEDF = float(robot_config.get(
    "self_trainer", "speedf", 3.0, env="SELF_TRAIN_SPEEDF"))
MAX_SESSION_SEC = float(robot_config.get(
    "self_trainer", "max_session_sec", 900.0, env="SELF_TRAIN_MAX_SESSION"))
SCENARIO_SOURCE = str(robot_config.get(
    "self_trainer", "scenario_source", "", env="SELF_TRAIN_SCENARIOS"))
CHARGING_ONLY = robot_config.get_bool(
    "self_trainer", "charging_only", False, env="SELF_TRAIN_CHARGING_ONLY")


# --------------------------------------------------------------------------
# pure decision helpers (unit-tested - no bus, no IO, no subprocess)
# --------------------------------------------------------------------------

def battery_healthy(battery, min_voltage=BATT_HEALTHY_V):
    """A stand-in for 'charging/docked' from picarx/state/world's battery block:
    present, fresh, not low/critical, and at/above min_voltage. Conservative -
    an absent or stale reading counts as NOT healthy, so charging_only errs
    toward not training rather than training on a guess."""
    if not isinstance(battery, dict):
        return False
    if battery.get("low") or battery.get("critical") or battery.get("stale"):
        return False
    v = battery.get("voltage")
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= min_voltage


def training_eligibility(now, last_activity, last_session_end, battery,
                         idle_after, cooldown, charging_only):
    """Pure decision: may a self-training session start right now?
    Returns (ok: bool, reason: str) - reason is a short tag for logging.

    Order encodes precedence: live activity blocks first (self-training must
    never compete with the robot doing something), then the between-sessions
    cooldown, then the optional charging guard (a footprint policy, not a
    safety gate). Only a fully idle, rested, (optionally) charging robot trains."""
    if now - last_activity < idle_after:
        return False, "busy"
    if now - last_session_end < cooldown:
        return False, "cooldown"
    if charging_only and not battery_healthy(battery):
        return False, "not-charging"
    return True, "idle"


def _is_motion_intent(action):
    """True if a picarx/intent/move action is the robot actually DRIVING (a
    non-zero forward/backward/turn), False for a stop / zero-speed / wheel-
    straighten. Pure/unit-tested - the idle detector keys on this so a parked
    robot's steady 'stop' stream doesn't read as activity."""
    if not isinstance(action, dict):
        return False
    direction = action.get("direction")
    if direction in ("forward", "backward"):
        return bool(action.get("speed"))
    if direction == "turn":
        return bool(action.get("angle"))
    return False   # stop / look / unknown -> not driving


def pick_scenario(scenario_paths, counter):
    """Deterministically rotate through the suite so successive sessions train
    on different scenarios. Sorted for stability. None if there are none."""
    if not scenario_paths:
        return None
    ordered = sorted(scenario_paths)
    return ordered[counter % len(ordered)]


def pack_to_messages(coach_policy, navigation_facts, lineage, mode="adopt"):
    """Translate a produced knowledge pack into the bus messages that route it
    back through the OWNING modules - coach for the policy, reflection for facts
    and patterns - so this module never writes a DB. Pure: dicts in, an ordered
    list of (topic, payload) out. Malformed facts/patterns are dropped."""
    messages = []
    if isinstance(coach_policy, dict) and coach_policy:
        messages.append(("picarx/coach/adopt",
                         {"coach_policy": coach_policy, "mode": mode,
                          "lineage": lineage}))
    nav = navigation_facts if isinstance(navigation_facts, dict) else {}
    for f in nav.get("facts") or []:
        subject = (f.get("subject") or "").strip()
        fact = (f.get("fact") or "").strip()
        if not subject or not fact:
            continue
        messages.append(("picarx/memory/note",
                         {"subject": subject, "fact": fact,
                          "confidence": f.get("confidence", 0.5),
                          "source": f.get("source") or "self_training"}))
    for p in nav.get("patterns") or []:
        condition = (p.get("condition") or "").strip()
        outcome = (p.get("outcome") or "").strip()
        if not condition or not outcome:
            continue
        messages.append(("picarx/memory/pattern",
                         {"condition": condition, "outcome": outcome,
                          "frequency": p.get("frequency", 0),
                          "confidence": p.get("confidence", 0.0)}))
    return messages


def resolve_training_repo(explicit=None, picarx_root=None):
    """Locate the picarx-training repo (the dir holding run_training.py).
    Mirrors sim/run_module._resolve_picarx_repo in reverse: an explicit path
    wins (validated), then the dev sibling-checkout and Pi home-dir layouts.
    Returns the repo root, or None if none has run_training.py."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if picarx_root:
        parent = os.path.dirname(picarx_root)
        candidates += [os.path.join(parent, "picarx-training"),
                       os.path.join(picarx_root, "picarx-training")]
    candidates.append(os.path.expanduser("~/picarx-training"))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "run_training.py")):
            return os.path.abspath(c)
    return None


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


# --------------------------------------------------------------------------
# thin file glue: snapshot live data into a scratch seed dir
# --------------------------------------------------------------------------

def _copy_live_data(src_dir, dst_dir):
    """Copy the seed files from the live data dir into a scratch dir. A SQLite
    db is snapshotted via the backup API (a consistent read even while coach /
    event_logger / reflection are mid-write), landing a standalone db with no
    -wal/-shm sidecar; JSON is copied verbatim (coach writes it atomically).
    Fail-soft per file - a snapshot that can't run falls back to a byte copy,
    and a missing file is just skipped."""
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in SEED_FILES:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src):
            continue
        try:
            if name.endswith(".db"):
                _snapshot_db(src, dst)
            else:
                shutil.copy2(src, dst)
            copied.append(name)
        except (OSError, sqlite3.Error) as e:
            print(f"Self-trainer: could not seed {name}: {e}")
    return copied


def _snapshot_db(src, dst):
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            dst_conn = sqlite3.connect(dst)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
    except sqlite3.Error:
        shutil.copy2(src, dst)   # not a usable sqlite db (or locked) - raw copy


def list_scenarios(scenario_source, training_repo):
    """Scenario JSON files to train on. `scenario_source` may be empty (use the
    repo's scenarios/ dir), a directory, a single .json file, or a glob. Thin
    IO around glob; returns a possibly-empty list."""
    if not scenario_source:
        return sorted(glob.glob(os.path.join(training_repo, "scenarios", "*.json")))
    if os.path.isdir(scenario_source):
        return sorted(glob.glob(os.path.join(scenario_source, "*.json")))
    if os.path.isfile(scenario_source):
        return [scenario_source]
    return sorted(glob.glob(scenario_source))


class SelfTrainer:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.last_activity = time.time()
        self.last_activity_topic = "startup"   # which topic last reset the idle clock
        self.last_session_end = 0.0
        self.latest_battery = {}
        self.proc = None                     # the running training subprocess, or None
        self._abort = threading.Event()      # set the instant activity interrupts a session
        self._session_counter = 0
        self.training_repo = resolve_training_repo(
            os.environ.get("PICARX_TRAINING_REPO"), PICARX_ROOT)

    # ---------- activity tracking (anything here means "not idle") ----------

    def on_move_intent(self, payload):
        """A DRIVING intent counts as activity; a 'stop' (or a zero-speed /
        straighten) does not. field_agent republishes 'stop' at 5Hz while parked
        or given-up, so counting stops would pin the idle clock and self-training
        would NEVER fire on a robot that is sitting still - exactly when we want
        it. Real motion (forward/backward/turn) still resets the clock and kills
        a running session, so live driving always preempts training."""
        try:
            if _is_motion_intent(payload.get("action")):
                self._mark_activity("intent/move")
        except Exception as e:
            print(f"Self-trainer: move handler error: {e}")

    def on_heard_activity(self, _payload):
        self._mark_activity("audio/heard")

    def on_coach_activity(self, _payload):
        self._mark_activity("coach/query")

    def _mark_activity(self, topic):
        """Live behaviour wins: reset the idle clock (recording WHICH topic did
        it, for the status heartbeat) and, if a session is running, kill it this
        instant so responsiveness never waits on training."""
        try:
            with self.lock:
                self.last_activity = time.time()
                self.last_activity_topic = topic
                proc = self.proc
            if proc is not None:
                self._abort.set()
                self._terminate(proc, f"activity ({topic})")
        except Exception as e:
            print(f"Self-trainer: activity handler error: {e}")

    # back-compat alias (older callers / tests)
    def on_activity(self, payload):
        self._mark_activity("activity")

    def on_world(self, payload):
        try:
            battery = payload.get("battery")
            if isinstance(battery, dict):
                with self.lock:
                    self.latest_battery = battery
        except Exception as e:
            print(f"Self-trainer: world handler error: {e}")

    # ---------- bus-visible status ----------

    def _publish_status(self, state, **extra):
        """Publish the self-trainer's current state to picarx/self_trainer/status
        so it's watchable with `mosquitto_sub`. Fail-soft: a status hiccup must
        never disturb (or block) the training loop. States: busy | cooldown |
        not-charging | training | published | aborted | timeout | failed |
        error | disabled | no-scenarios."""
        try:
            self.bus.publish(STATUS_TOPIC, {
                "state": state, "ts": time.time(),
                "repo": bool(self.training_repo), **extra})
        except Exception as e:
            print(f"Self-trainer: status publish failed: {e}")

    # ---------- session orchestration (thin) ----------

    def _terminate(self, proc, why):
        """SIGTERM the subprocess (run_training's handler tears it down cleanly)."""
        try:
            if proc.poll() is None:
                print(f"Self-trainer: {why} - stopping training session")
                proc.terminate()
        except Exception:
            pass

    def maybe_train(self, now=None):
        """One eligibility check; runs a session (blocking) if clear. Returns
        True if a session ran. Called off the main loop, so a blocking session
        never stalls the bus callbacks that can abort it."""
        now = now if now is not None else time.time()
        with self.lock:
            last_activity = self.last_activity
            last_activity_topic = self.last_activity_topic
            last_session_end = self.last_session_end
            battery = dict(self.latest_battery)
        ok, reason = training_eligibility(
            now, last_activity, last_session_end, battery,
            IDLE_AFTER_SEC, COOLDOWN_SEC, CHARGING_ONLY)
        if not self.training_repo:
            self._publish_status("disabled", reason="no picarx-training repo")
            return False
        if not ok:
            # Heartbeat WHY we're holding off, plus how long we've actually been
            # idle and which topic last reset it - so "stuck busy" is diagnosable
            # (e.g. last_activity "audio/heard" -> the mic is spraying messages).
            self._publish_status(
                reason,
                idle_for_sec=round(now - last_activity),
                last_activity=last_activity_topic,
                idle_needed_sec=IDLE_AFTER_SEC,
                cooldown_remaining_sec=round(
                    max(0.0, COOLDOWN_SEC - (now - last_session_end))))
            return False
        self._run_session()
        return True

    def _run_session(self):
        scratch = tempfile.mkdtemp(prefix="picarx_selftrain_")
        self._abort.clear()
        try:
            _copy_live_data(DATA_DIR, scratch)
            scenarios = list_scenarios(SCENARIO_SOURCE, self.training_repo)
            scenario = pick_scenario(scenarios, self._session_counter)
            self._session_counter += 1
            if not scenario:
                self._publish_status("no-scenarios")
                print("Self-trainer: no scenarios to train on - skipping")
                return
            scenario_name = os.path.basename(scenario)
            cmd = ["nice", "-n", str(NICE), sys.executable,
                   os.path.join(self.training_repo, "run_training.py"), scenario,
                   "--knowledge-dir", scratch, "--seed-from", scratch,
                   "--speedf", f"{SPEEDF:g}", "--quiet"]
            env = dict(os.environ)
            env.setdefault("PICARX_REPO", PICARX_ROOT)   # help the sim find our repo
            print(f"Self-trainer: training on {scenario_name} "
                  f"(speedf {SPEEDF:g}, nice {NICE})")
            self._publish_status("training", scenario=scenario_name, speedf=SPEEDF)
            proc = subprocess.Popen(cmd, cwd=self.training_repo, env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            with self.lock:
                self.proc = proc
            outcome = self._await_session(proc)
            with self.lock:
                self.proc = None
            if outcome != "done":
                self._publish_status(outcome, scenario=scenario_name)
                print(f"Self-trainer: session {outcome} - no learning published")
                return
            self._publish_pack(scratch, scenario_name)
        except Exception as e:
            self._publish_status("error", detail=str(e))
            print(f"Self-trainer: session error: {e}")
        finally:
            with self.lock:
                self.proc = None
                self.last_session_end = time.time()   # cooldown applies even on abort/failure
            shutil.rmtree(scratch, ignore_errors=True)

    def _await_session(self, proc):
        """Block until the subprocess finishes, is aborted by live activity, or
        blows the wall-clock budget. Returns 'done' | 'aborted' | 'timeout' |
        'failed'. Instant abort is handled in on_activity; this loop also
        enforces the deadline and reaps the process."""
        deadline = time.time() + MAX_SESSION_SEC
        timed_out = False
        while proc.poll() is None:
            if self._abort.is_set():
                self._terminate(proc, "activity")
                break
            if time.time() >= deadline:
                timed_out = True
                self._terminate(proc, "max session time")
                break
            time.sleep(POLL_INTERVAL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if self._abort.is_set():
            return "aborted"
        if timed_out:
            return "timeout"
        return "done" if proc.returncode == 0 else "failed"

    def _publish_pack(self, scratch, scenario=None):
        policy = _load_json(os.path.join(scratch, "coach_policy.json"), {})
        nav = _load_json(os.path.join(scratch, "navigation_facts.json"), {})
        manifest = _load_json(os.path.join(scratch, "knowledge_pack.json"), {})
        lineage = manifest.get("lineage") if isinstance(manifest, dict) else None
        messages = pack_to_messages(policy, nav, lineage, mode="adopt")
        for topic, payload in messages:
            self.bus.publish(topic, payload)
        notes = sum(1 for t, _ in messages if t == "picarx/memory/note")
        patterns = sum(1 for t, _ in messages if t == "picarx/memory/pattern")
        adopted = any(t == "picarx/coach/adopt" for t, _ in messages)
        print(f"Self-trainer: published refined learning (lineage {lineage or '?'}) "
              f"-> coach/adopt {'yes' if adopted else 'no'}, "
              f"{notes} notes, {patterns} patterns")
        self._publish_status("published", scenario=scenario, lineage=lineage,
                             adopted=adopted, notes=notes, patterns=patterns)

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/intent/move", self.on_move_intent)
        self.bus.subscribe("picarx/audio/heard", self.on_heard_activity)
        self.bus.subscribe("picarx/coach/query", self.on_coach_activity)
        self.bus.subscribe("picarx/state/world", self.on_world)

        if not self.training_repo:
            print("Self-trainer: picarx-training repo not found beside this repo "
                  "(set PICARX_TRAINING_REPO) - idle self-training disabled.")
            self._publish_status("disabled", reason="no picarx-training repo")
        else:
            print(f"Self-trainer active (repo {self.training_repo}), training when "
                  f"idle {IDLE_AFTER_SEC:.0f}s+, cooldown {COOLDOWN_SEC:.0f}s"
                  + (", only when charging" if CHARGING_ONLY else ""))
            self._publish_status("starting", idle_after_sec=IDLE_AFTER_SEC,
                                 cooldown_sec=COOLDOWN_SEC, charging_only=CHARGING_ONLY)
        while True:
            time.sleep(CHECK_INTERVAL)
            try:
                self.maybe_train()
            except Exception as e:
                print(f"Self-trainer: cycle error: {e}")


if __name__ == "__main__":
    SelfTrainer().run()
