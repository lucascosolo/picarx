#!/usr/bin/env python3
"""Need-driven Layer B process supervisor.

The registry is a capability catalog with lifecycle policy, not a list of
things a user must manually enable. Infrastructure stays up, state-owned
workers follow RobotState, and demand workers are started by their typed bus
request and stopped after their need expires.
"""
import os
import signal
import subprocess
import sys
import time

import robot_config
from broker_client import Bus
from module_lifecycle import NeedPlanner, STATE_TOPIC
from module_registry import load_registry as read_registry

REGISTRY_PATH = robot_config.base_path("module_registry.json")
LOCAL_REGISTRY_PATH = robot_config.base_path("module_registry.local.json")
MODULES_DIR = robot_config.base_path("modules")

running_processes = {}
running_mtimes = {}
last_good_registry = None
planner = None
lifecycle_bus = None
lifecycle_subscriptions = set()
pending_replays = []
deferred_stops = {}
last_lifecycle_status = None
last_bus_error_at = 0.0
LIFECYCLE_TICK_SEC = 0.5
DEMAND_STOP_GRACE_SEC = 0.75


def load_registry():
    global last_good_registry
    try:
        registry = read_registry(REGISTRY_PATH, LOCAL_REGISTRY_PATH)
        last_good_registry = registry
    except (OSError, ValueError) as e:
        print(f"Orchestrator: could not load {REGISTRY_PATH} ({e}) - "
              f"keeping last good registry ({0 if last_good_registry is None else len(last_good_registry)} entries)")
    return last_good_registry or []


def module_path(entry):
    return os.path.join(MODULES_DIR, entry["entrypoint"])


def get_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError as e:
        print(f"Could not stat {path}: {e}")
        return None


def start_module(entry, replay_payload=None):
    name = entry["name"]
    if name in running_processes:
        return
    path = module_path(entry)
    proc = subprocess.Popen([sys.executable, path])
    running_processes[name] = proc
    running_mtimes[name] = get_mtime(path)
    print(f"Started {name} (pid {proc.pid})")
    if replay_payload is not None:
        activation = entry.get("activation") or {}
        topic = activation.get("topic")
        if topic:
            pending_replays.append({
                "due": time.monotonic() + 0.35,
                "name": name, "topic": topic,
                "payload": dict(replay_payload),
            })


def stop_module(name):
    proc = running_processes.get(name)
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"{name} ignored SIGTERM for 10s, killing it")
        proc.kill()
        proc.wait()
    del running_processes[name]
    running_mtimes.pop(name, None)
    deferred_stops.pop(name, None)
    pending_replays[:] = [item for item in pending_replays if item["name"] != name]
    print(f"Stopped {name}")


def restart_module(entry):
    print(f"Detected updated file for {entry['name']}, restarting...")
    stop_module(entry["name"])
    start_module(entry)


def _entry(name):
    return planner.registry.get(name) if planner is not None else None


def _ensure_lifecycle_bus():
    """Keep the supervisor alive if MQTT starts after systemd does."""
    global lifecycle_bus, last_bus_error_at
    if lifecycle_bus is not None:
        return
    try:
        lifecycle_bus = Bus()
    except Exception as exc:
        now = time.monotonic()
        if now - last_bus_error_at >= 10.0:
            print(f"Orchestrator: lifecycle bus unavailable ({exc}); "
                  "continuing with infrastructure modules")
            last_bus_error_at = now


def _publish_lifecycle_status(force=False):
    global last_lifecycle_status
    if planner is None or lifecycle_bus is None:
        return
    status = planner.status()
    status["running"] = sorted(running_processes)
    signature = (tuple(status["desired"]), tuple(status["running"]),
                 status["state"], tuple(status["needs"]))
    if force or signature != last_lifecycle_status:
        last_lifecycle_status = signature
        lifecycle_bus.publish("picarx/lifecycle/status", status)


def _sync_lifecycle(now=None):
    now = time.time() if now is None else float(now)
    if planner is None:
        return
    desired = planner.desired_names(now)
    for name in desired:
        entry = _entry(name)
        if entry is not None and name not in running_processes:
            start_module(entry)

    for name in list(running_processes):
        if name in desired:
            continue
        entry = _entry(name) or {}
        activation = entry.get("activation") or {}
        if activation.get("mode") == "demand":
            due = deferred_stops.get(name)
            if due is None:
                deferred_stops[name] = time.monotonic() + DEMAND_STOP_GRACE_SEC
                continue
            if time.monotonic() < due:
                continue
        stop_module(name)
    _publish_lifecycle_status()


def _on_state(payload):
    if planner is not None:
        planner.set_state(payload)
        _sync_lifecycle()


def _on_demand(topic, payload):
    if planner is None:
        return
    now = time.time()
    before = set(planner.desired_names(now))
    name = planner.observe_demand(topic, payload, now)
    if name is None:
        return
    after = set(planner.desired_names(now))
    if name in after and name not in running_processes:
        entry = _entry(name)
        if entry is not None:
            start_module(entry, replay_payload=payload)
    elif name in before and name not in after and name in running_processes:
        # Give the daemon time to receive its explicit false/stop request and
        # publish safe cleanup before terminating its process.
        deferred_stops[name] = time.monotonic() + DEMAND_STOP_GRACE_SEC
    _sync_lifecycle(now)


def _on_state_signal(topic, payload):
    if planner is not None:
        planner.observe_state_signal(topic, payload)
        _sync_lifecycle()


def _on_lifecycle_topic(topic, payload):
    if planner is None:
        return
    is_demand = any((entry.get("activation") or {}).get("topic") == topic
                    for entry in planner.registry.values())
    if is_demand:
        _on_demand(topic, payload)
    else:
        _on_state_signal(topic, payload)


def _subscribe_lifecycle_topics(registry):
    if lifecycle_bus is None:
        return
    topics = {STATE_TOPIC}
    for entry in registry:
        activation = entry.get("activation") or {}
        if activation.get("mode") == "demand":
            for key in ("topic", "state_topic"):
                if activation.get(key):
                    topics.add(activation[key])
    for topic in sorted(topics - lifecycle_subscriptions):
        callback = _on_state if topic == STATE_TOPIC else \
            (lambda payload, topic=topic: _on_lifecycle_topic(topic, payload))
        lifecycle_bus.subscribe(topic, callback)
        lifecycle_subscriptions.add(topic)


def _run_pending_replays():
    if lifecycle_bus is None:
        return
    now = time.monotonic()
    due = [item for item in pending_replays if item["due"] <= now]
    pending_replays[:] = [item for item in pending_replays if item["due"] > now]
    for item in due:
        if item["name"] in running_processes:
            lifecycle_bus.publish(item["topic"], item["payload"])


def sync_with_registry():
    global planner
    registry = load_registry()
    if planner is None:
        planner = NeedPlanner(registry)
    else:
        planner.replace_registry(registry)
    _subscribe_lifecycle_topics(registry)
    _sync_lifecycle()

    desired = planner.desired_names()
    for name in list(running_processes):
        entry = _entry(name)
        if entry is None or name not in desired:
            continue
        path = module_path(entry)
        current_mtime = get_mtime(path)
        if current_mtime is not None and current_mtime != running_mtimes.get(name):
            restart_module(entry)
            continue
        proc = running_processes[name]
        if proc.poll() is not None:
            print(f"{name} exited unexpectedly (code {proc.returncode}), restarting...")
            del running_processes[name]
            running_mtimes.pop(name, None)
            start_module(entry)
    _run_pending_replays()


def shutdown(signum, frame):
    for name in list(running_processes):
        stop_module(name)
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)


def main():
    print("Orchestrator starting, syncing need-driven module state...")
    added = robot_config.sync_defaults()
    if added:
        print(f"Orchestrator: materialized {len(added)} new config default(s) "
              f"into config.json ({', '.join(f'{s}.{k}' for s, k in added)})")
    _ensure_lifecycle_bus()
    sync_with_registry()
    while True:
        time.sleep(LIFECYCLE_TICK_SEC)
        try:
            _ensure_lifecycle_bus()
            sync_with_registry()
        except Exception as e:
            print(f"Orchestrator: sync cycle failed ({e}), retrying next cycle")


if __name__ == "__main__":
    main()
