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
from repository_updater import (HEALTH_TOPIC, CONTROL_TOPIC,
                                 RepositoryUpdater)

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
maintenance_mode = False
repository_updater = None
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
    if maintenance_mode:
        return
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
    if maintenance_mode:
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
    if repository_updater is not None:
        repository_updater.on_state(payload)
    if planner is not None:
        planner.set_state(payload)
        _sync_lifecycle()


def _on_health(payload):
    if repository_updater is not None:
        repository_updater.on_health(payload)


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


def _quiesce_for_update():
    """Stop every Layer B child so camera/audio leases are released."""
    global maintenance_mode
    maintenance_mode = True
    # Give the arbiter an explicit safe command before terminating it.  The
    # safety daemon's 0.75s drive lease watchdog is an independent backstop,
    # but an update should not depend on waiting for that timeout.
    if lifecycle_bus is not None:
        lifecycle_bus.publish("picarx/intent/move", {
            "source": "repository_updater", "priority": 1000,
            "action": {"direction": "stop"}, "ttl": 1.0,
            "reason": "repository update quiescence", "ts": time.time()})
    for name in list(running_processes):
        stop_module(name)
    return True


def _resume_after_update_failure():
    global maintenance_mode
    maintenance_mode = False
    sync_with_registry()


def _restart_after_update():
    """Replace the supervisor with the newly pulled checkout.

    The production process is ``picarx-orchestrator.service``.  Re-execing in
    place keeps that systemd unit supervising the new code without requiring
    the Layer B user to have permission to invoke ``systemctl``.  The separate
    ``picarx-safety.service`` process is never stopped or re-execed here.
    """
    for name in list(running_processes):
        stop_module(name)
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
    raise RuntimeError("orchestrator re-exec returned unexpectedly")


def _services_healthy():
    required = ("robot_state", "camera_controller", "audio_nodes",
                "arbiter", "field_agent")
    missing = [name for name in required if name not in running_processes]
    dead = [name for name in required
            if name in running_processes and running_processes[name].poll() is not None]
    if missing or dead:
        return False, f"missing={missing}, dead={dead}"
    return True, "core Layer B services are running"


def _initialize_repository_updater():
    global repository_updater
    if lifecycle_bus is None:
        return
    repo_path = robot_config.get(
        "repository_updater", "repo_path",
        os.path.dirname(robot_config.BASE_DIR), env="PICARX_UPDATE_REPO")
    repository_updater = RepositoryUpdater(
        repo_path=repo_path,
        remote=robot_config.get("repository_updater", "remote", "origin",
                                env="PICARX_UPDATE_REMOTE"),
        branch=robot_config.get("repository_updater", "branch", "master",
                                env="PICARX_UPDATE_BRANCH"),
        enabled=robot_config.get_bool("repository_updater", "enabled", False,
                                      env="PICARX_UPDATE_ENABLED"),
        poll_interval=float(robot_config.get(
            "repository_updater", "poll_interval_sec", 3600.0,
            env="PICARX_UPDATE_INTERVAL")),
        health_timeout=float(robot_config.get(
            "repository_updater", "health_timeout_sec", 30.0,
            env="PICARX_UPDATE_HEALTH_TIMEOUT")),
        publish=lifecycle_bus.publish,
        quiesce=_quiesce_for_update,
        resume=_resume_after_update_failure,
        restart=_restart_after_update,
        runtime_health_check=_services_healthy,
    )
    lifecycle_bus.subscribe(CONTROL_TOPIC, repository_updater.on_control)
    lifecycle_bus.subscribe(HEALTH_TOPIC, _on_health)
    # RobotState does not rely on retained MQTT messages. Ask for a fresh
    # snapshot so an approved update is evaluated against current state.
    lifecycle_bus.publish("picarx/state/query", {"source": "repository_updater"})
    if not repository_updater.startup_recover():
        return
    repository_updater.start()


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
    _initialize_repository_updater()
    while True:
        time.sleep(LIFECYCLE_TICK_SEC)
        try:
            _ensure_lifecycle_bus()
            sync_with_registry()
        except Exception as e:
            print(f"Orchestrator: sync cycle failed ({e}), retrying next cycle")


if __name__ == "__main__":
    main()
