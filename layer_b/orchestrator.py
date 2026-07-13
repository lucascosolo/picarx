#!/usr/bin/env python3
import json
import os
import subprocess
import time
import signal
import sys

REGISTRY_PATH = "/home/picarx/layer_b/module_registry.json"
MODULES_DIR = "/home/picarx/layer_b/modules"

running_processes = {}
# name -> mtime of its entrypoint file at the time it was (re)started
running_mtimes = {}
# Last successfully parsed registry - the manifest is re-read every
# sync cycle (which is what lets a newly added module_registry.json
# entry get picked up and started without restarting this process),
# so a load MUST be able to fail softly: an editor/deploy writing the
# file non-atomically means we can catch it half-written, and one bad
# read must neither crash this process nor (worse) parse as
# empty/partial and stop every running module. On any load problem we
# keep running against this last good copy and retry next cycle.
last_good_registry = None


def load_registry():
    global last_good_registry
    try:
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        if not isinstance(registry, list) or not all(
            isinstance(e, dict) and e.get("name") and e.get("entrypoint") for e in registry
        ):
            raise ValueError("registry must be a list of {name, entrypoint, ...} entries")
        last_good_registry = registry
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Orchestrator: could not load {REGISTRY_PATH} ({e}) - "
              f"keeping last good registry ({0 if last_good_registry is None else len(last_good_registry)} entries)")
    return last_good_registry or []


def module_path(entry):
    return f"{MODULES_DIR}/{entry['entrypoint']}"


def get_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError as e:
        print(f"Could not stat {path}: {e}")
        return None


def start_module(entry):
    if entry["name"] in running_processes:
        return
    path = module_path(entry)
    proc = subprocess.Popen(["python3", path])
    running_processes[entry["name"]] = proc
    running_mtimes[entry["name"]] = get_mtime(path)
    print(f"Started {entry['name']} (pid {proc.pid})")


def stop_module(name):
    if name in running_processes:
        running_processes[name].terminate()
        running_processes[name].wait()
        del running_processes[name]
        running_mtimes.pop(name, None)
        print(f"Stopped {name}")


def restart_module(entry):
    print(f"Detected updated file for {entry['name']}, restarting...")
    stop_module(entry["name"])
    start_module(entry)


def sync_with_registry():
    registry = load_registry()
    enabled_names = {e["name"] for e in registry if e.get("enabled")}

    for entry in registry:
        name = entry["name"]

        if not entry.get("enabled"):
            continue

        if name not in running_processes:
            start_module(entry)
            continue

        # Already running - check whether its file has been changed
        # on disk since we last (re)started it, and restart if so.
        path = module_path(entry)
        current_mtime = get_mtime(path)
        last_mtime = running_mtimes.get(name)
        if current_mtime is not None and current_mtime != last_mtime:
            restart_module(entry)
            continue

        # Also catch a process that died on its own (crash) so it
        # gets relaunched rather than silently staying down.
        if running_processes[name].poll() is not None:
            print(f"{name} exited unexpectedly (code {running_processes[name].returncode}), restarting...")
            del running_processes[name]
            running_mtimes.pop(name, None)
            start_module(entry)

    for name in list(running_processes.keys()):
        if name not in enabled_names:
            stop_module(name)


def shutdown(signum, frame):
    for name in list(running_processes.keys()):
        stop_module(name)
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)


def main():
    print("Orchestrator starting, syncing initial module state...")
    sync_with_registry()
    while True:
        time.sleep(5)
        try:
            sync_with_registry()
        except Exception as e:
            # A single bad cycle (stat race on a file being replaced,
            # a process fighting termination, etc.) must not take down
            # the supervisor for every module.
            print(f"Orchestrator: sync cycle failed ({e}), retrying next cycle")


if __name__ == "__main__":
    main()