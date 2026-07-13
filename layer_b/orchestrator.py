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


def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


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
        sync_with_registry()


if __name__ == "__main__":
    main()