#!/usr/bin/env python3
"""Load the tracked module catalog plus an optional robot-local overlay.

``module_registry.json`` is intentionally safe to version and deploy. The
``activation`` field describes need-driven lifecycle policy; the orchestrator
normally decides when a module runs. A robot can keep machine-specific
availability overrides in the ignored ``module_registry.local.json`` file
instead, so changing the catalog no longer dirties the tracked manifest or
creates pull conflicts.
"""
import json
import os

import robot_config


REGISTRY_PATH = robot_config.base_path("module_registry.json")
LOCAL_REGISTRY_PATH = robot_config.base_path("module_registry.local.json")


def _read(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _validate(entries, source):
    if not isinstance(entries, list) or not all(
            isinstance(entry, dict) and entry.get("name") and
            entry.get("entrypoint") for entry in entries):
        raise ValueError(f"{source} must be a list of {{name, entrypoint, ...}} entries")
    return entries


def _overlay(base, local):
    """Apply either a compact name->override map or a registry list."""
    by_name = {entry["name"]: dict(entry) for entry in base}
    order = [entry["name"] for entry in base]

    if isinstance(local, list):
        overrides = ((entry.get("name"), entry) for entry in local
                     if isinstance(entry, dict))
    elif isinstance(local, dict):
        overrides = local.items()
    else:
        raise ValueError("local registry must be an object or list")

    for name, override in overrides:
        if not name:
            raise ValueError("local registry contains an entry without a name")
        if isinstance(override, bool):
            override = {"enabled": override}
        if not isinstance(override, dict):
            raise ValueError(f"local override for {name!r} must be an object or boolean")
        if name not in by_name:
            # A local-only module is useful while developing on the robot,
            # but it still needs an entrypoint before the orchestrator can
            # safely start it.
            if not override.get("entrypoint"):
                raise ValueError(f"local override for unknown module {name!r} lacks entrypoint")
            order.append(name)
            by_name[name] = {}
        by_name[name].update(override)
        by_name[name]["name"] = name

    return _validate([by_name[name] for name in order], "merged module registry")


def load_registry(registry_path=REGISTRY_PATH, local_path=LOCAL_REGISTRY_PATH):
    """Return defaults merged with the optional ignored local overlay.

    A missing local file is normal.  Invalid base data remains an error, while
    an invalid local file fails soft to the tracked defaults so a bad edit
    cannot stop the supervisor from starting the robot.
    """
    base = _validate(_read(registry_path), registry_path)
    if not os.path.exists(local_path):
        return base
    try:
        return _overlay(base, _read(local_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Module registry: ignoring invalid local overlay {local_path}: {exc}")
        return base
