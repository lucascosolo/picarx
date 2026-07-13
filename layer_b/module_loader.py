#!/usr/bin/env python3
# /home/picarx/layer_b/module_loader.py
"""
Dynamic module loader. New capabilities are added by dropping a file
in modules/ and adding an entry to module_registry.json - no core
process restart required, since load_all_enabled() can be re-invoked
at runtime after the registry changes.
"""
import json
import importlib.util
import os

REGISTRY_PATH = "/home/picarx/layer_b/module_registry.json"
MODULES_DIR = "/home/picarx/layer_b/modules"

def load_registry():
  with open(REGISTRY_PATH) as f:
    return json.load(f)

def load_module(entry):
  path = os.path.join(MODULES_DIR, entry["entrypoint"])
  spec = importlib.util.spec_from_file_location(entry["name"], path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

def load_all_enabled():
  loaded = {}
  for entry in load_registry():
    if entry.get("enabled"):
      try:
        loaded[entry["name"]] = load_module(entry)
        print(f"Loaded module: {entry['name']}, v{entry['version']}")
      except Exception as e:
        print(f"Failed to load {entry['name']}: {e}")
  return loaded

if __name__ == "__main__":
  modules = load_all_enabled()
  print(f"Active modules: {list(modules.keys())}")

