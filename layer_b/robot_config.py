#!/usr/bin/env python3
# /home/picarx/layer_b/robot_config.py
"""
Central tunables for Layer B: one JSON file, every knob visible.

All user-adjustable configuration lives in config.json NEXT TO THIS FILE
(layer_b/config.json), shipped with every key present at its default
value - open it to see everything you can change, edit, and restart the
orchestrator. Precedence per knob, highest first:

  1. The environment variable (the same names that were used before this
     file existed), so one-off experiments (`ESPEAK_VOICE=mb-en1 python3
     orchestrator.py`) and any existing systemd Environment= lines keep
     working unchanged.
  2. The value in config.json.
  3. The built-in default baked into the calling module.

Fail-soft like everything else in Layer B: a missing or corrupt
config.json just means built-in defaults (one console line says so); it
never crashes a module. Secrets (ANTHROPIC_API_KEY) deliberately stay
environment-only - a JSON file that lives in a git repo is where keys go
to leak, so this loader will never serve them.

Modules use it like:

    import robot_config
    VOICE = robot_config.get("audio", "espeak_voice", "mb-us1", env="ESPEAK_VOICE")

Values from the environment arrive as strings (exactly as os.environ.get
always returned them); values from config.json keep their JSON types.
Call sites keep their own int()/float()/str() coercion, which handles
both. get_bool() is the exception - env flag strings like "0"/"false"
need real parsing, so booleans get a dedicated helper.

The file path itself can be overridden with LAYER_B_CONFIG (mainly for
tests and unusual layouts).
"""
import json
import os

CONFIG_PATH = os.environ.get(
    "LAYER_B_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))

_cache = None


def reload():
    """Forget the cached file so the next get() re-reads it. For tests."""
    global _cache
    _cache = None


def _load():
    global _cache
    if _cache is None:
        try:
            with open(CONFIG_PATH) as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("top level must be a JSON object")
            _cache = loaded
        except FileNotFoundError:
            print(f"robot_config: {CONFIG_PATH} not found; using built-in defaults")
            _cache = {}
        except (OSError, ValueError) as e:
            print(f"robot_config: could not read {CONFIG_PATH} ({e}); "
                  f"using built-in defaults")
            _cache = {}
    return _cache


def get(section, key, default, env=None):
    """One tunable. Precedence: env var (set and non-empty) > config.json >
    default. A JSON null (or a missing key) falls through to the default,
    so a config file can list a key without pinning it."""
    if env:
        v = os.environ.get(env)
        if v not in (None, ""):
            return v
    section_obj = _load().get(section)
    if isinstance(section_obj, dict):
        v = section_obj.get(key)
        if v is not None:
            return v
    return default


_FALSY = ("0", "", "false", "no", "off")


def get_bool(section, key, default, env=None):
    """Boolean tunable. Env strings "0"/""/"false"/"no"/"off" (any case)
    are False and anything else set is True - matching how the old
    bool(os.environ.get(...)) call sites behaved, plus the explicit
    falsy words some flags already accepted. JSON true/false pass
    through; JSON null / missing key falls to the default."""
    if env:
        v = os.environ.get(env)
        if v is not None:
            return v.strip().lower() not in _FALSY
    section_obj = _load().get(section)
    if isinstance(section_obj, dict):
        v = section_obj.get(key)
        if v is not None:
            return bool(v)
    return bool(default)
