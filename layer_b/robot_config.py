#!/usr/bin/env python3
# layer_b/robot_config.py
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
import copy
import json
import os

CONFIG_PATH = os.environ.get(
    "LAYER_B_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))

# The directory THIS file lives in is the Layer B install root (the layer_b/
# tree), wherever the repo happens to be checked out. Everything derives its
# paths from here instead of hard-coding an absolute location, so the whole
# tree can be relocated just by moving it. Set LAYER_B_HOME to override (for a
# symlinked install, or code and data deliberately split apart).
BASE_DIR = os.environ.get("LAYER_B_HOME") or os.path.dirname(
    os.path.abspath(__file__))


def base_path(*parts):
    """Absolute path to something inside the Layer B tree, wherever it lives:
    base_path('modules', 'models') -> <install root>/modules/models."""
    return os.path.join(BASE_DIR, *parts)


def data_path(*parts):
    """Absolute path inside the on-robot data dir (layer_b/data/, gitignored)."""
    return os.path.join(BASE_DIR, "data", *parts)


_cache = None


# ---------------------------------------------------------------------------
# Knob registry - the single source of truth for EVERYTHING a user can tune.
#
# Every get()/get_bool() call site across the modules is listed here exactly
# once. The web console's Config page renders straight from this list, so it is
# by construction complete: every file-or-env tunable shows up, with the right
# type, help text, default, and - crucially - a warning when an environment
# variable is currently shadowing the file value (env still wins at runtime, so
# a stale `export` would otherwise silently defeat an edit made in the browser).
#
# Two tests keep this honest: one asserts the registry and config.json list the
# same knobs at the same defaults; another scans the source for every
# `env="..."` and asserts each is registered. Add a knob here the moment you
# add a get() call, and it appears on the Config page for free.
#
# `env=None` means the knob is config-file-only (no environment override).
# The Claude API key is deliberately absent: secrets never live in the file or
# on the page (see the module docstring).
#
# type is one of "str" | "int" | "float" | "bool".
# ---------------------------------------------------------------------------
KNOBS = [
    # ---- audio (audio_nodes.py) ----
    {"section": "audio", "key": "speaker_enable_cmd", "type": "str",
     "default": "robot_hat enable_speaker", "env": "SPEAKER_ENABLE_CMD",
     "desc": "Shell command run to power the speaker amp before speaking."},
    {"section": "audio", "key": "vosk_model_path", "type": "str",
     "default": base_path("modules", "models", "model-en-lgraph"),
     "env": "VOSK_MODEL_PATH", "desc": "Path to the Vosk speech-to-text model."},
    {"section": "audio", "key": "debug_levels", "type": "bool",
     "default": False, "env": "AUDIO_DEBUG_LEVELS",
     "desc": "Print live mic input/noise levels to help tune the gates."},
    {"section": "audio", "key": "espeak_voice", "type": "str",
     "default": "mb-us1", "env": "ESPEAK_VOICE",
     "desc": "TTS voice: mb-us1 (US female) / mb-us2, mb-us3 (US male) / "
             "mb-en1 (British male). `espeak --voices=mb` lists installed ones."},
    {"section": "audio", "key": "espeak_speed", "type": "int",
     "default": 130, "env": "ESPEAK_SPEED",
     "desc": "TTS words-per-minute (espeak default is ~175)."},
    {"section": "audio", "key": "espeak_pitch", "type": "str",
     "default": "", "env": "ESPEAK_PITCH",
     "desc": "TTS pitch 0-99; empty string = espeak's default."},
    {"section": "audio", "key": "gain", "type": "float",
     "default": 12.0, "env": "AUDIO_GAIN",
     "desc": "Digital mic amplification for low-gain USB mics."},
    {"section": "audio", "key": "bandpass", "type": "bool",
     "default": True, "env": "AUDIO_BANDPASS",
     "desc": "Band-pass filter each capture chunk to reject out-of-band room noise."},
    {"section": "audio", "key": "bandpass_hp_hz", "type": "int",
     "default": 150, "env": "AUDIO_BANDPASS_HP",
     "desc": "Band-pass high-pass cutoff (Hz): kill rumble below this."},
    {"section": "audio", "key": "bandpass_lp_hz", "type": "int",
     "default": 4000, "env": "AUDIO_BANDPASS_LP",
     "desc": "Band-pass low-pass cutoff (Hz): kill hiss above this."},
    {"section": "audio", "key": "heard_min_confidence", "type": "float",
     "default": 0.3, "env": "HEARD_MIN_CONFIDENCE",
     "desc": "Drop STT decodes below this mean word confidence (stop/halt always pass)."},
    {"section": "audio", "key": "heard_min_snr", "type": "float",
     "default": 2.5, "env": "HEARD_MIN_SNR",
     "desc": "An utterance's peak level must exceed this multiple of the noise floor."},
    # ---- companion (companion.py) ----
    {"section": "companion", "key": "model", "type": "str",
     "default": "claude-sonnet-5", "env": "COMPANION_MODEL",
     "desc": "Claude model for conversation."},
    {"section": "companion", "key": "intent_model", "type": "str",
     "default": "claude-haiku-4-5-20251001", "env": "INTENT_MODEL",
     "desc": "Cheaper Claude model for fast intent classification."},
    {"section": "companion", "key": "chat_noise_quality", "type": "float",
     "default": 0.2, "env": "CHAT_NOISE_QUALITY",
     "desc": "Below this utterance-quality score, drop silently (likely noise)."},
    {"section": "companion", "key": "chat_min_quality", "type": "float",
     "default": 0.45, "env": "CHAT_MIN_QUALITY",
     "desc": "Between noise_quality and this, say 'I didn't catch that' with no LLM call."},
    # ---- coach (coach.py) ----
    {"section": "coach", "key": "model", "type": "str",
     "default": "claude-haiku-4-5-20251001", "env": "COACH_MODEL",
     "desc": "Claude model for maneuver coaching."},
    # ---- reflection (reflection.py) ----
    {"section": "reflection", "key": "model", "type": "str",
     "default": "claude-haiku-4-5-20251001", "env": "REFLECTION_MODEL",
     "desc": "Claude model for idle-time reflection."},
    # ---- radio (radio.py) ----
    {"section": "radio", "key": "alsa_device", "type": "str",
     "default": "plug:robot_speaker", "env": "RADIO_ALSA_DEVICE",
     "desc": "ALSA output device the radio player writes to."},
    {"section": "radio", "key": "tts_settle_sec", "type": "float",
     "default": 2.0, "env": "RADIO_TTS_SETTLE",
     "desc": "Pause radio for this long around spoken replies so TTS is audible."},
    # ---- web console (web_console.py) ----
    {"section": "web_console", "key": "port", "type": "int",
     "default": 8088, "env": "WEB_CONSOLE_PORT",
     "desc": "TCP port this console listens on (restart to apply)."},
    # ---- bluetooth (tools/bluetooth_daemon.py) ----
    {"section": "bluetooth", "key": "connect_cmd", "type": "str",
     "default": "nmcli device connect {mac}", "env": "BT_CONNECT_CMD",
     "desc": "Shell command to connect a device; {mac} is substituted."},
    # ---- health (tools/health_daemon.py) ----
    {"section": "health", "key": "battery_adc", "type": "bool",
     "default": False, "env": "HEALTH_BATTERY_ADC",
     "desc": "Direct-ADC battery fallback for setups without world_state "
             "(leave off normally - it contends on the I2C bus)."},
    # ---- embeddings (embedding_util.py) ----
    {"section": "embeddings", "key": "model_path", "type": "str",
     "default": data_path("models", "minilm", "model.onnx"),
     "env": "EMBED_MODEL_PATH", "desc": "Path to the MiniLM ONNX embedding model."},
    {"section": "embeddings", "key": "tokenizer_path", "type": "str",
     "default": data_path("models", "minilm", "tokenizer.json"),
     "env": "EMBED_TOKENIZER_PATH", "desc": "Path to the embedding model's tokenizer."},
    # ---- kinematics (steering_controller.py) - file-only ----
    {"section": "kinematics", "key": "wheelbase_mm", "type": "int",
     "default": 95, "env": None,
     "desc": "Physical wheelbase in mm - measure your chassis."},
    {"section": "kinematics", "key": "max_steer_deg", "type": "int",
     "default": 30, "env": None, "desc": "Maximum steering angle (deg)."},
    {"section": "kinematics", "key": "steering_rate_deg_per_sec", "type": "int",
     "default": 60, "env": None,
     "desc": "Cap on commanded steering slew (lower = smoother arcs)."},
    # ---- steering (steering_controller.py) - file-only ----
    {"section": "steering", "key": "area_distance_k", "type": "float",
     "default": 35.0, "env": None,
     "desc": "Box-area->distance calibration: k = distance_cm * sqrt(area_ratio)."},
    {"section": "steering", "key": "clearance_m", "type": "float",
     "default": 0.15, "env": None, "desc": "Preferred lateral passing clearance (m)."},
    {"section": "steering", "key": "cruise_speed", "type": "int",
     "default": 25, "env": None, "desc": "Default forward speed while exploring."},
    {"section": "steering", "key": "curve_slowdown_gain", "type": "float",
     "default": 0.9, "env": None,
     "desc": "How hard speed drops with steering angle (0 = never, 1 = full)."},
]


def knobs():
    """The full knob registry as a deep copy (see KNOBS). The web console's
    Config page renders from this, so it always lists every tunable."""
    return copy.deepcopy(KNOBS)


def env_override(env_name):
    """The current value of environment variable `env_name` if it is SET and
    non-empty (which is exactly when it wins over config.json in get()), else
    None. Lets the Config page warn that a stale env var is shadowing a knob."""
    if not env_name:
        return None
    v = os.environ.get(env_name)
    return v if v not in (None, "") else None


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


# ---------- whole-file access (for the web console's Config page) ----------

def all_config():
    """The full parsed config.json as a deep COPY (so callers can't mutate the
    cache), or {} if the file is missing/corrupt. Includes the `_readme` block."""
    return copy.deepcopy(_load())


def merge_and_save(edits):
    """Apply `edits` (a {section: {key: value}} dict) onto the current
    config.json and write it back atomically, preserving every key the edits
    don't mention - including `_readme` and any sections the editor never
    showed. Returns the saved config. Raises ValueError on a malformed `edits`
    shape and OSError on a write failure; the caller (web console) reports
    either back to the browser rather than crashing.

    Env vars still override these values at runtime (see get()), and most
    modules read config only at startup, so a saved change lands when the
    module next restarts - the console says as much."""
    if not isinstance(edits, dict):
        raise ValueError("config edits must be an object")
    merged = copy.deepcopy(_load())
    for section, keys in edits.items():
        if section == "_readme" or not isinstance(keys, dict):
            raise ValueError(f"section {section!r} must map to an object of knobs")
        dest = merged.setdefault(section, {})
        if not isinstance(dest, dict):
            raise ValueError(f"section {section!r} is not a knob group")
        for key, value in keys.items():
            if isinstance(value, (dict, list)):
                raise ValueError(f"{section}.{key} must be a scalar value")
            dest[key] = value
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    os.replace(tmp, CONFIG_PATH)   # atomic: a reader never sees a half file
    reload()
    return merged
