"""Small, machine-readable catalog of voice-routable tool intents.

The catalog is deliberately conservative. It describes the semantic request
that an intent repair may produce; individual tool daemons still own their
payload validation and safety checks. Keeping this separate from the LLM
prompt makes the same safety classification available to runtime gates,
tests, and operator diagnostics without importing a module with MQTT side
effects.
"""
import json


_SPECS = (
    # Local, read-only queries are the only commands eligible for automatic
    # LLM repair. They can be repeated without changing robot state.
    ("stop", "safety_stop", False, ()),
    ("battery", "read_only", True, ()),
    ("status", "read_only", True, ()),
    ("history", "read_only", True, ()),
    ("objects", "read_only", True, ()),
    ("map", "read_only", True, ()),
    ("why", "read_only", True, ()),
    ("hello", "read_only", True, ()),
    ("who am i", "read_only", True, ()),
    ("where are you", "read_only", True, ()),
    ("what's playing", "read_only", True, ()),
    ("list stations", "read_only", True, ()),
    ("list reminders", "read_only", True, ()),
    ("list notes", "read_only", True, ()),
    ("where is ", "read_only", True, ("object",)),
    ("what's in ", "read_only", True, ("place",)),
    # These may be valid commands, but they change state, contact a service,
    # or can create/delete durable data. Intent repair must not execute them
    # from a guess; an explicit local parse or confirmation is required.
    ("play radio", "reversible_write", False, ()),
    ("stop radio", "reversible_write", False, ()),
    ("next station", "reversible_write", False, ()),
    ("tune to ", "reversible_write", False, ("frequency",)),
    ("radio find ", "reversible_write", False, ("keywords",)),
    ("station ", "reversible_write", False, ("station",)),
    ("start meeting notes", "reversible_write", False, ()),
    ("pause meeting notes", "reversible_write", False, ()),
    ("resume meeting notes", "reversible_write", False, ()),
    ("stop meeting notes", "reversible_write", False, ()),
    ("remind me in ", "reversible_write", False, ("time", "message")),
    ("remind me at ", "reversible_write", False, ("time", "message")),
    ("take a note ", "reversible_write", False, ("text",)),
    ("delete reminders ", "destructive", False, ("query",)),
    ("cancel reminders ", "destructive", False, ("query",)),
    ("delete notes ", "destructive", False, ("query",)),
    ("call this place ", "reversible_write", False, ("name",)),
    ("ssh into ", "remote", False, ("host",)),
    ("connect to ", "remote", False, ("host",)),
    ("disconnect remote assist", "remote", False, ()),
)


def _make_spec(pattern, safety_class, idempotent, required_fields):
    return {
        "pattern": pattern,
        "safety_class": safety_class,
        "idempotent": bool(idempotent),
        "required_fields": list(required_fields),
        "auto_repair": safety_class == "read_only" and bool(idempotent),
    }


CATALOG = tuple(_make_spec(*row) for row in _SPECS)


def command_spec(command):
    """Return a copy of the catalog entry matching an exact/prefix command."""
    value = " ".join(str(command or "").strip().lower().split())
    if not value:
        return None
    matches = [spec for spec in CATALOG
               if value == spec["pattern"].rstrip()
               or (spec["pattern"].endswith(" ")
                   and value.startswith(spec["pattern"]))]
    if not matches:
        return None
    match = max(matches, key=lambda spec: len(spec["pattern"]))
    result = dict(match)
    result["command"] = value
    return result


def prompt_text():
    """Compact catalog text safe to include in a bounded repair prompt."""
    rows = []
    for spec in CATALOG:
        rows.append({
            "command": spec["pattern"] + (
                "<value>" if spec["pattern"].endswith(" ") else ""),
            "safety": spec["safety_class"],
            "auto_repair": spec["auto_repair"],
        })
    return json.dumps(rows, separators=(",", ":"))

