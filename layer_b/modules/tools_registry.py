#!/usr/bin/env python3
# layer_b/modules/tools_registry.py
"""
Tools Registry (Layer B) - the pluggable, non-safety-critical ability
layer.

A "tool" is anything fun or useful that is NOT part of the drive/
explore/learn pipeline: radio, future games, party tricks. Each tool
is its own module listening on its own picarx/tools/<name> topic; this
registry is the single voice-command front door that routes utterances
to them, so adding a tool never means touching field_agent again.

Routing contract with field_agent: field_agent ignores any utterance
containing a tool keyword (TOOL_KEYWORDS there mirrors the names
here), so "stop radio" reaches the radio and never trips the
robot-wide "stop". Movement words are deliberately NOT routable as
tools - safety-relevant commands stay in field_agent's fast local
path.

Publishes picarx/tools/available at startup (and on request via
"what tools do you have"), so both humans and other modules can
discover what's installed. Tool invocations go to the decision
journal like every other choice the robot makes.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import speech_match

import re
import time

REMOTE_TOPIC = "picarx/tools/remote_assist"
REMINDER_SET_TOPIC = "picarx/tools/reminder/set"
REMINDER_CONTROL_TOPIC = "picarx/tools/reminder/control"
NOTES_TOPIC = "picarx/tools/notes"

# ---------- spoken-number → dial string ----------
# Vosk transcribes a frequency as WORDS ("ninety eight point seven",
# "one oh two point five"), sometimes as digits ("98.7"). This turns
# either into a canonical dial string like "98.7" so the radio can
# match it. Frequencies use two spoken conventions - grouped tens
# ("ninety eight" = 98) and digit-by-digit ("one oh two" = 102) - and
# this handles both, keyed off whether every token is a single digit.
_ONES = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD = {**_ONES, **_TEENS, **_TENS, "hundred": 100, "a": 1}


def _spoken_amount(value):
    """Parse the small numeric vocabulary used by reminder phrases."""
    value = str(value or "").strip().lower()
    try:
        return float(value)
    except ValueError:
        pass
    if value in _ONES:
        return float(_ONES[value])
    if value in _TEENS:
        return float(_TEENS[value])
    if value in _TENS:
        return float(_TENS[value])
    return None


def _side_to_digits(tokens):
    """Convert one side of the dial (before or after the point) to a
    digit string. Returns None if a token isn't a number word."""
    if not tokens:
        return ""
    if any(t not in _NUMWORD for t in tokens):
        return None
    # All single-digit words (ones/oh) -> digit-by-digit ("one oh two").
    if all(t in _ONES for t in tokens):
        return "".join(str(_ONES[t]) for t in tokens)
    # Otherwise grouped arithmetic ("ninety eight", "one hundred eight").
    current = 0
    for t in tokens:
        if t == "hundred":
            current = (current or 1) * 100
        else:
            current += _NUMWORD[t]
    return str(current)


def parse_dial(text):
    """Return a canonical dial string ('98.7') from a spoken/typed
    frequency, or None if the text doesn't clearly contain one."""
    text = text.lower()
    # Digits already present: "98.7", "98 7", "1025", "987".
    m = re.search(r"\b(\d{2,4})(?:[.\s](\d))?\b", text)
    if m and m.group(2):
        return f"{m.group(1)}.{m.group(2)}"
    if m and len(m.group(1)) >= 4:                    # e.g. "1025" -> 102.5
        return f"{m.group(1)[:-1]}.{m.group(1)[-1]}"
    if m and len(m.group(1)) == 3:                    # e.g. "987" -> 98.7
        return f"{m.group(1)[:2]}.{m.group(1)[2]}"
    if m:
        return m.group(1)                             # bare "98"
    # Word form. Split on point/dot, keep only number words either side.
    tokens = re.findall(r"[a-z]+", text)
    tokens = [t for t in tokens if t in _NUMWORD or t in ("point", "dot")]
    if not tokens:
        return None
    if "point" in tokens or "dot" in tokens:
        sep = "point" if "point" in tokens else "dot"
        i = tokens.index(sep)
        whole = _side_to_digits(tokens[:i])
        frac = _side_to_digits(tokens[i + 1:])
        if whole and frac:
            return f"{whole}.{frac}"
        return whole or None
    return _side_to_digits(tokens)


# ---------- rule table ----------
# Each rule: (compiled pattern, tool topic, payload builder). First
# match wins, top to bottom - put more specific patterns first. A
# builder may return None to signal "matched the shape but couldn't
# extract params" so routing falls through to the next rule.
def _tune_payload(m, text):
    dial = parse_dial(text)
    return {"command": "play", "dial": dial} if dial else None


# Words that aren't part of the genre/name in a search utterance:
# "radio find some soft rock for me please" -> keywords "soft rock".
_FIND_FILLER = {"radio", "station", "stations", "find", "search", "look",
                "for", "a", "an", "some", "me", "please", "the", "up",
                "on", "of", "to", "my", "play"}


def _find_payload(m, text):
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _FIND_FILLER]
    if not words or all(w.isdigit() for w in words):
        return None  # nothing searchable / it's a dial - let later rules tune it
    return {"command": "find", "keywords": " ".join(words)}


def _remote_payload(m, text):
    # Keep extraction deliberately narrow. Hostnames/IPs are validated again
    # by remote_assist.py; this only prevents ordinary navigation phrases such
    # as "connect to the kitchen" from becoming network requests.
    host = None
    if m:
        for i in range(1, (len(m.groups()) if m.lastindex else 0) + 1):
            if m.group(i):
                host = m.group(i).strip().rstrip(".,")
                break
    if not host:
        return None
    # speech_match.canonicalize() separates punctuation in IPv4 addresses
    # ("192.168.1.20" -> "192 168 1 20"). Put the four octets back together
    # before the remote module performs its own strict validation.
    if re.fullmatch(r"\d{1,3}(?:\s+\d{1,3}){3}", host):
        host = ".".join(host.split())
    return {"command": "connect", "host": host}


def _reminder_payload(m, text):
    amount = _spoken_amount(m.group("amount")) if m and m.group("amount") else None
    if amount is not None:
        unit = m.group("unit").lower()
        minutes = amount / 60.0 if unit.startswith("second") else amount
        if unit.startswith("hour"):
            minutes = amount * 60.0
        message = m.group("message").strip(" .,!?")
        if message and minutes > 0:
            return {"message": message, "delay_minutes": minutes}
    if m and m.group("clock"):
        hour, minute = m.group("clock").split()
        message = m.group("clock_message").strip(" .,!?")
        if message:
            return {"message": message, "at": f"{int(hour):02d}:{int(minute):02d}"}
    return None


def _note_payload(m, text):
    body = (m.group("body") if m else "").strip(" .,!?")
    return {"command": "create", "text": body, "source": "voice"} if body else None


def _query_payload(m, text):
    query = (m.group("query") if m and m.groupdict().get("query") else "").strip(" .,!?")
    query = re.sub(r"^(?:about|for|named|called)\s+", "", query)
    return {"command": "delete", "query": query, "confirmed": True,
            "source": "voice"} if query else None


# Patterns run against speech_match.canonicalize()d text ("play the
# radio for me please" arrives here as "play radio"), so they only need
# to cover meaningful word variants, not filler permutations.
RULES = [
    # Explicit, deterministic reminder phrases.  More ambiguous requests
    # such as "remind me to call mom" remain available to the companion LLM,
    # which can ask for a time instead of inventing one.
    (re.compile(r"\bremind(?:\s+me)?\s+in\s+(?P<amount>\d+(?:\.\d+)?|"
                r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                r"twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|"
                r"fifty|sixty)\s+(?P<unit>seconds?|minutes?|hours?)\s+"
                r"(?:to\s+)?(?P<message>.+)$|"
                r"\bremind(?:\s+me)?\s+at\s+(?P<clock>\d{1,2}\s+\d{2})\s+"
                r"(?:to\s+)?(?P<clock_message>.+)$"),
     REMINDER_SET_TOPIC, _reminder_payload),
    (re.compile(r"\b(?:list|show|what are)\b.*\breminders?\b|"
                r"\bwhat reminders\b"),
     REMINDER_CONTROL_TOPIC, lambda m, t: {"command": "list", "source": "voice"}),
    (re.compile(r"\b(?:cancel|delete|remove)\b\s+reminders?\s+(?P<query>.+)$"),
     REMINDER_CONTROL_TOPIC, _query_payload),
    # A note with content is a single memory entry; a bare "take notes" is a
    # consented continuous meeting session.
    (re.compile(r"\b(?:take|make|write|add)\s+(?:a\s+)?note\b"
                r"(?:\s+(?:that|saying|about))?\s*(?P<body>.+)$"),
     NOTES_TOPIC, _note_payload),
    (re.compile(r"\b(?:start|begin)\s+(?:taking\s+)?(?:meeting\s+)?notes?\b|"
                r"\btake\s+(?:meeting\s+)?notes?\b"),
     NOTES_TOPIC, lambda m, t: {"command": "start", "confirmed": True,
                                "source": "voice"}),
    (re.compile(r"\b(?:pause|resume|stop|end|finish)\b.*\bmeeting\s+notes?\b"),
     NOTES_TOPIC,
     lambda m, t: {"command": ("pause" if "pause" in m.group(0) else
                                "resume" if "resume" in m.group(0) else "stop"),
                    "source": "voice"}),
    (re.compile(r"\b(?:list|show|search)\b.*\bnotes?\b"),
     NOTES_TOPIC, lambda m, t: {"command": "list", "source": "voice"}),
    (re.compile(r"\b(?:delete|remove)\s+notes?\s+(?P<query>.+)$"),
     NOTES_TOPIC, _query_payload),
    # Remote project assistance. Require ssh/remote/host/computer wording or
    # an unmistakable "ssh into" phrase so this never captures a place goal.
    (re.compile(r"\b(?:revoke|remove|disable)\b.*\b(?:remote|ssh|host|project)\b"
                r".*\bwrite(?: access| permission)?\b"),
     REMOTE_TOPIC, lambda m, t: {"command": "revoke_write"}),
    (re.compile(r"\b(?:grant|give|allow|enable)\b.*\b(?:remote|ssh|host|project)\b"
                r".*\bwrite(?: access| permission)?\b|"
                r"\b(?:remote|ssh|host|project)\b.*\bwrite access\b"),
     REMOTE_TOPIC, lambda m, t: {"command": "authorize_write", "confirmed": True}),
    (re.compile(r"\b(?:ssh\s+(?:into\s+)?|(?:remote|host|computer)\s+)"
                r"((?:\d{1,3}(?:[ .]\d{1,3}){3})|[a-z][a-z0-9.-]{1,62})\b|"
                r"\bconnect\s+to\s+((?:\d{1,3}(?:[ .]\d{1,3}){3})|"
                r"[a-z0-9-]+\.[a-z0-9.-]+)\b"),
     REMOTE_TOPIC,
     _remote_payload),
    (re.compile(r"\b(?:disconnect|close)\b.*\b(?:ssh|remote|host|computer)\b|"
                r"\bstop\s+remote\s+assist\b"),
     REMOTE_TOPIC, lambda m, t: {"command": "disconnect"}),
    (re.compile(r"\b(?:stop|pause|turn off|shut off|kill)\b.*\b(?:radio|music)\b|"
                r"\b(?:radio|music) off\b"),
     "picarx/tools/radio", lambda m, t: {"command": "stop"}),
    # Live directory search: needs a find/search word AND radio/station
    # in the utterance ("radio find soft rock", "find me a jazz station").
    (re.compile(r"\b(?:find|search)\b(?=.*\b(?:radio|station)\b)|"
                r"\b(?:radio|station)\b(?=.*\b(?:find|search)\b)"),
     "picarx/tools/radio", _find_payload),
    (re.compile(r"\bwhat(?:'s| is)?\s+playing\b|\bradio status\b"),
     "picarx/tools/radio", lambda m, t: {"command": "status"}),
    (re.compile(r"\blist\b.*\bstations?\b|\bwhat stations\b"),
     "picarx/tools/radio", lambda m, t: {"command": "list"}),
    (re.compile(r"\b(?:next|change|switch|skip|another|different)\b.*"
                r"\b(?:station|radio|song|music)\b|"
                r"\b(?:station|song)\b.*\b(?:next|skip)\b"),
     "picarx/tools/radio", lambda m, t: {"command": "next"}),
    # Tune to a frequency/dial: needs a tuning word AND a number.
    # NOTE: "to" is deliberately NOT a tuning word - "\bto\b.*\d" matched
    # any utterance shaped like "... to <number> ..." ("set a timer to 20
    # minutes", "count to 10") and hijacked it into a radio tune. Real
    # tune requests always carry one of the actual radio words below.
    (re.compile(r"\b(?:tune|station|frequency|dial|fm)\b.*\d|"
                r"\b(?:tune|station|frequency|dial|fm)\b.*\b(?:one|two|three|four|five|six|"
                r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                r"eighty|ninety|hundred|oh|zero)\b"),
     "picarx/tools/radio", _tune_payload),
    (re.compile(r"\b(?:play|start|put on)\b.*\b(?:radio|music|tunes)\b|"
                r"\b(?:radio|music) on\b"),
     "picarx/tools/radio", lambda m, t: {"command": "play"}),
    # Named station: "station <name>" (only when it's not a number).
    (re.compile(r"\bstation\s+([a-z][a-z\s]*)"),
     "picarx/tools/radio",
     lambda m, t: {"command": "play", "station": m.group(1).strip()}),
]

TOOL_DESCRIPTIONS = [
    {"name": "notes", "topic": NOTES_TOPIC,
     "say": "take a note <text> / start, pause, resume, stop meeting notes / "
            "list or delete notes",
     "description": "stores local user notes and consented meeting transcripts"},
    {"name": "reminders", "topic": REMINDER_CONTROL_TOPIC,
     "say": "remind me in <time> to <text> / list or cancel reminders",
     "description": "persists bounded local reminders and speaks them at the requested time"},
    {"name": "remote_assist", "topic": REMOTE_TOPIC,
     "say": "ssh into <host> / give or revoke remote write access / disconnect remote assist",
     "description": "connects to a provisioned host helper over verified SSH so I can inspect a project, preview approved patches, and run bounded debugging commands"},
    {"name": "radio", "topic": "picarx/tools/radio",
     "say": "play radio / stop radio / next station / station <name> / "
            "tune to <number> / radio find <genre or name> / "
            "what's playing / list stations",
     "description": "streams internet radio through my speaker; tune saved "
                    "dials, or search the live radio-browser.info directory "
                    "by keyword and cycle results with next station"},
]


class ToolsRegistry:
    def __init__(self):
        self.bus = Bus()

    def publish_available(self):
        self.bus.publish("picarx/tools/available", {
            "tools": TOOL_DESCRIPTIONS, "ts": time.time()})

    # Vocabulary that marks an utterance as a tool attempt even when no rule
    # managed to parse it. These go to the LLM intent arbiter instead of
    # vanishing. A successful repair becomes a cached alias, so new phrasing
    # does not require another hardcoded regex rule.
    _TOOL_WORDS = (
        "radio", "station", "stations", "tune", "dial", "frequency", "music",
        "song", "remind", "reminder", "note", "notes", "meeting",
        "ssh", "remote", "host",
    )
    # Compatibility name for local overlays/tests that used the old marker.
    _RADIO_WORDS = _TOOL_WORDS[:7]

    def on_heard(self, payload):
        text = (payload.get("text") or "").lower().strip()
        if not text:
            return
        # Match on the canonicalized form ("play the radio for me
        # please" -> "play radio", "play the radial" -> "play radio"),
        # but keep the raw text for logs - canonical text is lossy.
        canon = speech_match.canonicalize(text)
        if "what tools" in canon or "list tools" in canon:
            self.publish_available()
            names = ", ".join(t["say"] for t in TOOL_DESCRIPTIONS)
            self.bus.publish("picarx/audio/speak", {
                "text": f"I can do: {names}.", "ts": time.time()})
            return
        for pattern, topic, build in RULES:
            m = pattern.search(canon)
            if not m:
                continue
            command = build(m, canon)
            if command is None:
                continue  # shape matched but params didn't - try next rule
            print(f"Tools registry: '{text}' (as '{canon}') -> {topic} {command}")
            self.bus.publish(topic, command)
            self.bus.publish("picarx/decision", {
                "source": "tools_registry", "kind": "tool_invocation",
                "choice": {"topic": topic, **command},
                "reason": f"voice command matched: '{text}'", "ts": time.time()})
            return
        # No rule fired, but the utterance clearly tried to use a registered
        # tool ("set a reminder", "save this as a note", "put the music on").
        # Escalate it to the intent arbiter. Repaired output is the loop guard
        # that prevents recursive escalation.
        if payload.get("source") != "intent_repair" and \
                any(w in canon for w in self._TOOL_WORDS):
            print(f"Tools registry: unparsed tool utterance -> arbiter: '{text}'")
            self.bus.publish("picarx/audio/uncertain", {
                "text": text, "confidence": payload.get("confidence"),
                "from": "tools_registry"})

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.publish_available()
        print(f"Tools registry active ({len(TOOL_DESCRIPTIONS)} tools routable)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    ToolsRegistry().run()
