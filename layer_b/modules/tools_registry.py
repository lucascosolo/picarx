#!/usr/bin/env python3
# /home/picarx/layer_b/modules/tools_registry.py
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
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import re
import time

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
    total, current = 0, 0
    for t in tokens:
        if t == "hundred":
            current = (current or 1) * 100
        else:
            current += _NUMWORD[t]
    return str(total + current)


def parse_dial(text):
    """Return a canonical dial string ('98.7') from a spoken/typed
    frequency, or None if the text doesn't clearly contain one."""
    text = text.lower()
    # Digits already present: "98.7", "98 7", "1025", "987".
    m = re.search(r"\b(\d{2,3})(?:[.\s](\d))?\b", text)
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


RULES = [
    (re.compile(r"\b(?:stop|pause|turn off|kill)\b.*\bradio\b|\bradio off\b"),
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
    (re.compile(r"\b(?:next|change|switch)\b.*\b(?:station|radio)\b"),
     "picarx/tools/radio", lambda m, t: {"command": "next"}),
    # Tune to a frequency/dial: needs a tuning word AND a number.
    (re.compile(r"\b(?:tune|station|frequency|dial|fm|to)\b.*\d|"
                r"\b(?:tune|station|frequency|dial|fm)\b.*\b(?:one|two|three|four|five|six|"
                r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                r"eighty|ninety|hundred|oh|zero)\b"),
     "picarx/tools/radio", _tune_payload),
    (re.compile(r"\b(?:play|start)\b.*\bradio\b|\bradio on\b"),
     "picarx/tools/radio", lambda m, t: {"command": "play"}),
    # Named station: "station <name>" (only when it's not a number).
    (re.compile(r"\bstation\s+([a-z][a-z\s]*)"),
     "picarx/tools/radio",
     lambda m, t: {"command": "play", "station": m.group(1).strip()}),
]

TOOL_DESCRIPTIONS = [
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

    def on_heard(self, payload):
        text = (payload.get("text") or "").lower().strip()
        if not text:
            return
        if "what tools" in text or "list tools" in text:
            self.publish_available()
            names = ", ".join(t["say"] for t in TOOL_DESCRIPTIONS)
            self.bus.publish("picarx/audio/speak", {
                "text": f"I can do: {names}.", "ts": time.time()})
            return
        for pattern, topic, build in RULES:
            m = pattern.search(text)
            if not m:
                continue
            command = build(m, text)
            if command is None:
                continue  # shape matched but params didn't - try next rule
            print(f"Tools registry: '{text}' -> {topic} {command}")
            self.bus.publish(topic, command)
            self.bus.publish("picarx/decision", {
                "source": "tools_registry", "kind": "tool_invocation",
                "choice": {"topic": topic, **command},
                "reason": f"voice command matched: '{text}'", "ts": time.time()})
            return

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.publish_available()
        print(f"Tools registry active ({len(TOOL_DESCRIPTIONS)} tools routable)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    ToolsRegistry().run()
