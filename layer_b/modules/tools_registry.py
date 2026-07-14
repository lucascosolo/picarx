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

# Each rule: (compiled pattern, tool topic, payload builder). First
# match wins, top to bottom - put more specific patterns first.
RULES = [
    (re.compile(r"\b(?:play|start)\b.*\bradio\b|\bradio on\b"),
     "picarx/tools/radio", lambda m: {"command": "play"}),
    (re.compile(r"\b(?:stop|pause|turn off|kill)\b.*\bradio\b|\bradio off\b"),
     "picarx/tools/radio", lambda m: {"command": "stop"}),
    (re.compile(r"\b(?:next|change|switch)\b.*\b(?:station|radio)\b"),
     "picarx/tools/radio", lambda m: {"command": "next"}),
    (re.compile(r"\bstation\s+(\w[\w\s]*)"),
     "picarx/tools/radio", lambda m: {"command": "play", "station": m.group(1).strip()}),
]

TOOL_DESCRIPTIONS = [
    {"name": "radio", "topic": "picarx/tools/radio",
     "say": "play radio / stop radio / next station / station <name>",
     "description": "streams internet radio through my speaker"},
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
            command = build(m)
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
