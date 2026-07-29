#!/usr/bin/env python3
# layer_b/modules/tools_registry.py
"""
Tools Registry (Layer B) - the bus front door for the capability router.

A "tool" is anything fun or useful that is NOT part of the drive/
explore/learn pipeline: radio, reminders, notes, remote assistance,
future games and party tricks. Each tool is its own module listening on
its own picarx/tools/<name> topic.

What each tool answers to now lives in ONE place, `layer_b/capabilities.py`:
its phrases, its vocabulary, its topic, and its self-description. This
module is deliberately thin - it subscribes to heard speech, asks the
router who owns the utterance, and publishes the result. Adding a tool
means declaring a capability, not editing routing logic in two or three
modules and keeping hand-copied keyword lists in sync.

Routing contract with field_agent: field_agent asks the same router
whether an utterance is already tool-owned and stays out of the way if
it is, so "stop radio" reaches the radio and never trips the robot-wide
"stop". Movement words are deliberately NOT routable as capabilities -
safety-relevant commands stay in field_agent's fast local path.

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
import capabilities
import speech_match

import time

# Topics and the dial parser stay importable from here: they were part of this
# module's public surface before the capability registry existed.
REMOTE_TOPIC = capabilities.REMOTE_TOPIC
REMINDER_SET_TOPIC = capabilities.REMINDER_SET_TOPIC
REMINDER_CONTROL_TOPIC = capabilities.REMINDER_CONTROL_TOPIC
NOTES_TOPIC = capabilities.NOTES_TOPIC
parse_dial = capabilities.parse_dial


class ToolsRegistry:
    def __init__(self):
        self.bus = Bus()
        self.router = capabilities.ROUTER

    def publish_available(self):
        self.bus.publish("picarx/tools/available", {
            "tools": self.router.describe(), "ts": time.time()})

    def on_heard(self, payload):
        text = (payload.get("text") or "").lower().strip()
        if not text:
            return
        # Route on the canonicalized form ("play the radio for me
        # please" -> "play radio", "play the radial" -> "play radio"),
        # but keep the raw text for logs - canonical text is lossy.
        canon = speech_match.canonicalize(text)
        if "what tools" in canon or "list tools" in canon:
            self.publish_available()
            names = ", ".join(t["say"] for t in self.router.describe())
            self.bus.publish("picarx/audio/speak", {
                "text": f"I can do: {names}.", "ts": time.time()})
            return

        decision = self.router.route(canon)
        if decision.matched:
            print(f"Tools registry: '{text}' (as '{canon}') -> "
                  f"{decision.topic} {decision.payload}")
            self.bus.publish(decision.topic, decision.payload)
            self.bus.publish("picarx/decision", {
                "source": "tools_registry", "kind": "tool_invocation",
                "choice": {"topic": decision.topic, **decision.payload},
                "capability": decision.capability.name,
                "reason": f"voice command matched: '{text}'", "ts": time.time()})
            return

        # No rule fired, but the utterance clearly tried to use a registered
        # capability ("set a reminder", "save this as a note", "put the music
        # on"). Escalate it to the intent arbiter. Repaired output is the loop
        # guard that prevents recursive escalation.
        if decision.claimed and payload.get("source") != "intent_repair":
            print(f"Tools registry: unparsed {decision.capability.name} "
                  f"utterance -> arbiter: '{text}'")
            self.bus.publish("picarx/audio/uncertain", {
                "text": text, "confidence": payload.get("confidence"),
                "capability": decision.capability.name,
                "from": "tools_registry"})

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.publish_available()
        print(f"Tools registry active "
              f"({len(self.router.capabilities)} capabilities routable)")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    ToolsRegistry().run()
