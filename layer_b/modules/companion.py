#!/usr/bin/env python3
# /home/picarx/layer_b/modules/companion.py
"""
Companion (Layer B) - natural conversation fallback.

field_agent.py handles a small fixed vocabulary of hard commands
("explore", "stop", "status", "objects", "history", "battery",
"hello") entirely locally, with zero network dependency, because
"stop" in particular must never wait on an LLM round-trip. Anything
that doesn't match one of those gets published to
picarx/audio/unhandled instead of silently doing nothing - that's
this module's entire job: turn it into a natural spoken reply.

This module never controls the robot. It cannot publish movement
intents at all - if someone asks it to drive somewhere in
conversation, its system prompt tells it to point them at the actual
command words instead of trying to comply itself. That split (fast
local safety-relevant commands vs. this slower, LLM-backed chat
layer) is deliberate and should not be blurred.

Each reply is grounded with a short snapshot of picarx/state/world
(face/objects/distance/battery) folded into the prompt, so it can
answer naturally ("are you doing okay?", "what's that thing you're
looking at?") without needing its own sensor access. Conversation
history is kept in memory only (a rolling window) - it resets on
restart, which is an intentional simplicity trade-off for now.

Requires ANTHROPIC_API_KEY in the environment, same as coach.py. If
it's missing, or a request fails/times out, this module just replies
with a short apology instead of raising - a quiet, unhelpful companion
is fine; a crashed process that stops handling any future messages is
not.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import threading
import queue
import time
from collections import deque

HISTORY_TURNS = 12          # user+assistant messages kept for context
WORKER_THREADS = 2
REPLY_TIMEOUT = 8.0
REPLY_MAX_TOKENS = 150

COMPANION_MODEL = os.environ.get("COMPANION_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are the voice and personality of a small autonomous robot car (PiCar-X).
You are friendly, a little playful, and curious about the world you're rolling around in.

You are talking out loud through a text-to-speech engine, so keep every reply SHORT -
one or two sentences, plain spoken English, no markdown, no lists, no emoji.

Each message you receive starts with a bracketed snapshot of your current sensors, like
"[current status: sees a face; tracking: chair, bottle; nearest obstacle ~40cm away;
battery 7.4V]", followed by what the person actually said. Use that snapshot naturally
when it's relevant to the conversation, but don't recite it like a status report unless
asked directly what you see/sense.

You do NOT control your own motors from this conversation - a separate, instant,
safety-critical command system handles "explore", "stop", "status", "objects",
"history", and "battery". If someone asks you to move, stop, explore, or asks a
question one of those commands already answers, tell them briefly to just say that
word directly instead of trying to comply here yourself.
"""


class Companion:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_TURNS)
        self.latest_world = None
        self.work_queue = queue.Queue()
        self._client = None
        self._warned_no_key = False

    # ---------- inbound ----------

    def on_world_state(self, payload):
        with self.lock:
            self.latest_world = payload

    def on_unhandled(self, payload):
        text = (payload.get("text") or "").strip()
        if text:
            self.work_queue.put(text)

    # ---------- Anthropic call ----------

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if not self._warned_no_key:
                print("Companion: ANTHROPIC_API_KEY not set - can't chat, will apologize instead.")
                self._warned_no_key = True
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Companion: 'anthropic' package not installed - pip install anthropic to enable chat.")
        return self._client

    def _context_blurb(self):
        with self.lock:
            snap = dict(self.latest_world) if self.latest_world else None
        if not snap:
            return "no sensor data yet"

        parts = []
        face = snap.get("face", {})
        parts.append("sees a face" if face.get("detected") and not face.get("stale", True) else "doesn't currently see a face")

        objects = snap.get("objects", {})
        if not objects.get("stale", True) and objects.get("items"):
            labels = [o.get("label", "something") for o in objects["items"]]
            parts.append(f"tracking: {', '.join(labels)}")

        distance = snap.get("distance_cm")
        if distance is not None and not snap.get("distance_stale", True):
            parts.append(f"nearest obstacle ~{distance:.0f}cm away")

        battery = snap.get("battery", {})
        if battery.get("voltage") is not None:
            low_note = " (low)" if battery.get("low") else ""
            parts.append(f"battery {battery['voltage']:.1f}V{low_note}")

        return "; ".join(parts)

    def _handle_utterance(self, text):
        client = self._get_client()
        if client is None:
            self.bus.publish("picarx/audio/speak", {"text": "Sorry, I can't chat right now."})
            return

        with self.lock:
            messages = list(self.history)
        messages = messages + [{"role": "user", "content": f"[current status: {self._context_blurb()}]\n{text}"}]

        try:
            response = client.messages.create(
                model=COMPANION_MODEL,
                max_tokens=REPLY_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                timeout=REPLY_TIMEOUT,
            )
            reply = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
        except Exception as e:
            print(f"Companion: chat failed: {e}")
            reply = "Sorry, I got a little confused there."

        if not reply:
            return

        with self.lock:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})

        print(f"Companion says: {reply}")
        self.bus.publish("picarx/audio/speak", {"text": reply})

    # ---------- worker pool ----------

    def _worker_loop(self):
        while True:
            text = self.work_queue.get()
            try:
                self._handle_utterance(text)
            except Exception as e:
                print(f"Companion: error handling utterance: {e}")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/unhandled", self.on_unhandled)
        self.bus.subscribe("picarx/state/world", self.on_world_state)

        for _ in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, daemon=True).start()

        print("Companion active, listening on picarx/audio/unhandled")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Companion().run()
