#!/usr/bin/env python3
# /home/picarx/layer_b/modules/coach.py
"""
LLM Coach (Layer B) - advises the onboard AI on novel situations and
fail states, and remembers what actually worked.

field_agent.py is the only thing that talks to this module, over two
topics:

  picarx/coach/query      - {"query_id", "source", "situation"
                             ("novel_object" | "collision_loop"),
                             "label", "urgent", "context": {...},
                             "extra": {...}, "requested_at"}
  picarx/coach/outcome    - {"query_id", "situation_key", "source",
                             "success"} - reported back once
                             field_agent has actually tried whatever
                             this module suggested.

This module replies on:

  picarx/coach/suggestion - {"query_id", "situation_key", "action",
                             "duration", "rationale", "cached"}

Local policy cache (the "training data")
-----------------------------------------
Every situation collapses to a coarse situation_key (e.g.
"novel_object:bottle" or "collision_loop:repeated_veto"). Each key
tracks an action plus how many times it's succeeded/failed when
actually tried:

  - A confident cache hit (enough successes, high enough success
    rate - see MIN_SUCCESSES/SUCCESS_RATE_THRESHOLD) is served
    immediately, with no LLM call at all - a suggestion that worked
    before IS the training signal that produces better behavior later,
    exactly by skipping the trip back out to the model once the robot
    already knows what to do.
  - Anything else (novel situation, or a cached action that hasn't
    proven itself yet) goes to the Anthropic API for a fresh
    suggestion, which becomes a new/updated (unproven) cache entry
    that starts earning its own success/failure record from here on.

The cache is persisted to disk (COACH_POLICY_PATH) so it survives
restarts - restarting the robot should not erase what it already
learned.

Requires ANTHROPIC_API_KEY in the environment to call out to the
model. If it's missing (or the API call fails/times out), and there's
no confident cache entry either, this module simply does not publish
a suggestion for that query - field_agent has its own bounded timeout
and canned evasion fallback, so a coach that's unreachable degrades to
"the robot handles it itself," never to "the robot is stuck waiting."
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import json
import time
import threading
import queue

DATA_DIR = "/home/picarx/layer_b/data"
COACH_POLICY_PATH = f"{DATA_DIR}/coach_policy.json"

MIN_SUCCESSES = 2          # need at least this many confirmed successes...
SUCCESS_RATE_THRESHOLD = 0.66   # ...and this success rate to trust the cache blindly

WORKER_THREADS = 2          # query handling runs off the MQTT callback thread

DEFAULT_SPEED = 25
DEFAULT_ANGLE = 0
DEFAULT_DURATION = 1.5
MIN_DURATION, MAX_DURATION = 0.3, 3.0
MIN_SPEED, MAX_SPEED = 10, 40
MIN_ANGLE, MAX_ANGLE = -30, 30
ALLOWED_DIRECTIONS = {"forward", "backward", "stop", "turn"}

COACH_MODEL = os.environ.get("COACH_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a driving coach for a small autonomous robot car (PiCar-X).
It has either gotten stuck repeatedly bumping something its sensors missed, or it
has spotted a kind of object it has never seen before and isn't sure how to react to.

Suggest exactly ONE short corrective action. Reply with JSON only, no prose, no
markdown fences, matching this schema exactly:

{
  "direction": "forward" | "backward" | "stop" | "turn",
  "speed": <int, 10-40, only meaningful for forward/backward>,
  "angle": <int, -30 to 30, only meaningful for turn, negative is left>,
  "duration": <float seconds, 0.3-3.0, how long to hold this action>,
  "rationale": <string, 15 words or fewer>
}

The robot's own safety layer will veto anything that's actually unsafe (too
close to an obstacle, a cliff underfoot), so you don't need to be overly
conservative - suggest a real corrective maneuver, not just another stop.
"""


class Coach:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.policy = self._load_policy()
        self.work_queue = queue.Queue()
        self._client = None
        self._warned_no_key = False

    # ---------- policy cache persistence ----------

    def _load_policy(self):
        try:
            with open(COACH_POLICY_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"Coach: failed to load policy cache, starting fresh: {e}")
            return {}

    def _save_policy(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = f"{COACH_POLICY_PATH}.tmp"
        with self.lock:
            snapshot = json.dumps(self.policy, indent=2)
        with open(tmp_path, "w") as f:
            f.write(snapshot)
        os.replace(tmp_path, COACH_POLICY_PATH)

    @staticmethod
    def _situation_key(payload):
        situation = payload.get("situation", "unknown")
        if situation == "novel_object":
            return f"novel_object:{payload.get('label') or 'unknown'}"
        reason = (payload.get("extra") or {}).get("reason", "unknown")
        return f"{situation}:{reason}"

    @staticmethod
    def _success_rate(entry):
        attempts = entry["successes"] + entry["failures"]
        return entry["successes"] / attempts if attempts else 0.0

    # ---------- Anthropic call ----------

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if not self._warned_no_key:
                print("Coach: ANTHROPIC_API_KEY not set - will only ever answer from the local policy cache.")
                self._warned_no_key = True
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Coach: 'anthropic' package not installed - pip install anthropic to enable live coaching.")
            self._client = None
        return self._client

    def _query_llm(self, payload, situation_key):
        client = self._get_client()
        if client is None:
            return None

        user_message = json.dumps({
            "situation": payload.get("situation"),
            "label": payload.get("label"),
            "extra": payload.get("extra"),
            "context": payload.get("context"),
        })

        try:
            response = client.messages.create(
                model=COACH_MODEL,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                timeout=8.0,
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            return self._parse_action(text)
        except Exception as e:
            print(f"Coach: LLM query failed for {situation_key}: {e}")
            return None

    @staticmethod
    def _parse_action(text):
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)

        direction = parsed.get("direction")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"bad direction: {direction}")

        action = {"direction": direction}
        if direction in ("forward", "backward"):
            speed = parsed.get("speed", DEFAULT_SPEED)
            action["speed"] = max(MIN_SPEED, min(MAX_SPEED, int(speed)))
        elif direction == "turn":
            angle = parsed.get("angle", DEFAULT_ANGLE)
            action["angle"] = max(MIN_ANGLE, min(MAX_ANGLE, int(angle)))

        duration = parsed.get("duration", DEFAULT_DURATION)
        duration = max(MIN_DURATION, min(MAX_DURATION, float(duration)))

        rationale = str(parsed.get("rationale", ""))[:200]
        return action, duration, rationale

    # ---------- inbound: queries ----------

    def on_query(self, payload):
        self.work_queue.put(payload)

    def _handle_query(self, payload):
        query_id = payload.get("query_id")
        if not query_id:
            return
        situation_key = self._situation_key(payload)

        with self.lock:
            entry = self.policy.get(situation_key)
            use_cached = (
                entry is not None
                and entry["successes"] >= MIN_SUCCESSES
                and self._success_rate(entry) >= SUCCESS_RATE_THRESHOLD
            )
            cached_action = dict(entry) if (use_cached and entry) else None

        if cached_action:
            print(f"Coach: serving cached policy for {situation_key} (no LLM call)")
            self.bus.publish("picarx/coach/suggestion", {
                "query_id": query_id,
                "situation_key": situation_key,
                "action": cached_action["action"],
                "duration": cached_action["duration"],
                "rationale": cached_action.get("last_rationale", "known-good response"),
                "cached": True,
            })
            return

        result = self._query_llm(payload, situation_key)
        if result is None:
            print(f"Coach: no suggestion available for {situation_key} (no key/cache/response)")
            return

        action, duration, rationale = result
        now = time.time()
        with self.lock:
            existing = self.policy.get(situation_key, {"successes": 0, "failures": 0})
            self.policy[situation_key] = {
                "action": action,
                "duration": duration,
                "last_rationale": rationale,
                "successes": existing["successes"],
                "failures": existing["failures"],
                "last_updated": now,
            }
        self._save_policy()

        self.bus.publish("picarx/coach/suggestion", {
            "query_id": query_id,
            "situation_key": situation_key,
            "action": action,
            "duration": duration,
            "rationale": rationale,
            "cached": False,
        })

    # ---------- inbound: outcomes ----------

    def on_outcome(self, payload):
        situation_key = payload.get("situation_key")
        if not situation_key:
            return
        with self.lock:
            entry = self.policy.get(situation_key)
            if entry is None:
                return
            if payload.get("success"):
                entry["successes"] += 1
            else:
                entry["failures"] += 1
            entry["last_updated"] = time.time()
        print(f"Coach: recorded {'success' if payload.get('success') else 'failure'} for {situation_key}")
        self._save_policy()

    # ---------- worker pool ----------

    def _worker_loop(self):
        while True:
            payload = self.work_queue.get()
            try:
                self._handle_query(payload)
            except Exception as e:
                print(f"Coach: error handling query: {e}")

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/coach/query", self.on_query)
        self.bus.subscribe("picarx/coach/outcome", self.on_outcome)

        for _ in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, daemon=True).start()

        print(f"Coach active ({len(self.policy)} learned situations), listening on picarx/coach/query")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Coach().run()
