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
tracks a small set of candidate actions ("arms", after multi-armed
bandit terminology) it has actually tried, each with its own
success/failure record:

  - A brand new situation, or one with fewer than MIN_ARMS_BEFORE_EXPLOIT
    tried arms, always goes to the Anthropic API for a fresh suggestion,
    which becomes a new arm starting its own 0/0 record.
  - Otherwise, most of the time (1 - NEW_ARM_EXPLORE_RATE) this module
    exploits what it already knows: it scores every existing arm with a
    UCB1 bonus (success rate + an uncertainty bonus that favors
    under-tried arms) and serves the best one, with NO LLM call at all.
    A suggestion that worked before IS the training signal that
    produces better behavior later, by skipping the trip back out to
    the model once the robot already knows a good answer for this
    situation.
  - The rest of the time (NEW_ARM_EXPLORE_RATE, capped once a
    situation has MAX_ARMS_PER_SITUATION arms) it still asks the LLM
    for a fresh candidate even though it already has a working answer -
    this is deliberate: without ever exploring, the first action that
    happens to work twice gets frozen in forever, even if a better one
    exists. This is a real (if small) multi-armed bandit, not just a
    frozen lookup table.

Every completed query (chosen arm, whether it came from the cache or
a fresh LLM call, and the eventual success/failure) is also published
on picarx/coach/episode, which event_logger.py persists to the shared
events DB - so the full history of what was tried and how it went is
actually inspectable later, not just collapsed into a counter.

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
import math
import random
import time
import threading
import queue

DATA_DIR = "/home/picarx/layer_b/data"
COACH_POLICY_PATH = f"{DATA_DIR}/coach_policy.json"

MIN_ARMS_BEFORE_EXPLOIT = 2   # always ask the LLM until a situation has this many tried arms
MAX_ARMS_PER_SITUATION = 4    # stop growing new arms past this many
NEW_ARM_EXPLORE_RATE = 0.2    # even once past MIN_ARMS_BEFORE_EXPLOIT, try something new this often
UCB_C = 1.4                   # exploration bonus weight (classic UCB1 uses sqrt(2) ~= 1.41)

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
        # query_id -> in-flight episode bookkeeping (in-memory only; if
        # coach.py restarts mid-episode, that one episode's outcome is
        # simply never recorded - the persisted policy itself is safe).
        self.pending_queries = {}
        self.work_queue = queue.Queue()
        self._client = None
        self._warned_no_key = False

    # ---------- policy cache persistence ----------

    def _load_policy(self):
        try:
            with open(COACH_POLICY_PATH) as f:
                raw = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"Coach: failed to load policy cache, starting fresh: {e}")
            return {}
        policy = {}
        for key, entry in raw.items():
            if isinstance(entry, dict) and "arms" in entry:
                policy[key] = entry
            else:
                print(f"Coach: dropping legacy-format policy entry for {key} (pre-bandit cache schema)")
        return policy

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
    def _arm_signature(action, duration):
        return json.dumps({"action": action, "duration": round(duration, 1)}, sort_keys=True)

    def _select_arm(self, situation_key):
        """Returns an arm signature to exploit, or None to signal 'ask the LLM'."""
        entry = self.policy.get(situation_key)
        arms = entry["arms"] if entry else {}
        if not arms or len(arms) < MIN_ARMS_BEFORE_EXPLOIT:
            return None
        if len(arms) < MAX_ARMS_PER_SITUATION and random.random() < NEW_ARM_EXPLORE_RATE:
            return None

        total_pulls = sum(a["successes"] + a["failures"] for a in arms.values())
        best_sig, best_score = None, -1.0
        for sig, arm in arms.items():
            pulls = arm["successes"] + arm["failures"]
            if pulls == 0:
                score = float("inf")  # never-resolved arm - try it before trusting anything else
            else:
                rate = arm["successes"] / pulls
                score = rate + UCB_C * math.sqrt(math.log(total_pulls + 1) / pulls)
            if score > best_score:
                best_score, best_sig = score, sig
        return best_sig

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
            arm_sig = self._select_arm(situation_key)
            chosen_arm = dict(self.policy[situation_key]["arms"][arm_sig]) if arm_sig else None

        if chosen_arm is not None:
            print(f"Coach: exploiting learned arm for {situation_key} (no LLM call)")
            self._dispatch_suggestion(
                query_id, situation_key, arm_sig, chosen_arm["action"], chosen_arm["duration"],
                chosen_arm.get("rationale", "known-good response"), cached=True, query_payload=payload,
            )
            return

        result = self._query_llm(payload, situation_key)
        if result is None:
            print(f"Coach: no suggestion available for {situation_key} (no key/response, no learned arm yet)")
            return

        action, duration, rationale = result
        arm_sig = self._arm_signature(action, duration)
        now = time.time()
        with self.lock:
            entry = self.policy.setdefault(situation_key, {"arms": {}})
            arm = entry["arms"].setdefault(arm_sig, {
                "action": action, "duration": duration, "rationale": rationale,
                "successes": 0, "failures": 0, "last_updated": now,
            })
            arm["rationale"] = rationale
            arm["last_updated"] = now
        self._save_policy()

        self._dispatch_suggestion(query_id, situation_key, arm_sig, action, duration, rationale,
                                   cached=False, query_payload=payload)

    def _dispatch_suggestion(self, query_id, situation_key, arm_sig, action, duration, rationale, cached, query_payload):
        with self.lock:
            self.pending_queries[query_id] = {
                "situation_key": situation_key,
                "arm_sig": arm_sig,
                "action": action,
                "duration": duration,
                "rationale": rationale,
                "cached": cached,
                "query_payload": query_payload,
                "issued_at": time.time(),
            }
        self.bus.publish("picarx/coach/suggestion", {
            "query_id": query_id,
            "situation_key": situation_key,
            "action": action,
            "duration": duration,
            "rationale": rationale,
            "cached": cached,
        })

    # ---------- inbound: outcomes ----------

    def on_outcome(self, payload):
        query_id = payload.get("query_id")
        situation_key = payload.get("situation_key")
        success = bool(payload.get("success"))

        with self.lock:
            pending = self.pending_queries.pop(query_id, None)
            if pending is None or pending["situation_key"] != situation_key:
                return  # stale/unknown episode (e.g. coach restarted mid-episode) - nothing to update
            entry = self.policy.setdefault(situation_key, {"arms": {}})
            arm = entry["arms"].get(pending["arm_sig"])
            if arm is not None:
                if success:
                    arm["successes"] += 1
                else:
                    arm["failures"] += 1
                arm["last_updated"] = time.time()
        self._save_policy()
        print(f"Coach: recorded {'success' if success else 'failure'} for {situation_key}")

        query_payload = pending["query_payload"]
        self.bus.publish("picarx/coach/episode", {
            "query_id": query_id,
            "situation_key": situation_key,
            "situation": query_payload.get("situation"),
            "label": query_payload.get("label"),
            "context": query_payload.get("context"),
            "action": pending["action"],
            "duration": pending["duration"],
            "rationale": pending["rationale"],
            "cached": pending["cached"],
            "success": success,
            "issued_at": pending["issued_at"],
            "finished_at": time.time(),
        })

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
