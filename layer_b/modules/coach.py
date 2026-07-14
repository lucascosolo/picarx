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

  picarx/coach/suggestion - {"query_id", "situation_key", "steps",
                             "rationale", "cached"}

    where "steps" is an ORDERED LIST of one or more primitive actions
    to run back-to-back:
        [{"action": {"direction","speed"/"angle"}, "duration": sec}, ...]
    A simple "just back up" answer is a 1-step list; a real maneuver
    ("reverse, then turn out, then straighten and go") is several. Each
    step still passes through the arbiter + safety daemon ONE primitive
    at a time - field_agent sequences them, the hardware layers never
    learn that sequences exist, so the safety isolation is unchanged.
    (Legacy single-action "action"/"duration" fields are also included
    for the first step so an old field_agent still works.)

Local policy cache (the "training data")
-----------------------------------------
Every situation collapses to a coarse situation_key (e.g.
"novel_object:bottle" or "collision_loop:repeated_veto"). Each key
tracks a small set of candidate maneuvers ("arms", multi-armed bandit
terminology), each an ordered STEP LIST with its own success/failure
record. UCB1 picks among tried arms; new arms come from the LLM. See
the original design notes in git history - the bandit mechanics
(exploit vs. explore, retirement of persistently-failing arms) are
unchanged, an "arm" is just a sequence now instead of one action.

Two capabilities layered on top of the exact-string cache:

  1. Richer decisions: the LLM query now includes what has already been
     tried for THIS situation (each arm's steps + win/loss record) and
     any durable facts reflection.py has learned about the world, so
     the model reasons from history + memory, not just the momentary
     snapshot.

  2. Semantic generalization (OPTIONAL, via embedding_util.Embedder):
     a brand-new situation with no arms yet is matched to the nearest
     situation the robot already has experience with (cosine similarity
     over a MiniLM embedding of the situation description). If close
     enough, that neighbor's arms are transferred as a warm-start prior
     instead of starting cold. Entirely fail-soft: with no embedding
     model installed this is skipped and behavior is identical to the
     exact-string cache. See SETUP_embeddings.md.

Every completed query is published on picarx/coach/episode, which
event_logger.py persists. The cache is persisted to disk
(COACH_POLICY_PATH) so learning survives restarts.

Requires ANTHROPIC_API_KEY to call the model; without it (or on failure)
it answers only from the cache, and if it has nothing cached it simply
stays silent - field_agent has its own bounded timeout + canned
fallback, so an unreachable coach degrades to "the robot handles it
itself," never "the robot is stuck waiting."
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus
from semantic_store import SemanticStore
from embedding_util import Embedder

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

RETIRE_MIN_FAILURES = 3        # need at least this many failures to consider retiring
RETIRE_MAX_SUCCESS_RATE = 0.2  # ...and a success rate at or below this

# Semantic transfer: when a NEW situation has no arms, borrow from the
# nearest known situation if their embeddings are at least this similar.
EMBED_SIMILARITY_THRESHOLD = 0.72

MAX_STEPS = 4                 # cap on how many actions one maneuver may chain

WORKER_THREADS = 2          # query handling runs off the MQTT callback thread

DEFAULT_SPEED = 25
DEFAULT_ANGLE = 0
DEFAULT_DURATION = 1.5
MIN_DURATION, MAX_DURATION = 0.3, 3.0
# The safety daemon hard-vetoes continuous reverse beyond 2.0s (no rear
# sensor - see MAX_CONTINUOUS_REVERSE_SEC there). Cap reverse steps below
# that so a suggested maneuver never generates its own reverse-limit
# vetoes (which would also wrongly mark the episode as failed).
MAX_BACKWARD_DURATION = 1.8
MIN_SPEED, MAX_SPEED = 10, 40
MIN_ANGLE, MAX_ANGLE = -30, 30
ALLOWED_DIRECTIONS = {"forward", "backward", "stop", "turn"}

COACH_MODEL = os.environ.get("COACH_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a driving coach for a small autonomous robot car (PiCar-X).
It has either gotten stuck repeatedly bumping something its sensors missed, or it
has spotted a kind of object it has never seen before and isn't sure how to react to.

The car uses Ackermann steering (like a real car): it can ONLY change heading while
it is moving. Turning the wheels while stopped does nothing. So a real escape is
usually a SEQUENCE, e.g. reverse a bit, then reverse-or-drive with the wheels turned
to swing the nose away, then straighten.

Reply with JSON only, no prose, no markdown fences, as an ordered list of 1 to 4
steps to perform back to back:

{
  "steps": [
    {
      "direction": "forward" | "backward" | "stop" | "turn",
      "speed": <int 10-40, for forward/backward>,
      "angle": <int -30..30, for turn, negative = left>,
      "duration": <float seconds 0.3-3.0>
    }
  ],
  "rationale": "<15 words or fewer, why this maneuver>"
}

Use a single step when a single move is genuinely enough; use several when the car
must actually reposition. You may be told what has already been tried here and
whether it worked - prefer maneuvers unlike the ones that have failed. The safety
layer vetoes anything truly unsafe, so suggest a real corrective maneuver.
"""


class Coach:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.policy = self._load_policy()
        self.semantic = SemanticStore(readonly=True)   # reflection.py's learned facts (fail-soft)
        self.embedder = Embedder()                     # optional; .available False if not set up
        # query_id -> in-flight episode bookkeeping (in-memory only).
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
            if not (isinstance(entry, dict) and "arms" in entry):
                print(f"Coach: dropping legacy-format policy entry for {key} (pre-bandit cache schema)")
                continue
            # Migrate any single-action arms ({"action","duration"}) to the
            # step-list schema ({"steps":[{"action","duration"}]}), keeping
            # all learned counts/rationale so nothing already learned is lost.
            for arm in entry["arms"].values():
                if "steps" not in arm and "action" in arm:
                    arm["steps"] = [{"action": arm.pop("action"),
                                     "duration": arm.pop("duration", DEFAULT_DURATION)}]
            policy[key] = entry
        return policy

    def _save_policy(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = f"{COACH_POLICY_PATH}.tmp"
        with self.lock:
            snapshot = json.dumps(self.policy, indent=2)
        with open(tmp_path, "w") as f:
            f.write(snapshot)
        os.replace(tmp_path, COACH_POLICY_PATH)

    # ---------- situation identity ----------

    @staticmethod
    def _situation_key(payload):
        situation = payload.get("situation", "unknown")
        if situation == "novel_object":
            return f"novel_object:{payload.get('label') or 'unknown'}"
        reason = (payload.get("extra") or {}).get("reason", "unknown")
        return f"{situation}:{reason}"

    @staticmethod
    def _situation_text(payload):
        """Natural-language description of a situation, for embedding."""
        situation = payload.get("situation", "unknown")
        if situation == "novel_object":
            return f"unfamiliar {payload.get('label') or 'object'} seen ahead while exploring"
        reason = (payload.get("extra") or {}).get("reason", "unknown")
        return f"robot stuck, collision loop, reason {reason}"

    @staticmethod
    def _sequence_signature(steps):
        norm = [{"action": s["action"], "duration": round(s["duration"], 1)} for s in steps]
        return json.dumps(norm, sort_keys=True)

    # ---------- bandit selection ----------

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
                score = float("inf")
            else:
                rate = arm["successes"] / pulls
                score = rate + UCB_C * math.sqrt(math.log(total_pulls + 1) / pulls)
            if score > best_score:
                best_score, best_sig = score, sig
        return best_sig

    def _maybe_retire_arm(self, entry, arm_sig):
        """Drop a consistently-failing arm (caller holds self.lock)."""
        arms = entry["arms"]
        arm = arms.get(arm_sig)
        if arm is None:
            return
        pulls = arm["successes"] + arm["failures"]
        if arm["failures"] < RETIRE_MIN_FAILURES:
            return
        if pulls == 0 or (arm["successes"] / pulls) > RETIRE_MAX_SUCCESS_RATE:
            return
        if len(arms) - 1 < MIN_ARMS_BEFORE_EXPLOIT:
            return
        del arms[arm_sig]
        print(f"Coach: retired persistently-failing arm ({arm['successes']}/{arm['failures']}) "
              f"- freeing a slot for a fresh candidate")

    # ---------- semantic transfer (optional) ----------

    def _ensure_embedding(self, situation_key, payload):
        """Compute + store this situation's embedding if we can and haven't."""
        if not self.embedder.available:
            return
        entry = self.policy.get(situation_key)
        if entry is None or entry.get("embedding"):
            return
        vec = self.embedder.encode(self._situation_text(payload))
        if vec:
            entry["embedding"] = vec

    def _find_similar_situation(self, situation_key, payload):
        """Nearest KNOWN situation (by embedding) that already has arms, or None."""
        if not self.embedder.available:
            return None
        query_vec = self.embedder.encode(self._situation_text(payload))
        if not query_vec:
            return None
        best_key, best_sim = None, EMBED_SIMILARITY_THRESHOLD
        for key, entry in self.policy.items():
            if key == situation_key or not entry.get("arms") or not entry.get("embedding"):
                continue
            sim = self.embedder.cosine(query_vec, entry["embedding"])
            if sim >= best_sim:
                best_key, best_sim = key, sim
        if best_key:
            print(f"Coach: '{situation_key}' looks similar to known '{best_key}' "
                  f"(sim={best_sim:.2f}) - transferring its maneuvers as a warm start")
        return best_key

    def _seed_from_neighbor(self, situation_key, neighbor_key, payload):
        """Copy a similar situation's arms into a new one as a prior."""
        neighbor = self.policy.get(neighbor_key)
        if not neighbor:
            return
        entry = self.policy.setdefault(situation_key, {"arms": {}})
        for sig, arm in neighbor["arms"].items():
            entry["arms"].setdefault(sig, {
                "steps": [dict(s) for s in arm["steps"]],
                "rationale": arm.get("rationale", "transferred from a similar situation"),
                # Keep the neighbor's record as a prior so UCB1 trusts it,
                # but it will be updated by THIS situation's own outcomes.
                "successes": arm.get("successes", 0),
                "failures": arm.get("failures", 0),
                "last_updated": time.time(),
                "transferred_from": neighbor_key,
            })
        entry["seeded_from"] = neighbor_key
        self._ensure_embedding(situation_key, payload)

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

    def _tried_before(self, situation_key):
        """Compact summary of what's already been tried here, for the prompt."""
        entry = self.policy.get(situation_key)
        if not entry or not entry.get("arms"):
            return []
        out = []
        for arm in entry["arms"].values():
            moves = ", ".join(
                s["action"].get("direction", "?") for s in arm.get("steps", []))
            out.append({
                "maneuver": moves,
                "rationale": arm.get("rationale", ""),
                "wins": arm.get("successes", 0),
                "losses": arm.get("failures", 0),
            })
        return out

    def _learned_facts(self, limit=4):
        try:
            return [f"{f['subject']}: {f['fact']}" for f in self.semantic.recent_facts(limit=limit)]
        except Exception:
            return []

    def _query_llm(self, payload, situation_key):
        client = self._get_client()
        if client is None:
            return None

        user_message = json.dumps({
            "situation": payload.get("situation"),
            "label": payload.get("label"),
            "extra": payload.get("extra"),
            "context": payload.get("context"),
            "already_tried_here": self._tried_before(situation_key),
            "things_ive_learned": self._learned_facts(),
        })

        try:
            response = client.messages.create(
                model=COACH_MODEL,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                timeout=10.0,
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            return self._parse_plan(text)
        except Exception as e:
            print(f"Coach: LLM query failed for {situation_key}: {e}")
            return None

    @staticmethod
    def _clamp_step(parsed):
        direction = parsed.get("direction")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"bad direction: {direction}")
        action = {"direction": direction}
        if direction in ("forward", "backward"):
            action["speed"] = max(MIN_SPEED, min(MAX_SPEED, int(parsed.get("speed", DEFAULT_SPEED))))
        elif direction == "turn":
            action["angle"] = max(MIN_ANGLE, min(MAX_ANGLE, int(parsed.get("angle", DEFAULT_ANGLE))))
        max_dur = MAX_BACKWARD_DURATION if direction == "backward" else MAX_DURATION
        duration = max(MIN_DURATION, min(max_dur, float(parsed.get("duration", DEFAULT_DURATION))))
        return {"action": action, "duration": duration}

    @classmethod
    def _parse_plan(cls, text):
        """Parse the LLM reply into (steps, rationale). Accepts the steps-list
        schema, a bare list of steps, or a single legacy action object."""
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)

        if isinstance(parsed, dict) and "steps" in parsed:
            raw_steps = parsed["steps"]
            rationale = str(parsed.get("rationale", ""))[:200]
        elif isinstance(parsed, list):
            raw_steps = parsed
            rationale = ""
        else:  # single legacy action object
            raw_steps = [parsed]
            rationale = str(parsed.get("rationale", ""))[:200]

        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("no steps in plan")
        steps = [cls._clamp_step(s) for s in raw_steps[:MAX_STEPS]]
        return steps, rationale

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
            # Brand-new situation with nothing learned: try to transfer
            # from a semantically similar known situation before paying
            # for an LLM call.
            if arm_sig is None and situation_key not in self.policy:
                neighbor = self._find_similar_situation(situation_key, payload)
                if neighbor:
                    self._seed_from_neighbor(situation_key, neighbor, payload)
                    arm_sig = self._select_arm(situation_key)
            chosen_arm = dict(self.policy[situation_key]["arms"][arm_sig]) if arm_sig else None

        if chosen_arm is not None:
            print(f"Coach: exploiting learned arm for {situation_key} (no LLM call)")
            self._dispatch_suggestion(
                query_id, situation_key, arm_sig, chosen_arm["steps"],
                chosen_arm.get("rationale", "known-good response"), cached=True, query_payload=payload,
            )
            return

        result = self._query_llm(payload, situation_key)
        if result is None:
            print(f"Coach: no suggestion available for {situation_key} (no key/response, no learned arm yet)")
            return

        steps, rationale = result
        arm_sig = self._sequence_signature(steps)
        now = time.time()
        with self.lock:
            entry = self.policy.setdefault(situation_key, {"arms": {}})
            arm = entry["arms"].setdefault(arm_sig, {
                "steps": steps, "rationale": rationale,
                "successes": 0, "failures": 0, "last_updated": now,
            })
            arm["rationale"] = rationale
            arm["last_updated"] = now
            self._ensure_embedding(situation_key, payload)
        self._save_policy()

        self._dispatch_suggestion(query_id, situation_key, arm_sig, steps, rationale,
                                   cached=False, query_payload=payload)

    def _dispatch_suggestion(self, query_id, situation_key, arm_sig, steps, rationale, cached, query_payload):
        with self.lock:
            self.pending_queries[query_id] = {
                "situation_key": situation_key,
                "arm_sig": arm_sig,
                "steps": steps,
                "rationale": rationale,
                "cached": cached,
                "query_payload": query_payload,
                "issued_at": time.time(),
            }
        first = steps[0]
        self.bus.publish("picarx/coach/suggestion", {
            "query_id": query_id,
            "situation_key": situation_key,
            "steps": steps,
            # Legacy fields (first step) so an old field_agent still works.
            "action": first["action"],
            "duration": first["duration"],
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
                return
            entry = self.policy.setdefault(situation_key, {"arms": {}})
            arm = entry["arms"].get(pending["arm_sig"])
            if arm is not None:
                if success:
                    arm["successes"] += 1
                else:
                    arm["failures"] += 1
                    self._maybe_retire_arm(entry, pending["arm_sig"])
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
            "steps": pending["steps"],
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

        print(f"Coach active ({len(self.policy)} learned situations, "
              f"embeddings {'on' if self.embedder.available else 'off'}), "
              f"listening on picarx/coach/query")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Coach().run()
