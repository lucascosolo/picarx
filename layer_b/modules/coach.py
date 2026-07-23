#!/usr/bin/env python3
# layer_b/modules/coach.py
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config
from semantic_store import SemanticStore
from embedding_util import Embedder
# combine_policy is a PURE function (no bus/hardware); reusing it keeps the
# online adopt path and the offline import_training tool folding packs in
# identically. See its two modes ("merge" sums, "adopt" replaces shared arms).
from import_training import combine_policy

import copy
import json
import math
import random
import time
import threading
import queue

DATA_DIR = robot_config.data_path()
COACH_POLICY_PATH = f"{DATA_DIR}/coach_policy.json"

# A/B experiment (experiment.py): a "control" session holds out the sim-trained
# arms so the round-trip can be measured against the pre-adoption baseline. The
# chosen condition is published so behavior_metrics.py can tag the session.
import experiment  # noqa: E402  (local module; kept with the other coach deps)
EXPERIMENT_STATE_PATH = f"{DATA_DIR}/experiment_state.json"
EXPERIMENT_TOPIC = "picarx/experiment/condition"
EXPERIMENT_ENABLED = robot_config.get_bool(
    "experiment", "enabled", True, env="EXPERIMENT_ENABLED")

MIN_ARMS_BEFORE_EXPLOIT = 2   # always ask the LLM until a situation has this many tried arms
MAX_ARMS_PER_SITUATION = 6    # stop growing new arms past this many (room for richer,
                              # multi-step repositioning tactics, not just a couple reverses)
NEW_ARM_EXPLORE_RATE = 0.2    # even once past MIN_ARMS_BEFORE_EXPLOIT, try something new this often
UCB_C = 1.4                   # exploration bonus weight (classic UCB1 uses sqrt(2) ~= 1.41)

RETIRE_MIN_FAILURES = 3        # need at least this many failures to consider retiring
RETIRE_MAX_SUCCESS_RATE = 0.2  # ...and a success rate at or below this

# Emergent-behavior knobs. NOVELTY_RATE: fraction of queries where,
# instead of exploiting a learned arm, the LLM is explicitly asked for
# a maneuver UNLIKE anything tried before (tagged experimental, spoken
# as an experiment, tracked like any arm - the safety layer bounds it
# exactly like everything else). Kept low per the roadmap so curiosity
# never crowds out reliability. One experimental slot may exceed
# MAX_ARMS_PER_SITUATION; growth stays bounded and retirement culls.
NOVELTY_RATE = 0.15
# Surprise detection: an arm this proven failing (or this disproven
# succeeding) is worth an event of its own for reflection to chew on.
SURPRISE_MIN_PULLS = 4
SURPRISE_HIGH_RATE = 0.7
SURPRISE_LOW_RATE = 0.2

# Semantic transfer: when a NEW situation has no arms, borrow from the
# nearest known situation if their embeddings are at least this similar.
# Per situation type: novelty reactions are low-stakes so they transfer
# eagerly; collision escapes are safety-adjacent so they transfer only
# on a much closer match.
EMBED_SIMILARITY_THRESHOLD = 0.72
EMBED_THRESHOLD_BY_SITUATION = {"novel_object": 0.70, "collision_loop": 0.78}

MAX_STEPS = 4                 # cap on how many actions one maneuver may chain

# Human demonstrations (picarx/rc/demonstration, from RC mode): the last
# few are persisted inside the policy file under a reserved "_" key and
# included in every LLM coaching query, so the model can IMITATE how a
# human actually drove out of a similar spot - its imitation becomes an
# ordinary arm the bandit then validates like anything else. Steps are
# clamped through the same bounds as the coach's own arms, and vetoed
# commands are never taught from.
DEMONSTRATIONS_KEY = "_demonstrations"
MAX_DEMONSTRATIONS = 10
DEMONSTRATIONS_IN_PROMPT = 4

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

COACH_MODEL = str(robot_config.get("coach", "model", "claude-haiku-4-5-20251001",
                                   env="COACH_MODEL"))

SYSTEM_PROMPT = """You are a driving coach for a small autonomous robot car (PiCar-X).
It has either gotten stuck repeatedly bumping something its sensors missed, or it
has spotted a kind of object it has never seen before and isn't sure how to react to.

The car uses Ackermann steering (like a real car): it can ONLY change heading while
it is moving. Turning the wheels while stopped does nothing. So a real escape is
usually a SEQUENCE, e.g. reverse a bit, then reverse-or-drive with the wheels turned
to swing the nose away, then straighten.

Aim for a maneuver that leaves the car POINTED AT OPEN SPACE so it can drive away
cleanly afterwards - not one that merely nudges back from the obstacle and leaves it
facing the same wall (that just gets stuck again). Think like a driver getting out of
a tight parking spot: reverse while turning to swing the nose around, or do a
multi-point turn (reverse-turn, pull-forward-turn, reverse-turn) to actually TURN
AROUND, then leave. Reversing STRAIGHT back is the weakest option - use it only for a
short unstick, and prefer a turning reverse or a full turn-around.

If already_tried_here shows a plain reverse (or reversing straight) has been tried and
is FAILING here, do NOT just propose reversing again harder or longer - switch tactic:
turn around and reverse INTO the open area, or arc out to one side. Escalate from
simple to repositioning maneuvers as simple ones prove they don't clear this spot.

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
whether it worked - prefer maneuvers unlike the ones that have failed. You may also
receive "learned_patterns" (statistics mined from the car's own history) and the
failure mode that triggered this (obstacle / cliff / reverse_limit) - a cliff needs
a different escape than an unseen obstacle, so weigh both.

"human_demonstrations" are maneuvers a HUMAN drove with the remote control to get
out of situations like these, with real step durations and where the obstacles sat
(side l/c/r). When one matches the current situation - especially a resolved one -
strongly prefer imitating its shape and timing over inventing something new.

The safety layer vetoes anything truly unsafe, so suggest a real corrective maneuver.
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
        # A/B experiment condition for this session (see experiment.py). Default
        # "adopt" (trained arms in play); run() may flip it to "control".
        self.experiment_condition = experiment.ADOPT
        self.control = False

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
            if key.startswith("_"):
                policy[key] = entry   # reserved sections (e.g. _demonstrations)
                continue
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
        extra = payload.get("extra") or {}
        reason = extra.get("reason", "unknown")
        key = f"{situation}:{reason}"
        # Failure-mode-specific recovery: a cliff veto and an unseen-
        # obstacle veto need different escapes, so they learn under
        # different keys. Older keys without the suffix stay valid (and
        # seed the new ones through embedding transfer).
        failure_mode = extra.get("failure_mode")
        if failure_mode:
            key = f"{key}:{failure_mode}"
        return key

    @staticmethod
    def _situation_text(payload):
        """Natural-language description of a situation, for embedding.
        Multi-modal on purpose: co-present objects, open/tight space,
        battery state and place all shape which past experience is
        genuinely 'similar' - a sofa in a tight corner on low battery
        is closer to a couch in a tight corner than to a sofa in open
        floor, whatever the labels say."""
        ctx = payload.get("context") or {}
        bits = []
        labels = sorted({o.get("label") for o in (ctx.get("objects") or []) if o.get("label")})
        if labels:
            bits.append(f"with {', '.join(labels[:5])} visible")
        distance = ctx.get("distance_cm")
        if distance is not None and not ctx.get("distance_stale", True):
            if distance < 50:
                bits.append("in a tight space")
            elif distance > 150:
                bits.append("in open space")
        if (ctx.get("battery") or {}).get("low"):
            bits.append("on low battery")
        place = (ctx.get("location") or {}).get("label")
        if place:
            bits.append(f"at {place}")
        suffix = (" " + " ".join(bits)) if bits else ""

        situation = payload.get("situation", "unknown")
        if situation == "novel_object":
            return f"unfamiliar {payload.get('label') or 'object'} seen ahead while exploring{suffix}"
        extra = payload.get("extra") or {}
        reason = extra.get("reason", "unknown")
        mode = extra.get("failure_mode")
        mode_txt = f" after a {mode} veto" if mode else ""
        return f"robot stuck, collision loop, reason {reason}{mode_txt}{suffix}"

    @staticmethod
    def _sequence_signature(steps):
        norm = [{"action": s["action"], "duration": round(s["duration"], 1)} for s in steps]
        return json.dumps(norm, sort_keys=True)

    # ---------- bandit selection ----------

    def _select_arm(self, situation_key):
        """Returns an arm signature to exploit, or None to signal 'ask the LLM'."""
        entry = self.policy.get(situation_key)
        arms = entry["arms"] if entry else {}
        # A/B control session: hold out the sim-trained arms so this session runs
        # the pre-adoption baseline. The arms stay STORED (on_adopt still folds
        # them) - they're just not selectable here - so adopt sessions and the
        # offline metrics can compare against them fairly.
        if self.control and arms:
            arms = {sig: a for sig, a in arms.items() if not a.get("trained_in_sim")}
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
        # Reality-gap guard: an arm carrying simulator-derived counts is never
        # auto-retired. Self-training may ADD or REFRESH an arm, never delete a
        # real one - a purely-sim record must not retire a maneuver the robot
        # learned on the carpet. Its poor record simply stops UCB1 from picking
        # it; only arms whose failures are entirely real get culled here.
        if arm.get("trained_in_sim"):
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
        threshold = EMBED_THRESHOLD_BY_SITUATION.get(
            payload.get("situation"), EMBED_SIMILARITY_THRESHOLD)
        best_key, best_sim = None, threshold
        for key, entry in self.policy.items():
            if (key == situation_key or not isinstance(entry, dict)
                    or not entry.get("arms") or not entry.get("embedding")):
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

    def _learned_patterns(self, limit=4):
        """Mined event-sequence patterns (reflection's offline analysis),
        fail-soft like facts. Only fresh, high-confidence ones surface."""
        try:
            return [f"{p['condition']}: {p['outcome']} "
                    f"({p['confidence']:.0%} over {p['frequency']} episodes)"
                    for p in self.semantic.top_patterns(limit=limit)]
        except Exception:
            return []

    def _query_llm(self, payload, situation_key, experiment=False):
        client = self._get_client()
        if client is None:
            return None

        message = {
            "situation": payload.get("situation"),
            "label": payload.get("label"),
            "extra": payload.get("extra"),
            "context": payload.get("context"),
            "already_tried_here": self._tried_before(situation_key),
            "things_ive_learned": self._learned_facts(),
            "learned_patterns": self._learned_patterns(),
            "human_demonstrations": self._recent_demonstrations(),
        }
        if experiment:
            message["experiment_request"] = (
                "This is an EXPERIMENT round: propose a maneuver clearly different "
                "from everything in already_tried_here - something plausible that "
                "has never been attempted in this situation.")
        user_message = json.dumps(message)

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

    # ---------- inbound: human demonstrations ----------

    def on_demonstration(self, payload):
        """A human drove out of an obstacle situation in RC mode. Keep
        the maneuver (clamped to the same safe bounds as any arm; vetoed
        commands excluded) so future LLM queries can imitate it."""
        steps = []
        for s in (payload.get("actions") or [])[:MAX_STEPS]:
            if s.get("status") not in (None, "executed"):
                continue   # the safety daemon refused it - not a lesson
            try:
                steps.append(self._clamp_step(
                    {**(s.get("action") or {}), "duration": s.get("duration")}))
            except (ValueError, TypeError):
                continue
        if not steps:
            return
        demo = {
            "situation": payload.get("situation"),
            "context": payload.get("context"),
            "steps": steps,
            "resolved": bool(payload.get("resolved")),
            "ts": payload.get("ts") or time.time(),
        }
        with self.lock:
            demos = self.policy.setdefault(DEMONSTRATIONS_KEY, [])
            demos.append(demo)
            del demos[:-MAX_DEMONSTRATIONS]
        self._save_policy()
        moves = ", ".join(f"{s['action'].get('direction')} {s['duration']:.1f}s"
                          for s in steps)
        print(f"Coach: stored human demonstration ({payload.get('situation')}: "
              f"{moves}, {'resolved' if demo['resolved'] else 'unresolved'})")

    def _recent_demonstrations(self):
        """Freshest few demonstrations for the prompt, resolved ones first."""
        with self.lock:
            demos = list(self.policy.get(DEMONSTRATIONS_KEY) or [])
        demos.sort(key=lambda d: (bool(d.get("resolved")), d.get("ts", 0)),
                   reverse=True)
        return [{"situation": d.get("situation"),
                 "context": d.get("context"),
                 "steps": [{"direction": s["action"].get("direction"),
                            **{k: v for k, v in s["action"].items()
                               if k in ("speed", "angle")},
                            "duration": s["duration"]} for s in d.get("steps", [])],
                 "resolved": d.get("resolved")}
                for d in demos[:DEMONSTRATIONS_IN_PROMPT]]

    # ---------- inbound: adopt trained learning (online intake) ----------

    @staticmethod
    def _tag_trained_in_sim(policy):
        """Deep copy of an incoming pack's policy with every bandit arm marked
        `trained_in_sim`, so the reality-gap guard (see _maybe_retire_arm) knows
        these counts came from simulation. Reserved '_' sections (e.g.
        _demonstrations) are copied through untouched."""
        out = {}
        for key, entry in policy.items():
            if key.startswith("_") or not isinstance(entry, dict):
                out[key] = copy.deepcopy(entry)
                continue
            e = copy.deepcopy(entry)
            for arm in (e.get("arms") or {}).values():
                if isinstance(arm, dict):
                    arm["trained_in_sim"] = True
            out[key] = e
        return out

    def on_adopt(self, payload):
        """Fold newly-trained learning (from the idle self_trainer, or an
        offline import routed over the bus) into the LIVE policy in-process, so
        coach stays the sole writer of coach_policy.json - the sender never
        touches the file. Payload:

            {"coach_policy": {...}, "mode": "adopt"|"merge", "lineage": "..."}

        mode defaults to "adopt" (right for this robot's own self-training
        round-trip: shared arms take the refined counts instead of summing a
        seed twice; "merge" sums, for an independently-trained pack). Fail-soft:
        a malformed payload is ignored. combine_policy never deletes an existing
        arm, and every adopted arm is tagged trained_in_sim, so sim learning can
        only add or refresh - never retire - a real maneuver."""
        incoming = payload.get("coach_policy")
        if not isinstance(incoming, dict) or not incoming:
            return
        mode = payload.get("mode", "adopt")
        if mode not in ("adopt", "merge"):
            mode = "adopt"
        lineage = payload.get("lineage")
        tagged = self._tag_trained_in_sim(incoming)
        try:
            with self.lock:
                self.policy, stats = combine_policy(self.policy, tagged, mode=mode)
        except Exception as e:
            print(f"Coach: failed to adopt trained learning: {e}")
            return
        self._save_policy()
        shared = stats["arms_replaced"] if mode == "adopt" else stats["arms_reinforced"]
        print(f"Coach: adopted self-training [{mode}] "
              f"(lineage {lineage or '?'}): +{stats['situations_added']} situations, "
              f"+{stats['arms_added']} arms, {shared} shared updated")

    # ---------- inbound: queries ----------

    def on_query(self, payload):
        self.work_queue.put(payload)

    @staticmethod
    def _arm_confidence(arm):
        """Honest confidence in an arm: its observed success rate, or
        None when it has never been pulled (a guess is a guess)."""
        pulls = arm.get("successes", 0) + arm.get("failures", 0)
        return round(arm["successes"] / pulls, 2) if pulls else None

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
            entry = self.policy.get(situation_key)
            # Occasionally step outside the learned playbook on purpose
            # (see NOVELTY_RATE) - but only where there IS a playbook to
            # step outside of, and never for urgent queries, where the
            # robot is stuck and reliability beats curiosity.
            experiment = (
                arm_sig is not None
                and not payload.get("urgent")
                and len(entry["arms"]) <= MAX_ARMS_PER_SITUATION
                and random.random() < NOVELTY_RATE
            )
            chosen_arm = dict(entry["arms"][arm_sig]) if (arm_sig and not experiment) else None

        if chosen_arm is not None:
            confidence = self._arm_confidence(chosen_arm)
            print(f"Coach: exploiting learned arm for {situation_key} (no LLM call)")
            self._publish_decision(situation_key, "exploit_learned_arm",
                                   f"best UCB1 arm here (observed success {confidence})")
            self._dispatch_suggestion(
                query_id, situation_key, arm_sig, chosen_arm["steps"],
                chosen_arm.get("rationale", "known-good response"), cached=True,
                query_payload=payload, confidence=confidence,
            )
            return

        result = self._query_llm(payload, situation_key, experiment=experiment)
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
            if experiment:
                arm["experimental"] = True
            self._ensure_embedding(situation_key, payload)
        self._save_policy()

        self._publish_decision(
            situation_key,
            "experiment" if experiment else "ask_llm",
            "deliberately trying something outside the learned playbook" if experiment
            else "no reliable learned arm yet - asking the model")
        self._dispatch_suggestion(query_id, situation_key, arm_sig, steps, rationale,
                                   cached=False, query_payload=payload,
                                   experimental=experiment)

    def _publish_decision(self, situation_key, choice, reason):
        # Same decision-journal topic the field agent uses; event_logger
        # persists it, so "why did you pick that maneuver?" has a real
        # answer.
        self.bus.publish("picarx/decision", {
            "source": "coach", "kind": "suggestion_strategy",
            "choice": choice, "reason": reason,
            "situation_key": situation_key, "ts": time.time(),
        })

    def _dispatch_suggestion(self, query_id, situation_key, arm_sig, steps, rationale,
                             cached, query_payload, confidence=None, experimental=False):
        with self.lock:
            self.pending_queries[query_id] = {
                "situation_key": situation_key,
                "arm_sig": arm_sig,
                "steps": steps,
                "rationale": rationale,
                "cached": cached,
                "experimental": experimental,
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
            # Introspection: how sure are we (observed success rate of a
            # learned arm; None = pure guess), and is this a deliberate
            # experiment? field_agent phrases its narration off these.
            "confidence": confidence,
            "experimental": experimental,
        })

    # ---------- inbound: outcomes ----------

    def on_outcome(self, payload):
        query_id = payload.get("query_id")
        situation_key = payload.get("situation_key")
        success = bool(payload.get("success"))

        surprise = None
        with self.lock:
            pending = self.pending_queries.pop(query_id, None)
            if pending is None or pending["situation_key"] != situation_key:
                return
            entry = self.policy.setdefault(situation_key, {"arms": {}})
            arm = entry["arms"].get(pending["arm_sig"])
            if arm is not None:
                # Judge surprise against the record BEFORE this outcome:
                # a proven maneuver failing (or a written-off one
                # working) is exactly the "this should have worked but
                # didn't" signal reflection feeds on.
                prior_rate = self._arm_confidence(arm)
                pulls = arm["successes"] + arm["failures"]
                if pulls >= SURPRISE_MIN_PULLS and prior_rate is not None:
                    if not success and prior_rate >= SURPRISE_HIGH_RATE:
                        surprise = {"kind": "proven_arm_failed", "prior_rate": prior_rate}
                    elif success and prior_rate <= SURPRISE_LOW_RATE:
                        surprise = {"kind": "written_off_arm_succeeded", "prior_rate": prior_rate}
                if success:
                    arm["successes"] += 1
                else:
                    arm["failures"] += 1
                    self._maybe_retire_arm(entry, pending["arm_sig"])
                arm["last_updated"] = time.time()
        self._save_policy()

        if surprise is not None:
            print(f"Coach: SURPRISE - {surprise['kind']} for {situation_key} "
                  f"(prior rate {surprise['prior_rate']})")
            self.bus.publish("picarx/coach/surprise", {
                **surprise,
                "situation_key": situation_key,
                "rationale": pending["rationale"],
                "steps": pending["steps"],
                "ts": time.time(),
            })
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
            "experimental": pending.get("experimental", False),
            "success": success,
            # HOW it ended, not just whether: which veto cut it short (if
            # any), whether the robot visibly moved, and how long it ran -
            # the difference between "reverse harder" and "reverse the
            # other way" when correcting later.
            "vetoed": payload.get("vetoed"),
            "veto_code": payload.get("veto_code"),
            "motion_max": payload.get("motion_max"),
            "duration": payload.get("duration"),
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

    def _begin_experiment_session(self):
        """Pick this session's A/B condition and announce it on the bus so
        behavior_metrics can tag the session. When the experiment is disabled we
        always adopt (learning fully on), but still publish the tag so metrics
        stay labelled. Fail-soft."""
        session_id = time.time()
        if EXPERIMENT_ENABLED:
            condition, counter = experiment.rotate(EXPERIMENT_STATE_PATH)
        else:
            condition, counter = experiment.ADOPT, None
        self.experiment_condition = condition
        self.control = (condition == experiment.CONTROL)
        self.bus.publish(EXPERIMENT_TOPIC, {
            "condition": condition, "session_id": session_id,
            "counter": counter, "enabled": EXPERIMENT_ENABLED, "ts": session_id})
        print(f"Coach: A/B session condition = {condition}"
              f"{' (sim-trained arms held out)' if self.control else ''}"
              f"{'' if EXPERIMENT_ENABLED else ' (experiment disabled)'}")

    def _heartbeat_status(self):
        """Compact self-reported detail folded into the module heartbeat (see
        heartbeat.py): this session's A/B condition (adopt vs control, the lever
        the learning loop is measured on) and how many situations the policy has
        learned - so the bus beacon shows the coach is not just alive but which
        experiment arm is live. Cheap; the heartbeat guards any error."""
        status = {"condition": self.experiment_condition}
        if self.control:
            status["arms_held_out"] = True    # sim-trained arms not selectable this session
        with self.lock:
            status["situations"] = len(self.policy)
        return status

    def run(self):
        self._begin_experiment_session()
        self.bus.set_heartbeat_status(self._heartbeat_status)
        self.bus.subscribe("picarx/coach/query", self.on_query)
        self.bus.subscribe("picarx/coach/outcome", self.on_outcome)
        self.bus.subscribe("picarx/rc/demonstration", self.on_demonstration)
        self.bus.subscribe("picarx/coach/adopt", self.on_adopt)

        for _ in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, daemon=True).start()

        print(f"Coach active ({len(self.policy)} learned situations, "
              f"embeddings {'on' if self.embedder.available else 'off'}), "
              f"listening on picarx/coach/query")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    Coach().run()
