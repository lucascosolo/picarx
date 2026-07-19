#!/usr/bin/env python3
# /home/picarx/layer_b/modules/reflection.py
"""
Idle reflection (Layer B) - turns the day's raw episodic events into a
few durable semantic facts, using ONE batched LLM call per reflection
window instead of reasoning in the hot path.

How it stays cheap (deliberate - preserve all of these):
  - LLM reflection only runs while the robot is IDLE: any movement
    intent, coach query, or heard speech resets the idle clock.
    Real-time behavior is never competing with the API call for CPU
    or attention. (Offline analysis has a looser trigger - see
    try_analyze - because it costs no API call and little CPU.)
  - Hard cooldown between reflections (REFLECTION_COOLDOWN) plus a
    minimum batch size (MIN_NEW_EVENTS): quiet days produce zero API
    calls, busy days produce at most a couple.
  - One request per window, digest capped in size, cheap model
    (REFLECTION_MODEL, haiku-class), bounded max_tokens.
  - No key / API failure => skips silently and retries next window.
    Reflection is an enrichment, never a dependency.

Data flow:
  events.db (read-only - event_logger.py stays the sole writer there)
    -> compact digest of NEW events since the last reflection
    -> LLM: "extract durable facts worth remembering"
    -> semantic.db via semantic_store.SemanticStore (this module is
       the SOLE writer to that DB, same single-writer convention)

Progress is tracked by event id in semantic.db's meta table, so a
restart never re-reflects the same events.

Consumers: companion.py folds recent facts into its conversational
grounding; future mapping/navigation code can query facts_for(...).
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus
import robot_config
from semantic_store import SemanticStore
from spatial_store import SpatialStore
import pattern_miner

import json
import sqlite3
import time
import threading

EVENTS_DB_PATH = "/home/picarx/layer_b/data/events.db"
# Coach's learned bandit policy (arm win/loss records). Read-only here -
# coach.py is its sole writer; we only aggregate it into a self-model.
COACH_POLICY_PATH = "/home/picarx/layer_b/data/coach_policy.json"

# Self-model thresholds: how much evidence before a tendency is worth
# stating in the first person. Kept conservative so the robot doesn't
# narrate noise from two coaching attempts.
SELF_MIN_PULLS_PER_DIRECTION = 3   # min tries before comparing escape directions
SELF_MIN_RATE_GAP = 0.15           # min success-rate gap to call one better
SELF_MIN_TOTAL_PULLS = 5           # min coaching attempts before an overall claim
SELF_VETO_PRONE_MIN = 2            # vetoes at one place before it's "troublesome"
SELF_MAX_FACTS = 5

# Non-LLM analysis (pattern mining + spatial connectivity facts) is
# pure Python but still only worth re-running occasionally. Unlike LLM
# reflection it does not wait for an idle window: it also fires once
# enough new events have accumulated, so statistics keep consolidating
# on a busy robot that never goes quiet.
ANALYSIS_COOLDOWN = 1800.0
ANALYSIS_MIN_NEW_EVENTS = 20
# An edge traversed at least this often becomes a durable layout fact.
CONNECTIVITY_MIN_TRAVERSALS = 2

CHECK_INTERVAL = 60.0          # how often the idle/eligibility check runs
IDLE_AFTER_SEC = 180.0         # no movement/speech/coach activity for this long = idle
REFLECTION_COOLDOWN = 1800.0   # min seconds between LLM reflection calls
MIN_NEW_EVENTS = 8             # don't bother reflecting on less than this
MAX_EVENTS_PER_DIGEST = 120    # newest N events considered per window
DIGEST_CHAR_BUDGET = 2800      # hard cap on digest text sent to the model
MAX_FACTS_PER_REFLECTION = 6

# Belief revision: how many current active facts to show the model as
# "existing memory" so it can spot a new fact that contradicts an old one.
# Kept small - this rides the SAME haiku call as extraction (no extra API
# request), so it only costs a few hundred input tokens.
EXISTING_FACTS_FOR_PROMPT = 25

# Autobiographical memory: a quiet stretch this long between consecutive
# events in the digest window marks a session boundary, and the reflection
# is asked to also emit one diary-style episode summary for that day.
SESSION_GAP_SEC = 3600.0

REFLECTION_MODEL = str(robot_config.get("reflection", "model",
                                        "claude-haiku-4-5-20251001", env="REFLECTION_MODEL"))

# Topics worth reflecting on (a subset of what event_logger records -
# periodic world snapshots are deliberately excluded, they're noise at
# this altitude).
INTERESTING_TOPICS = (
    "picarx/audio/heard",
    "picarx/coach/episode",
    "picarx/exploration/room_scan",
    "picarx/exploration/location_change",
    "picarx/exploration/hypothesis",
    "picarx/exploration/goal_progress",
    "picarx/coach/surprise",
    "picarx/action/result",
    "picarx/intent/feedback",
    "picarx/rc/demonstration",
)

SYSTEM_PROMPT = """You are the offline reflection process of a small autonomous robot car
(PiCar-X) that explores a home. You receive a digest of its recent event log: things it
heard people say, coached escape maneuvers and whether they worked, room scans (objects
seen at each camera angle), and movement results including safety vetoes.

Extract up to {max_facts} DURABLE facts worth remembering long-term. Good facts are
stable properties of the environment or its interactions ("the area with the sofa and
tvmonitor causes repeated collisions", "escape maneuvers that reverse work better than
turning here", "someone often asks about the battery"). Do NOT restate single transient
events, speculate beyond the data, or include timestamps.

USER DEMONSTRATION entries show how a human manually drove the robot out of
situations it struggles with - the highest-value material here. When the same
kind of demonstration repeats, extract the human's TACTIC as a fact (e.g.
"when blocked near the sofa, backing up then turning right works").

Up to 2 of the entries may instead use the subject "idea": a specific, safe
"what if ..." experiment the digest genuinely motivates (especially anything marked
SURPRISE - something that should have worked but didn't, or vice versa). Ideas feed
future coaching decisions, so keep them concrete and actionable, never fanciful.

BELIEF REVISION: the user message may include an EXISTING MEMORY block - facts you
already believe, each prefixed with a numeric id like "[42]". If a NEW fact you are
about to write directly CONTRADICTS one of those (same subject, incompatible claim -
e.g. you now learn a door that "connects" two rooms is actually always closed), add
"supersedes": <that id> to the new fact so the stale belief is retired. Use it ONLY
for genuine contradictions, never for mere elaboration or a fact you're just
reinforcing. Omit the field (or use null) otherwise.

AUTOBIOGRAPHICAL MEMORY: if the user message says SESSION BOUNDARY DETECTED, also
include exactly ONE extra entry whose subject is the "episode:<date>" string it gives
you. Its "fact" is a short (2-3 sentence) first-person, diary-style narrative of what
happened this session - warm and reflective, not a bullet list. This is the only entry
allowed to be narrative and multi-sentence; never emit an episode entry otherwise.

Reply with a JSON array only, no prose:
[{{"subject": "<short topic, e.g. 'living room' or 'escape tactics'>",
   "fact": "<one sentence>",
   "confidence": <0.0-1.0>,
   "supersedes": <id of a contradicted EXISTING MEMORY fact, or null>}}]
Return [] if nothing is worth remembering."""


def _move_with_duration(step):
    """'backward 1.2s' from a step dict; direction alone for old rows
    recorded before durations existed."""
    direction = (step.get("action") or {}).get("direction") or "?"
    duration = step.get("duration")
    return f"{direction} {duration:.1f}s" if duration is not None else direction


class Reflection:
    def __init__(self):
        self.bus = Bus()
        self.store = SemanticStore(readonly=False)
        self.spatial = SpatialStore(readonly=True)  # location_graph owns spatial.db
        self.lock = threading.Lock()
        self.last_activity = time.time()
        self._client = None
        self._warned_no_key = False

    # ---------- activity tracking (anything here means "not idle") ----------

    def on_activity(self, _payload):
        with self.lock:
            self.last_activity = time.time()

    # ---------- human-labeled sightings (curiosity.py / web console) ----------

    def on_label(self, payload):
        """A person told the robot what an ambiguous sighting actually is
        (picarx/perception/label). Unlike LLM reflection, this is written
        IMMEDIATELY at high confidence - a human identification is ground
        truth and shouldn't wait for the next idle window. Reflection stays
        the sole writer to semantic.db, so the write lands here; the sqlite
        connection is opened check_same_thread=False and is internally
        serialized, so writing from this callback thread alongside the main
        loop's writes is safe. Fail-soft: a bad payload is just ignored."""
        correct = (payload.get("correct_label") or "").strip().lower()
        if not correct:
            return
        guess = (payload.get("guess") or "").strip().lower()
        origin = payload.get("origin", "a person")
        try:
            confirmed = not guess or guess == correct
            self.store.upsert_fact(
                correct,
                (f"a person confirmed I identify this correctly as a {correct}"
                 if confirmed else
                 f"a person identified this as a {correct}, not the {guess} I guessed"),
                confidence=0.9, source="human_label")
            # A miss is itself a durable fact about my own perception: the
            # detector's habit of confusing these two is worth remembering.
            if not confirmed:
                self.store.upsert_fact(
                    "vision",
                    f"my detector sometimes mislabels a {correct} as a {guess}",
                    confidence=0.8, source="human_label")
            print(f"Reflection: human label ({origin}) '{correct}'"
                  f"{'' if confirmed else f' (I had guessed {guess})'} stored "
                  f"({self.store.fact_count()} facts known)")
        except Exception as e:
            print(f"Reflection: failed to store human label: {e}")

    # ---------- notes-to-self (expressions.py) ----------

    def on_note(self, payload):
        """A module asked to remember something durable (picarx/memory/note).
        Written straight through, since reflection is the sole writer to
        semantic.db and a note is already a decided fact, not something to
        infer. The store dedups on (subject, fact), so a repeated note just
        reinforces (seen_count++) rather than piling up. Fail-soft: a bad
        payload is ignored. Confidence is clamped modest - a note is a passing
        observation, not ground truth like a human label."""
        subject = (payload.get("subject") or "").strip()
        fact = (payload.get("fact") or "").strip()
        if not subject or not fact:
            return
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.1, min(0.7, confidence))
        source = (payload.get("source") or "note").strip()[:40] or "note"
        try:
            self.store.upsert_fact(subject, fact, confidence=confidence, source=source)
            print(f"Reflection: note ({source}) [{subject}] '{fact}' stored "
                  f"({self.store.fact_count()} facts known)")
        except Exception as e:
            print(f"Reflection: failed to store note: {e}")

    # ---------- events.db digest ----------

    def _fetch_new_events(self, since_id):
        try:
            conn = sqlite3.connect(f"file:{EVENTS_DB_PATH}?mode=ro", uri=True)
        except sqlite3.Error as e:
            print(f"Reflection: cannot open events.db read-only: {e}")
            return [], since_id
        try:
            placeholders = ",".join("?" * len(INTERESTING_TOPICS))
            rows = conn.execute(
                f"SELECT id, ts, topic, payload_json FROM events "
                f"WHERE id > ? AND topic IN ({placeholders}) "
                f"ORDER BY id DESC LIMIT ?",
                (since_id, *INTERESTING_TOPICS, MAX_EVENTS_PER_DIGEST),
            ).fetchall()
        except sqlite3.Error as e:
            print(f"Reflection: events query failed: {e}")
            rows = []
        finally:
            conn.close()
        if not rows:
            return [], since_id
        max_id = max(r[0] for r in rows)
        return list(reversed(rows)), max_id

    @staticmethod
    def _summarize_event(topic, payload_json):
        """One compact line per event - the model doesn't need raw JSON."""
        try:
            p = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        if topic == "picarx/audio/heard":
            return f"heard: {p.get('text')}"
        if topic == "picarx/coach/episode":
            # Episodes carry an ordered "steps" list; older rows may still
            # have the legacy single "action" field instead. Durations and
            # the failure cause ride along - "backward 1.0s failed, vetoed:
            # cliff" teaches something "failed" alone never could.
            steps = p.get("steps") or []
            if steps:
                moves = ",".join(_move_with_duration(s) for s in steps)
            else:
                moves = (p.get("action") or {}).get("direction")
            if p.get("success"):
                outcome = "worked"
            elif p.get("vetoed"):
                outcome = f"failed - vetoed ({p.get('veto_code') or 'unknown'})"
            elif p.get("motion_max") is not None and p["motion_max"] < 3.0:
                outcome = "failed - never visibly moved"
            else:
                outcome = "failed"
            return (f"coach[{p.get('situation_key')}]: {moves}"
                    f" -> {outcome}"
                    f" ({'cached' if p.get('cached') else 'fresh'})")
        if topic == "picarx/exploration/room_scan":
            parts = [f"{s.get('pan')}deg:{','.join(s.get('labels') or ['-'])}"
                     for s in p.get("sightings", [])]
            return "room scan: " + " | ".join(parts)
        if topic == "picarx/exploration/location_change":
            if p.get("is_new"):
                return f"discovered a new place: {p.get('label')}"
            if p.get("changed"):
                return f"moved to known place: {p.get('label')} (visit {p.get('visit_count')})"
            return None  # re-confirming the same spot is noise at this altitude
        if topic == "picarx/coach/surprise":
            return (f"SURPRISE [{p.get('situation_key')}]: {p.get('kind')} "
                    f"(prior success rate {p.get('prior_rate')})")
        if topic == "picarx/exploration/goal_progress":
            return (f"exploration subgoal '{p.get('label')}' {p.get('status')} "
                    f"after {p.get('elapsed')}s")
        if topic == "picarx/exploration/hypothesis":
            where = (p.get("location") or {}).get("label") or "unknown place"
            return (f"sensor hypothesis test at {where}: {p.get('question')} "
                    f"resolved to {p.get('resolution')}")
        if topic == "picarx/rc/demonstration":
            ctx = p.get("context") or {}
            where = (ctx.get("location") or {}).get("label") or "an unknown place"
            # Objects are {"label","side",...} dicts now; old rows in the
            # DB may still hold bare label strings.
            objects = ",".join(
                o if isinstance(o, str) else f"{o.get('label')}({o.get('side', '?')})"
                for o in ctx.get("objects") or []) or "nothing recognized"
            moves = ",".join(_move_with_duration(s) for s in p.get("actions") or [])
            return (f"USER DEMONSTRATION at {where} ({p.get('situation')}, "
                    f"seeing {objects}): the human drove {moves} -> "
                    f"{'cleared it' if p.get('resolved') else 'did not clear it'}")
        if topic == "picarx/intent/feedback":
            utterance = p.get("utterance")
            if not utterance:
                return None
            if p.get("verdict") == "incorrect":
                wanted = f" (they wanted: {p['correction']})" if p.get("correction") else ""
                return f"user flagged a MISUNDERSTOOD request: '{utterance}'{wanted}"
            return f"user confirmed a request was understood right: '{utterance}'"
        if topic == "picarx/action/result":
            result = p.get("result") or {}
            if result.get("status") == "vetoed":
                return f"safety veto: {result.get('reason')}"
            return None  # executed moves are noise at this altitude
        return None

    def _build_digest(self, rows):
        lines = []
        for _id, _ts, topic, payload_json in rows:
            line = self._summarize_event(topic, payload_json)
            if line:
                lines.append(line)
        digest = "\n".join(lines)
        if len(digest) > DIGEST_CHAR_BUDGET:
            digest = digest[-DIGEST_CHAR_BUDGET:]
            digest = digest[digest.index("\n") + 1:] if "\n" in digest else digest
        return digest, len(lines)

    # ---------- LLM ----------

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if not self._warned_no_key:
                print("Reflection: ANTHROPIC_API_KEY not set - reflection disabled.")
                self._warned_no_key = True
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Reflection: 'anthropic' package not installed - reflection disabled.")
        return self._client

    def _existing_memory_block(self):
        """Current active facts shown to the model for contradiction
        detection, each tagged with its id so the LLM can point
        'supersedes' at one. Returns (prompt_text, {id: fact_dict}).
        Episodes are excluded - a diary entry is never 'contradicted'."""
        facts = self.store.recent_facts(limit=EXISTING_FACTS_FOR_PROMPT)
        facts = [f for f in facts if not str(f["subject"]).startswith("episode:")]
        if not facts:
            return "", {}
        lines = [f"[{f['id']}] ({f['subject']}) {f['fact']}" for f in facts]
        by_id = {f["id"]: f for f in facts}
        return "EXISTING MEMORY (your current beliefs):\n" + "\n".join(lines), by_id

    @staticmethod
    def _session_boundary_subject(rows, now):
        """'episode:<YYYY-MM-DD>' if a significant idle gap splits this
        digest window into separate sessions, else None. rows are in
        chronological (id-ascending) order; ts is epoch seconds."""
        ts = [r[1] for r in rows if r[1]]
        if len(ts) < 2:
            return None
        if max(b - a for a, b in zip(ts, ts[1:])) < SESSION_GAP_SEC:
            return None
        return "episode:" + time.strftime("%Y-%m-%d", time.localtime(ts[-1]))

    def _extract_facts(self, digest, memory_block="", episode_subject=None):
        client = self._get_client()
        if client is None:
            return None
        content = digest
        if memory_block:
            content = f"{memory_block}\n\n---\nRECENT EVENTS:\n{digest}"
        if episode_subject:
            content += (f"\n\n---\nSESSION BOUNDARY DETECTED. Also emit one "
                        f"episode entry with subject \"{episode_subject}\".")
        try:
            response = client.messages.create(
                model=REFLECTION_MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT.format(max_facts=MAX_FACTS_PER_REFLECTION),
                messages=[{"role": "user", "content": content}],
                timeout=15.0,
            )
            text = "".join(b.text for b in response.content
                           if getattr(b, "type", None) == "text").strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            facts = json.loads(text)
            if not isinstance(facts, list):
                return None
            return facts  # capping happens in try_reflect (episodes are extra)
        except Exception as e:
            print(f"Reflection: LLM extraction failed: {e}")
            return None

    # ---------- offline analysis (pure Python, no LLM, no API key) ----------

    def _count_new_events(self, since_id):
        """(count, max_id) of events logged after since_id - any topic,
        since pattern mining scans the full stream, not just the digest
        topics. Fail-soft: unreadable events.db counts as nothing new."""
        try:
            conn = sqlite3.connect(f"file:{EVENTS_DB_PATH}?mode=ro", uri=True)
        except sqlite3.Error:
            return 0, since_id
        try:
            count, max_id = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), ?) FROM events WHERE id > ?",
                (since_id, since_id)).fetchone()
            return count, max_id
        except sqlite3.Error:
            return 0, since_id
        finally:
            conn.close()

    # ---------- self-model (pure Python, no LLM, no API key) ----------

    @staticmethod
    def _read_coach_policy():
        """Coach's bandit policy as a dict, fail-soft to {} (no file yet,
        or unreadable). We never write it - coach.py owns it."""
        try:
            with open(COACH_POLICY_PATH) as f:
                policy = json.load(f)
            return policy if isinstance(policy, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _aggregate_escape_directions(policy):
        """Sum coaching win/loss across every arm, keyed by the FIRST
        step's movement direction ('backward', 'forward', 'turn', ...).
        Returns {direction: [successes, failures]}."""
        agg = {}
        for entry in policy.values():
            if not isinstance(entry, dict):
                continue
            for arm in (entry.get("arms") or {}).values():
                steps = arm.get("steps") or []
                if not steps:
                    continue
                direction = (steps[0].get("action") or {}).get("direction")
                if not direction:
                    continue
                bucket = agg.setdefault(direction, [0, 0])
                bucket[0] += int(arm.get("successes", 0))
                bucket[1] += int(arm.get("failures", 0))
        return agg

    def _synthesize_self_facts(self):
        """3-5 first-person observations about the robot's own tendencies,
        aggregated purely from coach_policy.json, the spatial map and mined
        patterns. No API call. Returns a list of (text, confidence); may be
        shorter than 3 on a robot that simply hasn't done much yet - we
        state only what the data supports, never filler."""
        facts = []

        # --- escape-tactic tendencies, from the coach bandit records ---
        agg = self._aggregate_escape_directions(self._read_coach_policy())

        def rate(direction):
            s, f = agg.get(direction, [0, 0])
            total = s + f
            return (s / total if total else None), total

        b_rate, b_n = rate("backward")
        f_rate, f_n = rate("forward")
        if (b_n >= SELF_MIN_PULLS_PER_DIRECTION and f_n >= SELF_MIN_PULLS_PER_DIRECTION
                and b_rate is not None and f_rate is not None
                and abs(b_rate - f_rate) >= SELF_MIN_RATE_GAP):
            if b_rate > f_rate:
                facts.append(("I have learned that backing away first gets me unstuck "
                              "more reliably than pushing forward does.", 0.75))
            else:
                facts.append(("I have learned that easing forward gets me unstuck "
                              "more reliably than reversing does.", 0.75))

        total_s = sum(v[0] for v in agg.values())
        total_f = sum(v[1] for v in agg.values())
        total_pulls = total_s + total_f
        if total_pulls >= SELF_MIN_TOTAL_PULLS:
            overall = total_s / total_pulls
            if overall >= 0.6:
                facts.append(("I usually manage to work my own way out of a tight "
                              "spot once I try a maneuver I've learned.", 0.7))
            elif overall <= 0.4:
                facts.append(("I still get stuck fairly often even when I try what "
                              "I've learned, so I'm cautious in tight spaces.", 0.7))

        # --- where I have and haven't been, from the spatial map ---
        locations = self.spatial.all_locations()
        if locations:
            facts.append((f"I have mapped {len(locations)} different "
                          f"place{'s' if len(locations) != 1 else ''} while exploring.",
                          0.65))
            # A place found once but never returned to = still unexplored.
            lonely = [l for l in locations if l["visit_count"] <= 1]
            if lonely:
                l = min(lonely, key=lambda x: x["discovered_at"])
                facts.append((f"I still have not properly explored {l['label']} - "
                              f"I found it once but never went back.", 0.6))
            # A place that keeps vetoing me is worth admitting to.
            troublesome = [l for l in locations if l["veto_count"] >= SELF_VETO_PRONE_MIN]
            if troublesome:
                l = max(troublesome, key=lambda x: x["veto_count"])
                facts.append((f"Something about {l['label']} keeps tripping my safety "
                              f"sensors and stopping me.", 0.65))

        # --- one behavioural habit, from mined event patterns ---
        patterns = self.store.top_patterns(limit=1)
        if patterns:
            p = patterns[0]
            facts.append((f"I've noticed that {p['condition']} tends to lead to "
                          f"{p['outcome']}.", min(0.7, float(p['confidence']))))

        return facts[:SELF_MAX_FACTS]

    def try_analyze(self, now=None):
        """Pattern mining + spatial-connectivity facts. Does NOT need a
        key - a robot with no API access still consolidates statistics.

        Trigger (decoupled from LLM reflection): runs when the robot is
        idle OR when ANALYSIS_MIN_NEW_EVENTS have landed since the last
        analysis, whichever comes first, so cheap mining continues in
        the background of a busy day. ANALYSIS_COOLDOWN rate-limits both
        paths. Returns True if it ran (for tests)."""
        now = now if now is not None else time.time()
        last_run = float(self.store.get_meta("last_analysis_at", 0) or 0)
        if now - last_run < ANALYSIS_COOLDOWN:
            return False
        with self.lock:
            idle_for = now - self.last_activity
        since_id = int(self.store.get_meta("last_analyzed_event_id", 0) or 0)
        new_events, max_event_id = self._count_new_events(since_id)
        if idle_for < IDLE_AFTER_SEC and new_events < ANALYSIS_MIN_NEW_EVENTS:
            return False

        patterns = pattern_miner.mine_patterns(EVENTS_DB_PATH)
        for p in patterns:
            self.store.upsert_pattern(p["condition"], p["outcome"],
                                      p["frequency"], p["confidence"])

        # Room connectivity, straight from the location graph: an edge
        # crossed repeatedly is a stable property of the house.
        connectivity = 0
        for a, b, traversals in self.spatial.edge_list():
            if traversals < CONNECTIVITY_MIN_TRAVERSALS:
                continue
            loc_a, loc_b = self.spatial.get_location(a), self.spatial.get_location(b)
            if loc_a and loc_b:
                self.store.upsert_fact(
                    "layout", f"{loc_a['label']} connects to {loc_b['label']}",
                    confidence=min(0.9, 0.5 + 0.1 * traversals), source="location_graph")
                connectivity += 1

        # Self-model: recompute the robot's first-person sense of its own
        # tendencies from the freshly-written patterns + coach/spatial
        # stats, and make it the complete active "self" fact set (stale
        # snapshot retired). replace_subject no-ops if nothing synthesized,
        # so a quiet robot keeps whatever self-model it already had.
        self_facts = self._synthesize_self_facts()
        self.store.replace_subject("self", self_facts, source="self_model")

        self.store.set_meta("last_analysis_at", now)
        self.store.set_meta("last_analyzed_event_id", max_event_id)
        if patterns or connectivity or self_facts:
            print(f"Reflection: offline analysis stored {len(patterns)} patterns, "
                  f"{connectivity} connectivity facts, {len(self_facts)} self-facts")
        return True

    # ---------- one reflection attempt ----------

    def try_reflect(self, now=None):
        """LLM reflection - stays strictly idle-gated (IDLE_AFTER_SEC)
        plus REFLECTION_COOLDOWN; only try_analyze got the looser
        event-count trigger. Returns True if a reflection ran (for
        tests)."""
        now = now if now is not None else time.time()
        with self.lock:
            idle_for = now - self.last_activity
        if idle_for < IDLE_AFTER_SEC:
            return False
        last_run = float(self.store.get_meta("last_reflection_at", 0) or 0)
        if now - last_run < REFLECTION_COOLDOWN:
            return False

        since_id = int(self.store.get_meta("last_reflected_event_id", 0) or 0)
        rows, max_id = self._fetch_new_events(since_id)
        digest, n_lines = self._build_digest(rows) if rows else ("", 0)
        if n_lines < MIN_NEW_EVENTS:
            return False

        # Same haiku call, now also handed the current beliefs (for
        # contradiction detection) and, when the window straddles an idle
        # gap, a request for one diary-style episode summary. No extra API
        # call - it's all folded into the one request.
        memory_block, existing_by_id = self._existing_memory_block()
        episode_subject = self._session_boundary_subject(rows, now)
        facts = self._extract_facts(digest, memory_block, episode_subject)
        if facts is None:
            return False  # no key / API failure - leave events unconsumed, retry next window

        stored = 0
        superseded = 0
        episodes = 0
        n_facts = 0
        for f in facts:
            if not isinstance(f, dict):
                continue
            subject = (f.get("subject") or "").strip()
            fact = (f.get("fact") or "").strip()
            if not subject or not fact:
                continue
            is_episode = subject.startswith("episode:")
            # Ordinary facts are capped; the episode entry is extra and
            # always allowed through so a busy window still gets its diary.
            if not is_episode:
                if n_facts >= MAX_FACTS_PER_REFLECTION:
                    continue
                n_facts += 1
            try:
                confidence = max(0.0, min(1.0, float(f.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            # Only honor a supersede pointing at a real active fact we
            # actually showed the model, so a hallucinated id can't retire
            # an arbitrary belief. Episodes never supersede anything.
            supersedes = None
            if not is_episode:
                try:
                    cand = int(f.get("supersedes"))
                except (TypeError, ValueError):
                    cand = None
                if cand is not None and cand in existing_by_id:
                    supersedes = cand
            self.store.upsert_fact(subject, fact, confidence, supersedes=supersedes)
            stored += 1
            if supersedes is not None:
                superseded += 1
            if is_episode:
                episodes += 1

        self.store.set_meta("last_reflected_event_id", max_id)
        self.store.set_meta("last_reflection_at", now)
        print(f"Reflection: digested {n_lines} events into {stored} facts "
              f"({superseded} superseding, {episodes} episode) "
              f"({self.store.fact_count()} total known)")
        return True

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/intent/move", self.on_activity)
        self.bus.subscribe("picarx/coach/query", self.on_activity)
        self.bus.subscribe("picarx/audio/heard", self.on_activity)
        self.bus.subscribe("picarx/perception/label", self.on_label)
        self.bus.subscribe("picarx/memory/note", self.on_note)

        print(f"Reflection active ({self.store.fact_count()} facts known), "
              f"reflecting when idle {IDLE_AFTER_SEC:.0f}s+, analyzing when idle "
              f"or every {ANALYSIS_MIN_NEW_EVENTS} events")
        while True:
            time.sleep(CHECK_INTERVAL)
            try:
                self.try_analyze()
                self.try_reflect()
            except Exception as e:
                print(f"Reflection: cycle error: {e}")


if __name__ == "__main__":
    Reflection().run()
