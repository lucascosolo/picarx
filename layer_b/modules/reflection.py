#!/usr/bin/env python3
# /home/picarx/layer_b/modules/reflection.py
"""
Idle reflection (Layer B) - turns the day's raw episodic events into a
few durable semantic facts, using ONE batched LLM call per reflection
window instead of reasoning in the hot path.

How it stays cheap (deliberate - preserve all of these):
  - Only runs while the robot is IDLE: any movement intent, coach
    query, or heard speech resets the idle clock. Real-time behavior
    is never competing with reflection for CPU or attention.
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
from semantic_store import SemanticStore
from spatial_store import SpatialStore
import pattern_miner

import json
import sqlite3
import time
import threading

EVENTS_DB_PATH = "/home/picarx/layer_b/data/events.db"

# Non-LLM analysis (pattern mining + spatial connectivity facts) is
# pure Python but still only worth re-running occasionally.
ANALYSIS_COOLDOWN = 1800.0
# An edge traversed at least this often becomes a durable layout fact.
CONNECTIVITY_MIN_TRAVERSALS = 2

CHECK_INTERVAL = 60.0          # how often the idle/eligibility check runs
IDLE_AFTER_SEC = 180.0         # no movement/speech/coach activity for this long = idle
REFLECTION_COOLDOWN = 1800.0   # min seconds between LLM reflection calls
MIN_NEW_EVENTS = 8             # don't bother reflecting on less than this
MAX_EVENTS_PER_DIGEST = 120    # newest N events considered per window
DIGEST_CHAR_BUDGET = 2800      # hard cap on digest text sent to the model
MAX_FACTS_PER_REFLECTION = 6

REFLECTION_MODEL = os.environ.get("REFLECTION_MODEL", "claude-haiku-4-5-20251001")

# Topics worth reflecting on (a subset of what event_logger records -
# periodic world snapshots are deliberately excluded, they're noise at
# this altitude).
INTERESTING_TOPICS = (
    "picarx/audio/heard",
    "picarx/coach/episode",
    "picarx/exploration/room_scan",
    "picarx/exploration/location_change",
    "picarx/exploration/hypothesis",
    "picarx/action/result",
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

Reply with a JSON array only, no prose:
[{{"subject": "<short topic, e.g. 'living room' or 'escape tactics'>",
   "fact": "<one sentence>",
   "confidence": <0.0-1.0>}}]
Return [] if nothing is worth remembering."""


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
            act = p.get("action") or {}
            return (f"coach[{p.get('situation_key')}]: {act.get('direction')}"
                    f" -> {'worked' if p.get('success') else 'failed'}"
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
        if topic == "picarx/exploration/hypothesis":
            where = (p.get("location") or {}).get("label") or "unknown place"
            return (f"sensor hypothesis test at {where}: {p.get('question')} "
                    f"resolved to {p.get('resolution')}")
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

    def _extract_facts(self, digest):
        client = self._get_client()
        if client is None:
            return None
        try:
            response = client.messages.create(
                model=REFLECTION_MODEL,
                max_tokens=500,
                system=SYSTEM_PROMPT.format(max_facts=MAX_FACTS_PER_REFLECTION),
                messages=[{"role": "user", "content": digest}],
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
            return facts[:MAX_FACTS_PER_REFLECTION]
        except Exception as e:
            print(f"Reflection: LLM extraction failed: {e}")
            return None

    # ---------- offline analysis (pure Python, no LLM, no API key) ----------

    def try_analyze(self, now=None):
        """Pattern mining + spatial-connectivity facts. Runs on the
        same idle windows as LLM reflection but does NOT need a key -
        a robot with no API access still consolidates statistics.
        Returns True if it ran (for tests)."""
        now = now if now is not None else time.time()
        with self.lock:
            idle_for = now - self.last_activity
        if idle_for < IDLE_AFTER_SEC:
            return False
        last_run = float(self.store.get_meta("last_analysis_at", 0) or 0)
        if now - last_run < ANALYSIS_COOLDOWN:
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

        self.store.set_meta("last_analysis_at", now)
        if patterns or connectivity:
            print(f"Reflection: offline analysis stored {len(patterns)} patterns, "
                  f"{connectivity} connectivity facts")
        return True

    # ---------- one reflection attempt ----------

    def try_reflect(self, now=None):
        """Returns True if a reflection actually ran (for tests)."""
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

        facts = self._extract_facts(digest)
        if facts is None:
            return False  # no key / API failure - leave events unconsumed, retry next window

        stored = 0
        for f in facts:
            subject = (f.get("subject") or "").strip() if isinstance(f, dict) else ""
            fact = (f.get("fact") or "").strip() if isinstance(f, dict) else ""
            if not subject or not fact:
                continue
            try:
                confidence = max(0.0, min(1.0, float(f.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            self.store.upsert_fact(subject, fact, confidence)
            stored += 1

        self.store.set_meta("last_reflected_event_id", max_id)
        self.store.set_meta("last_reflection_at", now)
        print(f"Reflection: digested {n_lines} events into {stored} facts "
              f"({self.store.fact_count()} total known)")
        return True

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/intent/move", self.on_activity)
        self.bus.subscribe("picarx/coach/query", self.on_activity)
        self.bus.subscribe("picarx/audio/heard", self.on_activity)

        print(f"Reflection active ({self.store.fact_count()} facts known), "
              f"reflecting when idle {IDLE_AFTER_SEC:.0f}s+")
        while True:
            time.sleep(CHECK_INTERVAL)
            try:
                self.try_analyze()
                self.try_reflect()
            except Exception as e:
                print(f"Reflection: cycle error: {e}")


if __name__ == "__main__":
    Reflection().run()
