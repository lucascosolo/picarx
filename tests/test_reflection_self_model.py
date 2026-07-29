import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import reflection  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402
from spatial_store import SpatialStore  # noqa: E402


class SelfModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sem_db = os.path.join(self.tmp, "semantic.db")
        self.spa_db = os.path.join(self.tmp, "spatial.db")
        self.policy_path = os.path.join(self.tmp, "coach_policy.json")
        self._orig_policy_path = reflection.COACH_POLICY_PATH
        reflection.COACH_POLICY_PATH = self.policy_path

        self.r = reflection.Reflection.__new__(reflection.Reflection)
        self.r.store = SemanticStore(readonly=False, db_path=self.sem_db)
        self.r.spatial = SpatialStore(readonly=True, db_path=self.spa_db)

    def tearDown(self):
        reflection.COACH_POLICY_PATH = self._orig_policy_path

    def _write_policy(self, policy):
        with open(self.policy_path, "w") as f:
            json.dump(policy, f)

    def _seed_locations(self):
        w = SpatialStore(readonly=False, db_path=self.spa_db)
        w.conn.execute(
            "INSERT INTO locations (label, fingerprint_json, discovered_at, "
            "last_visited_at, visit_count, veto_count) VALUES (?,?,?,?,?,?)",
            ("place 1 (sofa)", "{}", 100.0, 200.0, 5, 0))
        w.conn.execute(
            "INSERT INTO locations (label, fingerprint_json, discovered_at, "
            "last_visited_at, visit_count, veto_count) VALUES (?,?,?,?,?,?)",
            ("place 4 (chair)", "{}", 50.0, 60.0, 1, 3))
        w.conn.commit()

    # ---- direction aggregation ----

    def test_aggregate_escape_directions(self):
        policy = {"s": {"arms": {
            "b": {"steps": [{"action": {"direction": "backward"}, "duration": 1.0}],
                  "successes": 7, "failures": 1},
            "f": {"steps": [{"action": {"direction": "forward"}, "duration": 1.0}],
                  "successes": 1, "failures": 6},
        }}}
        agg = reflection.Reflection._aggregate_escape_directions(policy)
        self.assertEqual(agg["backward"], [7, 1])
        self.assertEqual(agg["forward"], [1, 6])

    def test_read_coach_policy_failsoft(self):
        # No file -> {}, never raises.
        self.assertEqual(self.r._read_coach_policy(), {})

    # ---- full synthesis ----

    def test_synthesizes_backward_preference_and_location_facts(self):
        self._write_policy({"collision:stuck": {"arms": {
            "b": {"steps": [{"action": {"direction": "backward"}, "duration": 1.0}],
                  "successes": 7, "failures": 1},
            "f": {"steps": [{"action": {"direction": "forward"}, "duration": 1.0}],
                  "successes": 1, "failures": 6},
        }}})
        self._seed_locations()
        self.r.store.upsert_pattern("safety veto at place 4", "reverse maneuver", 5, 0.8)

        facts = self.r._synthesize_self_facts()
        text = " ".join(f for f, _ in facts).lower()
        self.assertGreaterEqual(len(facts), 3)
        self.assertLessEqual(len(facts), reflection.SELF_MAX_FACTS)
        self.assertIn("backing away first", text)     # escape-direction tendency
        self.assertIn("mapped 2", text)               # location count
        self.assertIn("place 4", text)                # unexplored / veto-prone place

    def test_no_data_yields_no_filler(self):
        # Empty policy + empty map -> honest silence, not invented facts.
        facts = self.r._synthesize_self_facts()
        self.assertEqual(facts, [])

    def test_replace_subject_makes_it_a_live_snapshot(self):
        self._write_policy({"s": {"arms": {
            "b": {"steps": [{"action": {"direction": "backward"}, "duration": 1.0}],
                  "successes": 7, "failures": 1},
            "f": {"steps": [{"action": {"direction": "forward"}, "duration": 1.0}],
                  "successes": 1, "failures": 6},
        }}})
        self._seed_locations()
        facts = self.r._synthesize_self_facts()
        self.r.store.replace_subject("self", facts, source="self_model")
        active = self.r.store.facts_for("self", limit=reflection.SELF_MAX_FACTS)
        self.assertEqual(len(active), len(facts))
        self.assertTrue(all(f["subject"] == "self" for f in active))

    # ---- session-boundary date (episodic) ----

    def test_session_boundary_detected_on_gap(self):
        now = 1_000_000.0
        rows = [(1, now - 8000, "t", "{}"),
                (2, now - 7900, "t", "{}"),
                (3, now - 10, "t", "{}")]
        subj = reflection.Reflection._session_boundary_subject(rows, now)
        self.assertIsNotNone(subj)
        self.assertTrue(subj.startswith("episode:"))

    def test_no_session_boundary_without_gap(self):
        now = 1_000_000.0
        rows = [(1, now - 30, "t", "{}"), (2, now - 20, "t", "{}"),
                (3, now - 10, "t", "{}")]
        self.assertIsNone(reflection.Reflection._session_boundary_subject(rows, now))

    # ---- digest line building ----

    def test_summarize_coach_episode_steps_schema(self):
        # Regression: episodes carry a "steps" list (coach.py's current
        # schema); the summarizer used to read the long-gone "action" field
        # and rendered "None" for the maneuver in every digest line.
        payload = json.dumps({
            "situation_key": "collision_loop:repeated_veto",
            "steps": [{"action": {"direction": "backward"}, "duration": 1.0},
                      {"action": {"direction": "turn", "angle": 20}, "duration": 0.5}],
            "success": True, "cached": False,
        })
        line = reflection.Reflection._summarize_event("picarx/coach/episode", payload)
        self.assertIn("backward 1.0s,turn 0.5s", line)
        self.assertNotIn("None", line)

    def test_summarize_coach_episode_legacy_action(self):
        payload = json.dumps({
            "situation_key": "novel_object:chair",
            "action": {"direction": "stop"}, "success": False, "cached": True,
        })
        line = reflection.Reflection._summarize_event("picarx/coach/episode", payload)
        self.assertIn("stop", line)
        self.assertIn("failed", line)


class ConversationDigestTest(unittest.TestCase):
    """A digest of "heard:" lines alone cannot tell a six-turn conversation
    from six things said near the robot. The conversation edges are what give
    the day a shape worth reflecting on."""

    def _line(self, payload):
        return reflection.Reflection._summarize_event(
            "picarx/dialog/conversation", json.dumps(payload))

    def test_the_opening_edge_brackets_what_follows(self):
        line = self._line({"open": True, "reason": "wake"})
        self.assertIn("a conversation started", line)
        self.assertIn("wake", line)

    def test_the_closing_edge_reports_length_and_why_it_ended(self):
        line = self._line({"open": False, "reason": "idle",
                           "turns": 6, "duration": 214.0})
        self.assertIn("6 turns", line)
        self.assertIn("214.0s", line)
        self.assertIn("went quiet", line)

    def test_being_asked_to_stop_reads_as_a_person_ending_it(self):
        # A meaningfully different ending from silence: somebody chose it.
        line = self._line({"open": False, "reason": "asked", "turns": 2,
                           "duration": 30.0})
        self.assertIn("asked me to stop listening", line)

    def test_a_close_without_the_shape_fields_still_summarizes(self):
        # Old rows (and any future publisher) must not crash the digest.
        line = self._line({"open": False, "reason": "idle"})
        self.assertIn("some turns", line)
        self.assertNotIn("None", line)


class ReflectionEvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sem_db = os.path.join(self.tmp, "semantic.db")
        self.spa_db = os.path.join(self.tmp, "spatial.db")
        spatial = SpatialStore(readonly=False, db_path=self.spa_db)
        spatial.conn.execute(
            "INSERT INTO locations (label, fingerprint_json, discovered_at, "
            "last_visited_at) VALUES (?,?,?,?)",
            ("place 1", "{}", 1.0, 1.0))
        spatial.conn.commit()
        spatial.note_veto(1, evidence={"labels": ["chair"]}, now=time.time())

        self.r = reflection.Reflection.__new__(reflection.Reflection)
        self.r.store = SemanticStore(readonly=False, db_path=self.sem_db)
        self.r.spatial = SpatialStore(readonly=True, db_path=self.spa_db)
        self.r.bus = harness.FakeBus()
        self.r.lock = threading.Lock()
        self.r.last_activity = time.time() - reflection.IDLE_AFTER_SEC - 1

        rows = [(1, time.time(), "picarx/action/result", json.dumps({
            "result": {"status": "vetoed", "reason": "obstacle"}}))]
        self.r._fetch_new_events = lambda _since: (rows, 1)
        self.r._build_digest = lambda _rows: ("safety veto: obstacle", 8)
        self.r._existing_memory_block = lambda: ("", {})
        self.r._session_boundary_subject = lambda _rows, _now: None

    def _run_with_fact(self, fact):
        self.r._extract_facts = lambda digest, memory, episode, evidence: [fact]
        self.assertTrue(self.r.try_reflect(now=time.time()))
        return self.r.store.facts_for(fact["subject"])[0]

    def test_uncorroborated_llm_confidence_is_capped_and_explained(self):
        stored = self._run_with_fact({
            "subject": "hall", "fact": "the hall is blocked", "confidence": 0.95})
        self.assertEqual(stored["confidence"], reflection.REFLECTION_HIGH_CONFIDENCE_CEILING)
        self.assertEqual(self.r.store.fact_evidence_for(stored["id"]), [])
        decision = self.r.bus.last("picarx/decision")
        self.assertEqual(decision["kind"], "fact_confidence_gate")

    def test_valid_sensor_citation_allows_high_confidence_and_is_copied(self):
        stored = self._run_with_fact({
            "subject": "hall", "fact": "the hall vetoes near the chair",
            "confidence": 0.95, "evidence": ["spatial:veto:1"]})
        self.assertAlmostEqual(stored["confidence"], 0.95)
        evidence = self.r.store.fact_evidence_for(stored["id"])
        self.assertEqual(evidence[0]["evidence_kind"], "veto")
        self.assertEqual(evidence[0]["evidence_db"], "spatial")
        self.assertEqual(evidence[0]["evidence_id"], "1")


if __name__ == "__main__":
    unittest.main()
