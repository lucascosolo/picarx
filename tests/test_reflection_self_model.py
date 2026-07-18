import json
import os
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
