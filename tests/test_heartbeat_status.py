"""Per-module heartbeat status_fn wiring (Bus.set_heartbeat_status): the small
self-reported dict each module folds into the unified liveness beacon so field
debugging sees what a module is DOING, not just that it's alive. The emitter
itself is covered in test_heartbeat; here we check each module's status_fn."""
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402
import coach  # noqa: E402
import reflection  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


class FieldAgentHeartbeatTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_standby_by_default(self):
        st = self.fa._heartbeat_status()
        self.assertEqual(st["mode"], "standby")
        self.assertEqual(st["state"], "CRUISING")
        self.assertNotIn("place", st)
        self.assertNotIn("given_up", st)

    def test_exploring_reports_state_and_place(self):
        self.fa.explore_mode = True
        self.fa.state = "EVADING"
        self.fa.current_location = {"label": "the kitchen"}
        st = self.fa._heartbeat_status()
        self.assertEqual(st["mode"], "exploring")
        self.assertEqual(st["state"], "EVADING")
        self.assertEqual(st["place"], "the kitchen")

    def test_rc_takes_precedence_over_explore(self):
        self.fa.explore_mode = True
        self.fa.rc_active = True
        self.assertEqual(self.fa._heartbeat_status()["mode"], "rc")

    def test_given_up_is_flagged(self):
        self.fa.given_up = True
        self.assertTrue(self.fa._heartbeat_status()["given_up"])

    def test_registered_on_run_via_bus(self):
        # The status_fn is handed to the bus so the heartbeat can carry it.
        fn = self.fa._heartbeat_status
        self.fa.bus.set_heartbeat_status(fn)
        self.assertIs(self.fa.bus.heartbeat_status_fn, fn)
        self.assertIn("state", self.fa.bus.heartbeat_status_fn())


class CoachHeartbeatTest(unittest.TestCase):
    def setUp(self):
        # Skip __init__ (loads policy from disk / builds an embedder); wire only
        # what the status_fn reads, mirroring the other coach A/B tests.
        self.c = coach.Coach.__new__(coach.Coach)
        self.c.lock = threading.Lock()
        self.c.policy = {}
        self.c.experiment_condition = "adopt"
        self.c.control = False

    def test_adopt_condition_and_situation_count(self):
        self.c.policy = {"stuck": {}, "corner": {}}
        st = self.c._heartbeat_status()
        self.assertEqual(st["condition"], "adopt")
        self.assertEqual(st["situations"], 2)
        self.assertNotIn("arms_held_out", st)

    def test_control_session_flags_held_out_arms(self):
        self.c.experiment_condition = "control"
        self.c.control = True
        st = self.c._heartbeat_status()
        self.assertEqual(st["condition"], "control")
        self.assertTrue(st["arms_held_out"])


class ReflectionHeartbeatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.r = reflection.Reflection.__new__(reflection.Reflection)
        self.r.lock = threading.Lock()
        self.r.store = SemanticStore(
            readonly=False, db_path=os.path.join(self.tmp, "semantic.db"))
        self.r.last_activity = 1000.0
        self.r._last_reflection = None
        self.r._last_analysis = None

    def test_idle_and_known_facts_without_a_run_yet(self):
        st = self.r._heartbeat_status(now=1030.0)
        self.assertEqual(st["idle_sec"], 30.0)
        self.assertEqual(st["facts_known"], 0)          # empty fresh store
        self.assertNotIn("reflected_sec_ago", st)
        self.assertNotIn("analyzed_sec_ago", st)

    def test_last_run_summaries_are_reported(self):
        self.r._last_reflection = (1000.0, 3)           # 3 facts, at t=1000
        self.r._last_analysis = (990.0, 2)              # 2 patterns, at t=990
        st = self.r._heartbeat_status(now=1010.0)
        self.assertEqual(st["reflected_sec_ago"], 10.0)
        self.assertEqual(st["reflected_facts"], 3)
        self.assertEqual(st["analyzed_sec_ago"], 20.0)
        self.assertEqual(st["mined_patterns"], 2)


if __name__ == "__main__":
    unittest.main()
