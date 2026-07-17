import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402
import goal_manager  # noqa: E402
import location_graph  # noqa: E402
from spatial_store import SpatialStore  # noqa: E402


class QueryParseTest(unittest.TestCase):
    def test_place_name_command(self):
        self.assertEqual(field_agent.parse_place_name_command(
            "call this place the kitchen"), "kitchen")
        self.assertEqual(field_agent.parse_place_name_command(
            "name this room lucas office"), "lucas office")
        self.assertEqual(field_agent.parse_place_name_command(
            "this room is the den"), "den")
        self.assertIsNone(field_agent.parse_place_name_command("what a nice place"))

    def test_go_to_command(self):
        self.assertEqual(field_agent.parse_go_to_command("go to the kitchen"), "kitchen")
        self.assertEqual(field_agent.parse_go_to_command(
            "go back to the living room"), "living room")
        self.assertIsNone(field_agent.parse_go_to_command("time to go"))

    def test_where_is_query(self):
        self.assertEqual(field_agent.parse_where_is_query("where is the bottle"), "bottle")
        self.assertEqual(field_agent.parse_where_is_query("where's my cup"), "cup")
        self.assertEqual(field_agent.parse_where_is_query(
            "where did you last see the chair"), "chair")
        self.assertIsNone(field_agent.parse_where_is_query("where are you"))

    def test_whats_in_query(self):
        self.assertEqual(field_agent.parse_whats_in_query(
            "what's in the kitchen"), "kitchen")
        self.assertEqual(field_agent.parse_whats_in_query(
            "what have you seen in the living room"), "living room")
        self.assertIsNone(field_agent.parse_whats_in_query("what is that"))

    def test_spoken_age(self):
        self.assertEqual(field_agent.spoken_age(30), "just now")
        self.assertEqual(field_agent.spoken_age(600), "10 minutes ago")
        self.assertIn("hour", field_agent.spoken_age(7200))


class FieldAgentMemoryCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "spatial.db")
        self.writer = SpatialStore(readonly=False, db_path=self.db)
        self.loc = self.writer.match_or_create(
            {"labels": ["c:sofa"], "range": "mid"}, now=time.time() - 300)
        self.writer.rename_location(self.loc["id"], "kitchen")
        self.writer.note_sightings(self.loc["id"], ["bottle", "chair"],
                                   now=time.time() - 300)
        self.fa = field_agent.FieldAgent()
        self.fa.spatial = SpatialStore(readonly=True, db_path=self.db)

    def _spoken(self):
        return " ".join(p["text"] for p in self.fa.bus.of("picarx/audio/speak"))

    def test_where_is_answers_from_sightings(self):
        self.fa.handle_voice_command("where is the bottle")
        speech = self._spoken()
        self.assertIn("bottle", speech)
        self.assertIn("kitchen", speech)
        self.assertIn("minutes ago", speech)

    def test_where_is_fuzzy_label(self):
        self.fa.handle_voice_command("where is the bottel")
        self.assertIn("kitchen", self._spoken())

    def test_where_is_unknown_object(self):
        self.fa.handle_voice_command("where is the unicorn")
        self.assertIn("haven't seen a unicorn", self._spoken())

    def test_whats_in_place(self):
        self.fa.handle_voice_command("what's in the kitchen")
        speech = self._spoken()
        self.assertIn("bottle", speech)
        self.assertIn("chair", speech)

    def test_name_place_routes_to_location_graph(self):
        self.fa.current_location = {"location_id": self.loc["id"], "label": "kitchen"}
        self.fa.handle_voice_command("call this place the pantry")
        msg = self.fa.bus.last("picarx/exploration/name_place")
        self.assertIsNotNone(msg)
        self.assertEqual(msg["name"], "pantry")
        self.assertEqual(msg["location_id"], self.loc["id"])

    def test_name_place_without_location_says_so(self):
        self.fa.handle_voice_command("call this place the pantry")
        self.assertIn("not sure where I am", self._spoken())
        self.assertIsNone(self.fa.bus.last("picarx/exploration/name_place"))

    def test_go_to_known_place_requests_goal_and_explores(self):
        self.fa.handle_voice_command("go to the kitchen")
        req = self.fa.bus.last("picarx/exploration/goal_request")
        self.assertEqual(req["location_id"], self.loc["id"])
        self.assertTrue(self.fa.explore_mode)
        self.assertEqual(self.fa.state, "SCANNING")

    def test_go_to_unknown_place_does_not_move(self):
        self.fa.handle_voice_command("go to the moon")
        self.assertIsNone(self.fa.bus.last("picarx/exploration/goal_request"))
        self.assertFalse(self.fa.explore_mode)
        self.assertIn("don't know a place called moon", self._spoken())

    def test_who_am_i_with_fresh_identity(self):
        self.fa.on_person({"name": "lucas", "confidence": 42.0})
        self.fa.bus.clear()
        self.fa.handle_voice_command("who am i")
        self.assertIn("lucas", self._spoken())

    def test_greeting_debounced(self):
        self.fa.on_person({"name": "lucas"})
        self.fa.on_person({"name": "lucas"})
        greetings = [p for p in self.fa.bus.of("picarx/audio/speak")
                     if "Hello, lucas" in p["text"]]
        self.assertEqual(len(greetings), 1)


class GoalRequestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "spatial.db")
        self.writer = SpatialStore(readonly=False, db_path=self.db)
        self.loc = self.writer.match_or_create(
            {"labels": ["c:sofa"], "range": "mid"}, now=time.time())
        self.gm = goal_manager.GoalManager.__new__(goal_manager.GoalManager)
        self.gm.bus = harness.FakeBus()
        self.gm.store = SpatialStore(readonly=True, db_path=self.db)
        import threading
        self.gm.lock = threading.Lock()
        self.gm.scores = {}
        self.gm.current_id = None
        self.gm.active = None
        self.gm.failures = {self.loc["id"]: goal_manager.MAX_GOAL_FAILURES}

    def test_user_goal_adopted_despite_failure_blacklist(self):
        self.gm.on_goal_request({"location_id": self.loc["id"]})
        self.assertIsNotNone(self.gm.active)
        goal = self.gm.bus.last("picarx/exploration/active_goal")
        self.assertEqual(goal["location_id"], self.loc["id"])

    def test_user_goal_supersedes_active_goal(self):
        self.gm.active = {"goal_id": "old", "location_id": 999, "label": "elsewhere",
                          "started_at": time.time(), "deadline": time.time() + 100}
        self.gm.on_goal_request({"location_id": self.loc["id"]})
        progress = self.gm.bus.last("picarx/exploration/goal_progress")
        self.assertEqual(progress["status"], "superseded")
        self.assertEqual(self.gm.active["location_id"], self.loc["id"])
        # superseding never counts as a failure against the old place
        self.assertNotIn(999, self.gm.failures)

    def test_unknown_location_ignored(self):
        self.gm.on_goal_request({"location_id": 424242})
        self.assertIsNone(self.gm.active)


class LocationGraphSightingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "spatial.db")
        self.lg = location_graph.LocationGraph.__new__(location_graph.LocationGraph)
        self.lg.bus = harness.FakeBus()
        self.lg.store = SpatialStore(readonly=False, db_path=self.db)
        import threading
        self.lg.lock = threading.Lock()
        self.lg.current_id = None
        self.lg.current_label = None

    def test_room_scan_records_sightings(self):
        self.lg.on_room_scan({"sightings": [
            {"pan": -35, "labels": ["sofa"]},
            {"pan": 35, "labels": ["bottle", "sofa"]},
        ], "distance_cm": 80})
        loc_id = self.lg.current_id
        labels = {o["label"] for o in self.lg.store.location_objects(loc_id)}
        self.assertEqual(labels, {"sofa", "bottle"})

    def test_name_place_renames_current(self):
        self.lg.on_room_scan({"sightings": [{"pan": 0, "labels": ["sofa"]}],
                              "distance_cm": 80})
        self.lg.on_name_place({"name": "kitchen"})
        self.assertEqual(self.lg.current_label, "kitchen")
        self.assertEqual(
            self.lg.store.get_location(self.lg.current_id)["label"], "kitchen")
        speech = " ".join(p["text"] for p in self.lg.bus.of("picarx/audio/speak"))
        self.assertIn("kitchen", speech)

    def test_name_place_with_no_location_says_so(self):
        self.lg.on_name_place({"name": "kitchen"})
        speech = " ".join(p["text"] for p in self.lg.bus.of("picarx/audio/speak"))
        self.assertIn("not sure where I am", speech)


if __name__ == "__main__":
    unittest.main()
