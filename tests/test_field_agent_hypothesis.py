import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402


class HypothesisFrameworkTest(unittest.TestCase):
    def setUp(self):
        # Full __init__ is safe off-robot: FakeBus + a readonly SpatialStore
        # that lazily fail-softs to [] on a missing DB (never writes).
        self.fa = field_agent.FieldAgent()
        self.bus = self.fa.bus  # FakeBus

    def _intents(self):
        return [p for p in self.bus.of("picarx/intent/move")]

    def _hyps(self):
        return self.bus.of("picarx/exploration/hypothesis")

    def _decision_kinds(self):
        return {p["kind"] for p in self.bus.of("picarx/decision")}

    # ---- sensor-disagreement probe (migrated original) ----

    def test_sensor_probe_phantom(self):
        self.fa.latest_world = {"distance_cm": 18, "distance_stale": False,
                                "objects": {"stale": False, "items": [],
                                            "close_object": False}}
        started = self.fa._maybe_start_sensor_probe(100.0, self.fa.latest_world, 18)
        self.assertTrue(started)
        self.assertEqual(self.fa.state, "HYPOTHESIS")
        self.assertEqual(self.fa.hypothesis.type, "sensor_disagreement")
        for t in (100.2, 100.5, 100.6, 101.3, 101.8):
            if t == 101.8:  # path opened right up by judge time -> phantom
                self.fa.latest_world = {"distance_cm": 90, "distance_stale": False,
                                        "objects": {"stale": False, "items": []}}
            self.fa._handle_hypothesis_tick(t)
        h = self._hyps()[-1]
        self.assertEqual(h["question"], "ultrasonic_obstacle_vs_empty_vision")
        self.assertEqual(h["resolution"], "phantom_reading")
        self.assertEqual(h["d0"], 18)
        self.assertEqual(h["d1"], 90)
        self.assertIsNone(self.fa.hypothesis)
        self.assertEqual(self.fa.state, "CRUISING")
        self.assertEqual(self._decision_kinds() & {"hypothesis_probe", "hypothesis_resolved"},
                         {"hypothesis_probe", "hypothesis_resolved"})

    def test_sensor_probe_speed_capped(self):
        self.fa.latest_world = {"distance_cm": 18, "distance_stale": False,
                                "objects": {"stale": False, "items": [],
                                            "close_object": False}}
        self.fa._maybe_start_sensor_probe(100.0, self.fa.latest_world, 18)
        for t in (100.2, 100.5, 100.6, 101.3, 101.8):
            self.fa._handle_hypothesis_tick(t)
        forwards = [i for i in self._intents() if i["action"].get("direction") == "forward"]
        self.assertTrue(forwards)
        self.assertTrue(all(i["action"]["speed"] <= 15 for i in forwards))

    def test_sensor_probe_veto_resolves_real(self):
        self.fa.latest_world = {"distance_cm": 18, "distance_stale": False,
                                "objects": {"stale": False, "items": [],
                                            "close_object": False}}
        self.fa._maybe_start_sensor_probe(200.0, self.fa.latest_world, 18)
        self.fa.veto_events.append(200.3)  # safety daemon vetoes the creep
        self.fa._handle_hypothesis_tick(200.5)
        h = self._hyps()[-1]
        self.assertEqual(h["resolution"], "real_obstacle")
        self.assertEqual(self.fa.state, "EVADING")  # follow-up evades

    # ---- veto-prone location probe (the new one) ----

    def _arm_veto_prone(self, veto_count=5):
        self.fa.current_location = {"location_id": 4, "label": "place 4 (chair)"}
        self.fa.spatial.get_location = lambda i: {
            "id": 4, "label": "place 4 (chair)", "veto_count": veto_count}

    def test_veto_prone_triggers_only_above_threshold(self):
        self._arm_veto_prone(veto_count=2)  # below VETO_PRONE_THRESHOLD (3)
        self.assertFalse(self.fa._maybe_start_veto_prone_probe(300.0))
        self._arm_veto_prone(veto_count=3)
        self.assertTrue(self.fa._maybe_start_veto_prone_probe(400.0))

    def test_veto_prone_maybe_clear(self):
        self._arm_veto_prone()
        self.assertTrue(self.fa._maybe_start_veto_prone_probe(300.0))
        self.assertEqual(self.fa.hypothesis.type, "veto_prone_location")
        for t in (300.5, 301.5, 302.5, 303.5):  # >3s window, no veto
            self.fa._handle_hypothesis_tick(t)
        h = self._hyps()[-1]
        self.assertEqual(h["question"], "is_veto_prone_area_still_blocked")
        self.assertEqual(h["resolution"], "maybe_clear")
        self.assertEqual(h["location_id"], 4)
        self.assertEqual(self.fa.state, "CRUISING")

    def test_veto_prone_speed_capped_at_10(self):
        self._arm_veto_prone()
        self.fa._maybe_start_veto_prone_probe(300.0)
        for t in (300.5, 301.5, 303.5):
            self.fa._handle_hypothesis_tick(t)
        forwards = [i for i in self._intents() if i["action"].get("direction") == "forward"]
        self.assertTrue(forwards)
        self.assertTrue(all(i["action"]["speed"] <= 10 for i in forwards))

    def test_veto_prone_still_blocked_on_veto(self):
        self._arm_veto_prone()
        self.fa._maybe_start_veto_prone_probe(300.0)
        self.fa._handle_hypothesis_tick(300.3)     # creeping
        self.fa.veto_events.append(300.4)          # daemon flags it
        self.fa._handle_hypothesis_tick(300.6)
        h = self._hyps()[-1]
        self.assertEqual(h["resolution"], "still_blocked")
        self.assertEqual(self.fa.state, "EVADING")

    def test_veto_prone_no_location_is_noop(self):
        self.fa.current_location = None
        self.assertFalse(self.fa._maybe_start_veto_prone_probe(300.0))

    def test_probe_never_touches_safety_socket(self):
        # Every motion the probes emit rides picarx/intent/move (vetoable),
        # never a direct hardware/safety channel.
        self._arm_veto_prone()
        self.fa._maybe_start_veto_prone_probe(300.0)
        for t in (300.5, 301.5, 303.5):
            self.fa._handle_hypothesis_tick(t)
        motion_topics = {t for (t, _) in self.bus.published if "intent" in t or "move" in t}
        self.assertTrue(motion_topics <= {"picarx/intent/move"})


if __name__ == "__main__":
    unittest.main()
