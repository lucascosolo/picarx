import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import world_state as ws  # noqa: E402


class OverheadApproachTest(unittest.TestCase):
    def setUp(self):
        self.w = ws.WorldState()   # Bus() -> FakeBus

    def _overhead(self):
        return self.w.build_snapshot()["objects"]["overhead"]

    def test_absent_overhead_is_none(self):
        self.w.on_objects({"objects": [], "close_object": False})
        self.assertIsNone(self._overhead())

    def test_overhead_passed_through(self):
        self.w.on_objects({"objects": [], "overhead": {"area_ratio": 0.4, "y_center_frac": 0.3}})
        o = self._overhead()
        self.assertEqual(o["area_ratio"], 0.4)
        self.assertIn("approaching", o)

    def test_growing_overhead_flagged_approaching(self):
        # Two samples where the mass grows fast enough to clear the
        # approach-rate bar between them.
        base = 1000.0
        self.w.on_objects({"objects": [], "overhead": {"area_ratio": 0.30, "y_center_frac": 0.3}})
        # Force a known previous timestamp so the rate is deterministic.
        self.w._overhead_prev["_ts"] = base
        # +0.30 area over 1s = 0.30/s, well above APPROACH_RATE_THRESHOLD.
        import time
        real_time = time.time
        time.time = lambda: base + 1.0
        try:
            self.w.on_objects({"objects": [], "overhead": {"area_ratio": 0.60, "y_center_frac": 0.3}})
        finally:
            time.time = real_time
        self.assertTrue(self._overhead()["approaching"])

    def test_static_overhead_not_approaching(self):
        self.w.on_objects({"objects": [], "overhead": {"area_ratio": 0.40, "y_center_frac": 0.3}})
        self.w.on_objects({"objects": [], "overhead": {"area_ratio": 0.40, "y_center_frac": 0.3}})
        self.assertFalse(self._overhead()["approaching"])

    def test_overhead_clears(self):
        self.w.on_objects({"objects": [], "overhead": {"area_ratio": 0.40, "y_center_frac": 0.3}})
        self.w.on_objects({"objects": [], "overhead": None})
        self.assertIsNone(self._overhead())
        self.assertIsNone(self.w._overhead_prev)


class PersonFreshnessTest(unittest.TestCase):
    """A recognized identity is held while the person stays visibly in frame,
    even after their face turns away and stops refreshing the name."""
    NOW = 10000.0

    def _fresh(self, age, visible, name="Sam"):
        person = {"name": name, "updated_at": self.NOW - age}
        return ws.WorldState._person_freshness(person, visible, self.NOW)

    def test_recent_identity_is_fresh(self):
        self.assertEqual(self._fresh(5, True), (False, False))   # not stale, not held

    def test_no_identity_yet_is_stale(self):
        self.assertEqual(
            ws.WorldState._person_freshness({"name": None, "updated_at": None},
                                            True, self.NOW), (True, False))

    def test_aged_but_visible_is_held(self):
        # 20s since the last face read, but a person is still in frame.
        self.assertEqual(self._fresh(20, True), (False, True))   # held

    def test_aged_and_gone_is_stale(self):
        # No person visible -> they actually left; drop the identity.
        self.assertEqual(self._fresh(20, False), (True, False))

    def test_hold_expires_after_window(self):
        self.assertEqual(self._fresh(ws.PERSON_HOLD_SEC + 5, True), (True, False))

    def test_no_name_is_never_held(self):
        self.assertEqual(self._fresh(20, True, name=None), (True, False))


class PersonHoldIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.w = ws.WorldState()

    def _person(self):
        return self.w.build_snapshot()["person"]

    def _age_identity(self, seconds):
        with self.w.lock:
            self.w.state["person"]["updated_at"] -= seconds

    def test_name_held_while_person_object_in_view(self):
        self.w.on_person({"name": "Sam", "confidence": 0.9})
        self.w.on_objects({"objects": [{"id": "object_1", "label": "person",
                                        "area_ratio": 0.3, "center_offset": 0}]})
        self._age_identity(20)   # face stopped refreshing 20s ago
        p = self._person()
        self.assertEqual(p["name"], "Sam")
        self.assertFalse(p["stale"])   # held, not forgotten
        self.assertTrue(p["held"])

    def test_name_dropped_when_person_leaves(self):
        self.w.on_person({"name": "Sam", "confidence": 0.9})
        self.w.on_objects({"objects": [{"id": "object_1", "label": "chair",
                                        "area_ratio": 0.2, "center_offset": 0}]})
        self._age_identity(20)   # aged out, and no person is in view
        p = self._person()
        self.assertTrue(p["stale"])
        self.assertFalse(p["held"])


if __name__ == "__main__":
    unittest.main()
