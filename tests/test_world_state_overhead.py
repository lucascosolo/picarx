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


if __name__ == "__main__":
    unittest.main()
