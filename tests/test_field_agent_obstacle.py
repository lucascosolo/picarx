import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402


def _world(distance=None, distance_stale=True, overhead=None,
           close_object=False, items=None):
    return {
        "distance_cm": distance,
        "distance_stale": distance_stale,
        "objects": {
            "stale": False,
            "items": items or [],
            "close_object": close_object,
            "overhead": overhead,
        },
    }


class VisionObstacleClassifyTest(unittest.TestCase):
    def test_overhead_takes_priority(self):
        snap = _world(overhead={"area_ratio": 0.4, "approaching": True},
                      close_object=True)
        obs = field_agent._vision_obstacle(snap)
        self.assertTrue(obs["overhead"])
        self.assertTrue(obs["approaching"])

    def test_close_object_not_overhead(self):
        obs = field_agent._vision_obstacle(_world(close_object=True))
        self.assertFalse(obs["overhead"])

    def test_stale_objects_return_none(self):
        snap = _world(overhead={"area_ratio": 0.9})
        snap["objects"]["stale"] = True
        self.assertIsNone(field_agent._vision_obstacle(snap))

    def test_nothing_returns_none(self):
        self.assertIsNone(field_agent._vision_obstacle(_world()))


class OverheadCrossCheckTest(unittest.TestCase):
    """The core behavioral fix: a clear-LONG ultrasonic reading must not
    dismiss a head-height overhang the way it dismisses a floor-level
    frame-filler."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.bus = self.fa.bus

    def _drive(self, world):
        self.fa.latest_world = world
        self.fa.state = "CRUISING"
        self.fa.last_scan_at = self.fa.start_time  # skip the cruise glance
        self.fa.explore_tick()

    def test_overhead_evades_even_with_clear_ultrasonic_when_looming(self):
        # Ultrasonic reads far (beam under the counter) but the overhang is
        # growing in frame -> evade before the head hits.
        self._drive(_world(distance=120, distance_stale=False,
                           overhead={"area_ratio": 0.5, "approaching": True}))
        self.assertEqual(self.fa.state, "EVADING")

    def test_overhead_evades_when_ultrasonic_not_clear(self):
        # No usable distance at all -> the overhang alone drives evasion.
        self._drive(_world(distance=None, distance_stale=True,
                           overhead={"area_ratio": 0.5, "approaching": False}))
        self.assertEqual(self.fa.state, "EVADING")

    def test_distant_high_wall_not_evaded(self):
        # Overhead-looking mass but ultrasonic reads clearly long AND the mass
        # isn't growing -> a far high wall, not an imminent overhang. Don't
        # evade (the old false-positive the ultrasonic cross-check guards).
        self._drive(_world(distance=200, distance_stale=False,
                           overhead={"area_ratio": 0.5, "approaching": False}))
        self.assertNotEqual(self.fa.state, "EVADING")

    def test_floor_frame_filler_still_dismissed_by_clear_ultrasonic(self):
        # Unchanged legacy behavior: a non-overhead close_object with a clear
        # long reading is the room across the way, not an obstacle.
        self._drive(_world(distance=200, distance_stale=False, close_object=True))
        self.assertNotEqual(self.fa.state, "EVADING")

    def test_floor_frame_filler_evaded_when_ultrasonic_not_clear(self):
        self._drive(_world(distance=None, distance_stale=True, close_object=True))
        self.assertEqual(self.fa.state, "EVADING")


if __name__ == "__main__":
    unittest.main()
