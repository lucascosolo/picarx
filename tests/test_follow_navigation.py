"""Regression tests for the 'follow me -> hard half-circle' failure.

Root causes covered here:
  1. The safety daemon's MotionSmoother holds the last steering angle
     FOREVER (the arbiter's fallback stop only zeroes speed), and
     follow_daemon assumed the wheels were straight at enable time - a
     stale angle plus a centered person meant zero correction demanded,
     so the robot arced on whatever the previous behaviour left behind.
  2. The proportional steering law could jump straight to full lock,
     swinging the body-fixed camera off the target.
  3. Nothing recentered the camera head, whose pan the follow geometry
     silently assumes to be zero.
  4. explore_mode kept running underneath follow, publishing competing
     headings and able to outrank follow via its EVADE/COACH priorities.
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

import field_agent  # noqa: E402
import follow_daemon as fd  # noqa: E402

T0 = 2000.0
DT = 1.0 / fd.CONTROL_HZ


def _person(offset, area=0.1, frame_w=640):
    return {"objects": [{"label": "person", "area_ratio": area,
                         "center_offset": offset, "frame_width": frame_w}]}


class StraightenOnEntryTest(unittest.TestCase):
    def setUp(self):
        self.d = fd.FollowDaemon()
        self.d.on_control({"enabled": True})

    def _actions(self):
        return [p["action"] for p in self.d.bus.of(fd.INTENT_TOPIC)]

    def test_first_drive_tick_zeros_the_wheels(self):
        # THE half-circle regression: wheels must be explicitly zeroed
        # before the first drive, never assumed straight.
        self.d.on_objects(_person(offset=0))
        self.d._tick(T0)
        self.assertEqual(self._actions()[-1], {"direction": "turn", "angle": 0})
        self.d._tick(T0 + DT)
        self.assertEqual(self._actions()[-1]["direction"], "forward")

    def test_enable_recenters_camera_head(self):
        looks = self.d.bus.of("picarx/intent/look")
        self.assertTrue(looks)
        self.assertEqual(looks[-1]["action"],
                         {"direction": "look", "pan": 0, "tilt": 0})

    def test_straighten_skipped_when_close_person_stops(self):
        # Stop must not be delayed by the straighten (turned wheels while
        # stationary are harmless).
        self.d.on_objects(_person(offset=0, area=fd.STOP_AREA_RATIO + 0.1))
        self.d._tick(T0)
        self.assertEqual(self._actions()[-1], {"direction": "stop"})


class SlewedSteeringTest(unittest.TestCase):
    def setUp(self):
        self.d = fd.FollowDaemon()
        self.d.on_control({"enabled": True})
        self.d.bus.clear()

    def _run(self, offset, ticks):
        for i in range(ticks):
            self.d.on_objects(_person(offset=offset))
            # keep the sighting fresh relative to simulated time
            self.d.person = self.d.person[:3] + (T0 + i * DT,)
            self.d._tick(T0 + i * DT)
        return [p["action"] for p in self.d.bus.of(fd.INTENT_TOPIC)]

    def test_edge_person_never_gets_instant_full_lock(self):
        actions = self._run(offset=310, ticks=12)   # right at the frame edge
        turns = [a["angle"] for a in actions
                 if a["direction"] == "turn" and a["angle"] != 0]
        self.assertTrue(turns)
        max_step = fd.FOLLOW_STEER_RATE * DT
        self.assertLessEqual(abs(turns[0]), max_step + 1e-6)   # ramps, no jump
        prev = 0.0
        for angle in turns:
            self.assertLessEqual(abs(angle - prev),
                                 fd.FOLLOW_STEER_RATE * DT * 2 + 1e-6)
            prev = angle
        # ...and it converges toward the proportional target over time.
        self.assertGreater(turns[-1], turns[0])
        self.assertGreater(turns[-1], 15)

    def test_steers_toward_person_sign(self):
        right = self._run(offset=200, ticks=6)
        self.assertTrue(any(a["direction"] == "turn" and a["angle"] > 0
                            for a in right))

    def test_turn_and_forward_alternate(self):
        actions = self._run(offset=310, ticks=12)
        kinds = [a["direction"] for a in actions]
        self.assertIn("forward", kinds)
        for a, b in zip(kinds, kinds[1:]):
            self.assertFalse(a == b == "turn", f"consecutive turns in {kinds}")


class DriverHandoffTest(unittest.TestCase):
    """One driver at a time: follow pauses explore, explore/go-to stop follow."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_follow_enable_pauses_exploration(self):
        self.fa.explore_mode = True
        self.fa.on_follow_state({"enabled": True})
        self.assertFalse(self.fa.explore_mode)
        self.assertTrue(self.fa.bus.of("picarx/intent/cancel"))

    def test_follow_disable_does_not_resume_exploration(self):
        self.fa.explore_mode = False
        self.fa.on_follow_state({"enabled": False})
        self.assertFalse(self.fa.explore_mode)

    def test_explore_command_stops_following(self):
        self.fa.handle_voice_command("explore")
        msg = self.fa.bus.last("picarx/tools/follow/set")
        self.assertEqual(msg, {"enabled": False})
        self.assertTrue(self.fa.explore_mode)

    def test_scan_entry_arms_wheel_straighten(self):
        # Every scan exit passes through the timed steering reset, so
        # cruising never resumes on a stale wheel angle.
        now = time.time()
        self.fa._enter_scanning(now, startup=True)
        self.assertEqual(self.fa.steering_active_until, now)

    def test_timed_reset_syncs_controller_model(self):
        self.fa.state = "CRUISING"
        self.fa.last_scan_at = time.time()
        self.fa.last_wander = time.time()
        if self.fa.steering is not None:
            self.fa.steering._angle = -15.0
        self.fa.steering_active_until = time.time() - 0.1
        self.fa.latest_world = {"distance_cm": 100, "distance_stale": False,
                                "objects": {"stale": False, "items": [],
                                            "close_object": False,
                                            "overhead": None}}
        self.fa.explore_tick()
        actions = [p["action"] for p in self.fa.bus.of("picarx/intent/move")]
        self.assertIn({"direction": "turn", "angle": 0}, actions)
        if self.fa.steering is not None:
            self.assertEqual(self.fa.steering._angle, 0.0)


if __name__ == "__main__":
    unittest.main()
