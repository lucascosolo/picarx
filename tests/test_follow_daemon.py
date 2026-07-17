import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

import follow_daemon as fd  # noqa: E402


class FollowControlMathTest(unittest.TestCase):
    def test_steer_centered_is_zero(self):
        self.assertEqual(fd.steer_angle(0, 640), 0)
        self.assertEqual(fd.steer_angle(fd.STEER_DEADBAND, 640), 0)  # within deadband

    def test_steer_right_positive_left_negative(self):
        self.assertGreater(fd.steer_angle(200, 640), 0)
        self.assertLess(fd.steer_angle(-200, 640), 0)

    def test_steer_clamped_to_limit(self):
        self.assertEqual(fd.steer_angle(100000, 640), fd.MAX_STEER_ANGLE)
        self.assertEqual(fd.steer_angle(-100000, 640), -fd.MAX_STEER_ANGLE)

    def test_steer_no_framewidth_is_zero(self):
        self.assertEqual(fd.steer_angle(200, None), 0)

    def test_drive_stops_when_close(self):
        d, s = fd.drive_decision(fd.STOP_AREA_RATIO + 0.1)
        self.assertEqual((d, s), ("stop", 0))

    def test_drive_forward_when_far(self):
        d, s = fd.drive_decision(0.05)
        self.assertEqual(d, "forward")
        self.assertLessEqual(s, fd.FOLLOW_SPEED)
        self.assertGreater(s, 0)

    def test_drive_forward_when_no_area(self):
        self.assertEqual(fd.drive_decision(None)[0], "forward")

    def test_pick_person_largest(self):
        payload = {"objects": [
            {"label": "chair", "area_ratio": 0.9, "center_offset": 0},
            {"label": "person", "area_ratio": 0.1, "center_offset": -50},
            {"label": "person", "area_ratio": 0.3, "center_offset": 80},
        ]}
        p = fd.pick_person(payload)
        self.assertEqual(p["center_offset"], 80)  # the bigger person box

    def test_pick_person_none(self):
        self.assertIsNone(fd.pick_person({"objects": [{"label": "sofa"}]}))
        self.assertIsNone(fd.pick_person({}))


class FollowDaemonBehaviourTest(unittest.TestCase):
    def setUp(self):
        self.d = fd.FollowDaemon()  # Bus() -> FakeBus

    def _intents(self):
        return [p for p in self.d.bus.of(fd.INTENT_TOPIC)]

    def test_disabled_daemon_emits_nothing(self):
        self.d.on_objects({"objects": [{"label": "person", "area_ratio": 0.1,
                                        "center_offset": 0, "frame_width": 640}]})
        self.d._tick(time.time()) if self.d.enabled else None
        self.assertEqual(self._intents(), [])

    def test_enable_routes_motion_through_intent_topic_only(self):
        self.d.on_control({"enabled": True})
        now = time.time()
        self.d.on_objects({"objects": [{"label": "person", "area_ratio": 0.1,
                                        "center_offset": 5, "frame_width": 640}]})
        for _ in range(4):
            self.d._tick(now)
        intents = self._intents()
        self.assertTrue(intents)
        # Safety: every command is a normal vetoable intent, never a direct
        # safety-socket write, and every one is tagged as ours.
        for i in intents:
            self.assertEqual(i["source"], fd.SOURCE_NAME)
            self.assertIn("action", i)
            self.assertLessEqual(i.get("priority"), 7)

    def test_close_person_stops(self):
        self.d.on_control({"enabled": True})
        now = time.time()
        self.d.on_objects({"objects": [{"label": "person",
                                        "area_ratio": fd.STOP_AREA_RATIO + 0.2,
                                        "center_offset": 0, "frame_width": 640}]})
        self.d._tick(now)
        self.assertEqual(self._intents()[-1]["action"], {"direction": "stop"})

    def test_follow_speed_is_bounded(self):
        self.d.on_control({"enabled": True})
        now = time.time()
        self.d.on_objects({"objects": [{"label": "person", "area_ratio": 0.05,
                                        "center_offset": 2, "frame_width": 640}]})
        for _ in range(6):
            self.d._tick(now)
        for i in self._intents():
            if i["action"].get("direction") == "forward":
                self.assertLessEqual(i["action"]["speed"], 20)

    def test_lost_target_holds_then_gives_up(self):
        self.d.on_control({"enabled": True})
        self.d.bus.clear()
        base = 1000.0
        self.d.person = (0, 640, 0.1, base)   # last seen at base
        self.d._tick(base + fd.LOST_HOLD_SEC + 0.1)   # briefly lost -> hold (stop)
        self.assertEqual(self._intents()[-1]["action"], {"direction": "stop"})
        self.d._tick(base + fd.LOST_GIVEUP_SEC + 0.1)  # long lost -> disable
        self.assertFalse(self.d.enabled)

    def test_enable_without_target_waits_instead_of_instant_giveup(self):
        # Regression: enabling follow before the detector has produced a
        # person (its SSD pass runs every ~1.5s) measured "lost" time from
        # epoch 0 and disabled itself with "I lost you" on the first tick.
        self.d.on_control({"enabled": True})
        self.d._tick(time.time())
        self.assertTrue(self.d.enabled)
        # Holds still while waiting to acquire, doesn't drive blind.
        self.assertEqual(self._intents()[-1]["action"], {"direction": "stop"})

    def test_enable_clears_stale_sightings(self):
        # A person box from a previous session must not seed the new one.
        self.d.person = (0, 640, 0.1, 12345.0)
        self.d.face = (0, 640, None, 12345.0)
        self.d.on_control({"enabled": True})
        self.assertIsNone(self.d.person)
        self.assertIsNone(self.d.face)

    def test_spoken_stop_disables(self):
        self.d.on_control({"enabled": True})
        self.assertTrue(self.d.enabled)
        self.d.on_heard({"text": "please stop"})
        self.assertFalse(self.d.enabled)


if __name__ == "__main__":
    unittest.main()
