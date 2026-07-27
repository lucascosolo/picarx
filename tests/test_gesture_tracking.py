import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from gesture_tracking import (AdaptiveFrameScheduler, CpuThermalMonitor,
                              GestureHeadController, GestureTracker, HeadLimits,
                              LatestFrame, hand_target)  # noqa: E402


class GestureTrackingTests(unittest.TestCase):
    def test_head_limits_and_deadzone(self):
        c = GestureHeadController(gain=8, max_step_deg=4)
        self.assertEqual(c.update(160, 120, 320, 240), (0.0, 0.0))
        self.assertEqual(c.update(169, 129, 320, 240), (0.0, 0.0))
        pan, tilt = c.update(319, 239, 320, 240)
        self.assertEqual((pan, tilt), (4.0, 4.0))
        for _ in range(100):
            pan, tilt = c.update(319, 239, 320, 240)
        self.assertEqual((pan, tilt), HeadLimits.clamp(pan, tilt))
        self.assertLessEqual(pan, 35)
        self.assertLessEqual(tilt, 30)

    def test_latest_frame_is_one_slot(self):
        f = LatestFrame()
        f.put("old")
        f.put("new")
        self.assertEqual(f.take()[1], "new")
        self.assertIsNone(f.take())

    def test_scheduler_starts_at_third_frame_and_throttles_after_hold(self):
        samples = iter([90.0, 95.0, 96.0, 97.0, 80.0, 70.0])
        monitor = CpuThermalMonitor(cpu_reader=lambda: next(samples),
                                    temp_reader=lambda: 50.0)
        s = AdaptiveFrameScheduler(monitor=monitor, high_hold_sec=3)
        self.assertFalse(s.should_process(0))
        self.assertFalse(s.should_process(1))
        self.assertTrue(s.should_process(2))
        s.should_process(3)
        s.should_process(4)
        s.should_process(5)
        self.assertGreaterEqual(s.skip_factor, 3)

    def test_thermal_sample_stops_processing(self):
        monitor = CpuThermalMonitor(cpu_reader=lambda: 20, temp_reader=lambda: 85)
        s = AdaptiveFrameScheduler(monitor=monitor)
        self.assertFalse(s.should_process(0))
        self.assertTrue(s.thermal_stop)
        self.assertGreaterEqual(s.skip_factor, 12)

    def test_missing_temperature_sensor_stops_processing(self):
        monitor = CpuThermalMonitor(cpu_reader=lambda: 20, temp_reader=lambda: None)
        s = AdaptiveFrameScheduler(monitor=monitor)
        self.assertFalse(s.should_process(0))
        self.assertTrue(s.thermal_stop)

    def test_hand_target_prefers_index_tip(self):
        points = [types.SimpleNamespace(x=0.5, y=0.5, visibility=1.0)
                  for _ in range(21)]
        points[8] = types.SimpleNamespace(x=0.8, y=0.25, visibility=1.0)
        result = types.SimpleNamespace(
            multi_hand_landmarks=[types.SimpleNamespace(landmark=points)])
        self.assertEqual(hand_target(result, 320, 240), (256.0, 60.0))
        self.assertIsNone(hand_target(types.SimpleNamespace(multi_hand_landmarks=[]), 320, 240))

    def test_hand_target_accepts_tasks_api_landmarks(self):
        points = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        points[8] = types.SimpleNamespace(x=0.25, y=0.75)
        result = types.SimpleNamespace(hand_landmarks=[points])
        self.assertEqual(hand_target(result, 320, 240), (80.0, 180.0))

    def test_hand_loss_recenters_after_short_hold(self):
        tracker = GestureTracker()
        tracker.enabled = True
        tracker.controller.pan, tracker.controller.tilt = 20.0, -10.0
        tracker._last_pose = (20.0, -10.0)
        tracker._last_hand_at = 10.0
        tracker._handle_hand_loss(12.0)
        look = tracker.bus.last("picarx/intent/look")
        self.assertEqual(look["action"], {"direction": "look", "pan": 0.0, "tilt": 0.0})
        self.assertEqual(tracker.controller.pose(), (0.0, 0.0))

    def test_model_import_failure_reports_missing_dependency(self):
        tracker = GestureTracker()
        original_import = __import__

        def missing_mediapipe(name, *args, **kwargs):
            if name == "mediapipe":
                raise ModuleNotFoundError("No module named 'mediapipe'", name="mediapipe")
            return original_import(name, *args, **kwargs)

        import builtins
        builtins.__import__, saved = missing_mediapipe, builtins.__import__
        try:
            self.assertFalse(tracker._load_hands())
        finally:
            builtins.__import__ = saved
        status = tracker.bus.last("picarx/gesture/status")
        self.assertEqual(status["state"], "model_error")
        self.assertIn("mediapipe", status["error"])
        self.assertIn(sys.executable, status["error"])


if __name__ == "__main__":
    unittest.main()
