import os
import sys
import time
import threading
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
import gesture_tracking as gt  # noqa: E402

from gesture_tracking import (AdaptiveFrameScheduler, CpuThermalMonitor,
                              GestureHeadController, GestureTracker, HeadLimits,
                              LatestFrame, hand_bbox, hand_target)  # noqa: E402


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

    def test_hand_bbox_accepts_tasks_api_landmarks(self):
        points = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        points[0] = types.SimpleNamespace(x=0.2, y=0.25)
        points[20] = types.SimpleNamespace(x=0.8, y=0.75)
        result = types.SimpleNamespace(hand_landmarks=[points])
        x, y, w, h = hand_bbox(result, 320, 240)
        self.assertLess(x, 64.0)
        self.assertLess(y, 60.0)
        self.assertGreater(w, 192.0)
        self.assertGreater(h, 120.0)

    def test_hand_helpers_accept_worker_landmark_payload(self):
        points = [{"x": 0.5, "y": 0.5} for _ in range(21)]
        points[0] = {"x": 0.2, "y": 0.25}
        points[8] = {"x": 0.75, "y": 0.25}
        points[20] = {"x": 0.8, "y": 0.75}
        result = {"hand_landmarks": [points]}
        self.assertEqual(hand_target(result, 320, 240), (240.0, 60.0))
        self.assertIsNotNone(hand_bbox(result, 320, 240))

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

    def test_enabled_tracker_renews_lease_when_preempted(self):
        tracker = GestureTracker()
        tracker.enabled = True
        tracker.on_state({"state": "SPEAKING"})
        claim = tracker.bus.last("picarx/state/claim")
        self.assertEqual(claim["owner"], "gesture_tracking")
        self.assertEqual(claim["state"], "GESTURE_TRACKING")

    def test_lease_heartbeat_survives_slow_model_startup(self):
        tracker = GestureTracker()
        tracker.enabled = True
        tracker._start_claim_loop()
        try:
            time.sleep(0.65)
        finally:
            tracker._stop_claim_loop()
        claims = tracker.bus.of("picarx/state/claim")
        self.assertGreaterEqual(len(claims), 2)
        self.assertTrue(all(claim["ttl"] == 3.0 for claim in claims))

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

    def test_async_model_loader_reports_backend_and_runtime_diagnostics(self):
        tracker = GestureTracker()
        tracker.enabled = True
        fake_mp = types.SimpleNamespace(
            __file__="/opt/picarx/venv/lib/python3.13/site-packages/mediapipe/__init__.py",
            __version__="0.10.test")
        fake_hands = types.SimpleNamespace()
        diagnostics = tracker._model_runtime_diagnostics(fake_mp, "tasks")
        tracker._create_hands = lambda progress=None: (
            fake_hands, fake_mp, "tasks", diagnostics)
        tracker._start_model_load(frame_age=0.42)
        thread = tracker._model_load_thread
        thread.join(1.0)
        self.assertIs(tracker._hands, fake_hands)
        self.assertEqual(tracker._hands_backend, "tasks")
        status = tracker.bus.last("picarx/gesture/status")
        self.assertEqual(status["state"], "model_ready")
        self.assertEqual(status["backend"], "tasks")
        self.assertEqual(status["mediapipe_version"], "0.10.test")
        self.assertEqual(status["interpreter"], sys.executable)
        self.assertEqual(status["frame_age_sec"], 0.42)

    def test_tasks_model_progress_merges_backend_diagnostics_once(self):
        # Regression for the production failure where the explicit backend
        # keyword collided with the same key returned by diagnostics.
        tracker = GestureTracker()
        tracker._ensure_hand_model = lambda progress=None: True
        mp = types.ModuleType("mediapipe")
        mp.__version__ = "test"
        tasks = types.ModuleType("mediapipe.tasks")
        tasks_python = types.ModuleType("mediapipe.tasks.python")
        vision = types.ModuleType("mediapipe.tasks.python.vision")

        class _Options:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _Landmarker:
            @classmethod
            def create_from_options(cls, options):
                return types.SimpleNamespace()

        tasks_python.BaseOptions = lambda model_asset_path: model_asset_path
        vision.HandLandmarkerOptions = _Options
        vision.RunningMode = types.SimpleNamespace(IMAGE="image")
        vision.HandLandmarker = _Landmarker
        tasks.python = tasks_python
        tasks_python.vision = vision
        mp.tasks = tasks
        names = ("mediapipe", "mediapipe.tasks", "mediapipe.tasks.python",
                 "mediapipe.tasks.python.vision")
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update({"mediapipe": mp, "mediapipe.tasks": tasks,
                            "mediapipe.tasks.python": tasks_python,
                            "mediapipe.tasks.python.vision": vision})
        phases = []
        try:
            result = tracker._create_hands(
                progress=lambda phase, **fields: phases.append((phase, fields)) or True)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.assertEqual(result[2], "tasks")
        constructing = dict(phases)["constructing"]
        self.assertEqual(constructing["backend"], "tasks")

    def test_model_load_timeout_reports_cleanup_and_ignores_late_result(self):
        original_timeout = gt.MODEL_LOAD_TIMEOUT_SEC
        gate = threading.Event()
        try:
            gt.MODEL_LOAD_TIMEOUT_SEC = 0.01
            tracker = GestureTracker()
            tracker.enabled = True
            fake_hands = types.SimpleNamespace(closed=False,
                                               close=lambda: setattr(
                                                   fake_hands, "closed", True))
            fake_mp = types.SimpleNamespace(__version__="late")
            tracker._create_hands = lambda progress=None: (
                gate.wait(1.0),
                (fake_hands, fake_mp, "tasks",
                 tracker._model_runtime_diagnostics(fake_mp, "tasks")))[1]
            tracker._start_model_load(frame_age=0.2)
            time.sleep(0.03)
            tracker._check_model_load_timeout()
            status = tracker.bus.last("picarx/gesture/status")
            self.assertEqual(status["state"], "model_error")
            self.assertTrue(status["timeout"])
            self.assertEqual(status["cleanup"], {
                "camera_released": True, "state_released": True})
            self.assertTrue(any(p.get("owner") == "gesture_tracking"
                                for p in tracker.bus.of("picarx/state/release")))
            gate.set()
            tracker._model_load_thread.join(1.0)
            self.assertIs(tracker._hands, False)
            self.assertTrue(fake_hands.closed)
        finally:
            gt.MODEL_LOAD_TIMEOUT_SEC = original_timeout

    def test_native_worker_crash_becomes_model_error_and_releases_resources(self):
        tracker = GestureTracker()
        tracker.enabled = True

        class DeadProcess:
            exitcode = -4

            @staticmethod
            def is_alive():
                return False

        class EmptyConnection:
            @staticmethod
            def poll(*args):
                return False

        tracker._hands = True
        tracker._hand_worker_process = DeadProcess()
        tracker._hand_worker_conn = EmptyConnection()
        self.assertIsNone(tracker._worker_infer(
            types.SimpleNamespace(shape=(240, 320, 3)), 1.0, 320, 240))
        status = tracker.bus.last("picarx/gesture/status")
        self.assertEqual(status["state"], "model_error")
        self.assertIn("exited during inference", status["error"])
        self.assertEqual(status["cleanup"], {
            "camera_released": True, "state_released": True})


if __name__ == "__main__":
    unittest.main()
