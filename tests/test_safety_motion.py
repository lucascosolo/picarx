"""Safety daemon motion smoothing: the time-based ramp and the
skip-redundant-writes apply path. The veto logic (obstacle/cliff/reverse)
is intentionally NOT exercised here - these tests only cover the motion
executor, and must not be taken to relax those safety checks.

The daemon imports picarx/robot_hat at load, so we stub them (no hardware
off-robot) before importing it."""
import os
import sys
import types
import unittest

for _name in ("picarx", "robot_hat"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["picarx"].Picarx = type(
    "Picarx", (), {"__init__": lambda self, *a, **k: None})
sys.modules["robot_hat"].ADC = type(
    "ADC", (), {"__init__": lambda self, *a, **k: None, "read": lambda self: 0})

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "safety"))
import safety_daemon  # noqa: E402


class FakePx:
    def __init__(self):
        self.calls = []

    def forward(self, s):
        self.calls.append(("forward", s))

    def backward(self, s):
        self.calls.append(("backward", s))

    def stop(self):
        self.calls.append(("stop",))

    def set_dir_servo_angle(self, a):
        self.calls.append(("angle", a))


class RampTowardTest(unittest.TestCase):
    def test_advance_is_bounded_by_rate_times_dt(self):
        self.assertEqual(safety_daemon.ramp_toward(0, 100, 100, 0.1), 10)

    def test_lands_exactly_on_target(self):
        self.assertEqual(safety_daemon.ramp_toward(95, 100, 100, 1.0), 100)

    def test_ramps_down_too(self):
        self.assertEqual(safety_daemon.ramp_toward(50, 0, 100, 0.1), 40)

    def test_stable_at_target(self):
        self.assertEqual(safety_daemon.ramp_toward(25, 25, 100, 0.1), 25)


class MotionSmootherTest(unittest.TestCase):
    def _sm(self):
        px = FakePx()
        return safety_daemon.MotionSmoother(px), px

    def test_steady_command_writes_once(self):
        sm, px = self._sm()
        sm._apply(25, 0)
        sm._apply(25, 0)  # unchanged - motor holds, don't re-send
        self.assertEqual([c for c in px.calls if c[0] == "forward"],
                         [("forward", 25)])

    def test_zero_crossing_switches_drive_direction(self):
        sm, px = self._sm()
        sm._apply(20, 0)
        sm._apply(-30, 0)
        kinds = [c[0] for c in px.calls]
        self.assertIn("forward", kinds)
        self.assertIn("backward", kinds)

    def test_subdegree_servo_jitter_is_skipped(self):
        sm, px = self._sm()
        sm._apply(0, 0.3)     # first write always lands
        px.calls.clear()
        sm._apply(0, 0.6)     # +0.3 deg < epsilon -> skipped
        self.assertEqual(px.calls, [])

    def test_reached_target_always_writes_even_if_tiny_step(self):
        sm, px = self._sm()
        sm.target_angle = 10.0
        sm._apply(0, 9.8)     # establishes applied_angle = 9.8
        px.calls.clear()
        sm._apply(0, 10.0)    # +0.2 < epsilon, but it's the target -> write
        self.assertIn(("angle", 10.0), px.calls)

    def test_reversal_snaps_through_zero(self):
        sm, _ = self._sm()
        sm.current_speed = 20.0
        sm.update_targets(speed=-30)
        self.assertEqual(sm.current_speed, 0.0)
        self.assertEqual(sm.target_speed, -30.0)

    def test_tick_ramps_toward_target_in_real_time(self):
        sm, _ = self._sm()
        sm.update_targets(speed=100, angle=0)
        sm._tick(0.1)         # 100 units/s * 0.1s = 10
        self.assertEqual(sm.current_speed, 10.0)

    def test_emergency_stop_is_immediate(self):
        sm, px = self._sm()
        sm.current_speed = 40.0
        sm.emergency_stop()
        self.assertEqual(sm.current_speed, 0.0)
        self.assertEqual(sm.target_speed, 0.0)
        self.assertIn(("stop",), px.calls)


if __name__ == "__main__":
    unittest.main()
