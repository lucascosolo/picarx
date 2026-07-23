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


class _FakeSensors:
    """px stand-in for is_safe's non-hardware branches: a clear ultrasonic
    reading and non-cliff grayscale, so the forward path returns 'ok' without a
    real HAT."""
    class _Ultrasonic:
        @staticmethod
        def read():
            return 100.0

    ultrasonic = _Ultrasonic()

    @staticmethod
    def get_grayscale_data():
        return [1000, 1000, 1000]


class ReverseBackstopTest(unittest.TestCase):
    """The no-rear-sensor continuous-reverse backstop, and the fix that a
    camera 'look' (pan/tilt only, not a drive command) must NOT re-arm it."""

    def setUp(self):
        self._real_time = safety_daemon.time.time
        self._real_px = safety_daemon.px
        self._now = 1000.0
        safety_daemon.time.time = lambda: self._now
        safety_daemon.px = _FakeSensors()
        safety_daemon._reverse_state["since"] = None

    def tearDown(self):
        safety_daemon.time.time = self._real_time
        safety_daemon.px = self._real_px
        safety_daemon._reverse_state["since"] = None

    def _reverse(self):
        return safety_daemon.is_safe({"direction": "backward"})

    def test_sustained_reverse_is_vetoed_past_the_limit(self):
        self.assertEqual(self._reverse(), (True, "ok"))       # arms the timer
        self._now += safety_daemon.MAX_CONTINUOUS_REVERSE_SEC + 0.1
        safe, reason = self._reverse()
        self.assertFalse(safe)
        self.assertIn("reverse", reason)

    def test_look_does_not_reset_the_reverse_timer(self):
        # A head glance interleaved during a sustained reverse must not re-arm
        # the backstop - a look moves only the camera servos.
        self._reverse()                                       # arms at t=1000
        self._now += 1.0
        self.assertEqual(safety_daemon.is_safe({"direction": "look",
                                                "pan": 20, "tilt": 0}), (True, "ok"))
        self._now += safety_daemon.MAX_CONTINUOUS_REVERSE_SEC - 0.5   # total > limit
        safe, _ = self._reverse()
        self.assertFalse(safe)                                # still bounded

    def test_a_drive_command_does_reset_the_reverse_timer(self):
        # stop/turn/forward ARE real reverse interrupters and clear the run.
        self._reverse()
        self._now += 1.0
        self.assertEqual(safety_daemon.is_safe({"direction": "turn", "angle": 10}),
                         (True, "ok"))
        self.assertIsNone(safety_daemon._reverse_state["since"])
        self._now += safety_daemon.MAX_CONTINUOUS_REVERSE_SEC - 0.5
        self.assertEqual(self._reverse(), (True, "ok"))       # fresh window


class _FakeI2C:
    """robot_hat.I2C stand-in: write([reg,...]) sets the register pointer from
    the first byte; read(n) returns n bytes from a reg->byte map."""
    def __init__(self, regs=None):
        self.regs = dict(regs or {})
        self._ptr = 0
        self.writes = []

    def write(self, data):
        self.writes.append(list(data))
        if data:
            self._ptr = data[0]

    def read(self, n):
        return [self.regs.get(self._ptr + i, 0) for i in range(n)]


class ImuReadTest(unittest.TestCase):
    """The daemon's MPU-6050 register protocol + datasheet scaling (Layer A owns
    the hardware read; Layer B's imu.py just consumes this over the socket)."""

    def test_signed16(self):
        self.assertEqual(safety_daemon._signed16(0x0083), 131)
        self.assertEqual(safety_daemon._signed16(0xC000), -16384)

    def test_words_are_pointer_then_burst_read(self):
        i2c = _FakeI2C({0x3B: 0xC0, 0x3C: 0x00, 0x3F: 0x40, 0x40: 0x00})
        words = safety_daemon._mpu_words(i2c, 0x3B, 3)
        self.assertEqual(words, [-16384, 0, 16384])      # X=-1g, Y=0, Z=+1g raw
        self.assertIn([0x3B], i2c.writes)                # pointer was set first

    def test_scaling_to_physical_units(self):
        out = safety_daemon.imu_from_words(0, 0, 16384, 131, 0, 0, 0)
        self.assertAlmostEqual(out["accel"]["z"], 9.80665, places=3)   # +1g
        self.assertAlmostEqual(out["gyro"]["x"], 1.0, places=3)        # 131 LSB/dps
        self.assertAlmostEqual(out["temp"], 36.53, places=2)

    def test_read_imu_reports_error_when_bus_absent(self):
        # robot_hat is a bare stub here (no I2C), so opening the handle fails ->
        # a fail-soft {"error": ...}, never a raise.
        out = safety_daemon.read_imu(0x68)
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
