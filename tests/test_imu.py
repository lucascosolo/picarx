"""The head-mounted MPU-6050 IMU: calibration zeroes an imperfect mount, the
derived signals (moving / impact / tilt) are frame-independent magnitudes,
head tilt is removed to estimate chassis tilt, brief impacts fire an
edge-triggered event, and a missing chip degrades to silence - never a crash.
All the decision math is pure and driven here with a fake sensor + FakeBus."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import imu  # noqa: E402
import world_state as ws  # noqa: E402


class _FakeSensor:
    """Stands in for mpu6050: returns queued (or fixed) readings; can raise."""
    def __init__(self, accel=(0, 0, 9.8), gyro=(0, 0, 0), temp=25.0, fail=False):
        self.accel, self.gyro, self.temp, self.fail = accel, gyro, temp, fail

    def _d(self, v):
        return {"x": v[0], "y": v[1], "z": v[2]}

    def get_accel_data(self):
        if self.fail:
            raise IOError("i2c NAK")
        return self._d(self.accel)

    def get_gyro_data(self):
        return self._d(self.gyro)

    def get_temperature(self):
        return self.temp


class PureHelperTest(unittest.TestCase):
    def test_magnitude_and_angle(self):
        self.assertAlmostEqual(imu.magnitude((3, 4, 0)), 5.0)
        self.assertAlmostEqual(imu.angle_between_deg((0, 0, 1), (1, 0, 0)), 90.0)
        self.assertEqual(imu.angle_between_deg((0, 0, 0), (1, 0, 0)), 0.0)  # degenerate

    def test_calibrate_captures_rest_and_bias(self):
        calib = imu.calibrate([(0.2, 0, 9.7)] * 4, [(0.5, -0.3, 0.1)] * 4)
        self.assertAlmostEqual(calib["accel_rest"][0], 0.2, places=3)
        self.assertAlmostEqual(calib["g_mag"], imu.magnitude((0.2, 0, 9.7)), places=3)
        self.assertAlmostEqual(calib["gyro_bias"][0], 0.5, places=3)

    def test_body_tilt_removes_head_tilt(self):
        # head tilted down 20deg, measured 22deg from rest -> ~2deg chassis tilt
        self.assertAlmostEqual(imu.body_tilt_deg(22.0, 20.0), 2.0)
        self.assertEqual(imu.body_tilt_deg(15.0, 20.0), 0.0)   # clamped >= 0

    def test_derived_still_vs_moving_vs_impact(self):
        calib = imu.calibrate([(0, 0, 9.8)] * 3, [(0, 0, 0)] * 3)
        still = imu.compute_derived((0, 0, 9.8), (0, 0, 0), calib, 0.0)
        self.assertFalse(still["moving"] or still["impact"] or still["tilted"])
        moving = imu.compute_derived((0.9, 0, 9.8), (0, 0, 40), calib, 0.0)
        self.assertTrue(moving["moving"])
        self.assertGreater(moving["rotation_rate_dps"], 30)
        jolt = imu.compute_derived((0, 0, 20), (0, 0, 0), calib, 0.0)
        self.assertTrue(jolt["impact"])

    def test_derived_tilt_is_head_compensated(self):
        calib = imu.calibrate([(0, 0, 9.8)] * 3, [(0, 0, 0)] * 3)
        # gravity swung 30deg but the head was commanded 30deg down -> not tilted
        rotated = (9.8 * 0.5, 0, 9.8 * 0.866)   # ~30deg from vertical
        d = imu.compute_derived(rotated, (0, 0, 0), calib, head_tilt_cmd=30.0)
        self.assertGreater(d["tilt_from_rest_deg"], 25)
        self.assertLess(d["body_tilt_deg"], 10)
        self.assertFalse(d["tilted"])

    def test_detect_event_rising_edge_only(self):
        self.assertEqual(imu.detect_event({"impact": False}, {"impact": True}), "impact")
        self.assertIsNone(imu.detect_event({"impact": True}, {"impact": True}))  # held
        self.assertEqual(imu.detect_event({"tilted": False}, {"tilted": True}), "tilted")


class IMUModuleTest(unittest.TestCase):
    def _imu(self, sensor):
        m = imu.IMU(sensor=sensor)
        self.assertTrue(m.calibrate_at_rest(samples=3, delay=0))
        m.bus.clear()
        return m

    def test_head_pose_tracked_from_look(self):
        m = imu.IMU(sensor=_FakeSensor())
        m.on_look({"action": {"direction": "look", "pan": 40, "tilt": -15}})
        self.assertEqual((m.head_pan, m.head_tilt), (40.0, -15.0))
        m.on_look({"action": {"direction": "stop"}})     # not a look -> ignored
        self.assertEqual(m.head_tilt, -15.0)

    def test_publish_carries_derived_signals(self):
        m = self._imu(_FakeSensor(accel=(0, 0, 9.8)))
        m._publish_reading((0.9, 0, 9.8), (0, 0, 40), 26.0)
        msg = m.bus.last("picarx/sensors/imu")
        self.assertTrue(msg["calibrated"])
        self.assertTrue(msg["moving"])
        self.assertIn("rotation_rate_dps", msg)
        self.assertEqual(msg["head_pose"], {"pan": 0.0, "tilt": 0.0})

    def test_impact_fires_one_throttled_event(self):
        m = self._imu(_FakeSensor())
        m._publish_reading((0, 0, 9.8), (0, 0, 0), 25.0)   # calm
        m._publish_reading((0, 0, 22), (0, 0, 0), 25.0)    # jolt -> event
        m._publish_reading((0, 0, 22), (0, 0, 0), 25.0)    # still high, no re-fire
        events = m.bus.of("picarx/sensors/imu/event")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "impact")

    def test_no_publish_before_calibration(self):
        m = imu.IMU(sensor=_FakeSensor())
        m._publish_reading((0, 0, 9.8), (0, 0, 0), 25.0)   # calib is None
        self.assertIsNone(m.bus.last("picarx/sensors/imu"))

    def test_read_failure_is_soft(self):
        m = imu.IMU(sensor=_FakeSensor(fail=True))
        self.assertIsNone(m._read())                       # no raise
        self.assertFalse(m.calibrate_at_rest(samples=2, delay=0))  # no samples

    def test_missing_chip_reports_reason_and_beacons_status(self):
        m = imu.IMU()                                      # no injected sensor
        ok, reason = m._open_sensor()                      # mpu6050 not installed here
        self.assertFalse(ok)
        self.assertIn("mpu6050", reason.lower())
        m._publish_status(False, reason)
        status = m.bus.last("picarx/sensors/imu/status")
        self.assertFalse(status["available"])
        self.assertEqual(status["reason"], reason)


class WorldStateImuTest(unittest.TestCase):
    def test_imu_folds_into_snapshot_with_staleness(self):
        w = ws.WorldState()
        # absent until seen -> present as stale
        self.assertTrue(w.build_snapshot()["imu"]["stale"])
        w.on_imu({"moving": True, "impact": False, "body_tilt_deg": 3.0})
        snap = w.build_snapshot()
        self.assertFalse(snap["imu"]["stale"])
        self.assertTrue(snap["imu"]["moving"])

    def test_imu_goes_stale_after_threshold(self):
        w = ws.WorldState()
        w.on_imu({"moving": True})
        # force the timestamp well past STALE_AFTER["imu"]
        w.state["imu"]["updated_at"] = 1.0
        self.assertTrue(w.build_snapshot()["imu"]["stale"])


if __name__ == "__main__":
    unittest.main()
