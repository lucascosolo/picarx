import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

import health_daemon as hd  # noqa: E402


class VitalHelpersTest(unittest.TestCase):
    def test_battery_percent_range(self):
        self.assertEqual(hd.battery_percent(hd.BATT_FULL_V), 100)
        self.assertEqual(hd.battery_percent(hd.BATT_EMPTY_V), 0)
        self.assertEqual(hd.battery_percent(9.9), 100)   # clamps high
        self.assertEqual(hd.battery_percent(4.0), 0)     # clamps low
        self.assertIsNone(hd.battery_percent(None))
        mid = hd.battery_percent((hd.BATT_FULL_V + hd.BATT_EMPTY_V) / 2)
        self.assertTrue(45 <= mid <= 55)

    def test_read_cpu_temp(self):
        p = os.path.join(tempfile.mkdtemp(), "temp")
        with open(p, "w") as f:
            f.write("48123\n")
        self.assertEqual(hd.read_cpu_temp_c(p), 48.1)
        self.assertIsNone(hd.read_cpu_temp_c("/no/such/thermal"))

    def test_read_disk(self):
        free_gb, used_pct = hd.read_disk("/")
        self.assertIsInstance(free_gb, float)
        self.assertTrue(0 <= used_pct <= 100)

    def test_summarize(self):
        s = hd.summarize({"battery_v": 7.4, "battery_pct": 58, "temp_c": 52.0,
                          "disk_free_gb": 12.3, "low_power": False})
        self.assertIn("battery 7.4 volts", s.lower())
        self.assertIn("58 percent", s)
        self.assertIn("52 degrees", s)
        self.assertIn("low", hd.summarize({"low_power": True}).lower())
        self.assertIn("don't have", hd.summarize(None))


class LowPowerStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.d = hd.HealthDaemon()   # Bus() -> FakeBus

    def _low_power_msgs(self):
        return self.d.bus.of(hd.LOW_POWER_TOPIC)

    def test_enters_low_power_below_threshold(self):
        self.d.on_world_state({"battery": {"voltage": hd.LOW_BATTERY_V - 0.1}})
        self.assertTrue(self.d.low_power)
        self.assertEqual(self._low_power_msgs()[-1]["active"], True)
        self.assertTrue(any("low" in s["text"].lower()
                            for s in self.d.bus.of(hd.SPEAK_TOPIC)))

    def test_hysteresis_stays_low_until_recovered(self):
        self.d.on_world_state({"battery": {"voltage": 6.4}})     # enter
        self.assertTrue(self.d.low_power)
        self.d.on_world_state({"battery": {"voltage": 6.8}})     # between thresholds
        self.assertTrue(self.d.low_power)                        # still low (hysteresis)
        self.d.on_world_state({"battery": {"voltage": 7.1}})     # above recover
        self.assertFalse(self.d.low_power)

    def test_healthy_battery_stays_normal(self):
        self.d.on_world_state({"battery": {"voltage": 7.8}})
        self.assertFalse(self.d.low_power)
        self.assertEqual(self._low_power_msgs(), [])

    def test_critical_flag_forces_low_power(self):
        self.d.on_world_state({"battery": {"voltage": 7.8, "critical": True}})
        self.assertTrue(self.d.low_power)   # low despite healthy voltage

    def test_manual_request_enters_low_power(self):
        # Unknown battery: manual latch drives low power on its own.
        self.d.on_lowpower_request({"active": True})
        self.assertTrue(self.d.low_power)
        self.assertEqual(self._low_power_msgs()[-1]["active"], True)

    def test_manual_request_clears(self):
        self.d.on_lowpower_request({"active": True})
        self.assertTrue(self.d.low_power)
        self.d.on_lowpower_request({"active": False})
        self.assertFalse(self.d.low_power)

    def test_manual_latch_autoclears_when_healthy(self):
        self.d.on_lowpower_request({"active": True})
        self.assertTrue(self.d.low_power)
        # A healthy battery reading clears the manual latch and exits low power.
        self.d.on_world_state({"battery": {"voltage": 7.6}})
        self.assertFalse(self.d.low_power)

    def test_zero_volt_glitch_does_not_trip_low_power(self):
        # A healthy reading, then a spurious 0.0V glitch: the glitch must be
        # ignored, not treated as a dead battery.
        self.d.on_world_state({"battery": {"voltage": 7.6}})
        self.assertFalse(self.d.low_power)
        self.d.on_world_state({"battery": {"voltage": 0.0}})
        self.assertFalse(self.d.low_power)
        self.assertEqual(self._low_power_msgs(), [])   # no transition announced

    def test_glitch_keeps_last_good_voltage(self):
        self.d.on_world_state({"battery": {"voltage": 7.4}})
        self.d.on_world_state({"battery": {"voltage": 0.0}})     # glitch
        self.assertEqual(self.d.battery_v, 7.4)                  # unchanged
        self.assertEqual(self.d._collect()["battery_v"], 7.4)

    def test_glitch_with_critical_flag_is_ignored(self):
        # 'critical' riding along with a 0.0V glitch is computed from the same
        # bad sample - it must not force low power.
        self.d.on_world_state({"battery": {"voltage": 0.0, "critical": True}})
        self.assertFalse(self.d.low_power)
        self.assertFalse(self.d.battery_critical)

    def test_implausible_high_reading_ignored(self):
        self.d.on_world_state({"battery": {"voltage": 7.2}})
        self.d.on_world_state({"battery": {"voltage": 42.0}})    # impossible spike
        self.assertEqual(self.d.battery_v, 7.2)

    def test_plausible_voltage_helper(self):
        self.assertTrue(hd.plausible_voltage(7.4))
        self.assertTrue(hd.plausible_voltage(hd.BATT_EMPTY_V))
        self.assertFalse(hd.plausible_voltage(0.0))
        self.assertFalse(hd.plausible_voltage(1.5))
        self.assertFalse(hd.plausible_voltage(50.0))
        self.assertFalse(hd.plausible_voltage(None))

    def test_genuine_low_still_trips_after_glitch(self):
        # The glitch filter must not swallow a REAL low battery that follows.
        self.d.on_world_state({"battery": {"voltage": 0.0}})     # glitch, ignored
        self.assertFalse(self.d.low_power)
        self.d.on_world_state({"battery": {"voltage": 6.4}})     # real low reading
        self.assertTrue(self.d.low_power)

    def test_collect_shape(self):
        self.d.on_world_state({"battery": {"voltage": 7.4}})
        v = self.d._collect()
        self.assertEqual(set(v) >= {"battery_v", "battery_pct", "temp_c",
                                    "disk_free_gb", "low_power", "ts"}, True)
        self.assertEqual(v["battery_v"], 7.4)


if __name__ == "__main__":
    unittest.main()
