import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

import bluetooth_daemon as bt  # noqa: E402


class BluetoothConfigTest(unittest.TestCase):
    def test_load_missing_writes_template(self):
        path = os.path.join(tempfile.mkdtemp(), "bluetooth.json")
        cfg = bt.load_config(path)
        self.assertIn("devices", cfg)
        self.assertTrue(cfg["auto_failover"])
        self.assertTrue(os.path.exists(path))

    def test_load_merges_partial(self):
        path = os.path.join(tempfile.mkdtemp(), "bluetooth.json")
        with open(path, "w") as f:
            json.dump({"devices": [{"mac": "AA:BB:CC:DD:EE:FF", "name": "Pixel"}]}, f)
        cfg = bt.load_config(path)
        self.assertEqual(cfg["devices"][0]["name"], "Pixel")
        self.assertIn("check_interval", cfg)

    def test_load_corrupt_returns_defaults(self):
        path = os.path.join(tempfile.mkdtemp(), "bluetooth.json")
        with open(path, "w") as f:
            f.write("{nope")
        self.assertEqual(bt.load_config(path)["devices"], bt.DEFAULT_CONFIG["devices"])

    def test_pick_device_by_name_and_default(self):
        cfg = {"devices": [{"mac": "11:11:11:11:11:11", "name": "Home"},
                           {"mac": "22:22:22:22:22:22", "name": "Pixel"}]}
        self.assertEqual(bt.pick_device(cfg)["name"], "Home")
        self.assertEqual(bt.pick_device(cfg, "pixel")["name"], "Pixel")
        self.assertEqual(bt.pick_device(cfg, "22:22:22:22:22:22")["name"], "Pixel")
        self.assertEqual(bt.pick_device(cfg, "nope")["name"], "Home")

    def test_pick_device_none(self):
        self.assertIsNone(bt.pick_device({"devices": []}))

    def test_build_pan_connect_cmd_default_and_template(self):
        self.assertEqual(bt.build_pan_connect_cmd("AA:BB:CC:DD:EE:FF"),
                         ["nmcli", "device", "connect", "AA:BB:CC:DD:EE:FF"])
        self.assertEqual(
            bt.build_pan_connect_cmd("AA:BB", template="bt-network -c {mac} nap"),
            ["bt-network", "-c", "AA:BB", "nap"])


class BluetoothMonitorTest(unittest.TestCase):
    def _daemon(self, reachable, **cfg):
        path = os.path.join(tempfile.mkdtemp(), "bluetooth.json")
        merged = dict(bt.DEFAULT_CONFIG)
        merged.update(cfg)
        with open(path, "w") as f:
            json.dump(merged, f)
        bt.BLUETOOTH_PATH = path
        return bt.BluetoothDaemon(reachable=reachable)

    def test_offline_then_online_announces_recovery(self):
        state = {"up": False}
        d = self._daemon(lambda: state["up"])
        d._check_once(1.0)
        self.assertIs(d.online, False)
        state["up"] = True
        d._check_once(2.0)
        self.assertIs(d.online, True)
        self.assertTrue(any("back online" in s["text"] for s in d.bus.of(bt.SPEAK_TOPIC)))

    def test_auto_failover_after_grace(self):
        calls = []
        d = self._daemon(lambda: False, offline_grace=30, auto_failover=True,
                         devices=[{"mac": "AA:BB:CC:DD:EE:FF", "name": "Pixel"}])
        d.have_tool = True
        d._tether = lambda name=None: calls.append(name) or True
        d._check_once(100.0)
        self.assertEqual(calls, [])
        d._check_once(131.0)
        self.assertEqual(len(calls), 1)

    def test_no_failover_without_tool(self):
        calls = []
        d = self._daemon(lambda: False, offline_grace=1, auto_failover=True,
                         devices=[{"mac": "AA:BB:CC:DD:EE:FF"}])
        d.have_tool = False
        d._tether = lambda name=None: calls.append(name) or True
        d._check_once(100.0)
        d._check_once(200.0)
        self.assertEqual(calls, [])

    def test_tether_without_devices_speaks_help(self):
        d = self._daemon(lambda: True, devices=[])
        self.assertFalse(d._tether())
        self.assertTrue(d.bus.of(bt.SPEAK_TOPIC))

    def test_does_not_touch_wifi_radio(self):
        # The connect command targets Bluetooth PAN, never a wifi interface.
        cmd = bt.build_pan_connect_cmd("AA:BB:CC:DD:EE:FF")
        self.assertNotIn("wifi", cmd)


if __name__ == "__main__":
    unittest.main()
