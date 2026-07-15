import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

import network_daemon as nd  # noqa: E402


class NetworkConfigTest(unittest.TestCase):
    def test_load_missing_writes_template_and_returns_defaults(self):
        path = os.path.join(tempfile.mkdtemp(), "networks.json")
        cfg = nd.load_config(path)
        self.assertIn("hotspots", cfg)
        self.assertTrue(cfg["auto_failover"])
        self.assertTrue(os.path.exists(path))  # template written for easy editing

    def test_load_merges_partial_config(self):
        path = os.path.join(tempfile.mkdtemp(), "networks.json")
        with open(path, "w") as f:
            json.dump({"hotspots": [{"ssid": "Pixel", "password": "pw"}]}, f)
        cfg = nd.load_config(path)
        self.assertEqual(cfg["hotspots"][0]["ssid"], "Pixel")
        self.assertIn("check_interval", cfg)  # default filled in

    def test_load_corrupt_returns_defaults(self):
        path = os.path.join(tempfile.mkdtemp(), "networks.json")
        with open(path, "w") as f:
            f.write("{not json")
        self.assertEqual(nd.load_config(path)["hotspots"],
                         nd.DEFAULT_CONFIG["hotspots"])

    def test_pick_hotspot_by_name_and_default(self):
        cfg = {"hotspots": [{"ssid": "Home"}, {"ssid": "Pixel"}]}
        self.assertEqual(nd.pick_hotspot(cfg)["ssid"], "Home")           # first
        self.assertEqual(nd.pick_hotspot(cfg, "pixel")["ssid"], "Pixel")  # by name
        self.assertEqual(nd.pick_hotspot(cfg, "nope")["ssid"], "Home")    # fallback

    def test_pick_hotspot_none(self):
        self.assertIsNone(nd.pick_hotspot({"hotspots": []}))

    def test_build_wifi_connect_cmd(self):
        self.assertEqual(nd.build_wifi_connect_cmd("Pixel", "pw"),
                         ["nmcli", "device", "wifi", "connect", "Pixel", "password", "pw"])
        self.assertEqual(nd.build_wifi_connect_cmd("Open", None),
                         ["nmcli", "device", "wifi", "connect", "Open"])


class NetworkMonitorTest(unittest.TestCase):
    def _daemon(self, reachable, **cfg):
        path = os.path.join(tempfile.mkdtemp(), "networks.json")
        merged = dict(nd.DEFAULT_CONFIG)
        merged.update(cfg)
        with open(path, "w") as f:
            json.dump(merged, f)
        nd.NETWORKS_PATH = path
        return nd.NetworkDaemon(reachable=reachable)

    def test_offline_then_online_announces_recovery(self):
        state = {"up": False}
        d = self._daemon(lambda: state["up"])
        d._check_once(1.0)                 # first check: offline
        self.assertIs(d.online, False)
        state["up"] = True
        d._check_once(2.0)                 # recovered
        self.assertIs(d.online, True)
        self.assertTrue(any("back online" in s["text"]
                            for s in d.bus.of(nd.SPEAK_TOPIC)))

    def test_auto_failover_after_grace(self):
        calls = []
        d = self._daemon(lambda: False, offline_grace=30, auto_failover=True,
                         hotspots=[{"ssid": "Pixel", "password": "pw"}])
        d.have_nmcli = True                # pretend we can switch
        d._join_hotspot = lambda name=None: calls.append(name) or True
        d._check_once(100.0)               # goes offline, grace starts
        self.assertEqual(calls, [])        # not yet past grace
        d._check_once(100.0 + 31)          # past grace -> failover
        self.assertEqual(len(calls), 1)

    def test_no_failover_without_nmcli(self):
        calls = []
        d = self._daemon(lambda: False, offline_grace=1, auto_failover=True,
                         hotspots=[{"ssid": "Pixel", "password": "pw"}])
        d.have_nmcli = False
        d._join_hotspot = lambda name=None: calls.append(name) or True
        d._check_once(100.0)
        d._check_once(200.0)
        self.assertEqual(calls, [])        # can't switch -> never calls join

    def test_join_without_hotspots_speaks_help(self):
        d = self._daemon(lambda: True, hotspots=[])
        ok = d._join_hotspot()
        self.assertFalse(ok)
        self.assertTrue(d.bus.of(nd.SPEAK_TOPIC))


if __name__ == "__main__":
    unittest.main()
