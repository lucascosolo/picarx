"""Unified per-module heartbeat (heartbeat.py) and debug_monitor's liveness
consumer: a module going silent must be detectable in the field."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import heartbeat  # noqa: E402
import debug_monitor  # noqa: E402


class ModuleNameTest(unittest.TestCase):
    def test_derives_from_entrypoint(self):
        self.assertEqual(heartbeat.module_name("/opt/picarx/modules/field_agent.py"),
                         "field_agent")
        self.assertEqual(heartbeat.module_name("audio_nodes.py"), "audio_nodes")

    def test_non_py_and_empty(self):
        self.assertEqual(heartbeat.module_name("python3"), "python3")
        self.assertEqual(heartbeat.module_name(""), "unknown")


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class EmitterTest(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.clock = _Clock(1000.0)
        self.em = heartbeat.HeartbeatEmitter(
            lambda topic, p: self.sent.append((topic, p)),
            name="field_agent", pid=42, interval=10.0, clock=self.clock)

    def test_first_tick_emits_then_paces_by_interval(self):
        self.assertIsNotNone(self.em.tick())
        self.assertEqual(len(self.sent), 1)
        self.clock.t += 5                      # too soon
        self.assertIsNone(self.em.tick())
        self.assertEqual(len(self.sent), 1)
        self.clock.t += 5                      # now due
        self.assertIsNotNone(self.em.tick())
        self.assertEqual(len(self.sent), 2)

    def test_payload_shape_and_topic(self):
        self.em.tick()
        topic, p = self.sent[-1]
        self.assertEqual(topic, heartbeat.HEARTBEAT_TOPIC)
        self.assertEqual(p["name"], "field_agent")
        self.assertEqual(p["pid"], 42)
        self.assertEqual(p["uptime_sec"], 0.0)
        self.assertIn("seq", p)

    def test_seq_increments(self):
        self.em.tick()
        self.clock.t += 10
        self.em.tick()
        self.assertEqual(self.sent[0][1]["seq"], 0)
        self.assertEqual(self.sent[1][1]["seq"], 1)

    def test_status_fn_is_folded_in(self):
        self.em.status_fn = lambda: {"state": "cruising"}
        self.em.tick()
        self.assertEqual(self.sent[-1][1]["status"], {"state": "cruising"})

    def test_broken_status_fn_does_not_silence_heartbeat(self):
        def boom():
            raise RuntimeError("nope")
        self.em.status_fn = boom
        self.assertIsNotNone(self.em.tick())       # still emits
        self.assertNotIn("status", self.sent[-1][1])

    def test_publish_error_is_swallowed(self):
        def boom(topic, p):
            raise IOError("bus down")
        em = heartbeat.HeartbeatEmitter(boom, "x", 1, clock=self.clock)
        self.assertIsNotNone(em.tick())            # returns payload, doesn't raise


class StartTest(unittest.TestCase):
    def test_start_spawns_and_returns_emitter(self):
        captured = []
        sent = []
        em = heartbeat.start(lambda t, p: sent.append(p), name="coach",
                             interval=3.0, spawn=lambda fn: captured.append(fn))
        self.assertEqual(em.name, "coach")
        self.assertEqual(em.interval, 3.0)
        self.assertEqual(len(captured), 1)         # a loop was handed to spawn
        em.tick()                                  # drive one beat manually
        self.assertEqual(sent[-1]["name"], "coach")


class EvaluateLivenessTest(unittest.TestCase):
    def test_splits_alive_and_silent(self):
        now = 1000.0
        hbs = {"a": {"last_seen": now - 5}, "b": {"last_seen": now - 100}}
        live = debug_monitor.evaluate_liveness(hbs, now, stale_after=30.0)
        self.assertEqual(live["alive"], ["a"])
        self.assertEqual(live["silent"], ["b"])

    def test_empty(self):
        self.assertEqual(debug_monitor.evaluate_liveness({}, 1000.0),
                         {"alive": [], "silent": []})


class DebugMonitorLivenessTest(unittest.TestCase):
    def setUp(self):
        self.dm = debug_monitor.DebugMonitor()     # FakeBus via harness
        self.writes = []
        self.dm._write = lambda entry: self.writes.append(entry)   # no real file IO

    def test_heartbeat_recorded_and_alive(self):
        self.dm.on_heartbeat({"name": "field_agent", "ts": 1000.0, "pid": 7, "seq": 0})
        live = self.dm._check_liveness(1000.0)
        self.assertEqual(live["alive"], ["field_agent"])
        self.assertEqual(live["silent"], [])

    def test_going_silent_is_flagged_once_then_recovers(self):
        self.dm.on_heartbeat({"name": "field_agent", "ts": 1000.0, "pid": 7, "seq": 0})
        self.dm._check_liveness(1000.0)
        # Long enough with no beat -> silent, logged as a transition.
        t = 1000.0 + debug_monitor.HEARTBEAT_STALE_SEC + 1
        live = self.dm._check_liveness(t)
        self.assertEqual(live["silent"], ["field_agent"])
        self.assertTrue(any(w["type"] == "module_liveness" for w in self.writes))
        # A repeat while still silent doesn't log the transition again.
        self.writes.clear()
        self.dm._check_liveness(t + 1)
        self.assertEqual(self.writes, [])
        # A fresh heartbeat brings it back.
        self.dm.on_heartbeat({"name": "field_agent", "ts": t + 2, "pid": 7, "seq": 5})
        live = self.dm._check_liveness(t + 2)
        self.assertEqual(live["alive"], ["field_agent"])

    def test_malformed_heartbeat_ignored(self):
        self.dm.on_heartbeat({"ts": 1000.0})       # no name
        self.assertEqual(self.dm.heartbeats, {})


if __name__ == "__main__":
    unittest.main()
