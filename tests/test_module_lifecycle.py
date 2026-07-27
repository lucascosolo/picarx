import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402,F401

from module_lifecycle import NeedPlanner  # noqa: E402


class NeedPlannerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.registry = [
            {"name": "core", "entrypoint": "core.py", "enabled": True,
             "activation": {"mode": "always"}},
            {"name": "vision", "entrypoint": "vision.py", "enabled": True,
             "activation": {"mode": "state", "states": ["IDLE", "RC"]}},
            {"name": "gesture", "entrypoint": "gesture.py", "enabled": True,
             "activation": {"mode": "demand", "topic": "gesture/control",
                             "enabled_field": "enabled", "persistent": True}},
            {"name": "radio", "entrypoint": "radio.py", "enabled": True,
             "activation": {"mode": "demand", "topic": "radio/control",
                             "state_topic": "radio/state", "state_field": "playing",
                             "ttl_sec": 10, "stop_commands": ["stop"]}},
        ]
        self.planner = NeedPlanner(self.registry, clock=lambda: self.now)

    def test_always_and_state_modules_follow_robot_state(self):
        self.assertEqual(self.planner.desired_names(), {"core", "vision"})
        self.planner.set_state({"state": "GESTURE_TRACKING"})
        self.assertEqual(self.planner.desired_names(), {"core"})
        self.planner.set_state({"state": "RC"})
        self.assertEqual(self.planner.desired_names(), {"core", "vision"})

    def test_persistent_demand_stays_until_explicit_disable(self):
        self.planner.observe_demand("gesture/control", {"enabled": True})
        self.assertIn("gesture", self.planner.desired_names())
        self.now += 10000
        self.assertIn("gesture", self.planner.desired_names())
        self.planner.observe_demand("gesture/control", {"enabled": False})
        self.assertNotIn("gesture", self.planner.desired_names())

    def test_one_shot_demand_expires_and_state_refreshes_it(self):
        payload = {"command": "play", "station": "lush"}
        self.planner.observe_demand("radio/control", payload)
        self.assertEqual(self.planner.replay_payload("radio"), payload)
        self.now += 11
        self.assertNotIn("radio", self.planner.desired_names())
        self.planner.observe_demand("radio/control", payload)
        self.planner.observe_state_signal("radio/state", {"playing": True})
        self.now += 9
        self.assertIn("radio", self.planner.desired_names())
        self.planner.observe_state_signal("radio/state", {"playing": False})
        self.assertNotIn("radio", self.planner.desired_names())

    def test_state_signal_without_active_field_does_not_clear_need(self):
        self.planner.observe_demand("radio/control", {"command": "play"})
        self.planner.observe_state_signal("radio/state", {"station": "lush"})
        self.assertIn("radio", self.planner.desired_names())

    def test_legacy_entry_remains_always_on(self):
        self.planner.replace_registry([
            {"name": "legacy", "entrypoint": "legacy.py", "enabled": True}
        ])
        self.assertEqual(self.planner.desired_names(), {"legacy"})


if __name__ == "__main__":
    unittest.main()
