"""The head intent channel honors exclusive RobotState ownership."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import arbiter  # noqa: E402


class ArbiterStateGateTests(unittest.TestCase):
    def setUp(self):
        self.a = arbiter.Arbiter()
        self.sent = []
        self.a.send_to_safety = lambda action: self.sent.append(dict(action))

    def _look(self, source):
        self.a.on_look({"source": source,
                        "action": {"direction": "look", "pan": 12, "tilt": 4}})

    def test_legacy_install_without_state_still_accepts_look(self):
        self._look("expressions")
        self.assertEqual(self.sent, [{"direction": "look", "pan": 12, "tilt": 4}])

    def test_gesture_state_allows_only_gesture_owner(self):
        self.a.on_robot_state({"state": "GESTURE_TRACKING", "owner": "gesture_tracking"})
        self._look("expressions")
        self._look("gesture_tracking")
        self.assertEqual(self.sent, [{"direction": "look", "pan": 12, "tilt": 4}])

    def test_speaking_and_rc_drop_competing_head_commands(self):
        for mode, owner in (("SPEAKING", "audio_nodes"), ("RC", "web_console_rc"),
                            ("REMOTE_ASSIST", "remote_assist"),
                            ("LOCAL_CAPTURE", "clip_daemon"),
                            ("SAFETY_STOP", "safety")):
            self.a.last_look_sent = None
            self.a.on_robot_state({"state": mode, "owner": owner})
            self._look("expressions")
        self.assertEqual(self.sent, [])

    def test_exclusive_modes_clear_and_reject_autonomous_drive(self):
        self.a.on_intent({"source": "follow", "action": {"direction": "forward"}})
        self.assertIn("follow", self.a.intents)
        self.a.on_robot_state({"state": "GESTURE_TRACKING", "owner": "gesture_tracking"})
        self.assertEqual(self.a.intents, {})

    def test_local_capture_clears_existing_motion_and_rejects_new_motion(self):
        self.a.on_intent({"source": "follow", "action": {"direction": "forward"}})
        self.a.on_robot_state({"state": "LOCAL_CAPTURE", "owner": "clip_daemon"})
        self.assertEqual(self.a.intents, {})
        self.a.on_intent({"source": "follow", "action": {"direction": "forward"}})
        self.assertEqual(self.a.intents, {})
        self.a.on_intent({"source": "follow", "action": {"direction": "forward"}})
        self.assertEqual(self.a.intents, {})

    def test_rc_state_keeps_only_manual_drive_source(self):
        self.a.on_intent({"source": "follow", "action": {"direction": "forward"}})
        self.a.on_robot_state({"state": "RC", "owner": "web_console_rc"})
        self.a.on_intent({"source": "follow", "action": {"direction": "forward"}})
        self.a.on_intent({"source": "rc", "action": {"direction": "forward"}})
        self.assertEqual(set(self.a.intents), {"rc"})


if __name__ == "__main__":
    unittest.main()
