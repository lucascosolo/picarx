import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from robot_state import RobotState, StateManager, parse_state  # noqa: E402


class RobotStateTests(unittest.TestCase):
    def test_parse_and_reject_invalid_claims(self):
        self.assertEqual(parse_state("gesture_tracking"), RobotState.GESTURE_TRACKING)
        self.assertEqual(parse_state("local_capture"), RobotState.LOCAL_CAPTURE)
        manager = StateManager()
        self.assertFalse(manager.claim("", "IDLE")["accepted"])
        self.assertFalse(manager.claim("x", "not-a-state")["accepted"])
        self.assertFalse(manager.claim("x", "IDLE", ttl=0)["accepted"])

    def test_higher_priority_preempts_and_release_restores(self):
        m = StateManager()
        m.claim("gesture", RobotState.GESTURE_TRACKING, ttl=10, now=0)
        self.assertEqual(m.winner(1)["state"], "GESTURE_TRACKING")
        m.claim("speech", RobotState.SPEAKING, ttl=2, now=1)
        self.assertEqual(m.winner(1.5)["state"], "SPEAKING")
        m.release("speech", now=1.6)
        self.assertEqual(m.winner(1.6)["state"], "GESTURE_TRACKING")

    def test_expired_lease_returns_idle(self):
        m = StateManager()
        m.claim("gesture", "GESTURE_TRACKING", ttl=1, now=10)
        self.assertEqual(m.winner(10.9)["state"], "GESTURE_TRACKING")
        self.assertEqual(m.winner(11)["state"], "IDLE")

    def test_safety_and_rc_preempt_everything(self):
        m = StateManager()
        m.claim("gesture", "GESTURE_TRACKING", ttl=10, now=0)
        m.claim("remote", "REMOTE_ASSIST", ttl=10, now=1)
        m.claim("rc", "RC", ttl=10, now=2)
        self.assertEqual(m.winner(2)["state"], "RC")
        m.claim("safety", "SAFETY_STOP", ttl=10, now=3)
        self.assertEqual(m.winner(3)["state"], "SAFETY_STOP")

    def test_same_owner_renewal_and_snapshot_are_bounded(self):
        m = StateManager()
        m.claim("x", "GESTURE_TRACKING", ttl=50000, reason="a", now=0)
        m.claim("x", "GESTURE_TRACKING", ttl=1, reason="b", now=1)
        self.assertEqual(len(m.snapshot(1)), 1)
        self.assertEqual(m.winner(1)["reason"], "b")
        self.assertLessEqual(m.winner(1)["expires_at"], 3601)


if __name__ == "__main__":
    unittest.main()
