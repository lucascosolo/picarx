import os
import sys
import time
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402  - stubs + sys.path
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

import reminder_daemon as rd  # noqa: E402


class ReminderParsingTest(unittest.TestCase):
    def setUp(self):
        # A fixed "now": 2026-07-15 12:00:00 local.
        self.now = datetime(2026, 7, 15, 12, 0, 0).timestamp()

    def test_parse_at_datetime(self):
        got = rd.parse_at("2026-07-15 18:30", self.now)
        self.assertEqual(datetime.fromtimestamp(got),
                         datetime(2026, 7, 15, 18, 30))

    def test_parse_at_iso_t_separator(self):
        got = rd.parse_at("2026-07-15T18:30", self.now)
        self.assertEqual(datetime.fromtimestamp(got), datetime(2026, 7, 15, 18, 30))

    def test_parse_at_clock_later_today(self):
        got = rd.parse_at("18:30", self.now)
        self.assertEqual(datetime.fromtimestamp(got), datetime(2026, 7, 15, 18, 30))

    def test_parse_at_clock_already_passed_rolls_tomorrow(self):
        got = rd.parse_at("09:00", self.now)  # 9am already passed at noon
        self.assertEqual(datetime.fromtimestamp(got), datetime(2026, 7, 16, 9, 0))

    def test_parse_at_garbage(self):
        self.assertIsNone(rd.parse_at("whenever", self.now))
        self.assertIsNone(rd.parse_at("", self.now))
        self.assertIsNone(rd.parse_at(None, self.now))

    def test_resolve_delay_minutes(self):
        self.assertAlmostEqual(rd.resolve_fire_at({"delay_minutes": 20}, self.now),
                               self.now + 1200)

    def test_resolve_rejects_nonpositive_and_absurd_delay(self):
        self.assertIsNone(rd.resolve_fire_at({"delay_minutes": 0}, self.now))
        self.assertIsNone(rd.resolve_fire_at({"delay_minutes": -5}, self.now))
        self.assertIsNone(rd.resolve_fire_at({"delay_minutes": 99999}, self.now))

    def test_resolve_at(self):
        got = rd.resolve_fire_at({"at": "18:30"}, self.now)
        self.assertEqual(datetime.fromtimestamp(got), datetime(2026, 7, 15, 18, 30))

    def test_resolve_none_when_no_time(self):
        self.assertIsNone(rd.resolve_fire_at({"message": "x"}, self.now))

    def test_humanize(self):
        self.assertEqual(rd.humanize(1200), "20 minutes")
        self.assertEqual(rd.humanize(60), "1 minute")
        self.assertIn("hour", rd.humanize(7200))


class ReminderDaemonTest(unittest.TestCase):
    def setUp(self):
        rd.REMINDERS_PATH = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"reminders_test_{os.getpid()}.json")
        rd.DATA_DIR = os.path.dirname(rd.REMINDERS_PATH)
        if os.path.exists(rd.REMINDERS_PATH):
            os.remove(rd.REMINDERS_PATH)
        self.d = rd.ReminderDaemon()   # Bus() is the stubbed FakeBus

    def tearDown(self):
        for r in list(self.d.reminders.values()):
            r["timer"].cancel()
        if os.path.exists(rd.REMINDERS_PATH):
            os.remove(rd.REMINDERS_PATH)

    def test_set_arms_a_reminder(self):
        self.d.on_set({"message": "take the cake out", "delay_minutes": 30})
        self.assertEqual(len(self.d.reminders), 1)
        state = self.d.bus.of(rd.STATE_TOPIC)
        self.assertTrue(any(s["event"] == "set" for s in state))

    def test_set_without_time_speaks_and_does_not_arm(self):
        self.d.on_set({"message": "something"})
        self.assertEqual(len(self.d.reminders), 0)
        self.assertTrue(self.d.bus.of(rd.SPEAK_TOPIC))

    def test_empty_message_ignored(self):
        self.d.on_set({"delay_minutes": 5})
        self.assertEqual(len(self.d.reminders), 0)

    def test_fire_speaks_and_clears(self):
        # Arm with a tiny delay and fire directly.
        self.d.on_set({"message": "stretch", "delay_minutes": 0.001})
        rid = next(iter(self.d.reminders))
        self.d._fire(rid)
        spoken = self.d.bus.of(rd.SPEAK_TOPIC)
        self.assertTrue(any("stretch" in s["text"] for s in spoken))
        self.assertEqual(len(self.d.reminders), 0)

    def test_persistence_reload(self):
        self.d.on_set({"message": "persisted", "delay_minutes": 60})
        # A brand-new daemon should reload the pending reminder from disk.
        d2 = rd.ReminderDaemon()
        try:
            self.assertEqual(len(d2.reminders), 1)
            self.assertTrue(any(r["message"] == "persisted"
                                for r in d2.reminders.values()))
        finally:
            for r in d2.reminders.values():
                r["timer"].cancel()


if __name__ == "__main__":
    unittest.main()
