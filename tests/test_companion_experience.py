"""Companion talking about its OWN experience: live motion/orientation, a
bump/pickup it just felt, its episodic diary, and what it learned from an idle
self-training run - all folded into the per-turn context, all fail-soft when the
IMU (currently flaky) is unavailable."""
import os
import sys
import threading
import time
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import companion  # noqa: E402


class FakeSemantic:
    def __init__(self, episodes=None):
        self.episodes = episodes or {}   # subject -> [ {"fact": ...} ]

    def recent_facts(self, limit=4):
        return []

    def facts_for(self, subject, limit=1):
        return self.episodes.get(subject, [])[:limit]


class _Base(unittest.TestCase):
    def setUp(self):
        self.c = companion.Companion.__new__(companion.Companion)
        self.c.lock = threading.Lock()
        self.c.bus = harness.FakeBus()
        self.c.recent_physical_events = deque(maxlen=8)
        self.c.latest_training = None
        self.c._last_training_announce_at = 0.0
        self.c.semantic = FakeSemantic()

    def notes(self, snap, now=None):
        return self.c._experience_notes(now or time.time(), snap)


class LiveMotionTest(_Base):
    def test_imu_lifted(self):
        snap = {"imu": {"stale": False, "tilted": True, "body_tilt_deg": 40}}
        self.assertIn("I'm being tilted or lifted right now", self.notes(snap))

    def test_imu_moving_vs_still(self):
        self.assertIn("I'm moving",
                      self.notes({"imu": {"stale": False, "moving": True}}))
        self.assertIn("I'm sitting still",
                      self.notes({"imu": {"stale": False, "moving": False}}))

    def test_imu_stale_falls_back_to_vision_motion(self):
        # The whole point of fail-soft: a dead IMU still lets it say whether it's
        # moving, using vision's scene_motion instead.
        moving = self.notes({"imu": {"stale": True},
                             "objects": {"scene_motion": 8.0}})
        self.assertIn("I seem to be moving", moving)
        still = self.notes({"imu": {"stale": True},
                            "objects": {"scene_motion": 0.5}})
        self.assertIn("I'm sitting still", still)

    def test_no_motion_signal_is_silent_not_crashing(self):
        # IMU dead AND no vision motion -> simply no motion note, no error.
        notes = self.notes({"imu": {"stale": True}, "objects": {}})
        self.assertFalse(any("moving" in n or "sitting still" in n for n in notes))

    def test_missing_imu_block_entirely(self):
        self.assertEqual(self.notes({}), [])   # nothing to say, no crash


class PhysicalEventTest(_Base):
    def test_recent_bump_is_mentioned(self):
        self.c.on_imu_event({"kind": "impact", "ts": time.time()})
        notes = self.notes({})
        self.assertTrue(any("felt a bump" in n for n in notes))

    def test_pickup_wording(self):
        self.c.on_imu_event({"kind": "tilted", "ts": time.time()})
        notes = self.notes({})
        self.assertTrue(any("picked up or tilted" in n for n in notes))

    def test_stale_event_is_forgotten(self):
        old = time.time() - companion.PHYSICAL_EVENT_MEMORY_SEC - 5
        self.c.on_imu_event({"kind": "impact", "ts": old})
        self.assertFalse(any("bump" in n for n in self.notes({})))

    def test_unknown_event_kind_ignored(self):
        self.c.on_imu_event({"kind": "wobble", "ts": time.time()})
        self.assertEqual(len(self.c.recent_physical_events), 0)


class EpisodicDiaryTest(_Base):
    def test_today_episode_folded_in(self):
        key = "episode:" + time.strftime("%Y-%m-%d")
        self.c.semantic = FakeSemantic({key: [{"fact": "I mapped the kitchen "
                                                       "and got stuck in a corner."}]})
        notes = self.notes({})
        self.assertTrue(any("earlier today: I mapped the kitchen" in n for n in notes))

    def test_long_diary_is_truncated(self):
        key = "episode:" + time.strftime("%Y-%m-%d")
        self.c.semantic = FakeSemantic({key: [{"fact": "x" * 400}]})
        note = next(n for n in self.notes({}) if n.startswith("earlier today:"))
        self.assertTrue(note.endswith("..."))
        self.assertLess(len(note), 230)


class SelfTrainingReportTest(_Base):
    def _publish(self, **extra):
        self.c.on_self_trainer_status(
            {"state": "published", "scenario": "corner_escape",
             "notes": 2, "adopted": True, **extra})

    def test_published_announces_once_and_is_remembered(self):
        self._publish()
        spoken = self.c.bus.of("picarx/audio/speak")
        self.assertEqual(len(spoken), 1)
        self.assertIn("practising", spoken[0]["text"])
        self.assertIsNotNone(self.c.latest_training)
        # And it can reference it in later chat context.
        self.assertTrue(any("practised" in n for n in self.notes({})))

    def test_announcement_is_throttled(self):
        self._publish()
        self._publish()
        self.assertEqual(len(self.c.bus.of("picarx/audio/speak")), 1)

    def test_non_published_states_are_ignored(self):
        self.c.on_self_trainer_status({"state": "training", "scenario": "x"})
        self.c.on_self_trainer_status({"state": "aborted"})
        self.assertEqual(self.c.bus.of("picarx/audio/speak"), [])
        self.assertIsNone(self.c.latest_training)

    def test_stale_training_result_drops_from_context(self):
        self._publish()
        self.c.latest_training["ts"] = time.time() - companion.TRAINING_REPORT_TTL_SEC - 5
        self.assertFalse(any("practised" in n for n in self.notes({})))

    def test_report_wording_without_adopted_or_notes(self):
        report = self.c._training_report({"scenario": None, "notes": 0,
                                          "adopted": False})
        self.assertIn("how I drive", report)


if __name__ == "__main__":
    unittest.main()
