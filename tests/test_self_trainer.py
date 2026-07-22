"""The idle self-trainer's DECISIONS are pure and unit-tested here: when it is
eligible to train (idle + rested + optionally charging), which scenario it
picks, and how a produced pack becomes bus messages routed back through the
owning modules. The subprocess/file glue stays thin and is exercised lightly
(activity aborts a running session; a bad repo degrades to no-op)."""
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import self_trainer  # noqa: E402

NOW = 100000.0


class EligibilityTest(unittest.TestCase):
    def _elig(self, idle_for=9999, since_session=9999, battery=None,
              idle_after=600, cooldown=3600, charging_only=False):
        return self_trainer.training_eligibility(
            NOW, NOW - idle_for, NOW - since_session, battery or {},
            idle_after, cooldown, charging_only)

    def test_idle_rested_robot_is_eligible(self):
        ok, reason = self._elig()
        self.assertTrue(ok)
        self.assertEqual(reason, "idle")

    def test_recent_activity_blocks(self):
        ok, reason = self._elig(idle_for=10)          # moved 10s ago
        self.assertFalse(ok)
        self.assertEqual(reason, "busy")

    def test_cooldown_blocks(self):
        ok, reason = self._elig(since_session=60)     # trained a minute ago
        self.assertFalse(ok)
        self.assertEqual(reason, "cooldown")

    def test_charging_only_blocks_without_healthy_battery(self):
        ok, reason = self._elig(charging_only=True, battery={"voltage": 6.2})
        self.assertFalse(ok)
        self.assertEqual(reason, "not-charging")

    def test_charging_only_allows_healthy_battery(self):
        ok, _ = self._elig(charging_only=True,
                           battery={"voltage": 8.1, "low": False,
                                    "critical": False, "stale": False})
        self.assertTrue(ok)

    def test_activity_precedence_over_charging(self):
        # busy wins even if the battery is fine - live work always first
        ok, reason = self._elig(idle_for=5, charging_only=True,
                                battery={"voltage": 8.2})
        self.assertEqual(reason, "busy")


class BatteryHealthyTest(unittest.TestCase):
    def test_healthy(self):
        self.assertTrue(self_trainer.battery_healthy(
            {"voltage": 7.5, "low": False, "critical": False, "stale": False}))

    def test_low_or_critical_or_stale_or_absent(self):
        self.assertFalse(self_trainer.battery_healthy({"voltage": 8.0, "low": True}))
        self.assertFalse(self_trainer.battery_healthy({"voltage": 8.0, "critical": True}))
        self.assertFalse(self_trainer.battery_healthy({"voltage": 8.0, "stale": True}))
        self.assertFalse(self_trainer.battery_healthy({"voltage": None}))
        self.assertFalse(self_trainer.battery_healthy({}))
        self.assertFalse(self_trainer.battery_healthy(None))

    def test_below_threshold(self):
        self.assertFalse(self_trainer.battery_healthy({"voltage": 6.5}))


class ScenarioPickTest(unittest.TestCase):
    def test_rotates_deterministically(self):
        paths = ["b.json", "a.json", "c.json"]     # sorted -> a,b,c
        self.assertEqual(self_trainer.pick_scenario(paths, 0), "a.json")
        self.assertEqual(self_trainer.pick_scenario(paths, 1), "b.json")
        self.assertEqual(self_trainer.pick_scenario(paths, 3), "a.json")  # wraps

    def test_empty_is_none(self):
        self.assertIsNone(self_trainer.pick_scenario([], 0))


class PackToMessagesTest(unittest.TestCase):
    def test_full_pack_routes_to_owning_modules(self):
        policy = {"k": {"arms": {"s": {"successes": 5, "failures": 1}}}}
        nav = {"facts": [{"subject": "escape tactics", "fact": "reverse first",
                          "confidence": 0.7, "source": "training"}],
               "patterns": [{"condition": "stuck:x", "outcome": "reverse works",
                             "frequency": 4, "confidence": 0.8}]}
        msgs = self_trainer.pack_to_messages(policy, nav, "abc123")
        topics = [t for t, _ in msgs]
        self.assertEqual(topics, ["picarx/coach/adopt", "picarx/memory/note",
                                  "picarx/memory/pattern"])
        adopt = msgs[0][1]
        self.assertEqual(adopt["mode"], "adopt")          # this robot's round-trip
        self.assertEqual(adopt["lineage"], "abc123")
        self.assertEqual(adopt["coach_policy"], policy)
        self.assertEqual(msgs[1][1]["subject"], "escape tactics")
        self.assertEqual(msgs[2][1]["condition"], "stuck:x")

    def test_empty_policy_yields_no_adopt(self):
        msgs = self_trainer.pack_to_messages({}, {"facts": [], "patterns": []}, None)
        self.assertEqual(msgs, [])

    def test_malformed_facts_and_patterns_dropped(self):
        nav = {"facts": [{"subject": "", "fact": "x"}, {"subject": "s", "fact": ""}],
               "patterns": [{"condition": "", "outcome": "o"}]}
        msgs = self_trainer.pack_to_messages({"k": {"arms": {}}}, nav, "l")
        self.assertEqual([t for t, _ in msgs], ["picarx/coach/adopt"])  # only the policy

    def test_fact_default_source_is_self_training(self):
        nav = {"facts": [{"subject": "s", "fact": "f"}]}   # no source
        msgs = self_trainer.pack_to_messages({}, nav, None)
        self.assertEqual(msgs[0][1]["source"], "self_training")


class RepoResolutionTest(unittest.TestCase):
    def test_finds_sibling_with_runner(self):
        parent = tempfile.mkdtemp()
        picarx = os.path.join(parent, "picarx")
        training = os.path.join(parent, "picarx-training")
        os.makedirs(picarx)
        os.makedirs(training)
        open(os.path.join(training, "run_training.py"), "w").close()
        self.assertEqual(self_trainer.resolve_training_repo(None, picarx), training)

    def test_none_when_absent(self):
        parent = tempfile.mkdtemp()
        picarx = os.path.join(parent, "picarx")
        os.makedirs(picarx)
        self.assertIsNone(self_trainer.resolve_training_repo(None, picarx))


class ActivityAbortTest(unittest.TestCase):
    """The thin glue: an activity signal mid-session terminates the subprocess
    and sets the abort flag, without needing a real training run."""

    class _FakeProc:
        def __init__(self):
            self.terminated = False
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self.terminated = True
            self._alive = False

    def _trainer(self):
        st = self_trainer.SelfTrainer.__new__(self_trainer.SelfTrainer)
        st.bus = harness.FakeBus()
        st.lock = threading.Lock()
        st._abort = threading.Event()
        st.last_activity = 0.0
        st.proc = None
        st.latest_battery = {}
        return st

    def test_activity_kills_running_session(self):
        st = self._trainer()
        proc = self._FakeProc()
        st.proc = proc
        st.on_activity({})
        self.assertTrue(proc.terminated)
        self.assertTrue(st._abort.is_set())

    def test_activity_without_session_just_updates_clock(self):
        st = self._trainer()
        st.on_activity({})           # no proc - must not raise
        self.assertGreater(st.last_activity, 0.0)

    def test_on_world_records_battery(self):
        st = self._trainer()
        st.on_world({"battery": {"voltage": 7.9}})
        self.assertEqual(st.latest_battery["voltage"], 7.9)


class IdleActivityTest(unittest.TestCase):
    """A parked robot's steady 'stop' stream must NOT count as activity, or
    self-training could never fire when the robot is sitting still. Real motion
    still does (and kills a running session)."""

    def test_is_motion_intent(self):
        self.assertTrue(self_trainer._is_motion_intent({"direction": "forward", "speed": 25}))
        self.assertTrue(self_trainer._is_motion_intent({"direction": "backward", "speed": 30}))
        self.assertTrue(self_trainer._is_motion_intent({"direction": "turn", "angle": -20}))
        # holds / straightens / stops are NOT driving
        self.assertFalse(self_trainer._is_motion_intent({"direction": "stop"}))
        self.assertFalse(self_trainer._is_motion_intent({"direction": "forward", "speed": 0}))
        self.assertFalse(self_trainer._is_motion_intent({"direction": "turn", "angle": 0}))
        self.assertFalse(self_trainer._is_motion_intent(None))

    def _trainer(self):
        st = self_trainer.SelfTrainer.__new__(self_trainer.SelfTrainer)
        st.bus = harness.FakeBus()
        st.lock = threading.Lock()
        st._abort = threading.Event()
        st.last_activity = 0.0
        st.proc = None
        return st

    def test_stop_intents_do_not_count_as_activity(self):
        st = self._trainer()
        st.on_move_intent({"source": "field_agent",
                           "action": {"direction": "stop"}})   # parked robot
        self.assertEqual(st.last_activity, 0.0)                 # clock NOT bumped

    def test_driving_intent_counts_and_kills_session(self):
        st = self._trainer()
        proc = ActivityAbortTest._FakeProc()
        st.proc = proc
        st.on_move_intent({"source": "field_agent",
                           "action": {"direction": "forward", "speed": 25}})
        self.assertGreater(st.last_activity, 0.0)              # clock bumped
        self.assertTrue(proc.terminated)                      # session killed


class StatusPublishTest(unittest.TestCase):
    def _trainer(self, repo="/fake/picarx-training", last_activity=0.0,
                 last_session_end=0.0):
        st = self_trainer.SelfTrainer.__new__(self_trainer.SelfTrainer)
        st.bus = harness.FakeBus()
        st.lock = threading.Lock()
        st._abort = threading.Event()
        st.last_activity = last_activity
        st.last_activity_topic = "startup"
        st.last_session_end = last_session_end
        st.latest_battery = {}
        st.proc = None
        st._session_counter = 0
        st.training_repo = repo
        return st

    def test_busy_heartbeat_when_recently_active(self):
        st = self._trainer(last_activity=NOW - 42)      # active 42s ago
        st.last_activity_topic = "audio/heard"
        st.maybe_train(now=NOW)
        s = st.bus.last(self_trainer.STATUS_TOPIC)
        self.assertEqual(s["state"], "busy")
        self.assertTrue(s["repo"])
        # the diagnostic fields that reveal WHAT is keeping it busy
        self.assertEqual(s["idle_for_sec"], 42)
        self.assertEqual(s["last_activity"], "audio/heard")
        self.assertEqual(s["idle_needed_sec"], self_trainer.IDLE_AFTER_SEC)

    def test_cooldown_heartbeat_reports_remaining(self):
        # idle long enough, but a session just ended -> cooldown, with an ETA
        st = self._trainer(last_activity=NOW - 100000, last_session_end=NOW - 5)
        st.maybe_train(now=NOW)
        s = st.bus.last(self_trainer.STATUS_TOPIC)
        self.assertEqual(s["state"], "cooldown")
        self.assertGreater(s["cooldown_remaining_sec"], 0)

    def test_disabled_when_no_repo(self):
        st = self._trainer(repo=None)
        st.maybe_train(now=NOW)
        self.assertEqual(st.bus.last(self_trainer.STATUS_TOPIC)["state"], "disabled")

    def test_publish_pack_emits_published_status(self):
        import json
        import os
        import tempfile
        st = self._trainer()
        d = tempfile.mkdtemp()
        json.dump({"k": {"arms": {}}}, open(os.path.join(d, "coach_policy.json"), "w"))
        json.dump({"facts": [{"subject": "s", "fact": "f"}], "patterns": []},
                  open(os.path.join(d, "navigation_facts.json"), "w"))
        json.dump({"lineage": "abc123"}, open(os.path.join(d, "knowledge_pack.json"), "w"))
        st._publish_pack(d, scenario="box_corner.json")
        s = st.bus.last(self_trainer.STATUS_TOPIC)
        self.assertEqual(s["state"], "published")
        self.assertEqual(s["scenario"], "box_corner.json")
        self.assertTrue(s["adopted"])
        self.assertEqual(s["notes"], 1)


if __name__ == "__main__":
    unittest.main()
