import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402,F401

from repository_updater import (RepositoryUpdater, STATUS_TOPIC,
                                safe_to_update)  # noqa: E402


class GitFake:
    def __init__(self, old="old", target="new", ancestor=True):
        self.old = old
        self.target = target
        self.ancestor = ancestor
        self.commands = []

    def __call__(self, command, **kwargs):
        args = command[3:]
        self.commands.append(tuple(args))
        if args[:2] == ["status", "--porcelain"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["rev-parse", "HEAD"]:
            return types.SimpleNamespace(returncode=0, stdout=self.old, stderr="")
        if args[:2] == ["rev-parse", "origin/master"]:
            return types.SimpleNamespace(returncode=0, stdout=self.target, stderr="")
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return types.SimpleNamespace(returncode=0 if self.ancestor else 1,
                                         stdout="", stderr="not an ancestor")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


class RepositoryUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events = []
        self.resumed = []
        self.git = GitFake()
        self.updater = RepositoryUpdater(
            self.tmp.name, run_command=self.git,
            publish=lambda topic, payload: self.events.append((topic, payload)),
            quiesce=lambda: True,
            resume=lambda: self.resumed.append(True),
            health_check=lambda: (False, "simulated failed check"),
            marker_path=os.path.join(self.tmp.name, "marker.json"))
        self.updater.on_state({"state": "IDLE", "claims": []})

    def tearDown(self):
        self.tmp.cleanup()

    def test_safe_gate_rejects_motion_and_low_power(self):
        self.assertTrue(safe_to_update({"state": "IDLE", "claims": []},
                                       {"low_power": False}))
        self.assertFalse(safe_to_update({"state": "RC", "claims": []}, {}))
        self.assertFalse(safe_to_update({"state": "IDLE", "claims": []},
                                        {"low_power": True}))
        self.assertFalse(safe_to_update({"state": "IDLE", "claims": [
            {"state": "GESTURE_TRACKING"}]}, {}))
        self.assertFalse(safe_to_update({"state": "IDLE", "claims": ["bad"]}, {}))

    def test_failed_pre_restart_health_rolls_back_and_resumes(self):
        self.updater._run_update("approved", "req-1")
        commands = self.git.commands
        self.assertIn(("merge", "--ff-only", "origin/master"), commands)
        self.assertIn(("reset", "--hard", "old"), commands)
        self.assertEqual(self.resumed, [True])
        status = [payload for topic, payload in self.events if topic == STATUS_TOPIC]
        self.assertEqual(status[-1]["state"], "rollback")
        self.assertIn("simulated failed check", status[-1]["error"])

    def test_non_fast_forward_never_quiesces_or_merges(self):
        self.updater = RepositoryUpdater(
            self.tmp.name, run_command=GitFake(ancestor=False),
            publish=lambda topic, payload: self.events.append((topic, payload)),
            quiesce=lambda: self.fail("must not quiesce a divergent branch"),
            marker_path=os.path.join(self.tmp.name, "marker.json"))
        self.updater.on_state({"state": "IDLE", "claims": []})
        self.updater._run_update("approved", "req-2")
        status = [payload for topic, payload in self.events if topic == STATUS_TOPIC]
        self.assertEqual(status[-1]["state"], "error")
        self.assertIn("fast-forward", status[-1]["error"])

    def test_control_requires_explicit_confirmation(self):
        self.updater.on_control({"operation": "update", "request_id": "req-3"})
        self.assertEqual(self.events[-1][1]["state"], "rejected")
        self.assertIn("confirmed=true", self.events[-1][1]["error"])

    def test_startup_health_failure_restores_marked_revision(self):
        with open(self.updater.marker_path, "w", encoding="utf-8") as stream:
            stream.write('{"previous_commit":"old","new_commit":"new"}')
        restarted = []
        self.updater.restart = lambda: restarted.append(True)
        self.assertFalse(self.updater.startup_recover())
        self.assertIn(("reset", "--hard", "old"), self.git.commands)
        self.assertEqual(restarted, [True])
        self.assertFalse(os.path.exists(self.updater.marker_path))
        self.assertEqual(self.events[-1][1]["state"], "rollback")

    def test_startup_health_success_commits_marked_revision(self):
        with open(self.updater.marker_path, "w", encoding="utf-8") as stream:
            stream.write('{"previous_commit":"old","new_commit":"new"}')
        self.updater.health_check = lambda: (True, "ok")
        self.updater.runtime_health_check = lambda: (True, "services ok")
        self.assertTrue(self.updater.startup_recover())
        self.assertFalse(os.path.exists(self.updater.marker_path))
        self.assertEqual(self.events[-1][1]["state"], "success")


if __name__ == "__main__":
    unittest.main()
