"""Prove-the-learning-loop scaffolding: the A/B experiment rotation, coach
holding out sim-trained arms in a control session, the behavior-metrics
collector, and the offline A/B report."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import experiment  # noqa: E402
import ab_report  # noqa: E402
import coach  # noqa: E402
import behavior_metrics  # noqa: E402


class ExperimentRotationTest(unittest.TestCase):
    def test_assign_condition_alternates(self):
        self.assertEqual(experiment.assign_condition(0), experiment.ADOPT)
        self.assertEqual(experiment.assign_condition(1), experiment.CONTROL)
        self.assertEqual(experiment.assign_condition(2), experiment.ADOPT)

    def test_rotate_persists_and_alternates(self):
        path = os.path.join(tempfile.mkdtemp(), "experiment_state.json")
        self.assertEqual(experiment.rotate(path), ("adopt", 0))
        self.assertEqual(experiment.rotate(path), ("control", 1))
        self.assertEqual(experiment.rotate(path), ("adopt", 2))

    def test_missing_state_starts_fresh(self):
        path = os.path.join(tempfile.mkdtemp(), "nope.json")
        self.assertEqual(experiment.load_state(path), {})


class CoachControlHoldoutTest(unittest.TestCase):
    """A control session must run the pre-adoption baseline: sim-trained arms
    are held out of selection (but not deleted)."""

    def setUp(self):
        self.c = coach.Coach.__new__(coach.Coach)
        self._orig_rate = coach.NEW_ARM_EXPLORE_RATE
        coach.NEW_ARM_EXPLORE_RATE = 0.0        # force exploitation, deterministic
        # A weak organic arm and a strong sim-trained one for the same situation.
        self.c.policy = {"stuck": {"arms": {
            "organic": {"successes": 1, "failures": 4},
            "trained": {"successes": 9, "failures": 0, "trained_in_sim": True},
        }}}

    def tearDown(self):
        coach.NEW_ARM_EXPLORE_RATE = self._orig_rate

    def test_adopt_session_can_exploit_the_trained_arm(self):
        self.c.control = False
        self.assertEqual(self.c._select_arm("stuck"), "trained")

    def test_control_session_holds_out_the_trained_arm(self):
        # Only the weak organic arm remains -> under MIN_ARMS -> ask the LLM
        # (exactly the pre-adoption behaviour).
        self.c.control = True
        self.assertIsNone(self.c._select_arm("stuck"))

    def test_control_still_uses_organic_arms(self):
        self.c.policy["stuck"]["arms"]["organic2"] = {"successes": 3, "failures": 1}
        self.c.control = True
        self.assertIn(self.c._select_arm("stuck"), ("organic", "organic2"))


class SessionMetricsTest(unittest.TestCase):
    def setUp(self):
        self.m = behavior_metrics.SessionMetrics(
            session_id=1.0, condition="adopt", started_at=100.0)

    def test_motion_attempts_and_vetoes(self):
        self.m.record_action("field_agent", {"direction": "forward"}, "ok")
        self.m.record_action("field_agent", {"direction": "turn", "angle": 20},
                             "vetoed", reason_code="obstacle")
        self.assertEqual(self.m.move_attempts, 2)
        self.assertEqual(self.m.vetoes, 1)
        self.assertEqual(self.m.veto_reasons, {"obstacle": 1})

    def test_stop_and_non_motion_do_not_count(self):
        self.m.record_action("field_agent", {"direction": "stop"}, "ok")
        self.m.record_action("field_agent", {"direction": "look"}, "ok")
        self.assertEqual(self.m.move_attempts, 0)

    def test_human_rc_is_excluded(self):
        self.m.record_action("rc", {"direction": "forward"}, "vetoed",
                             reason_code="obstacle")
        self.assertEqual(self.m.move_attempts, 0)
        self.assertEqual(self.m.vetoes, 0)

    def test_summary_rates(self):
        for _ in range(3):
            self.m.record_action("field_agent", {"direction": "forward"}, "ok")
        self.m.record_action("field_agent", {"direction": "forward"}, "vetoed",
                             reason_code="cliff")
        self.m.record_impact()
        self.m.record_fail_loop()
        s = self.m.summary(160.0)
        self.assertEqual(s["move_attempts"], 4)
        self.assertEqual(s["vetoes"], 1)
        self.assertEqual(s["veto_rate"], 0.25)
        self.assertEqual(s["impacts"], 1)
        self.assertEqual(s["fail_loops"], 1)
        self.assertEqual(s["condition"], "adopt")
        self.assertEqual(s["uptime_sec"], 60.0)

    def test_zero_attempts_has_zero_rate(self):
        self.assertEqual(self.m.summary(160.0)["veto_rate"], 0.0)


class BehaviorMetricsHandlersTest(unittest.TestCase):
    def setUp(self):
        self.bm = behavior_metrics.BehaviorMetrics()   # FakeBus via harness

    def test_action_and_impact_and_fail_loop(self):
        self.bm.on_action_result({"source": "field_agent",
                                  "action": {"direction": "forward"},
                                  "result": {"status": "vetoed", "reason_code": "x"}})
        self.bm.on_imu_event({"kind": "impact"})
        self.bm.on_imu_event({"kind": "tilted"})       # not a collision
        self.bm.on_coach_query({"situation": "collision_loop"})
        self.bm.on_coach_query({"situation": "novel_object"})   # not a fail loop
        self.assertEqual(self.bm.metrics.vetoes, 1)
        self.assertEqual(self.bm.metrics.impacts, 1)
        self.assertEqual(self.bm.metrics.fail_loops, 1)

    def test_experiment_tags_the_session(self):
        self.bm.on_experiment({"condition": "control", "session_id": 4242})
        self.assertEqual(self.bm.metrics.condition, "control")
        self.assertEqual(self.bm.metrics.session_id, 4242)


class AbReportTest(unittest.TestCase):
    def test_latest_per_session_keeps_newest(self):
        recs = [
            {"session_id": 1, "ts": 10, "vetoes": 1},
            {"session_id": 1, "ts": 20, "vetoes": 3},   # newer checkpoint wins
            {"session_id": 2, "ts": 5, "vetoes": 0},
        ]
        latest = ab_report.latest_per_session(recs)
        by_id = {r["session_id"]: r for r in latest}
        self.assertEqual(by_id[1]["vetoes"], 3)
        self.assertEqual(len(latest), 2)

    def test_aggregate_rates_by_condition(self):
        sessions = [
            {"condition": "adopt", "move_attempts": 100, "vetoes": 5, "impacts": 1},
            {"condition": "adopt", "move_attempts": 100, "vetoes": 5, "impacts": 1},
            {"condition": "control", "move_attempts": 100, "vetoes": 20, "impacts": 4},
        ]
        g = ab_report.aggregate_by_condition(sessions)
        self.assertEqual(g["adopt"]["sessions"], 2)
        self.assertEqual(g["adopt"]["veto_rate"], round(10 / 200, 4))
        self.assertEqual(g["control"]["veto_rate"], 0.2)

    def test_verdict_not_enough_data(self):
        g = ab_report.aggregate_by_condition([
            {"condition": "adopt", "move_attempts": 10, "vetoes": 0}])
        self.assertIn("Not enough data", ab_report._verdict(g))

    def test_verdict_adopt_better(self):
        sessions = ([{"condition": "adopt", "move_attempts": 100, "vetoes": 2}] * 5 +
                    [{"condition": "control", "move_attempts": 100, "vetoes": 10}] * 5)
        verdict = ab_report._verdict(ab_report.aggregate_by_condition(sessions))
        self.assertIn("BETTER", verdict)

    def test_verdict_adopt_worse(self):
        sessions = ([{"condition": "adopt", "move_attempts": 100, "vetoes": 15}] * 5 +
                    [{"condition": "control", "move_attempts": 100, "vetoes": 5}] * 5)
        verdict = ab_report._verdict(ab_report.aggregate_by_condition(sessions))
        self.assertIn("WORSE", verdict)

    def test_format_report_smoke(self):
        out = ab_report.format_report(ab_report.aggregate_by_condition([
            {"condition": "adopt", "move_attempts": 50, "vetoes": 1}]))
        self.assertIn("condition", out)
        self.assertIn("adopt", out)

    def test_empty(self):
        self.assertIn("No behavior metrics", ab_report.format_report({}))


if __name__ == "__main__":
    unittest.main()
