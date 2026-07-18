"""The learning stack must record enough to make MEANINGFUL corrections:
real durations ("backed up for 1.2s"), geometry (obstacle on the left vs
right), and failure causes (vetoed by what, vs. never moved) - and human
demonstrations must actually reach the coach."""
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import coach  # noqa: E402
import field_agent  # noqa: E402
import reflection  # noqa: E402

T0 = 7000.0


def _world(items=None, distance=100, distance_stale=False):
    return {
        "distance_cm": distance,
        "distance_stale": distance_stale,
        "objects": {"stale": False, "items": items or [],
                    "close_object": False, "overhead": None},
    }


class CompressionDurationTest(unittest.TestCase):
    def test_durations_come_from_timestamps(self):
        back = {"direction": "backward", "speed": 25}
        turn = {"direction": "turn", "angle": 25}
        raw = [(T0 + 0.1 * i, back, "executed") for i in range(12)]   # 1.1s held
        raw += [(T0 + 1.3, turn, "executed"), (T0 + 1.4, turn, "executed")]
        steps = field_agent.FieldAgent._compress_rc_actions(raw)
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(steps[0]["duration"], 1.2, places=2)
        self.assertAlmostEqual(steps[1]["duration"], 0.2, places=2)
        self.assertEqual(steps[0]["count"], 12)

    def test_single_tick_step_gets_minimum_duration(self):
        steps = field_agent.FieldAgent._compress_rc_actions(
            [(T0, {"direction": "forward", "speed": 25}, "executed")])
        self.assertAlmostEqual(steps[0]["duration"], 0.1, places=2)


class DemoGeometryTest(unittest.TestCase):
    def test_side_size_and_urgency_recorded(self):
        obj = {"label": "chair", "center_offset": -80, "frame_width": 320,
               "area_ratio": 0.31, "approaching": True}
        rec = field_agent.FieldAgent._demo_object(obj)
        self.assertEqual(rec, {"label": "chair", "side": "l",
                               "area_ratio": 0.31, "approaching": True})

    def test_center_object_is_c(self):
        rec = field_agent.FieldAgent._demo_object(
            {"label": "box", "center_offset": 10, "frame_width": 320})
        self.assertEqual(rec["side"], "c")

    def test_demo_context_carries_geometry(self):
        fa = field_agent.FieldAgent()
        fa.on_rc_mode({"active": True})
        fa.latest_world = _world(
            items=[{"label": "chair", "center_offset": 90, "frame_width": 320,
                    "area_ratio": 0.3}], distance=20)
        fa._rc_observer_tick(T0)
        self.assertEqual(fa.rc_demo["context"]["objects"],
                         [{"label": "chair", "side": "r", "area_ratio": 0.3,
                           "approaching": False}])


class OutcomeCauseTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.fa.coach_steps = [{"action": {"direction": "backward", "speed": 25},
                                "duration": 1.0}]
        self.fa.coach_action_started_at = T0
        self.fa.last_coach_query_id_used = "q1"
        self.fa.last_coach_situation_key = "collision_loop:test"

    def test_vetoed_episode_reports_code_and_duration(self):
        self.fa.veto_events.append((T0 + 0.5, "cliff"))
        self.fa.coach_motion_max = 8.0
        self.fa._finish_coach_episode(T0 + 2.0)
        out = self.fa.bus.last("picarx/coach/outcome")
        self.assertFalse(out["success"])
        self.assertTrue(out["vetoed"])
        self.assertEqual(out["veto_code"], "cliff")
        self.assertEqual(out["duration"], 2.0)
        self.assertEqual(out["motion_max"], 8.0)

    def test_no_motion_failure_distinguished_from_veto(self):
        self.fa.coach_motion_max = 0.4     # ground never moved in view
        self.fa._finish_coach_episode(T0 + 2.0)
        out = self.fa.bus.last("picarx/coach/outcome")
        self.assertFalse(out["success"])
        self.assertFalse(out["vetoed"])
        self.assertIsNone(out["veto_code"])

    def test_clean_success_reports_motion_evidence(self):
        self.fa.coach_motion_max = 7.5
        self.fa._finish_coach_episode(T0 + 1.5)
        out = self.fa.bus.last("picarx/coach/outcome")
        self.assertTrue(out["success"])
        self.assertEqual(out["motion_max"], 7.5)


class EvadeJournalTest(unittest.TestCase):
    def test_evasion_choice_is_journaled_with_trigger_and_angle(self):
        fa = field_agent.FieldAgent()
        fa.state = "CRUISING"
        fa._begin_evasion("ultrasonic", away_hint=-30)
        fa.state_until = 0.0    # expire stage 0 so the next tick picks the angle
        fa.evade_stage = 0
        fa.latest_world = _world()
        fa.explore_tick()
        evades = [p for p in fa.bus.of("picarx/decision") if p["kind"] == "evade"]
        self.assertEqual(len(evades), 1)
        self.assertEqual(evades[0]["choice"], {"angle": -30, "trigger": "ultrasonic"})
        self.assertIn("left", evades[0]["reason"])


class CoachDemonstrationIntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig = (coach.DATA_DIR, coach.COACH_POLICY_PATH)
        coach.DATA_DIR = self.tmp
        coach.COACH_POLICY_PATH = os.path.join(self.tmp, "coach_policy.json")
        self.c = coach.Coach.__new__(coach.Coach)
        self.c.bus = harness.FakeBus()
        self.c.lock = threading.Lock()
        self.c.policy = {}

    def tearDown(self):
        coach.DATA_DIR, coach.COACH_POLICY_PATH = self._orig

    def _demo(self, **overrides):
        payload = {
            "situation": "obstacle_ahead",
            "context": {"objects": [{"label": "chair", "side": "r"}]},
            "actions": [
                {"action": {"direction": "backward", "speed": 25},
                 "status": "executed", "duration": 1.2},
                {"action": {"direction": "turn", "angle": 25},
                 "status": "executed", "duration": 0.5},
            ],
            "resolved": True, "ts": T0,
        }
        payload.update(overrides)
        return payload

    def test_demonstration_stored_with_clamped_steps(self):
        self.c.on_demonstration(self._demo())
        demos = self.c.policy[coach.DEMONSTRATIONS_KEY]
        self.assertEqual(len(demos), 1)
        steps = demos[0]["steps"]
        self.assertEqual(steps[0]["action"]["direction"], "backward")
        self.assertEqual(steps[0]["duration"], 1.2)
        self.assertEqual(steps[1]["action"]["angle"], 25)

    def test_vetoed_commands_never_taught(self):
        self.c.on_demonstration(self._demo(actions=[
            {"action": {"direction": "forward", "speed": 25},
             "status": "vetoed", "duration": 0.4}]))
        self.assertNotIn(coach.DEMONSTRATIONS_KEY, self.c.policy)

    def test_durations_clamped_to_safe_bounds(self):
        self.c.on_demonstration(self._demo(actions=[
            {"action": {"direction": "backward", "speed": 25},
             "status": "executed", "duration": 5.0}]))    # human held it 5s
        step = self.c.policy[coach.DEMONSTRATIONS_KEY][0]["steps"][0]
        self.assertLessEqual(step["duration"], coach.MAX_BACKWARD_DURATION)

    def test_rolling_cap(self):
        for i in range(coach.MAX_DEMONSTRATIONS + 3):
            self.c.on_demonstration(self._demo(ts=T0 + i))
        self.assertEqual(len(self.c.policy[coach.DEMONSTRATIONS_KEY]),
                         coach.MAX_DEMONSTRATIONS)

    def test_demonstrations_reach_the_llm_prompt(self):
        self.c.on_demonstration(self._demo())
        prompt_demos = self.c._recent_demonstrations()
        self.assertEqual(len(prompt_demos), 1)
        self.assertEqual(prompt_demos[0]["steps"][0],
                         {"direction": "backward", "speed": 25, "duration": 1.2})
        self.assertTrue(prompt_demos[0]["resolved"])

    def test_reserved_key_survives_policy_reload(self):
        self.c.on_demonstration(self._demo())
        reloaded = coach.Coach._load_policy(self.c)
        self.assertIn(coach.DEMONSTRATIONS_KEY, reloaded)
        # ...and a normal arm entry alongside it still loads.
        self.c.policy["novel_object:chair"] = {"arms": {}}
        self.c._save_policy()
        reloaded = coach.Coach._load_policy(self.c)
        self.assertIn("novel_object:chair", reloaded)

    def test_similarity_search_skips_reserved_sections(self):
        # _find_similar_situation iterates the whole policy; the list
        # under the reserved key must not crash it.
        self.c.policy[coach.DEMONSTRATIONS_KEY] = [{"steps": []}]
        self.c.embedder = type("E", (), {"available": False})()
        self.assertIsNone(self.c._find_similar_situation("x", {}))


class DigestCauseTest(unittest.TestCase):
    def test_vetoed_episode_line_names_the_veto(self):
        line = reflection.Reflection._summarize_event(
            "picarx/coach/episode",
            json.dumps({"situation_key": "k", "success": False, "vetoed": True,
                        "veto_code": "cliff", "cached": True,
                        "steps": [{"action": {"direction": "backward"},
                                   "duration": 1.0}]}))
        self.assertIn("backward 1.0s", line)
        self.assertIn("vetoed (cliff)", line)

    def test_no_motion_failure_line(self):
        line = reflection.Reflection._summarize_event(
            "picarx/coach/episode",
            json.dumps({"situation_key": "k", "success": False, "vetoed": False,
                        "motion_max": 0.5, "cached": False,
                        "steps": [{"action": {"direction": "forward"},
                                   "duration": 2.0}]}))
        self.assertIn("never visibly moved", line)

    def test_demo_line_shows_durations_and_sides(self):
        line = reflection.Reflection._summarize_event(
            "picarx/rc/demonstration",
            json.dumps({"situation": "obstacle_ahead", "resolved": True,
                        "context": {"location": {"label": "kitchen"},
                                    "objects": [{"label": "chair", "side": "r"}]},
                        "actions": [{"action": {"direction": "backward"},
                                     "duration": 1.2}]}))
        self.assertIn("chair(r)", line)
        self.assertIn("backward 1.2s", line)


if __name__ == "__main__":
    unittest.main()
