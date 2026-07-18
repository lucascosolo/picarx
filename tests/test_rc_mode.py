import json
import os
import queue
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import audio_nodes  # noqa: E402
import field_agent  # noqa: E402
import reflection  # noqa: E402
import web_console  # noqa: E402

T0 = 5000.0
FRAME_W = 320


def _world(items=None, distance=100, distance_stale=False):
    return {
        "distance_cm": distance,
        "distance_stale": distance_stale,
        "objects": {"stale": False, "items": items or [],
                    "close_object": False, "overhead": None},
    }


class RcControllerTest(unittest.TestCase):
    def setUp(self):
        self.rc = web_console.RcController(harness.FakeBus())

    def _actions(self):
        return [p["action"] for p in self.rc.bus.of("picarx/intent/move")]

    def test_mode_toggle_publishes_and_cleans_up(self):
        self.rc.set_mode(True, now=T0)
        self.assertEqual(self.rc.bus.last(web_console.RC_MODE_TOPIC)["active"], True)
        self.rc.set_mode(False, now=T0 + 1)
        self.assertEqual(self.rc.bus.last(web_console.RC_MODE_TOPIC)["active"], False)
        actions = self._actions()
        self.assertIn({"direction": "turn", "angle": 0}, actions)
        self.assertEqual(actions[-1], {"direction": "stop"})
        self.assertTrue(self.rc.bus.of("picarx/intent/cancel"))

    def test_intents_are_vetoable_and_outrank_ai(self):
        self.rc.set_mode(True, now=T0)
        self.rc.update(1, 0, now=T0)
        self.rc.step(now=T0)
        for p in self.rc.bus.of("picarx/intent/move"):
            self.assertEqual(p["source"], "rc")
            self.assertEqual(p["priority"], web_console.RC_PRIORITY)
            self.assertGreater(p["priority"], 9)   # above coach, the top AI source
            self.assertEqual(p["ttl"], web_console.RC_INTENT_TTL)

    def test_first_step_straightens_then_drives(self):
        self.rc.set_mode(True, now=T0)
        self.rc.update(1, 0, now=T0)
        self.assertEqual(self.rc.step(now=T0), {"direction": "turn", "angle": 0})
        self.assertEqual(self.rc.step(now=T0 + 0.1),
                         {"direction": "forward", "speed": web_console.RC_SPEED})

    def test_steer_and_drive_alternate(self):
        self.rc.set_mode(True, now=T0)
        self.rc.update(1, 1, now=T0)
        kinds = [self.rc.step(now=T0 + i * 0.1)["direction"] for i in range(6)]
        self.assertIn("turn", kinds)
        self.assertIn("forward", kinds)
        for a, b in zip(kinds, kinds[1:]):
            self.assertFalse(a == b == "turn")

    def test_release_stops_immediately(self):
        self.rc.set_mode(True, now=T0)
        self.rc.update(1, 0, now=T0)
        self.rc.update(0, 0, now=T0 + 0.2)   # keys released
        self.assertEqual(self._actions()[-1], {"direction": "stop"})

    def test_deadman_stops_a_silent_client(self):
        self.rc.set_mode(True, now=T0)
        self.rc.update(1, 0, now=T0)
        out = self.rc.step(now=T0 + web_console.RC_DEADMAN_SEC + 0.1)
        self.assertEqual(out, {"direction": "stop"})
        self.assertEqual((self.rc.f, self.rc.t), (0, 0))

    def test_mode_times_out_without_a_client(self):
        self.rc.set_mode(True, now=T0)
        self.rc.step(now=T0 + web_console.RC_MODE_TIMEOUT_SEC + 1)
        self.assertFalse(self.rc.enabled)
        self.assertEqual(self.rc.bus.last(web_console.RC_MODE_TOPIC)["active"], False)

    def test_updates_ignored_when_disabled(self):
        self.rc.update(1, 0, now=T0)
        self.assertIsNone(self.rc.step(now=T0))
        self.assertEqual(self._actions(), [])


class BuildBoxesTest(unittest.TestCase):
    def test_objects_and_named_person(self):
        world = {
            "objects": {"stale": False, "items": [
                {"x": 10, "y": 20, "w": 50, "h": 60, "label": "chair",
                 "confidence": 0.8, "frame_width": 320, "frame_height": 240},
                {"x": 100, "y": 30, "w": 80, "h": 150, "label": "person",
                 "confidence": 0.7, "frame_width": 320, "frame_height": 240},
            ]},
            "person": {"name": "lucas", "stale": False},
            "face": {"detected": True, "stale": False,
                     "x": 120, "y": 40, "w": 30, "h": 30, "frame_width": 320},
        }
        out = web_console.build_boxes(world)
        self.assertEqual(out["frame_w"], 320)
        labels = [b["label"] for b in out["boxes"]]
        self.assertEqual(labels, ["chair", "lucas", "lucas"])
        kinds = [b["kind"] for b in out["boxes"]]
        self.assertEqual(kinds, ["object", "object", "face"])

    def test_stale_world_yields_no_boxes(self):
        out = web_console.build_boxes({
            "objects": {"stale": True, "items": [
                {"x": 1, "y": 1, "w": 5, "h": 5, "label": "chair"}]},
            "face": {"detected": True, "stale": True, "x": 1, "y": 1, "w": 2, "h": 2},
        })
        self.assertEqual(out["boxes"], [])

    def test_empty_world(self):
        self.assertEqual(web_console.build_boxes({})["boxes"], [])


class RcObserverTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.fa.on_rc_mode({"active": True})
        self.fa.bus.clear()

    def _rc_action(self, direction, status="executed", **kw):
        self.fa.on_action_result({
            "source": "rc", "action": {"direction": direction, **kw},
            "result": {"status": status}})

    def test_rc_mode_pauses_explore_and_follow(self):
        fa = field_agent.FieldAgent()
        fa.explore_mode = True
        fa.on_rc_mode({"active": True})
        self.assertFalse(fa.explore_mode)
        self.assertTrue(fa.rc_active)
        self.assertTrue(fa.bus.of("picarx/intent/cancel"))
        self.assertEqual(fa.bus.last("picarx/tools/follow/set"), {"enabled": False})

    def test_explore_command_refused_during_rc(self):
        self.fa.handle_voice_command("explore")
        self.assertFalse(self.fa.explore_mode)
        speech = " ".join(p["text"] for p in self.fa.bus.of("picarx/audio/speak"))
        self.assertIn("R C mode", speech)

    def test_demonstration_recorded_and_compressed(self):
        self.fa.latest_world = _world(distance=20)     # obstacle ahead
        self.fa._rc_observer_tick(T0)
        self.assertIsNotNone(self.fa.rc_demo)
        for _ in range(3):
            self._rc_action("backward", speed=25)
        self._rc_action("turn", angle=25)
        self.fa._rc_observer_tick(T0 + 1)              # still blocked, collecting
        self.fa.latest_world = _world(distance=100)    # human drove clear
        self.fa._rc_observer_tick(T0 + 2)
        demo = self.fa.bus.last("picarx/rc/demonstration")
        self.assertIsNotNone(demo)
        self.assertEqual(demo["situation"], "obstacle_ahead")
        self.assertTrue(demo["resolved"])
        self.assertEqual([(s["action"]["direction"], s["count"])
                          for s in demo["actions"]],
                         [("backward", 3), ("turn", 1)])
        self.assertEqual(demo["context"]["distance_cm"], 20)

    def test_empty_episode_not_published(self):
        self.fa.latest_world = _world(distance=20)
        self.fa._rc_observer_tick(T0)
        self.fa.latest_world = _world(distance=100)
        self.fa._rc_observer_tick(T0 + 1)              # cleared, no human action
        self.assertIsNone(self.fa.bus.last("picarx/rc/demonstration"))

    def test_cooldown_between_episodes(self):
        self.test_demonstration_recorded_and_compressed()
        self.fa.bus.clear()
        self.fa.latest_world = _world(distance=20)
        self.fa._rc_observer_tick(T0 + 3)              # within cooldown
        self.assertIsNone(self.fa.rc_demo)

    def test_timeout_closes_unresolved_episode(self):
        self.fa.latest_world = _world(distance=20)
        self.fa._rc_observer_tick(T0)
        self._rc_action("forward", speed=25, status="vetoed")
        self.fa._rc_observer_tick(T0 + field_agent.RC_DEMO_MAX_SEC + 1)
        demo = self.fa.bus.last("picarx/rc/demonstration")
        self.assertIsNotNone(demo)
        self.assertFalse(demo["resolved"])
        self.assertEqual(demo["actions"][0]["status"], "vetoed")

    def test_rc_actions_ignored_outside_episodes(self):
        self._rc_action("forward", speed=25)
        self.assertEqual(len(self.fa.rc_pending_actions), 0)


class SpeakerToggleTest(unittest.TestCase):
    def setUp(self):
        self.node = audio_nodes.AudioNode.__new__(audio_nodes.AudioNode)
        self.node.bus = harness.FakeBus()
        self.node.speaker_enabled = True
        self.node.mute_until = 0.0
        self.node._last_amp_assert_at = 0.0
        self.node._tts_queue = queue.Queue()
        self.node._tts_worker_started = True
        self.enables = []
        self.node._enable_speakers_once = lambda: self.enables.append(1) or True

    def test_disable_drops_speech(self):
        self.node.on_speaker_control({"enabled": False})
        self.node.handle_speak_request({"text": "hello"})
        self.assertTrue(self.node._tts_queue.empty())
        state = self.node.bus.last("picarx/audio/speaker_state")
        self.assertFalse(state["enabled"])

    def test_reenable_runs_amp_enable_command(self):
        self.node.on_speaker_control({"enabled": False})
        self.assertEqual(self.enables, [])          # muting never touches the amp
        self.node.on_speaker_control({"enabled": True})
        self.assertEqual(self.enables, [1])         # off->on press re-asserts it
        self.node.handle_speak_request({"text": "hello"})
        self.assertFalse(self.node._tts_queue.empty())

    def test_redundant_enable_is_a_state_echo_only(self):
        self.node.on_speaker_control({"enabled": True})
        self.assertEqual(self.enables, [])          # already on: no amp churn
        self.assertTrue(self.node.bus.last("picarx/audio/speaker_state")["enabled"])


class ReflectionDemonstrationDigestTest(unittest.TestCase):
    def test_summarize_demonstration(self):
        line = reflection.Reflection._summarize_event(
            "picarx/rc/demonstration",
            json.dumps({"situation": "obstacle_ahead", "resolved": True,
                        "context": {"location": {"label": "kitchen"},
                                    "objects": ["chair"]},
                        "actions": [{"action": {"direction": "backward"}},
                                    {"action": {"direction": "turn"}}]}))
        self.assertIn("USER DEMONSTRATION", line)
        self.assertIn("kitchen", line)
        self.assertIn("backward,turn", line)
        self.assertIn("cleared", line)


if __name__ == "__main__":
    unittest.main()
