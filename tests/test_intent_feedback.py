import os
import queue
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import speech_match  # noqa: E402
import field_agent  # noqa: E402
import companion  # noqa: E402
import reflection  # noqa: E402
import web_console  # noqa: E402


class ParseFeedbackTest(unittest.TestCase):
    def test_incorrect_phrases(self):
        for text in ("that's not what i meant",
                     "thats wrong",
                     "no, that's not what i wanted",
                     "bad robot",
                     "you misunderstood me"):
            self.assertEqual(speech_match.parse_feedback(text), "incorrect", text)

    def test_correct_phrases(self):
        for text in ("that's right", "good robot", "well done",
                     "yes that's what i meant"):
            self.assertEqual(speech_match.parse_feedback(text), "correct", text)

    def test_not_right_never_reads_as_right(self):
        self.assertEqual(speech_match.parse_feedback("that's not right"), "incorrect")

    def test_ordinary_speech_is_not_feedback(self):
        for text in ("you took the wrong turn back there",
                     "is that the right way",
                     "explore", "where is the bottle", ""):
            self.assertIsNone(speech_match.parse_feedback(text), text)


class VoiceFeedbackRoutingTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_feedback_carries_last_utterance(self):
        self.fa.handle_voice_command("battery")
        self.fa.handle_voice_command("that's not what i meant")
        fb = self.fa.bus.last("picarx/intent/feedback")
        self.assertIsNotNone(fb)
        self.assertEqual(fb["verdict"], "incorrect")
        self.assertEqual(fb["utterance"], "battery")
        self.assertEqual(fb["origin"], "voice")

    def test_feedback_never_forwarded_as_chat(self):
        self.fa._mark_interaction()  # open the conversation window
        self.fa.handle_voice_command("good robot")
        self.assertEqual(self.fa.bus.of("picarx/audio/unhandled"), [])
        self.assertEqual(self.fa.bus.of("picarx/audio/uncertain"), [])
        self.assertEqual(self.fa.bus.last("picarx/intent/feedback")["verdict"],
                         "correct")

    def test_feedback_phrase_not_tracked_as_utterance(self):
        self.fa.handle_voice_command("battery")
        self.fa.handle_voice_command("that's wrong")
        self.fa.handle_voice_command("that's wrong")
        # Both judgments refer to "battery", not to the first "that's wrong".
        for fb in self.fa.bus.of("picarx/intent/feedback"):
            self.assertEqual(fb["utterance"], "battery")

    def test_repaired_text_cannot_emit_feedback(self):
        self.fa.handle_voice_command("that's wrong", source="intent_repair")
        self.assertEqual(self.fa.bus.of("picarx/intent/feedback"), [])


class CompanionTeacherTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self._orig = (companion.DATA_DIR, companion.LEARNED_INTENTS_PATH)
        companion.DATA_DIR = tmp
        companion.LEARNED_INTENTS_PATH = os.path.join(tmp, "learned_intents.json")
        self.c = companion.Companion.__new__(companion.Companion)
        self.c.lock = threading.Lock()
        self.c.bus = harness.FakeBus()
        self.c.work_queue = queue.Queue()
        self.c.learned_intents = {}
        self.c.awaiting_correction = None
        self.c._client = None
        self.c._warned_no_key = False

    def tearDown(self):
        companion.DATA_DIR, companion.LEARNED_INTENTS_PATH = self._orig

    def _key(self, utterance):
        return companion.speech_match.canonicalize(utterance)

    def test_correct_reinforces_cached_mapping(self):
        key = self._key("hows your charge")
        self.c.learned_intents[key] = {"command": "battery", "count": 1, "last": 0}
        self.c.on_feedback({"verdict": "correct", "utterance": "hows your charge",
                            "origin": "web"})
        self.assertEqual(self.c.learned_intents[key]["count"], 2)
        self.assertTrue(self.c.learned_intents[key]["confirmed"])

    def test_incorrect_unlearns_cached_mapping(self):
        key = self._key("hows your charge")
        self.c.learned_intents[key] = {"command": "play radio", "count": 3, "last": 0}
        self.c.on_feedback({"verdict": "incorrect", "utterance": "hows your charge",
                            "origin": "web"})
        self.assertNotIn(key, self.c.learned_intents)

    def test_voice_incorrect_asks_what_they_wanted(self):
        self.c.on_feedback({"verdict": "incorrect", "utterance": "hows your charge",
                            "origin": "voice"})
        self.assertIsNotNone(self.c.awaiting_correction)
        self.assertEqual(self.c.awaiting_correction["utterance"], "hows your charge")
        speech = " ".join(p["text"] for p in self.c.bus.of("picarx/audio/speak"))
        self.assertIn("What did you want", speech)

    def test_web_incorrect_with_correction_queues_learning(self):
        self.c.on_feedback({"verdict": "incorrect", "utterance": "hows your charge",
                            "correction": "battery", "origin": "web"})
        kind, item = self.c.work_queue.get_nowait()
        self.assertEqual(kind, "learn")
        self.assertEqual(item, ("hows your charge", "battery"))

    def test_answer_to_question_queues_learning(self):
        self.c.awaiting_correction = {"utterance": "hows your charge",
                                      "until": companion.time.time() + 30}
        self.c.on_heard({"text": "i wanted the battery"})
        kind, item = self.c.work_queue.get_nowait()
        self.assertEqual(kind, "learn")
        self.assertEqual(item, ("hows your charge", "i wanted the battery"))
        self.assertIsNone(self.c.awaiting_correction)

    def test_answer_capture_ignores_repairs_and_feedback(self):
        self.c.awaiting_correction = {"utterance": "x",
                                      "until": companion.time.time() + 30}
        self.c.on_heard({"text": "battery", "source": "intent_repair"})
        self.c.on_heard({"text": "that's wrong"})
        self.assertTrue(self.c.work_queue.empty())
        self.assertIsNotNone(self.c.awaiting_correction)

    def test_learn_clean_command_without_llm(self):
        ok = self.c._learn_correction("hows your charge", "battery")
        self.assertTrue(ok)
        entry = self.c.learned_intents[self._key("hows your charge")]
        self.assertEqual(entry["command"], "battery")
        self.assertTrue(entry["taught"])

    def test_learn_motion_command_refused(self):
        # No client in tests, and "explore" is outside the allowlist -
        # the cache must never gain a motion mapping.
        ok = self.c._learn_correction("get moving", "explore")
        self.assertFalse(ok)
        self.assertEqual(self.c.learned_intents, {})

    def test_taught_mapping_survives_save_and_reload(self):
        self.c._learn_correction("hows your charge", "battery")
        reloaded = companion.Companion._load_learned_intents(self.c)
        self.assertEqual(reloaded[self._key("hows your charge")]["command"],
                         "battery")


class ConsoleLogPairingTest(unittest.TestCase):
    def setUp(self):
        self.state = web_console.ConsoleState()

    def test_robot_lines_reference_last_user_text(self):
        self.state.add_log("you", "battery")
        self.state.add_log("robot", "My battery is at 7.8 volts.")
        entry = self.state.log[0]
        self.assertEqual(entry["kind"], "robot")
        self.assertEqual(entry["re"], "battery")

    def test_mark_feedback_targets_newest_unjudged_match(self):
        self.state.add_log("heard", "battery")
        self.state.add_log("robot", "same response")
        self.state.add_log("robot", "same response")
        self.assertTrue(self.state.mark_feedback("same response", "incorrect"))
        marked = [e for e in self.state.log if e.get("fb")]
        self.assertEqual(len(marked), 1)
        # Second judgment lands on the OTHER line, not the same one twice.
        self.assertTrue(self.state.mark_feedback("same response", "correct"))
        self.assertEqual(len([e for e in self.state.log if e.get("fb")]), 2)
        self.assertFalse(self.state.mark_feedback("same response", "correct"))

    def test_mark_feedback_unknown_response(self):
        self.assertFalse(self.state.mark_feedback("never said this", "correct"))


class ReflectionFeedbackDigestTest(unittest.TestCase):
    def test_incorrect_with_correction(self):
        import json
        line = reflection.Reflection._summarize_event(
            "picarx/intent/feedback",
            json.dumps({"verdict": "incorrect", "utterance": "hows your charge",
                        "correction": "battery"}))
        self.assertIn("MISUNDERSTOOD", line)
        self.assertIn("hows your charge", line)
        self.assertIn("battery", line)

    def test_correct(self):
        import json
        line = reflection.Reflection._summarize_event(
            "picarx/intent/feedback",
            json.dumps({"verdict": "correct", "utterance": "battery"}))
        self.assertIn("understood right", line)

    def test_missing_utterance_skipped(self):
        line = reflection.Reflection._summarize_event(
            "picarx/intent/feedback", "{\"verdict\": \"correct\"}")
        self.assertIsNone(line)


if __name__ == "__main__":
    unittest.main()
