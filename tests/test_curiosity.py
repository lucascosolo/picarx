"""Perception feedback loop: vision's ambiguity signal, curiosity's spoken
questions and answer capture, reflection's immediate human-label writes, and
the web console's identification-vs-interpretation feedback split."""
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import vision_basic  # noqa: E402
import curiosity  # noqa: E402
import reflection  # noqa: E402
import web_console  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


class ContestedLabelTest(unittest.TestCase):
    def test_dominant_winner_is_not_contested(self):
        self.assertIsNone(vision_basic.contested_label({"chair": 9, "sofa": 1}))

    def test_close_runner_up_is_contested(self):
        self.assertEqual(
            vision_basic.contested_label({"chair": 5, "tvmonitor": 4}), "tvmonitor")

    def test_single_label_never_contested(self):
        self.assertIsNone(vision_basic.contested_label({"chair": 3}))
        self.assertIsNone(vision_basic.contested_label({}))
        self.assertIsNone(vision_basic.contested_label(None))

    def test_returns_the_actual_runner_up(self):
        # Three-way split: the runner-up (not just "any other") is offered.
        alt = vision_basic.contested_label({"chair": 6, "sofa": 5, "bench": 1})
        self.assertEqual(alt, "sofa")


class ParseLabelAnswerTest(unittest.TestCase):
    def test_offered_option_wins(self):
        self.assertEqual(
            curiosity.parse_label_answer("it's the speaker", ["chair", "speaker"]),
            "speaker")

    def test_fresh_label_from_free_speech(self):
        self.assertEqual(
            curiosity.parse_label_answer("that's a coffee mug", ["chair"]),
            "coffee mug")

    def test_bare_no_yields_nothing(self):
        self.assertIsNone(curiosity.parse_label_answer("no", ["chair"]))

    def test_option_substring_not_falsely_matched(self):
        # "chairman" must not match the option "chair".
        self.assertEqual(
            curiosity.parse_label_answer("a chairman", ["chair", "sofa"]),
            "chairman")


class CuriosityAskTest(unittest.TestCase):
    def setUp(self):
        self.c = curiosity.Curiosity()  # __init__ only builds a FakeBus + locks

    def _ambiguous(self, oid="object_0"):
        return {"objects": [{"id": oid, "label": "chair", "alt_label": "speaker",
                             "confidence": 0.7}]}

    def test_ambiguous_object_asks_a_question(self):
        self.c.on_objects(self._ambiguous())
        spoken = self.c.bus.of("picarx/audio/speak")
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0]["text"], "Is that a chair or a speaker?")
        self.assertEqual(spoken[0]["kind"], "question")
        self.assertIsNotNone(self.c.pending)

    def test_low_confidence_single_guess_asks(self):
        self.c.on_objects({"objects": [{"id": "object_1", "label": "bottle",
                                        "alt_label": None, "confidence": 0.52}]})
        self.assertIn("bottle", self.c.bus.last("picarx/audio/speak")["text"])
        self.assertEqual(self.c.pending["options"], ["bottle"])

    def test_confident_unambiguous_object_is_left_alone(self):
        self.c.on_objects({"objects": [{"id": "object_2", "label": "chair",
                                        "alt_label": None, "confidence": 0.95}]})
        self.assertEqual(self.c.bus.of("picarx/audio/speak"), [])
        self.assertIsNone(self.c.pending)

    def test_one_question_at_a_time(self):
        self.c.on_objects(self._ambiguous("object_0"))
        self.c.on_objects(self._ambiguous("object_9"))  # different object, pending open
        self.assertEqual(len(self.c.bus.of("picarx/audio/speak")), 1)

    def test_same_object_never_asked_twice(self):
        self.c.on_objects(self._ambiguous("object_0"))
        # Clear the pending question and cooldown, but keep the asked-memory.
        self.c.pending = None
        self.c.last_ask_at = 0.0
        self.c.on_objects(self._ambiguous("object_0"))
        self.assertEqual(len(self.c.bus.of("picarx/audio/speak")), 1)


class CuriosityAnswerTest(unittest.TestCase):
    def setUp(self):
        self.c = curiosity.Curiosity()
        self.c.on_objects({"objects": [{"id": "object_0", "label": "chair",
                                        "alt_label": "speaker", "confidence": 0.7}]})
        self.c.bus.clear()

    def test_naming_the_other_option_publishes_correction(self):
        self.c.on_heard({"text": "it's a speaker"})
        label = self.c.bus.last("picarx/perception/label")
        self.assertEqual(label["correct_label"], "speaker")
        self.assertEqual(label["guess"], "chair")
        self.assertEqual(label["object_id"], "object_0")
        self.assertEqual(label["origin"], "voice")
        self.assertIsNone(self.c.pending)

    def test_yes_confirms_the_guess(self):
        self.c.on_heard({"text": "yes that's right"})
        label = self.c.bus.last("picarx/perception/label")
        self.assertEqual(label["correct_label"], "chair")
        # A plain confirmation needs no spoken reply (chatter reduction).
        self.assertEqual(self.c.bus.of("picarx/audio/speak"), [])

    def test_correction_gets_a_terse_acknowledgement(self):
        self.c.on_heard({"text": "a speaker"})
        self.assertIn("speaker", self.c.bus.last("picarx/audio/speak")["text"])

    def test_repair_echo_is_not_taken_as_answer(self):
        self.c.on_heard({"text": "speaker", "source": "intent_repair"})
        self.assertIsNone(self.c.bus.last("picarx/perception/label"))
        self.assertIsNotNone(self.c.pending)  # still waiting for a real answer

    def test_stale_question_expires(self):
        self.c.pending["until"] = curiosity.time.time() - 1
        self.c.on_heard({"text": "a speaker"})
        self.assertIsNone(self.c.bus.last("picarx/perception/label"))


class ReflectionHumanLabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.r = reflection.Reflection.__new__(reflection.Reflection)
        self.r.store = SemanticStore(readonly=False,
                                     db_path=os.path.join(self.tmp, "semantic.db"))

    def _facts_text(self, subject):
        return " ".join(f["fact"] for f in self.r.store.facts_for(subject, limit=9))

    def test_correction_writes_identity_and_confusion_facts(self):
        self.r.on_label({"correct_label": "speaker", "guess": "chair",
                         "origin": "voice"})
        self.assertIn("speaker", self._facts_text("speaker"))
        # The detector's mistake becomes a durable fact about my own vision.
        vision = self._facts_text("vision")
        self.assertIn("speaker", vision)
        self.assertIn("chair", vision)

    def test_high_confidence_lands_immediately(self):
        self.r.on_label({"correct_label": "mug", "guess": "cup"})
        facts = self.r.store.facts_for("mug", limit=1)
        self.assertTrue(facts)
        self.assertGreaterEqual(facts[0]["confidence"], 0.85)

    def test_confirmation_writes_no_confusion_fact(self):
        self.r.on_label({"correct_label": "chair", "guess": "chair"})
        self.assertTrue(self.r.store.facts_for("chair", limit=1))
        self.assertEqual(self.r.store.facts_for("vision", limit=1), [])

    def test_empty_label_is_ignored(self):
        self.r.on_label({"correct_label": "  ", "guess": "chair"})
        self.assertEqual(self.r.store.fact_count(), 0)


class ConsoleObservationFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.state = web_console.ConsoleState()

    def test_observation_speak_is_tagged_on_the_log_line(self):
        prev = web_console.STATE
        web_console.STATE = self.state
        try:
            web_console.on_speak({"text": "looks like a chair",
                                  "kind": "observation", "label": "chair"})
        finally:
            web_console.STATE = prev
        entry = self.state.log[0]
        self.assertEqual(entry["obs"]["label"], "chair")
        self.assertEqual(entry["obs"]["kind"], "observation")

    def test_plain_robot_line_has_no_obs_tag(self):
        self.state.add_log("robot", "My battery is at 7.8 volts.")
        self.assertNotIn("obs", self.state.log[0])


if __name__ == "__main__":
    unittest.main()
