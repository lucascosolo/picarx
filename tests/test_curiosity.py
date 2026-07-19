"""Perception feedback loop: vision's ambiguity signal, curiosity's spoken
questions and answer capture, reflection's immediate human-label writes, and
the web console's identification-vs-interpretation feedback split."""
import os
import queue
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import vision_basic  # noqa: E402
import curiosity  # noqa: E402
import companion  # noqa: E402
import field_agent  # noqa: E402
import reflection  # noqa: E402
import web_console  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResp:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeClient:
    """Minimal Anthropic client stand-in: returns a fixed reply text."""
    def __init__(self, reply):
        self._reply = reply
        self.messages = self

    def create(self, **kwargs):
        return _FakeResp(self._reply)


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


class CuriosityLlmEscalationTest(unittest.TestCase):
    def setUp(self):
        self.c = curiosity.Curiosity()
        self.c.on_objects({"objects": [{"id": "object_0", "label": "chair",
                                        "alt_label": "speaker", "confidence": 0.7}]})
        self.c.bus.clear()

    def _expire(self):
        self.c.pending["until"] = curiosity.time.time() - 1

    def test_unanswered_question_escalates_to_llm(self):
        self._expire()
        self.c.on_objects({"objects": []})
        req = self.c.bus.last("picarx/perception/identify_request")
        self.assertIsNotNone(req)
        self.assertEqual(req["object_id"], "object_0")
        self.assertEqual(req["guess"], "chair")
        self.assertIsNone(self.c.pending)

    def test_answered_question_never_escalates(self):
        self.c.on_heard({"text": "it's a speaker"})   # answered before timeout
        self.c.on_objects({"objects": []})
        self.assertIsNone(self.c.bus.last("picarx/perception/identify_request"))

    def test_escalation_is_throttled(self):
        self._expire()
        self.c.on_objects({"objects": []})            # escalates once
        # A second timed-out question inside the cooldown must not re-escalate.
        self.c.pending = {"object_id": "object_9", "guess": "bottle",
                          "options": ["bottle"], "until": curiosity.time.time() - 1}
        self.c.on_objects({"objects": []})
        self.assertEqual(len(self.c.bus.of("picarx/perception/identify_request")), 1)


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


class CompanionIdentifyTest(unittest.TestCase):
    def _make(self, reply="watering can"):
        c = companion.Companion.__new__(companion.Companion)
        c.bus = harness.FakeBus()
        c.lock = threading.Lock()
        c.work_queue = queue.Queue()
        c._last_identify_at = 0.0
        c._client = _FakeClient(reply)
        c.latest_frame_b64 = "ZmFrZQ=="   # a fresh frame so _get_camera_frame returns at once
        c.latest_frame_at = time.time()
        return c

    def test_clean_label_normalizes(self):
        self.assertEqual(companion.Companion._clean_identify_label("Watering Can."),
                         "watering can")

    def test_clean_label_rejects_unknown_and_sentences(self):
        self.assertIsNone(companion.Companion._clean_identify_label("unknown"))
        self.assertIsNone(companion.Companion._clean_identify_label(
            "it is a coffee mug on the table"))
        self.assertIsNone(companion.Companion._clean_identify_label(""))

    def test_on_identify_queues_then_throttles(self):
        c = self._make()
        c.on_identify({"object_id": "object_0", "guess": "chair"})
        self.assertEqual(c.work_queue.get_nowait()[0], "identify")
        c.on_identify({"object_id": "object_0", "guess": "chair"})  # within cooldown
        self.assertTrue(c.work_queue.empty())

    def test_identify_publishes_llm_label_and_speaks(self):
        c = self._make("watering can")
        c._identify_object({"object_id": "object_0", "guess": "chair"})
        label = c.bus.last("picarx/perception/label")
        self.assertEqual(label["correct_label"], "watering can")
        self.assertEqual(label["guess"], "chair")
        self.assertEqual(label["origin"], "llm")
        speak = c.bus.last("picarx/audio/speak")
        self.assertIn("watering can", speak["text"])
        self.assertEqual(speak["kind"], "observation")

    def test_identify_unsure_publishes_nothing(self):
        c = self._make("unknown")
        c._identify_object({"object_id": "object_0", "guess": "chair"})
        self.assertIsNone(c.bus.last("picarx/perception/label"))


class FieldAgentAnnounceTaggingTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_single_object_claim_is_tagged_with_id(self):
        # The reported bug: a mid-evasion "a sofa is closing in" must carry the
        # object so its X asks WHAT it is (and can retrain memory by id).
        self.fa.announce("A sofa is closing in, backing away.",
                         kind="observation", label="sofa", object_id="object_2")
        msg = self.fa.bus.last("picarx/audio/speak")
        self.assertEqual(msg["kind"], "observation")
        self.assertEqual(msg["objects"], [{"label": "sofa", "id": "object_2"}])

    def test_multi_object_claim_lists_each(self):
        self.fa.announce("I see 2: a chair, a bottle.", force=True,
                         kind="observation",
                         objects=[{"label": "chair", "id": "object_1"},
                                  {"label": "bottle", "id": "object_2"}])
        self.assertEqual(len(self.fa.bus.last("picarx/audio/speak")["objects"]), 2)

    def test_plain_announcement_carries_no_object_tag(self):
        self.fa.announce("Stopping.", force=True)
        self.assertNotIn("objects", self.fa.bus.last("picarx/audio/speak"))


class ConsoleObservationFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.state = web_console.ConsoleState()

    def test_observation_speak_is_tagged_on_the_log_line(self):
        prev = web_console.STATE
        web_console.STATE = self.state
        try:
            web_console.on_speak({"text": "looks like a chair", "kind": "observation",
                                  "objects": [{"label": "chair", "id": "object_3"}]})
        finally:
            web_console.STATE = prev
        entry = self.state.log[0]
        self.assertEqual(entry["obs"]["items"][0]["label"], "chair")
        self.assertEqual(entry["obs"]["items"][0]["id"], "object_3")
        self.assertEqual(entry["obs"]["kind"], "observation")

    def test_plain_robot_line_has_no_obs_tag(self):
        self.state.add_log("robot", "My battery is at 7.8 volts.")
        self.assertNotIn("obs", self.state.log[0])


if __name__ == "__main__":
    unittest.main()
