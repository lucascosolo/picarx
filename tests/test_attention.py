"""The shared attention model (layer_b/attention.py): the "is this addressed to
me?" and "is this an answer to my open question?" truth tables that field_agent
and the dialog broker both depend on. Pure functions - no bus, no clock."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import attention  # noqa: E402

WAKE = ("robot", "hey robot", "computer")


class NormalizeWakePhrasesTest(unittest.TestCase):
    def test_comma_string_from_env(self):
        self.assertEqual(attention.normalize_wake_phrases("robot, computer , Hey Robot"),
                         ("robot", "computer", "hey robot"))

    def test_json_list_from_config(self):
        self.assertEqual(attention.normalize_wake_phrases(["Robot", "Computer"]),
                         ("robot", "computer"))

    def test_empty_and_none(self):
        self.assertEqual(attention.normalize_wake_phrases(None), ())
        self.assertEqual(attention.normalize_wake_phrases(""), ())
        self.assertEqual(attention.normalize_wake_phrases([]), ())


class StripWakePhraseTest(unittest.TestCase):
    def test_wake_prefix_stripped(self):
        self.assertEqual(attention.strip_wake_phrase("robot what do you see", WAKE),
                         "what do you see")

    def test_bare_wake_becomes_hello(self):
        self.assertEqual(attention.strip_wake_phrase("robot", WAKE), "hello")
        self.assertEqual(attention.strip_wake_phrase("computer!", WAKE), "hello")

    def test_whole_word_only(self):
        # "robotics" starts with "robot" but was never addressed to the robot.
        self.assertIsNone(attention.strip_wake_phrase("robotics class was fun", WAKE))

    def test_no_wake_phrase(self):
        self.assertIsNone(attention.strip_wake_phrase("what time is it", WAKE))


class ClassifyTest(unittest.TestCase):
    def test_wake_wins(self):
        a = attention.classify("robot what do you see", wake_phrases=WAKE)
        self.assertEqual(a.reason, attention.WAKE)
        self.assertTrue(a.addressed)
        self.assertEqual(a.remainder, "what do you see")

    def test_conversation_window(self):
        a = attention.classify("what's the weather", wake_phrases=WAKE,
                               in_conversation=True)
        self.assertEqual(a.reason, attention.CONVERSATION)
        self.assertEqual(a.remainder, "what's the weather")

    def test_command_vocabulary_shape(self):
        a = attention.classify("explore", wake_phrases=WAKE)
        self.assertEqual(a.reason, attention.COMMAND_SHAPE)

    def test_imperative_shape_without_domain_word(self):
        a = attention.classify("take me to the kitchen", wake_phrases=WAKE)
        self.assertEqual(a.reason, attention.COMMAND_SHAPE)

    def test_plain_chatter_not_addressed(self):
        a = attention.classify("the weather is nice today", wake_phrases=WAKE)
        self.assertFalse(a.addressed)
        self.assertIsNone(a.reason)

    def test_precedence_wake_over_conversation(self):
        # A wake phrase reports WAKE even mid-conversation (so the remainder is
        # stripped), not CONVERSATION.
        a = attention.classify("robot stop", wake_phrases=WAKE, in_conversation=True)
        self.assertEqual(a.reason, attention.WAKE)
        self.assertEqual(a.remainder, "stop")

    def test_is_addressed_helper(self):
        self.assertEqual(attention.is_addressed("explore", wake_phrases=WAKE),
                         (True, attention.COMMAND_SHAPE))
        self.assertEqual(attention.is_addressed("lovely day", wake_phrases=WAKE),
                         (False, None))


def _q(kind, options=None):
    return attention.Question("id", "asker", kind=kind, options=options or [])


class AnswersQuestionLabelTest(unittest.TestCase):
    def test_plain_noun_option_and_affirmations_pass(self):
        q = _q(attention.LABEL, ["chair", "speaker"])
        for text in ("a coffee mug", "it's the speaker", "yes", "no, a mug",
                     "yes that's right"):
            self.assertTrue(attention.answers_question(text, q), text)

    def test_commands_and_questions_rejected(self):
        q = _q(attention.LABEL, ["chair"])
        for text in ("who am i", "what do you see", "stop", "follow me",
                     "come here", "go to the kitchen", "turn left"):
            self.assertFalse(attention.answers_question(text, q), text)

    def test_empty_rejected(self):
        self.assertFalse(attention.answers_question("", _q(attention.LABEL)))


class AnswersQuestionCorrectionTest(unittest.TestCase):
    def test_clarification_including_a_command_word_counts(self):
        q = _q(attention.CORRECTION)
        self.assertTrue(attention.answers_question("i wanted the battery", q))
        self.assertTrue(attention.answers_question("battery", q))

    def test_feedback_verdict_is_not_the_answer(self):
        # "that's wrong" is intent feedback, graded on its own path.
        q = _q(attention.CORRECTION)
        self.assertFalse(attention.answers_question("that's wrong", q))
        self.assertFalse(attention.answers_question("good robot", q))

    def test_empty_rejected(self):
        self.assertFalse(attention.answers_question("", _q(attention.CORRECTION)))


class AnswersQuestionYesNoFreeformTest(unittest.TestCase):
    def test_yes_no(self):
        q = _q(attention.YES_NO)
        self.assertTrue(attention.answers_question("yes", q))
        self.assertTrue(attention.answers_question("no", q))
        self.assertFalse(attention.answers_question("banana", q))

    def test_freeform_takes_anything(self):
        q = _q(attention.FREEFORM)
        self.assertTrue(attention.answers_question("tell me a story", q))
        self.assertFalse(attention.answers_question("", q))


if __name__ == "__main__":
    unittest.main()
