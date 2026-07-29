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

    def test_second_person_talk_is_addressed(self):
        for text in ("do you like jokes", "what do you think",
                     "how are you feeling", "you are funny"):
            a = attention.classify(text, wake_phrases=WAKE)
            self.assertEqual(a.reason, attention.CHAT_SHAPE, text)
            self.assertEqual(a.remainder, text)

    def test_question_shape_is_addressed(self):
        a = attention.classify("why is the sky blue", wake_phrases=WAKE)
        self.assertEqual(a.reason, attention.CHAT_SHAPE)

    def test_overheard_statements_stay_unaddressed(self):
        # No second person, no question opener - the television keeps talking
        # to itself.
        for text in ("the weather is nice today", "i had lunch already",
                     "we should leave at six"):
            self.assertFalse(attention.classify(text, wake_phrases=WAKE).addressed,
                             text)

    def test_fragments_are_too_short_to_be_talk(self):
        for text in ("you", "why", "how come"):
            self.assertFalse(attention.looks_conversational(text), text)

    def test_command_shape_still_wins_over_chat_shape(self):
        # "what time is it" is a capability, not conversation - it must keep
        # routing to the command path. So does a question that merely mentions
        # robot vocabulary ("do you like music"): it goes to the intent
        # arbiter, which has its own chat verdict for exactly this case.
        for text in ("what time is it", "do you like music"):
            a = attention.classify(text, wake_phrases=WAKE)
            self.assertEqual(a.reason, attention.COMMAND_SHAPE, text)

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
