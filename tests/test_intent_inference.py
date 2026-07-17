import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import speech_match  # noqa: E402
import field_agent  # noqa: E402
import companion  # noqa: E402


class DirectedCommandShapeTest(unittest.TestCase):
    def test_imperative_openers_detected(self):
        for text in ("take me to the kitchen",
                     "come with me",
                     "bring me the ball",
                     "follow me around"):
            self.assertTrue(speech_match.looks_directed_command(text), text)

    def test_politeness_prefixes_stripped(self):
        for text in ("hey robot take me to the kitchen",
                     "could you follow me please",
                     "please tell me a story",
                     "ok now go to the door"):
            self.assertTrue(speech_match.looks_directed_command(text), text)

    def test_declarative_chatter_ignored(self):
        for text in ("the weather is nice today",
                     "we stopped by earlier",
                     "i think that's fine",
                     "he wanted to go home",
                     ""):
            self.assertFalse(speech_match.looks_directed_command(text), text)


class EscalationGateTest(unittest.TestCase):
    """Unmatched command-shaped speech must reach the intent arbiter
    instead of being dropped - by vocabulary OR by sentence shape."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()  # fresh: conversation window closed

    def _uncertain(self):
        return self.fa.bus.of("picarx/audio/uncertain")

    def test_paraphrase_without_domain_words_escalates(self):
        self.fa.handle_voice_command("take me to the kitchen")
        self.assertTrue(self._uncertain())
        self.assertEqual(self._uncertain()[-1]["from"], "field_agent")

    def test_domain_vocabulary_still_escalates(self):
        self.fa.handle_voice_command("how's your charge holding up")
        self.assertTrue(self._uncertain())

    def test_plain_chatter_still_dropped(self):
        self.fa.handle_voice_command("the weather is nice today")
        self.assertEqual(self._uncertain(), [])
        self.assertEqual(self.fa.bus.of("picarx/audio/unhandled"), [])

    def test_repaired_text_never_reescalates(self):
        self.fa.handle_voice_command("take me to the kitchen",
                                     source="intent_repair")
        self.assertEqual(self._uncertain(), [])


class TokenPrecisionTest(unittest.TestCase):
    """Whole-token matching: word fragments inside other words must not
    fire commands."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_stopped_does_not_halt(self):
        self.fa.explore_mode = True
        self.fa.handle_voice_command("we stopped by earlier")
        self.assertTrue(self.fa.explore_mode)

    def test_literal_stop_still_halts(self):
        self.fa.explore_mode = True
        self.fa.handle_voice_command("please stop")
        self.assertFalse(self.fa.explore_mode)

    def test_in_charge_does_not_report_battery(self):
        self.fa.handle_voice_command("who's in charge here")
        speech = " ".join(p["text"] for p in self.fa.bus.of("picarx/audio/speak"))
        self.assertNotIn("battery", speech.lower())
        # ...but it does escalate (contains domain vocabulary), so the
        # arbiter can still decide it was noise or chat.
        self.assertTrue(self.fa.bus.of("picarx/audio/uncertain"))

    def test_battery_token_still_reports(self):
        self.fa.handle_voice_command("battery")
        speech = " ".join(p["text"] for p in self.fa.bus.of("picarx/audio/speak"))
        self.assertIn("battery", speech.lower())


class RepairMotionGuardTest(unittest.TestCase):
    """Motion must never start from the arbiter's rewrite of a garbled
    transcript - enforced in field_agent, not just by the allowlist."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_repaired_explore_does_not_drive(self):
        self.fa.handle_voice_command("explore", source="intent_repair")
        self.assertFalse(self.fa.explore_mode)

    def test_repaired_go_to_does_not_request_goal(self):
        self.fa.handle_voice_command("go to the kitchen", source="intent_repair")
        self.assertIsNone(self.fa.bus.last("picarx/exploration/goal_request"))
        self.assertFalse(self.fa.explore_mode)

    def test_heard_explore_still_drives(self):
        self.fa.handle_voice_command("explore")
        self.assertTrue(self.fa.explore_mode)


class ArbiterAllowlistTest(unittest.TestCase):
    def test_new_memory_commands_allowed(self):
        for cmd in ("where is the bottle", "what's in the kitchen",
                    "call this place kitchen", "who am i", "where are you"):
            self.assertTrue(companion.Companion._intent_allowed(cmd), cmd)

    def test_motion_commands_still_excluded(self):
        for cmd in ("explore", "go to the kitchen", "forward", "turn left",
                    "follow me"):
            self.assertFalse(companion.Companion._intent_allowed(cmd), cmd)

    def test_parameterless_prefix_rejected(self):
        self.assertFalse(companion.Companion._intent_allowed("where is "))
        self.assertFalse(companion.Companion._intent_allowed("call this place "))


if __name__ == "__main__":
    unittest.main()
