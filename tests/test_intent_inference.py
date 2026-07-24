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


class ForwardsMissesToBrokerTest(unittest.TestCase):
    """field_agent no longer classifies addressing itself: it hands every
    command-miss to the dialog broker on picarx/audio/directed, and the broker
    decides chat vs intent-arbiter vs drop (see test_dialog). Here we only check
    field_agent forwards the right thing."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def _misses(self):
        return [m for m in self.fa.bus.of("picarx/audio/directed")
                if not m.get("handled")]

    def test_unmatched_speech_is_forwarded_to_the_broker(self):
        self.fa.handle_voice_command("take me to the kitchen")
        self.assertTrue(self._misses())
        self.assertEqual(self._misses()[-1]["text"], "take me to the kitchen")
        self.assertFalse(self._misses()[-1]["from_repair"])

    def test_domain_vocabulary_is_also_forwarded(self):
        self.fa.handle_voice_command("how's your charge holding up")
        self.assertTrue(self._misses())

    def test_plain_chatter_is_forwarded_for_the_broker_to_judge(self):
        # field_agent no longer drops locally; the broker classifies and drops.
        self.fa.handle_voice_command("the weather is nice today")
        self.assertEqual(self._misses()[-1]["text"], "the weather is nice today")

    def test_repaired_miss_carries_the_repair_flag(self):
        # The broker reads this to keep the loop guard (no re-escalation).
        self.fa.handle_voice_command("take me to the kitchen",
                                     source="intent_repair")
        self.assertTrue(self._misses()[-1]["from_repair"])

    def test_matched_command_is_not_forwarded_as_a_miss(self):
        self.fa.handle_voice_command("battery")
        self.assertEqual(self._misses(), [])
        # ...but it tells the broker a command WAS handled (holds the window).
        self.assertTrue(any(m.get("handled")
                            for m in self.fa.bus.of("picarx/audio/directed")))


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
        # ...but it IS forwarded to the broker (which escalates command-shaped
        # misses to the intent arbiter) rather than acted on locally.
        self.assertTrue([m for m in self.fa.bus.of("picarx/audio/directed")
                         if not m.get("handled")])

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
