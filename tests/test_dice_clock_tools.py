import os
import random
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import capabilities  # noqa: E402
import speech_match  # noqa: E402
import tools_registry as tr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "layer_b", "modules", "tools"))
import clock as clock_tool  # noqa: E402
import dice as dice_tool  # noqa: E402


def route(text):
    """What the registry publishes for a spoken utterance."""
    registry = tr.ToolsRegistry()
    registry.on_heard({"text": text})
    return registry.bus


class DiceRoutingTest(unittest.TestCase):
    def _payload(self, text):
        return route(text).last(capabilities.DICE_TOPIC)

    def test_plain_roll_is_one_six_sided_die(self):
        self.assertEqual(self._payload("roll the dice"),
                         {"command": "roll", "count": 1, "sides": 6})

    def test_spoken_count(self):
        self.assertEqual(self._payload("roll two dice")["count"], 2)

    def test_digit_count(self):
        self.assertEqual(self._payload("roll 3 dice")["count"], 3)

    def test_pair_of_dice(self):
        self.assertEqual(self._payload("roll a pair of dice")["count"], 2)

    def test_dice_shorthand_sets_sides(self):
        self.assertEqual(self._payload("roll a d20"),
                         {"command": "roll", "count": 1, "sides": 20})

    def test_spelled_sided_die_is_not_read_as_a_count(self):
        payload = self._payload("roll a 20 sided die")
        self.assertEqual(payload["sides"], 20)
        self.assertEqual(payload["count"], 1)

    def test_count_is_bounded(self):
        self.assertEqual(self._payload("roll 99 dice")["count"],
                         capabilities.MAX_DICE)

    def test_coin_flip(self):
        self.assertEqual(self._payload("flip a coin"), {"command": "flip"})
        self.assertEqual(self._payload("heads or tails"), {"command": "flip"})

    def test_unrelated_speech_does_not_roll(self):
        self.assertIsNone(self._payload("the dice game was fun last night"))
        self.assertIsNone(self._payload("tell me about coins"))


class ClockRoutingTest(unittest.TestCase):
    def _payload(self, text):
        return route(text).last(capabilities.CLOCK_TOPIC)

    def test_time_questions(self):
        for text in ("what time is it", "what's the time",
                     "tell me the time", "do you know what time it is"):
            self.assertEqual(self._payload(text), {"command": "time"}, text)

    def test_date_questions(self):
        for text in ("what's the date", "what day is it", "what is today's date"):
            self.assertEqual(self._payload(text), {"command": "date"}, text)

    def test_reminder_phrases_still_belong_to_reminders(self):
        bus = route("remind me in 10 minutes to take out the trash")
        self.assertIsNone(bus.last(capabilities.CLOCK_TOPIC))
        self.assertIsNotNone(bus.last(capabilities.REMINDER_SET_TOPIC))

    def test_radio_status_question_still_belongs_to_radio(self):
        bus = route("what's playing")
        self.assertIsNone(bus.last(capabilities.CLOCK_TOPIC))
        self.assertEqual(bus.last(capabilities.RADIO_TOPIC)["command"], "status")


class DiceToolTest(unittest.TestCase):
    def _dice(self, seed=1):
        return dice_tool.Dice(rng=random.Random(seed))

    def test_roll_speaks_and_publishes_a_result(self):
        d = self._dice()
        d.on_request({"command": "roll", "count": 2, "sides": 6})
        result = d.bus.last(dice_tool.RESULT_TOPIC)
        self.assertEqual(len(result["values"]), 2)
        self.assertEqual(result["total"], sum(result["values"]))
        for value in result["values"]:
            self.assertTrue(1 <= value <= 6)
        self.assertIn(str(result["total"]), d.bus.last("picarx/audio/speak")["text"])

    def test_single_die_is_spoken_as_one_number(self):
        d = self._dice()
        d.on_request({"command": "roll"})
        spoken = d.bus.last("picarx/audio/speak")["text"]
        self.assertNotIn("total", spoken)

    def test_model_supplied_bounds_are_clamped_not_trusted(self):
        d = self._dice()
        result = d.roll(count=10_000, sides=10_000)
        self.assertEqual(len(result["values"]), dice_tool.MAX_DICE)
        self.assertEqual(result["sides"], dice_tool.MAX_SIDES)

    def test_garbage_parameters_fall_back_to_defaults(self):
        result = self._dice().roll(count="lots", sides=None)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["sides"], dice_tool.DEFAULT_SIDES)

    def test_flip_is_one_of_two_faces(self):
        d = self._dice()
        d.on_request({"command": "flip"})
        self.assertIn(d.bus.last(dice_tool.RESULT_TOPIC)["face"],
                      ("heads", "tails"))

    def test_unknown_command_is_ignored_silently(self):
        d = self._dice()
        d.on_request({"command": "explode"})
        self.assertIsNone(d.bus.last("picarx/audio/speak"))

    def test_malformed_payload_does_not_raise(self):
        d = self._dice()
        d.on_request("roll")
        d.on_request(None)
        self.assertIsNone(d.bus.last(dice_tool.RESULT_TOPIC))

    def test_result_is_journaled_for_later_evidence(self):
        d = self._dice()
        d.on_request({"command": "flip"})
        decision = d.bus.last("picarx/decision")
        self.assertEqual(decision["source"], "dice")


class ClockToolTest(unittest.TestCase):
    def _clock(self, when):
        return clock_tool.Clock(clock=lambda: when)

    def test_spoken_time_reads_like_a_person(self):
        self.assertEqual(clock_tool.spoken_time(datetime(2026, 7, 28, 16, 20)),
                         "twenty past four in the afternoon")
        self.assertEqual(clock_tool.spoken_time(datetime(2026, 7, 28, 9, 0)),
                         "nine o'clock in the morning")
        self.assertEqual(clock_tool.spoken_time(datetime(2026, 7, 28, 8, 45)),
                         "quarter to nine in the morning")
        self.assertEqual(clock_tool.spoken_time(datetime(2026, 7, 28, 22, 30)),
                         "half past ten at night")

    def test_midnight_and_noon_are_twelve(self):
        self.assertTrue(clock_tool.spoken_time(
            datetime(2026, 7, 28, 0, 5)).startswith("five past twelve"))
        self.assertTrue(clock_tool.spoken_time(
            datetime(2026, 7, 28, 12, 5)).startswith("five past twelve"))

    def test_spoken_date_ordinals(self):
        self.assertEqual(clock_tool.spoken_date(datetime(2026, 7, 1)),
                         "Wednesday, July 1st")
        self.assertEqual(clock_tool.spoken_date(datetime(2026, 7, 28)),
                         "Tuesday, July 28th")
        self.assertEqual(clock_tool.spoken_date(datetime(2026, 7, 11)),
                         "Saturday, July 11th")

    def test_time_request_speaks_and_publishes_iso(self):
        c = self._clock(datetime(2026, 7, 28, 16, 20))
        c.on_request({"command": "time"})
        self.assertIn("twenty past four", c.bus.last("picarx/audio/speak")["text"])
        self.assertEqual(c.bus.last(clock_tool.RESULT_TOPIC)["iso"],
                         "2026-07-28T16:20:00")

    def test_date_and_datetime_requests(self):
        c = self._clock(datetime(2026, 7, 28, 16, 20))
        self.assertIn("July 28th", c.answer("date")["spoken"])
        both = c.answer("datetime")["spoken"]
        self.assertIn("twenty past four", both)
        self.assertIn("July 28th", both)

    def test_unknown_command_falls_back_to_the_time(self):
        c = self._clock(datetime(2026, 7, 28, 16, 20))
        self.assertEqual(c.answer("weather")["command"], "time")

    def test_malformed_payload_does_not_raise(self):
        c = self._clock(datetime(2026, 7, 28, 16, 20))
        c.on_request(["time"])
        self.assertIsNone(c.bus.last(clock_tool.RESULT_TOPIC))


class VocabularyTest(unittest.TestCase):
    def test_new_capabilities_are_in_the_shared_catalog(self):
        names = [entry["name"] for entry in capabilities.describe()]
        self.assertIn("dice", names)
        self.assertIn("clock", names)

    def test_time_questions_read_as_commands_without_a_wake_word(self):
        canon = speech_match.canonicalize("what time is it")
        self.assertTrue(speech_match.looks_command_like(canon))

    def test_near_miss_snaps_onto_the_new_vocabulary(self):
        # Conservative by design: only a close miss snaps, and "dise" (0.75
        # similar) deliberately does not.
        self.assertEqual(speech_match.canonicalize("check the clok"),
                         "check clock")


if __name__ == "__main__":
    unittest.main()
