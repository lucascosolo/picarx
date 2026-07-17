import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import tools_registry as tr  # noqa: E402


class ParseDialTest(unittest.TestCase):
    def test_decimal_digits(self):
        self.assertEqual(tr.parse_dial("tune to 98.7"), "98.7")

    def test_space_separated_digits(self):
        self.assertEqual(tr.parse_dial("tune to 98 7"), "98.7")

    def test_three_digit_run(self):
        self.assertEqual(tr.parse_dial("tune to 987"), "98.7")

    def test_four_digit_run(self):
        # Regression: the old pattern capped at 3 digits, so "1025" never
        # matched and the documented "1025 -> 102.5" branch was dead code.
        self.assertEqual(tr.parse_dial("tune to 1025"), "102.5")

    def test_bare_two_digits(self):
        self.assertEqual(tr.parse_dial("tune to 98"), "98")

    def test_word_form_grouped(self):
        self.assertEqual(tr.parse_dial("tune to ninety eight point seven"), "98.7")

    def test_word_form_digit_by_digit(self):
        self.assertEqual(tr.parse_dial("tune to one oh two point five"), "102.5")

    def test_word_form_hundred(self):
        self.assertEqual(tr.parse_dial("tune to one hundred eight"), "108")

    def test_no_dial(self):
        self.assertIsNone(tr.parse_dial("play some jazz"))


class TuneRuleScopeTest(unittest.TestCase):
    """The tune rule must only fire on actual radio vocabulary - a bare
    "... to <number> ..." is any utterance, not a tune request."""

    def _route(self, text):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": text})
        return registry.bus.of("picarx/tools/radio")

    def test_tune_to_number_routes(self):
        cmds = self._route("tune to 98.7")
        self.assertTrue(cmds)
        self.assertEqual(cmds[-1]["command"], "play")
        self.assertEqual(cmds[-1]["dial"], "98.7")

    def test_plain_to_number_does_not_route(self):
        # Regression: "\bto\b.*\d" used to hijack these into radio tunes.
        self.assertEqual(self._route("set a timer to 20 minutes"), [])
        self.assertEqual(self._route("count to 10"), [])


if __name__ == "__main__":
    unittest.main()
