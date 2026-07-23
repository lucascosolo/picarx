"""Proactive intent narration: field_agent names the place it's about to
explore ("Exploring the kitchen") when it knows where it is, and stays generic
(fail-soft) when it doesn't."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402


class ExploreNarrationTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def _speech(self):
        return " | ".join(p["text"] for p in self.fa.bus.of("picarx/audio/speak"))

    def test_names_the_place_when_known(self):
        self.fa.current_location = {"location_id": 1, "label": "the kitchen"}
        self.fa.handle_voice_command("explore")
        self.assertIn("Exploring the kitchen", self._speech())

    def test_generic_when_place_unknown(self):
        self.fa.current_location = None
        self.fa.handle_voice_command("explore")
        self.assertIn("Starting exploration", self._speech())


class CurrentPlaceLabelTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()

    def test_none_when_not_localized(self):
        self.fa.current_location = None
        self.assertIsNone(self.fa._current_place_label())

    def test_blank_label_is_none(self):
        self.fa.current_location = {"label": "   "}
        self.assertIsNone(self.fa._current_place_label())

    def test_real_label_returned_trimmed(self):
        self.fa.current_location = {"label": " den "}
        self.assertEqual(self.fa._current_place_label(), "den")


if __name__ == "__main__":
    unittest.main()
