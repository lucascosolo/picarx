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


class RemoteRuleTest(unittest.TestCase):
    def test_ssh_ip_routes_to_remote_assist(self):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": "ssh into 192.168.1.20"})
        self.assertEqual(registry.bus.last("picarx/tools/remote_assist"),
                         {"command": "connect", "host": "192.168.1.20"})

    def test_place_connection_does_not_route(self):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": "connect to the kitchen"})
        self.assertIsNone(registry.bus.last("picarx/tools/remote_assist"))

    def test_explicit_write_grant_routes_to_session_authorization(self):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": "give the remote host write access"})
        self.assertEqual(registry.bus.last("picarx/tools/remote_assist"),
                         {"command": "authorize_write", "confirmed": True})

    def test_write_revoke_routes_to_session_revoke(self):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": "revoke remote write access"})
        self.assertEqual(registry.bus.last("picarx/tools/remote_assist"),
                         {"command": "revoke_write"})


class NotesReminderRuleTest(unittest.TestCase):
    def _route(self, text, topic):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": text})
        return registry.bus.last(topic)

    def test_relative_reminder_example_routes_without_llm(self):
        payload = self._route("remind me in 10 minutes to take out the trash",
                              tr.REMINDER_SET_TOPIC)
        self.assertEqual(payload["message"], "take out trash")
        self.assertEqual(payload["delay_minutes"], 10.0)

    def test_spoken_relative_reminder_and_clock_time(self):
        payload = self._route("remind me in ten minutes to call Sam",
                              tr.REMINDER_SET_TOPIC)
        self.assertEqual(payload["delay_minutes"], 10.0)
        payload = self._route("remind me at 18:30 to call Sam", tr.REMINDER_SET_TOPIC)
        self.assertEqual(payload["at"], "18:30")

    def test_single_note_and_meeting_controls_are_distinct(self):
        note = self._route("take a note buy milk", tr.NOTES_TOPIC)
        self.assertEqual(note["command"], "create")
        self.assertEqual(note["text"], "buy milk")
        start = self._route("start taking meeting notes", tr.NOTES_TOPIC)
        self.assertEqual(start["command"], "start")
        self.assertTrue(start["confirmed"])

    def test_delete_reminder_is_explicitly_confirmed(self):
        payload = self._route("cancel reminder about trash",
                              tr.REMINDER_CONTROL_TOPIC)
        self.assertEqual(payload["command"], "delete")
        self.assertEqual(payload["query"], "trash")
        self.assertTrue(payload["confirmed"])

    def test_unparsed_tool_phrase_enters_learning_path(self):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": "set a reminder for later"})
        uncertain = registry.bus.last("picarx/audio/uncertain")
        self.assertEqual(uncertain["text"], "set a reminder for later")
        self.assertEqual(uncertain["from"], "tools_registry")

    def test_repaired_tool_phrase_does_not_recurse(self):
        registry = tr.ToolsRegistry()
        registry.on_heard({"text": "set a reminder for later",
                           "source": "intent_repair"})
        self.assertIsNone(registry.bus.last("picarx/audio/uncertain"))


if __name__ == "__main__":
    unittest.main()
