"""The robot's identity (layer_b/identity.py): the one place its name lives.
Continuity of self starts with a name it knows and answers to - so this locks
down that the name is configurable (not a buried constant), fail-soft, and
that it actually reaches the two places that make it real: companion's
personality prompt and the dialog broker's wake phrases."""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import identity  # noqa: E402


class IdentityTest(unittest.TestCase):
    def tearDown(self):
        for var in ("ROBOT_NAME", "ROBOT_PRONOUNS"):
            os.environ.pop(var, None)

    def test_default_name_is_marco(self):
        self.assertEqual(identity.name(), "Marco")

    def test_name_is_a_configurable_knob(self):
        os.environ["ROBOT_NAME"] = "Ada"
        self.assertEqual(identity.name(), "Ada")

    def test_blank_name_falls_back_to_default(self):
        os.environ["ROBOT_NAME"] = "   "
        self.assertEqual(identity.name(), identity.DEFAULT_NAME)

    def test_pronouns_are_configurable(self):
        os.environ["ROBOT_PRONOUNS"] = "they/them"
        self.assertEqual(identity.pronouns(), "they/them")

    def test_self_intro_names_the_robot_in_the_first_person(self):
        intro = identity.self_intro()
        self.assertIn("Marco", intro)
        self.assertIn("he/him", intro)
        # It seeds a first-person voice, not a third-person description.
        self.assertIn("Your name is", intro)


class IdentityReachesPersonalityTest(unittest.TestCase):
    def test_system_prompt_grounds_the_robot_in_its_name(self):
        import companion
        c = companion.Companion.__new__(companion.Companion)
        c.semantic = None  # _self_model_notes is not exercised here
        # Patch out the learned-self read so this test is about identity only.
        c._self_model_notes = lambda: []
        prompt = c._compose_system_prompt()
        self.assertIn("Marco", prompt)
        self.assertTrue(prompt.startswith("Your name is Marco"))


class IdentityReachesAddressingTest(unittest.TestCase):
    def test_the_name_is_a_wake_phrase(self):
        # Reimport dialog so WAKE_PHRASES is rebuilt against the current config.
        import dialog
        importlib.reload(dialog)
        self.assertIn("marco", dialog.WAKE_PHRASES)
        # The generic wake phrases are still present.
        self.assertIn("robot", dialog.WAKE_PHRASES)

    def test_calling_the_name_addresses_the_robot(self):
        import attention
        import dialog
        importlib.reload(dialog)
        a = attention.classify("marco come here", wake_phrases=dialog.WAKE_PHRASES)
        self.assertTrue(a.addressed)
        self.assertEqual(a.reason, attention.WAKE)
        self.assertEqual(a.remainder, "come here")


if __name__ == "__main__":
    unittest.main()
