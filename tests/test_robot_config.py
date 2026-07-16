import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import robot_config  # noqa: E402


class RobotConfigTest(unittest.TestCase):
    def setUp(self):
        self._orig_path = robot_config.CONFIG_PATH
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "config.json")
        robot_config.CONFIG_PATH = self.path
        robot_config.reload()

    def tearDown(self):
        robot_config.CONFIG_PATH = self._orig_path
        robot_config.reload()
        for var in ("TEST_KNOB", "TEST_FLAG"):
            os.environ.pop(var, None)

    def _write(self, obj):
        with open(self.path, "w") as f:
            json.dump(obj, f)
        robot_config.reload()

    # ---- precedence ladder ----

    def test_default_when_file_missing(self):
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 12.0)

    def test_json_value_beats_default(self):
        self._write({"audio": {"gain": 5.5}})
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 5.5)

    def test_env_beats_json(self):
        self._write({"audio": {"gain": 5.5}})
        os.environ["TEST_KNOB"] = "7"
        got = robot_config.get("audio", "gain", 12.0, env="TEST_KNOB")
        self.assertEqual(float(got), 7.0)   # env values arrive as strings

    def test_empty_env_is_unset(self):
        self._write({"audio": {"gain": 5.5}})
        os.environ["TEST_KNOB"] = ""
        self.assertEqual(robot_config.get("audio", "gain", 12.0, env="TEST_KNOB"), 5.5)

    def test_json_null_falls_to_default(self):
        # A key can be listed without pinning it.
        self._write({"audio": {"gain": None}})
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 12.0)

    # ---- fail-soft ----

    def test_corrupt_file_uses_defaults(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        robot_config.reload()
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 12.0)  # no raise

    def test_non_object_top_level_uses_defaults(self):
        with open(self.path, "w") as f:
            f.write("[1, 2, 3]")
        robot_config.reload()
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 12.0)

    def test_wrong_typed_section_uses_defaults(self):
        self._write({"audio": "oops"})
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 12.0)

    # ---- booleans ----

    def test_bool_env_falsy_words(self):
        for word in ("0", "false", "no", "off", "", "False", " OFF "):
            os.environ["TEST_FLAG"] = word
            self.assertFalse(robot_config.get_bool("x", "flag", True, env="TEST_FLAG"),
                             f"env {word!r} should read as False")
        os.environ["TEST_FLAG"] = "1"
        self.assertTrue(robot_config.get_bool("x", "flag", False, env="TEST_FLAG"))

    def test_bool_json_and_default(self):
        self._write({"x": {"flag": True}})
        self.assertTrue(robot_config.get_bool("x", "flag", False))
        self._write({"x": {"flag": False}})
        self.assertFalse(robot_config.get_bool("x", "flag", True))
        self._write({})
        self.assertFalse(robot_config.get_bool("x", "flag", False))

    # ---- the shipped file ----

    def test_shipped_config_parses_and_matches_key_defaults(self):
        shipped = os.path.join(harness.LAYER_B, "config.json")
        with open(shipped) as f:
            cfg = json.load(f)
        self.assertIsInstance(cfg, dict)
        # Spot-check that the shipped file states the same defaults the
        # modules bake in - the file's whole point is being an honest menu.
        self.assertEqual(cfg["audio"]["espeak_voice"], "mb-us1")
        self.assertEqual(cfg["health"]["battery_adc"], False)
        self.assertEqual(cfg["web_console"]["port"], 8088)
        self.assertEqual(cfg["radio"]["alsa_device"], "plug:robot_speaker")


if __name__ == "__main__":
    unittest.main()
