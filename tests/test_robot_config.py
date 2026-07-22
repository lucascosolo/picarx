import glob
import json
import os
import re
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

    # ---- whole-file edit/save (web console Config page) ----

    def test_merge_and_save_writes_and_reloads(self):
        self._write({"audio": {"gain": 12.0, "espeak_speed": 130}})
        robot_config.merge_and_save({"audio": {"gain": 8.0}})
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 8.0)
        # A round-trip re-read from disk sees it too (not just the cache).
        robot_config.reload()
        self.assertEqual(robot_config.get("audio", "gain", 12.0), 8.0)

    def test_merge_preserves_untouched_keys_and_readme(self):
        self._write({"_readme": ["note"], "audio": {"gain": 12.0, "espeak_speed": 130},
                     "radio": {"tts_settle_sec": 2.0}})
        robot_config.merge_and_save({"audio": {"gain": 5.0}})
        with open(self.path) as f:
            saved = json.load(f)
        self.assertEqual(saved["_readme"], ["note"])          # docs kept
        self.assertEqual(saved["audio"]["espeak_speed"], 130)  # sibling kept
        self.assertEqual(saved["radio"]["tts_settle_sec"], 2.0)  # section kept
        self.assertEqual(saved["audio"]["gain"], 5.0)

    def test_merge_can_add_a_new_section(self):
        self._write({"audio": {"gain": 12.0}})
        robot_config.merge_and_save({"steering": {"cruise_speed": 30}})
        self.assertEqual(robot_config.get("steering", "cruise_speed", 25), 30)

    def test_merge_rejects_editing_readme(self):
        self._write({"_readme": ["note"]})
        with self.assertRaises(ValueError):
            robot_config.merge_and_save({"_readme": ["hacked"]})

    def test_merge_rejects_non_scalar_value(self):
        self._write({"audio": {"gain": 12.0}})
        with self.assertRaises(ValueError):
            robot_config.merge_and_save({"audio": {"gain": {"nested": 1}}})

    def test_merge_rejects_bad_top_shape(self):
        with self.assertRaises(ValueError):
            robot_config.merge_and_save({"audio": 5})

    # ---- runtime materialization (config.json is derived from KNOBS) ----

    def test_sync_defaults_materializes_every_knob(self):
        # A fresh/absent config.json is topped up with every registered knob at
        # its default, so pulling code that adds a knob needs no manual edit.
        added = robot_config.sync_defaults()
        self.assertTrue(os.path.exists(self.path))
        with open(self.path) as f:
            cfg = json.load(f)
        file_knobs = {(sec, key) for sec, kv in cfg.items()
                      if sec != "_readme" and isinstance(kv, dict) for key in kv}
        reg_knobs = {(k["section"], k["key"]) for k in robot_config.KNOBS}
        self.assertEqual(reg_knobs, file_knobs)          # exact 1:1 coverage
        self.assertEqual(set(added), reg_knobs)          # all were newly added

    def test_sync_defaults_values_match_registry(self):
        robot_config.sync_defaults()
        with open(self.path) as f:
            cfg = json.load(f)
        for k in robot_config.KNOBS:
            written = cfg[k["section"]][k["key"]]
            # Path knobs carry config_default=None -> materialized as JSON null.
            expected = k.get("config_default", k["default"]) \
                if "config_default" in k else k["default"]
            self.assertEqual(written, expected,
                             f"{k['section']}.{k['key']} materialized wrongly")

    def test_sync_defaults_preserves_user_values_and_readme(self):
        self._write({"_readme": ["note"], "audio": {"gain": 5.0}})
        robot_config.sync_defaults()
        with open(self.path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["_readme"], ["note"])       # docs preserved
        self.assertEqual(cfg["audio"]["gain"], 5.0)      # user edit preserved
        self.assertEqual(cfg["web_console"]["port"], 8088)  # missing knob added

    def test_sync_defaults_is_idempotent(self):
        robot_config.sync_defaults()
        self.assertEqual(robot_config.sync_defaults(), [])   # nothing left to add


class KnobRegistryTest(unittest.TestCase):
    """The registry is the single source of truth behind the Config page and
    behind config.json (materialized via sync_defaults); these guard it against
    drift so 'every tunable is on the page' stays true."""

    def test_registry_has_no_duplicate_knobs(self):
        keys = [(k["section"], k["key"]) for k in robot_config.KNOBS]
        self.assertEqual(len(keys), len(set(keys)), "duplicate knob in registry")

    def test_every_env_knob_in_source_is_registered(self):
        # Scan every module for `env="NAME"` (the get()/get_bool() overrides)
        # and assert each is a registered knob - so a new env-tunable can never
        # be added without it showing up on the Config page. The Claude API key
        # is read via os.environ.get directly (never env=), so it's not here.
        found = set()
        for path in glob.glob(os.path.join(harness.REPO_ROOT, "**", "*.py"),
                              recursive=True):
            if "/tests/" in path:
                continue
            with open(path) as f:
                found |= set(re.findall(r'env="([A-Z_]+)"', f.read()))
        registered = {k["env"] for k in robot_config.KNOBS if k["env"]}
        self.assertEqual(found - registered, set(),
                         "env-var knob(s) read in source but absent from the registry")

    def test_registry_knob_shapes_are_valid(self):
        for k in robot_config.KNOBS:
            self.assertEqual(set(k) >= {"section", "key", "type", "default",
                                        "env", "desc"}, True)
            self.assertIn(k["type"], ("str", "int", "float", "bool"))
            self.assertTrue(k["desc"], f"{k['section']}.{k['key']} has no help text")


if __name__ == "__main__":
    unittest.main()
