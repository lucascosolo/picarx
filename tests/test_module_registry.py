import json
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402,F401

from module_registry import load_registry  # noqa: E402


class ModuleRegistryOverlayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = os.path.join(self.tmp.name, "module_registry.json")
        self.local = os.path.join(self.tmp.name, "module_registry.local.json")
        self.defaults = [
            {"name": "field_agent", "entrypoint": "field_agent.py", "enabled": True},
            {"name": "self_trainer", "entrypoint": "self_trainer.py", "enabled": False},
        ]
        with open(self.base, "w", encoding="utf-8") as stream:
            json.dump(self.defaults, stream)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_overlay_keeps_tracked_defaults(self):
        self.assertEqual(load_registry(self.base, self.local), self.defaults)

    def test_compact_overlay_changes_only_local_enablement(self):
        with open(self.local, "w", encoding="utf-8") as stream:
            json.dump({"self_trainer": {"enabled": True}}, stream)
        merged = load_registry(self.base, self.local)
        self.assertTrue(merged[1]["enabled"])
        self.assertEqual(merged[0], self.defaults[0])

    def test_invalid_overlay_fails_soft_to_defaults(self):
        with open(self.local, "w", encoding="utf-8") as stream:
            stream.write("not json")
        self.assertEqual(load_registry(self.base, self.local), self.defaults)

    def test_tracked_self_trainer_is_available_without_local_overlay(self):
        registry = load_registry()
        trainer = next(entry for entry in registry
                       if entry["name"] == "self_trainer")
        self.assertTrue(trainer["enabled"])
        self.assertEqual(trainer["activation"]["mode"], "always")


if __name__ == "__main__":
    unittest.main()
