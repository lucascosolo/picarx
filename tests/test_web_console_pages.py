"""Multi-page web console: the server-side helpers behind the new pages -
objects-in-view (Training relabelling), the semantic-memory facts view, the
Config page's tree/help, and the page/asset route table."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import robot_config  # noqa: E402
import web_console  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


class ObjectsSnapshotTest(unittest.TestCase):
    def _world(self, items, stale=False):
        return {"objects": {"items": items, "stale": stale}}

    def test_lists_fresh_tracked_objects_with_ids(self):
        world = self._world([
            {"id": "object_1", "label": "chair", "confidence": 0.82,
             "alt_label": "sofa", "area_ratio": 0.2},
            {"id": "object_2", "label": "bottle", "confidence": 0.6},
        ])
        objs = web_console.objects_snapshot(world)
        self.assertEqual([o["id"] for o in objs], ["object_1", "object_2"])
        self.assertEqual(objs[0]["label"], "chair")
        self.assertEqual(objs[0]["alt_label"], "sofa")

    def test_stale_objects_give_nothing(self):
        self.assertEqual(web_console.objects_snapshot(
            self._world([{"id": "o1", "label": "chair"}], stale=True)), [])

    def test_items_without_id_or_label_are_skipped(self):
        world = self._world([{"label": "chair"}, {"id": "o2"},
                             {"id": "o3", "label": "mug", "confidence": 0.9}])
        objs = web_console.objects_snapshot(world)
        self.assertEqual([o["id"] for o in objs], ["o3"])

    def test_missing_objects_is_safe(self):
        self.assertEqual(web_console.objects_snapshot({}), [])


class FactsSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        store = SemanticStore(readonly=False,
                              db_path=os.path.join(self.tmp, "semantic.db"))
        store.upsert_fact("chair", "I have seen a chair", confidence=0.6)
        store.upsert_fact("kitchen", "the kitchen has a table", confidence=0.7)
        self._orig = web_console.SEMANTIC
        web_console.SEMANTIC = store

    def tearDown(self):
        web_console.SEMANTIC = self._orig

    def test_recent_facts_when_no_query(self):
        snap = web_console.facts_snapshot()
        self.assertEqual(snap["count"], 2)
        subjects = {f["subject"] for f in snap["facts"]}
        self.assertEqual(subjects, {"chair", "kitchen"})

    def test_search_filters(self):
        snap = web_console.facts_snapshot("chair")
        self.assertTrue(all("chair" in (f["subject"] + f["fact"]) for f in snap["facts"]))
        self.assertTrue(snap["facts"])

    def test_failsoft_on_broken_store(self):
        web_console.SEMANTIC = object()   # no methods -> caught
        self.assertEqual(web_console.facts_snapshot(), {"facts": [], "count": 0})


class ConfigDataTest(unittest.TestCase):
    def setUp(self):
        self._orig_path = robot_config.CONFIG_PATH
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "config.json")
        with open(self.path, "w") as f:
            json.dump({"audio": {"gain": 5.0}}, f)   # only one knob pinned
        robot_config.CONFIG_PATH = self.path
        robot_config.reload()
        self._saved_env = {}

    def tearDown(self):
        robot_config.CONFIG_PATH = self._orig_path
        robot_config.reload()
        for name in self._saved_env:
            os.environ.pop(name, None)

    def _knob(self, d, section, key):
        return next(k for k in d["knobs"] if k["section"] == section and k["key"] == key)

    def test_lists_every_registry_knob(self):
        d = web_console.config_data()
        self.assertEqual(len(d["knobs"]), len(robot_config.KNOBS))
        pairs = {(k["section"], k["key"]) for k in d["knobs"]}
        self.assertEqual(pairs, {(k["section"], k["key"]) for k in robot_config.KNOBS})
        self.assertTrue(d["note"])

    def test_file_value_overrides_default_else_default_shown(self):
        d = web_console.config_data()
        self.assertEqual(self._knob(d, "audio", "gain")["value"], 5.0)   # from file
        # A knob the file doesn't pin falls back to the registry default.
        self.assertEqual(self._knob(d, "steering", "cruise_speed")["value"], 25)

    def test_env_override_is_surfaced_and_cleared(self):
        os.environ["ESPEAK_VOICE"] = "mb-en1"
        self._saved_env["ESPEAK_VOICE"] = True
        knob = self._knob(web_console.config_data(), "audio", "espeak_voice")
        self.assertEqual(knob["env_override"], "mb-en1")
        os.environ.pop("ESPEAK_VOICE")
        knob = self._knob(web_console.config_data(), "audio", "espeak_voice")
        self.assertIsNone(knob["env_override"])

    def test_empty_env_is_not_treated_as_override(self):
        os.environ["AUDIO_GAIN"] = ""   # set-but-empty falls through in get()
        self._saved_env["AUDIO_GAIN"] = True
        knob = self._knob(web_console.config_data(), "audio", "gain")
        self.assertIsNone(knob["env_override"])


class RouteTableTest(unittest.TestCase):
    def test_every_page_and_asset_file_exists(self):
        # WEB_UI_DIR is the robot's absolute deploy path; the files live under
        # the repo here, so resolve against that instead.
        web_ui = os.path.join(harness.LAYER_B, "web_ui")
        for fname in list(web_console.PAGES.values()) + \
                [a[0] for a in web_console.ASSETS.values()]:
            self.assertTrue(os.path.exists(os.path.join(web_ui, fname)),
                            f"{fname} referenced by a route but missing on disk")

    def test_core_pages_are_routed(self):
        for path in ("/", "/drive", "/training", "/people", "/audio", "/config"):
            self.assertIn(path, web_console.PAGES)


if __name__ == "__main__":
    unittest.main()
